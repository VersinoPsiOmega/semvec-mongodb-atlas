"""Transactional turn commit and recovery.

``TurnCommitter`` is the write-path composition root: it drives the
engine update (in-process, deterministic) and persists
the results atomically in the specified order **event → item
mutations → optional snapshot** inside one multi-document transaction.

Recovery is the inverse: newest valid snapshot (engine-checksummed
``from_bytes``) plus deterministic replay of the event tail through
the wheel's ``EventReplayService``. The engine checksum of the
recovered state must equal the live state's checksum — the
determinism gate is a release blocker. Since the engine's
full-precision codec (semvec >= 0.8.2) both recovery modes meet this gate.

Replay fidelity note (verified against the wheel): replay folds
``(embedding, content)`` pairs only. The committer therefore calls
``update()`` *without* per-memory meta — session/turn tagging lives in
the persistence layer (item documents carry ``ns``/``turn_added``),
keeping replay checksum-faithful by construction.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from pymongo import MongoClient
from pymongo.database import Database
from semvec.compliance.event_replay import EventReplayService
from semvec.compliance.event_store import EventStore, MemoryEvent

from semvec_atlas.embedders import ensure_tenant_embedder
from semvec_atlas.event_store import MongoEventStore
from semvec_atlas.firewall import (
    AnchorStore,
    FirewallVerdict,
    QuarantineVault,
    SemanticFirewall,
    redact,
)
from semvec_atlas.guard import TenantScope
from semvec_atlas.item_sync import MemoryItemSync, SyncReport
from semvec_atlas.persistence import AtlasPersistence, SnapshotMeta

__all__ = ["QuarantineInfo", "RecoveryInfo", "TurnCommitResult", "TurnCommitter"]

_ZERO_NORM_EPS = 1e-8  # mirrors SemvecSession.update_state

QUARANTINE_MARKER = "quarantine"
INOCULATION_MARKER = "inoculation"


@dataclass(frozen=True)
class QuarantineInfo:
    """A firewall block: why, and where the held input lives.

    ``quarantine_id`` is the vault entry (level ``quarantine``);
    ``None`` when the level dropped the input without holding it
    (``filter``/``lockdown`` — audited via the marker event only).
    ``anchor_id`` is the tenant-shared anchor learned from this block
    (``inoculate=True``), ``None`` otherwise.
    """

    quarantine_id: str | None
    verdict: FirewallVerdict
    anchor_id: str | None = None


@dataclass(frozen=True)
class TurnCommitResult:
    """Outcome of one committed turn.

    ``metrics is None`` means nothing reached the engine: either the
    input was zero-norm (the wheel's None-means-skip contract) or the
    firewall held it (then ``quarantine`` says why).
    """

    metrics: dict[str, Any] | None
    event_seq: int | None = None
    sync: SyncReport | None = None
    snapshot: SnapshotMeta | None = None
    # The engine's per-turn mutation feed (semvec >= 0.8.5);
    # None when the engine skipped the input.
    mutations: list[dict[str, Any]] | None = None
    quarantine: QuarantineInfo | None = None
    # Wall-clock of the commit transaction alone (event → items →
    # snapshot), excluding embedding and firewall work; ``None`` when
    # nothing was committed (zero-norm skip or firewall hold).
    txn_ms: float | None = None


@dataclass(frozen=True)
class RecoveryInfo:
    snapshot_turn: int | None
    replayed_events: int


class TurnCommitter:
    """Write path bound to one :class:`TenantScope`.

    ``snapshot_every=N`` persists a snapshot on every Nth turn;
    ``0`` disables periodic snapshots (recovery then replays the full
    log). ``snapshot_retention`` keeps the newest N snapshots per
    session.
    """

    def __init__(
        self,
        client: MongoClient,
        db: Database,
        scope: TenantScope,
        *,
        embedder: Any,
        snapshot_every: int = 1,
        snapshot_retention: int | None = None,
        safety_net_every: int = 25,
        firewall: SemanticFirewall | None = None,
        inoculate: bool = False,
    ) -> None:
        self._client = client
        self._db = db
        self._scope = scope
        self._embedder = embedder
        # Fail-closed embedder identity check (see embedders module):
        # a silently switched model would corrupt similarity search.
        ensure_tenant_embedder(db, scope, embedder)
        self._snapshot_every = snapshot_every
        # Periodic full tier diff even when the feed reports quiet
        # turns — consistency safety net for the feed fast path.
        self._safety_net_every = safety_net_every
        self._event_store = MongoEventStore(db, scope)
        self._item_sync = MemoryItemSync(db, scope)
        self._persistence = AtlasPersistence(
            db, scope, snapshot_retention=snapshot_retention
        )
        # Optional ingest gate. ``inoculate`` additionally
        # persists each block's embedding as a tenant-shared anchor.
        self._firewall = firewall
        self._quarantine = QuarantineVault(db, scope) if firewall else None
        self._anchors = AnchorStore(db, scope) if firewall and inoculate else None

    @property
    def scope(self) -> TenantScope:
        return self._scope

    @property
    def event_store(self) -> MongoEventStore:
        return self._event_store

    # ------------------------------------------------------------------
    # write path

    def run_turn(
        self,
        state: Any,
        *,
        session: str,
        turn: int,
        text: str,
        embedding: np.ndarray | None = None,
        event_meta: dict[str, Any] | None = None,
    ) -> TurnCommitResult:
        """Engine update, then atomic persistence of the results."""
        if embedding is None:
            embedding = np.asarray(
                self._embedder.get_embedding(text), dtype=np.float64
            )
        else:
            embedding = np.asarray(embedding, dtype=np.float64)
        if float(np.linalg.norm(embedding)) < _ZERO_NORM_EPS:
            return TurnCommitResult(metrics=None)

        # Ingest gate: a held input reaches neither the engine
        # nor the memory event log — replay determinism by construction.
        if self._firewall is not None:
            verdict = self._firewall.inspect(text, embedding=embedding)
            if not verdict.allowed:
                return self._hold_turn(
                    text, embedding, verdict, session=session, turn=turn
                )

        # Engine first: persistence stores results, never drives them.
        # include_mutations is trajectory-neutral (verified up to n=419)
        # and feeds the item-sync fast path.
        metrics = state.update(embedding, text, include_mutations=True)
        mutations = metrics.get("mutations")

        meta = {"session": session, "turn": turn}
        if event_meta:
            meta.update(event_meta)
        event = MemoryEvent.fresh(
            user_id=self._scope.user_id,
            content=text,
            embedding=[float(x) for x in embedding],
            meta=meta,
        )

        txn_start = time.perf_counter()
        with self._client.start_session() as ms, ms.start_transaction():
            self._event_store.append(event, mongo_session=ms)
            seq = self._event_store.last_appended_seq
            sync = self._item_sync.sync(
                state,
                session=session,
                turn=turn,
                mongo_session=ms,
                mutations=mutations,
                force_diff=(
                    self._safety_net_every > 0
                    and turn % self._safety_net_every == 0
                ),
            )
            snapshot: SnapshotMeta | None = None
            if self._snapshot_every > 0 and turn % self._snapshot_every == 0:
                snapshot = self._persistence.save_snapshot(
                    state,
                    session=session,
                    turn=turn,
                    last_event_seq=seq,
                    mongo_session=ms,
                )
        return TurnCommitResult(
            metrics=metrics,
            event_seq=seq,
            sync=sync,
            snapshot=snapshot,
            mutations=mutations,
            txn_ms=(time.perf_counter() - txn_start) * 1000.0,
        )

    def _hold_turn(
        self,
        text: str,
        embedding: np.ndarray,
        verdict: FirewallVerdict,
        *,
        session: str,
        turn: int,
    ) -> TurnCommitResult:
        """Persist a firewall block: vault hold + content-free marker.

        The marker event mirrors the forget marker: it proves
        *that* something was held without carrying *what* — preview is
        redaction-capped, correlation runs over the SHA-256 hash. Like
        all marker events it is excluded from replay.
        """
        preview, digest = redact(text)
        marker_meta = {
            "semvec_atlas": {
                "marker": QUARANTINE_MARKER,
                "text_preview": preview,
                "text_hash": digest,
                "score": verdict.score,
                "reason": verdict.reason,
                "level": verdict.level,
                "session": session,
                "turn": turn,
            }
        }
        with self._client.start_session() as ms, ms.start_transaction():
            quarantine_id: str | None = None
            # Only level "quarantine" holds for review; "filter" and
            # "lockdown" drop the input (audited via the marker alone).
            if verdict.level == "quarantine":
                quarantine_id = self._quarantine.hold(
                    text,
                    embedding,
                    verdict,
                    session=session,
                    turn=turn,
                    mongo_session=ms,
                )
            anchor_id: str | None = None
            if self._anchors is not None:
                anchor_id = self._anchors.add(
                    embedding, kind="learned", text_hash=digest, mongo_session=ms
                )
            self._event_store.append(
                MemoryEvent.fresh(
                    user_id=self._scope.user_id,
                    content="",  # content-free by design
                    embedding=[],
                    meta=marker_meta,
                ),
                mongo_session=ms,
            )
            if anchor_id is not None:
                # The learning event itself joins the audit chain: from
                # this moment every agent of the tenant is immune to
                # near-variants (inoculation, recorded as a marker).
                self._event_store.append(
                    MemoryEvent.fresh(
                        user_id=self._scope.user_id,
                        content="",  # content-free by design
                        embedding=[],
                        meta={
                            "semvec_atlas": {
                                "marker": INOCULATION_MARKER,
                                "anchor_id": anchor_id,
                                "text_hash": digest,
                                "session": session,
                                "turn": turn,
                            }
                        },
                    ),
                    mongo_session=ms,
                )
        if self._anchors is not None and self._firewall.anchor_store is None:
            # No live store view: inoculate the in-process firewall so
            # the very next turn of this committer is already covered.
            # (With a live anchor_store the persisted anchor is seen on
            # the next inspection anyway — adding it here would double-
            # count and, worse, survive a release un-learn.)
            self._firewall.add_exclusion_vector(embedding)
        return TurnCommitResult(
            metrics=None,
            quarantine=QuarantineInfo(
                quarantine_id=quarantine_id, verdict=verdict, anchor_id=anchor_id
            ),
        )

    # ------------------------------------------------------------------
    # recovery

    def recover(
        self, *, session: str, mode: str = "replay"
    ) -> tuple[Any, RecoveryInfo]:
        """Rebuild the engine state for ``session``.

        Two modes, both checksum-identical to the live state since the
        engine's full-precision codec (semvec >= 0.8.2 —
        verified by tests/test_commit.py):

        * ``"replay"`` (default) — full event-log replay into a fresh
          state. The determinism gate; version-independent (the
          log is the source of truth, so it also covers snapshots
          written by older engine codecs).
        * ``"snapshot_tail"`` — newest valid snapshot + tail replay.
          The fast path; requires the snapshot codec to be readable by
          the running engine (see ``SnapshotMeta.codec_version``).

        A gapped event log (tampering / partial deletion outside the
        forget path) refuses recovery instead of certifying a wrong
        state.
        """
        if mode not in ("replay", "snapshot_tail"):
            raise ValueError(f"unknown recovery mode {mode!r}")
        gaps = self._event_store.verify_contiguity(self._scope.user_id)
        if gaps:
            raise RuntimeError(
                f"event log has sequence gap(s) {gaps} for {self._scope.user_id!r} — "
                "refusing replay; investigate before recovery"
            )

        state = None
        after_seq = 0
        snapshot_turn: int | None = None
        if mode == "snapshot_tail":
            loaded = self._persistence.load_latest_snapshot(session=session)
            if loaded is not None:
                state, meta = loaded
                after_seq = meta.last_event_seq or 0
                snapshot_turn = meta.turn

        if state is None:
            from semvec import SemvecConfig, SemvecState

            state = SemvecState(
                SemvecConfig(dimension=self._embedder.get_dimension())
            )

        tail = _TailView(self._event_store, after_seq=after_seq)
        replayed = EventReplayService(store=tail).replay(
            state, user_id=self._scope.user_id
        )
        return state, RecoveryInfo(
            snapshot_turn=snapshot_turn, replayed_events=replayed
        )


class _TailView(EventStore):
    """Read-only ``EventStore`` view of events with ``seq > after_seq``.

    Lets the wheel's ``EventReplayService`` (which replays a *full*
    ``list_by_user``) resume from a snapshot boundary without
    re-implementing replay semantics here.
    """

    def __init__(self, inner: MongoEventStore, *, after_seq: int) -> None:
        self._inner = inner
        self._after_seq = after_seq

    def init_schema(self) -> None:  # pragma: no cover - delegation only
        self._inner.init_schema()

    def append(self, event: MemoryEvent) -> MemoryEvent:
        raise RuntimeError("_TailView is read-only")

    def get(self, event_id: uuid.UUID) -> MemoryEvent | None:
        return self._inner.get(event_id)

    def list_by_user(self, user_id: str) -> list[MemoryEvent]:
        events = self._inner.list_by_user(user_id)
        # Forget markers are bookkeeping, not content: content-free by
        # design, they must never be folded into a
        # replayed state.
        events = [
            e for e in events if not (e.meta or {}).get("semvec_atlas", {}).get("marker")
        ]
        if self._after_seq <= 0:
            return events
        seqs = self._inner.seq_index(user_id)
        return [e for e in events if seqs.get(str(e.event_id), 0) > self._after_seq]

    def query_by_user(
        self, user_id: str, *, query_embedding: list[float] | None, limit: int
    ) -> list[MemoryEvent]:  # pragma: no cover - delegation only
        return self._inner.query_by_user(
            user_id, query_embedding=query_embedding, limit=limit
        )

    def count_for_user(self, user_id: str) -> int:  # pragma: no cover
        return len(self.list_by_user(user_id))

    def delete(self, event_id: uuid.UUID) -> bool:
        raise RuntimeError("_TailView is read-only")

    def delete_by_user(self, user_id: str) -> int:
        raise RuntimeError("_TailView is read-only")

    def delete_older_than(self, user_id: str, cutoff: datetime) -> int:
        raise RuntimeError("_TailView is read-only")
