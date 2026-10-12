"""Transactional turn commit and recovery.

``TurnCommitter`` is the write-path composition root: it drives the
engine update (in-process, deterministic — guardrail L1) and persists
the results atomically in the specified order **event → item
mutations → optional snapshot** inside one multi-document transaction.

Recovery is the inverse: newest valid snapshot (engine-checksummed
``from_bytes``) plus deterministic replay of the event tail through
the wheel's ``EventReplayService``. The engine checksum of the
recovered state must equal the live state's checksum — the
determinism gate is a release blocker, and both recovery modes meet
it.

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
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pymongo import MongoClient
from pymongo.database import Database
from semvec.compliance.event_replay import EventReplayService
from semvec.compliance.event_store import EventStore, MemoryEvent

from semvec_atlas.client_metadata import announce
from semvec_atlas.embedders import ensure_tenant_embedder
from semvec_atlas.event_store import MongoEventStore
from semvec_atlas.firewall import (
    AnchorStore,
    FirewallVerdict,
    QuarantineVault,
    SemanticFirewall,
    redact,
)
from semvec_atlas.guard import TenantGuardViolation, TenantScope
from semvec_atlas.indexes import DOMAIN_COLLECTIONS
from semvec_atlas.item_sync import MemoryItemSync, SyncReport
from semvec_atlas.persistence import AtlasPersistence, SnapshotMeta

__all__ = [
    "DomainWrite",
    "QuarantineInfo",
    "RecoveryInfo",
    "TurnCommitResult",
    "TurnCommitter",
]

_ZERO_NORM_EPS = 1e-8  # mirrors SemvecSession.update_state

QUARANTINE_MARKER = "quarantine"
INOCULATION_MARKER = "inoculation"


@dataclass(frozen=True)
class DomainWrite:
    """One business-record upsert to commit together with the turn.

    ``key`` identifies the record (``{"claim_id": "CL-88371"}``) and
    ``doc`` carries the fields to set. The tenant is added by the guard
    and can be set by neither, so a caller cannot address another
    tenant's record even by supplying the field explicitly.

    The upsert is deliberate: an agent turn that mentions a claim should
    not have to know whether that claim already exists. Fields the write
    does not name survive, so two agents updating different aspects of
    one case do not overwrite each other.
    """

    collection: str
    key: dict[str, Any]
    doc: dict[str, Any]


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
    # Business records written inside the turn's transaction; 0 when the
    # caller passed none.
    domain_writes: int = 0
    # The engine's per-turn mutation feed;
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
        announce(client)
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
        domain_writes: list[DomainWrite] | None = None,
        domain_link: dict[str, Any] | None = None,
    ) -> TurnCommitResult:
        """Engine update, then atomic persistence of the results.

        ``domain_writes`` are business records committed inside the same
        transaction as the memory event, the item mutations and the
        snapshot: a failing domain write rolls the turn back whole, so
        there is no state in which the record exists without the memory
        of the turn that produced it, or the other way round.

        ``domain_link`` stamps identifying fields (``{"claim_id": …}``)
        onto this turn's memory items, which is what lets retrieval join
        a ranked memory hit to the record it concerns in one
        aggregation. It never reaches the engine — replay folds
        ``(embedding, content)`` pairs only, so the link cannot affect
        the recovered state.
        """
        # Validate before the engine runs: a refused write must not leave
        # the in-process state one turn ahead of the database.
        for write in domain_writes or ():
            self._check_domain_write(write)

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

        # Engine first (L1): persistence stores results, never drives them.
        # include_mutations is trajectory-neutral and feeds the
        # item-sync fast path.
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
                domain_link=domain_link,
                force_diff=(
                    self._safety_net_every > 0
                    and turn % self._safety_net_every == 0
                ),
            )
            for write in domain_writes or ():
                self._apply_domain_write(write, mongo_session=ms)
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
            domain_writes=len(domain_writes or ()),
            txn_ms=(time.perf_counter() - txn_start) * 1000.0,
        )

    # ------------------------------------------------------------------
    # domain records

    def _check_domain_write(self, write: DomainWrite) -> None:
        """Refuse anything the turn transaction must not be able to write.

        Two rules, both fail-closed. The collection must be on the
        allowlist, because ``domain_writes`` is otherwise a generic
        upsert primitive that could address the append-only audit log.
        And neither ``key`` nor ``doc`` may carry a scope field: the
        guard owns those, and a caller-supplied ``tenant`` is exactly
        how a cross-tenant write would look.
        """
        if write.collection not in DOMAIN_COLLECTIONS:
            raise TenantGuardViolation(
                f"{write.collection!r} is not a domain collection "
                f"(allowed: {', '.join(DOMAIN_COLLECTIONS)})"
            )
        for field in ("tenant", "agent"):
            if field in write.key or field in write.doc:
                raise TenantGuardViolation(
                    f"a domain write may not set {field!r} — the scope owns it"
                )
        # A key identifies one record; it is not a query. A value like
        # ``{"$exists": True}`` clears the checks above — it names no
        # scope field — and would then match some *other* record of the
        # same tenant and overwrite it.
        for field, value in write.key.items():
            if isinstance(value, (dict, list)):
                raise TenantGuardViolation(
                    f"domain write key {field!r} must be a scalar identifier, "
                    f"got {type(value).__name__} — a key is not a query"
                )

    def _apply_domain_write(self, write: DomainWrite, *, mongo_session: Any) -> None:
        """Upsert one business record inside the turn's transaction.

        Scoped tenant-wide rather than per agent: the specialists of one
        tenant work the same case, so the record must not belong to
        whichever agent happened to touch it first.
        """
        tenant_wide = TenantScope(tenant=self._scope.tenant)
        self._db[write.collection].update_one(
            tenant_wide.filter(dict(write.key)),
            {
                "$set": dict(write.doc),
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
            session=mongo_session,
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

        Two modes, both checksum-identical to the live state (the
        engine's snapshot codec is full-precision; verified by
        tests/test_commit.py):

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
    re-implementing replay semantics here (L1).
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
