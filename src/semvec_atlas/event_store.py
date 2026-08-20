"""MongoDB implementation of the semvec ``EventStore`` ABC.

The wheel ships ``SqliteEventStore`` as reference; this class is the
Atlas-backed sibling and adds what the specification's data model
requires on top of the ABC contract:

* a **monotonic sequence** per (tenant, agent), assigned atomically via
  a ``$inc`` counter document (``MemoryEvent`` itself carries no
  seq);
* **gap detection** for replay (:meth:`verify_contiguity`);
* the **tenant guard** — a store instance is bound to one
  :class:`TenantScope` and rejects foreign ``user_id``s;
* the 0.8.8 sibling contract :meth:`append_in_session`, so the store
  plugs into ``MongoStateStore.save(..., events=...)`` — the wheel's
  atomic state+events turn commit — without losing this layout.

Append-only: the public surface has no update/replace path. Deletions
exist only because the ABC's forget/retention contract requires them —
on Atlas they run under a separate worker role.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from bson.codec_options import CodecOptions
from pymongo.database import Database
from semvec.compliance.event_store import EventStore, MemoryEvent

from semvec_atlas.guard import TenantGuardViolation, TenantScope

__all__ = ["MongoEventStore"]

_COUNTERS = "counters"


class MongoEventStore(EventStore):
    """Append-only event store on MongoDB, bound to one tenant scope."""

    def __init__(self, db: Database, scope: TenantScope) -> None:
        if scope.agent is None:
            raise TenantGuardViolation("MongoEventStore requires an agent-bound scope")
        self._db = db
        self.scope = scope
        opts: CodecOptions = CodecOptions(tz_aware=True, tzinfo=timezone.utc)
        self._col = db.get_collection("compliance_events", codec_options=opts)
        self._counters = db.get_collection(_COUNTERS)

    # ------------------------------------------------------------------
    # helpers

    def _require_scope(self, user_id: str) -> None:
        if user_id != self.scope.user_id:
            raise TenantGuardViolation(
                f"event user_id {user_id!r} does not match store scope "
                f"{self.scope.user_id!r}"
            )

    def _next_seq(self, mongo_session: Any = None) -> int:
        doc = self._counters.find_one_and_update(
            {"_id": f"evt:{self.scope.user_id}"},
            {"$inc": {"value": 1}},
            upsert=True,
            return_document=True,
            session=mongo_session,
        )
        return int(doc["value"])

    def last_seq(self) -> int:
        """Highest assigned sequence for this scope (0 when empty)."""
        doc = self._col.find_one(
            self.scope.filter(), sort=[("seq", -1)], projection={"seq": 1}
        )
        return int(doc["seq"]) if doc else 0

    def _to_doc(self, event: MemoryEvent, seq: int) -> dict[str, Any]:
        return self.scope.filter(
            {
                "seq": seq,
                "event_id": str(event.event_id),
                "created_at": event.created_at,
                "content": event.content,
                "embedding": list(event.embedding),
                "meta": dict(event.meta),
                "source_event_id": (
                    str(event.source_event_id) if event.source_event_id else None
                ),
            }
        )

    @staticmethod
    def _to_event(doc: dict[str, Any]) -> MemoryEvent:
        created = doc["created_at"]
        if created.tzinfo is None:  # defensive: raw inserts without codec opts
            created = created.replace(tzinfo=timezone.utc)
        sid = doc.get("source_event_id")
        return MemoryEvent(
            event_id=uuid.UUID(doc["event_id"]),
            user_id=f"{doc['tenant']}/{doc['agent']}",
            created_at=created,
            content=doc["content"],
            embedding=list(doc["embedding"]),
            meta=dict(doc["meta"]),
            source_event_id=uuid.UUID(sid) if sid else None,
        )

    # ------------------------------------------------------------------
    # EventStore ABC

    def init_schema(self) -> None:
        self._col.create_index(
            [("tenant", 1), ("agent", 1), ("seq", 1)], name="ix_evt", unique=True
        )
        self._col.create_index([("event_id", 1)], name="ix_evt_id", unique=True)

    def append(self, event: MemoryEvent, *, mongo_session: Any = None) -> MemoryEvent:
        self._require_scope(event.user_id)
        seq = self._next_seq(mongo_session)
        self._col.insert_one(self._to_doc(event, seq), session=mongo_session)
        # The ABC returns the (immutable) MemoryEvent, which carries no
        # seq — expose the assigned value for the committer.
        self.last_appended_seq = seq
        return event

    def append_in_session(
        self, events: list[MemoryEvent], *, mongo_session: Any
    ) -> list[MemoryEvent]:
        """Persist several events inside a caller's Mongo transaction, in order.

        The sibling contract semvec 0.8.8 introduced for atomic turn
        commit: ``MongoStateStore.save(..., events=...)`` hands its
        transaction session here so the audit rows and the state doc
        land — or roll back — together. Every event draws its seq from
        the same per-scope ``$inc`` counter :meth:`append` uses,
        enlisted in ``mongo_session`` so an aborted commit leaves no
        gap. Returns the events carrying their assigned sequences
        (frozen, so replaced rather than mutated), oldest first.

        The tenant guard runs over the whole batch *before* the first
        write: outside a transaction a mid-batch refusal would leave a
        partial batch behind, and a guard that only sometimes protects
        is not a guard.
        """
        from dataclasses import replace

        for event in events:
            self._require_scope(event.user_id)
        written: list[MemoryEvent] = []
        for event in events:
            seq = self._next_seq(mongo_session)
            self._col.insert_one(self._to_doc(event, seq), session=mongo_session)
            self.last_appended_seq = seq
            written.append(replace(event, sequence=seq))
        return written

    def seq_index(self, user_id: str) -> dict[str, int]:
        """Map ``event_id -> seq`` for this scope (replay boundaries)."""
        self._require_scope(user_id)
        return {
            d["event_id"]: int(d["seq"])
            for d in self._col.find(
                self.scope.filter(), projection={"event_id": 1, "seq": 1}
            )
        }

    def get(self, event_id: uuid.UUID) -> MemoryEvent | None:
        doc = self._col.find_one(self.scope.filter({"event_id": str(event_id)}))
        return self._to_event(doc) if doc else None

    def list_by_user(self, user_id: str) -> list[MemoryEvent]:
        self._require_scope(user_id)
        return [
            self._to_event(d)
            for d in self._col.find(self.scope.filter(), sort=[("seq", 1)])
        ]

    def query_by_user(
        self,
        user_id: str,
        *,
        query_embedding: list[float] | None,
        limit: int,
    ) -> list[MemoryEvent]:
        self._require_scope(user_id)
        if query_embedding is None:
            newest_first = list(
                self._col.find(self.scope.filter(), sort=[("seq", -1)], limit=limit)
            )
            return [self._to_event(d) for d in reversed(newest_first)]

        # Cosine ranking, computed client-side: the event store is the
        # compliance log, not the retrieval path — hybrid retrieval over
        # memory_items has its own engine. Volumes here are
        # bounded per (tenant, agent).
        import numpy as np

        q = np.asarray(query_embedding, dtype=float)
        qn = np.linalg.norm(q)
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self._col.find(self.scope.filter()):
            v = np.asarray(doc["embedding"], dtype=float)
            vn = np.linalg.norm(v)
            score = float(q @ v / (qn * vn)) if qn > 0 and vn > 0 else -1.0
            scored.append((score, doc))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [self._to_event(d) for _, d in scored[:limit]]

    def count_for_user(self, user_id: str) -> int:
        self._require_scope(user_id)
        return self._col.count_documents(self.scope.filter())

    def delete(self, event_id: uuid.UUID) -> bool:
        res = self._col.delete_one(self.scope.filter({"event_id": str(event_id)}))
        return res.deleted_count == 1

    def delete_by_user(self, user_id: str) -> int:
        self._require_scope(user_id)
        return self._col.delete_many(self.scope.filter()).deleted_count

    def delete_older_than(self, user_id: str, cutoff: datetime) -> int:
        self._require_scope(user_id)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        return self._col.delete_many(
            self.scope.filter({"created_at": {"$lt": cutoff}})
        ).deleted_count

    # ------------------------------------------------------------------
    # Atlas-specific surface

    def verify_contiguity(self, user_id: str) -> list[int]:
        """Return missing sequence numbers (gap detection for replay).

        A non-empty result means the log was tampered with or partially
        deleted outside the forget path — replay must refuse to certify
        determinism until the gap is explained (a forget event records
        its own deletions, so legitimate forgets never leave silent
        gaps).
        """
        self._require_scope(user_id)
        seqs = [d["seq"] for d in self._col.find(self.scope.filter(), sort=[("seq", 1)])]
        if not seqs:
            return []
        expected = range(seqs[0], seqs[-1] + 1)
        present = set(seqs)
        return [s for s in expected if s not in present]
