"""MongoEventStore.

``MongoEventStore`` subclasses the semvec wheel's ``EventStore`` ABC —
the extension point the package provides for custom backends. On top of
the ABC contract it adds what the Atlas data model requires: a
monotonic per-(tenant, agent) sequence number, gap
detection for replay, and the tenant guard.

The ``user_id`` convention: ``"{tenant}/{agent}"``, exposed
as ``TenantScope.user_id``. A store instance is bound to one scope and
rejects foreign user_ids.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from semvec.compliance.event_store import MemoryEvent

from semvec_atlas.event_store import MongoEventStore
from semvec_atlas.guard import TenantGuardViolation, TenantScope
from semvec_atlas.indexes import provision

pytestmark = pytest.mark.mongo


@pytest.fixture()
def scope():
    return TenantScope(tenant="acme", agent="advisor-01")


@pytest.fixture()
def store(db, scope):
    provision(db, dimension=32, search_indexes=False)
    s = MongoEventStore(db, scope)
    s.init_schema()
    return s


def _event(user_id: str, content: str = "hello", dim: int = 4) -> MemoryEvent:
    return MemoryEvent.fresh(
        user_id=user_id,
        content=content,
        embedding=[0.1] * dim,
        meta={"channel": "test"},
    )


class TestScopeUserId:
    def test_scope_exposes_user_id(self, scope):
        assert scope.user_id == "acme/advisor-01"

    def test_user_id_requires_agent(self):
        with pytest.raises(TenantGuardViolation):
            _ = TenantScope(tenant="acme").user_id


class TestAppend:
    def test_append_assigns_monotonic_seq(self, store, scope, db):
        for _ in range(3):
            store.append(_event(scope.user_id))
        seqs = [d["seq"] for d in db.compliance_events.find(sort=[("seq", 1)])]
        assert seqs == [1, 2, 3]

    def test_append_returns_event_unchanged(self, store, scope):
        ev = _event(scope.user_id)
        out = store.append(ev)
        assert out.event_id == ev.event_id
        assert out.content == ev.content

    def test_append_rejects_foreign_user_id(self, store):
        with pytest.raises(TenantGuardViolation):
            store.append(_event("other-tenant/agent"))

    def test_seq_isolated_per_agent(self, db, scope):
        provision(db, dimension=32, search_indexes=False)
        s1 = MongoEventStore(db, scope)
        s2 = MongoEventStore(db, TenantScope(tenant="acme", agent="advisor-02"))
        s1.append(_event(s1.scope.user_id))
        s2.append(_event(s2.scope.user_id))
        s2.append(_event(s2.scope.user_id))
        assert s1.last_seq() == 1
        assert s2.last_seq() == 2


class TestReadPaths:
    def test_get_roundtrips_event(self, store, scope):
        ev = store.append(_event(scope.user_id, content="fact"))
        got = store.get(ev.event_id)
        assert got is not None
        assert got.event_id == ev.event_id
        assert got.content == "fact"
        assert got.meta == {"channel": "test"}
        assert got.created_at.tzinfo is not None
        assert got.embedding == ev.embedding

    def test_get_unknown_returns_none(self, store):
        assert store.get(uuid.uuid4()) is None

    def test_list_by_user_ordered_by_seq(self, store, scope):
        for i in range(3):
            store.append(_event(scope.user_id, content=f"e{i}"))
        events = store.list_by_user(scope.user_id)
        assert [e.content for e in events] == ["e0", "e1", "e2"]

    def test_count_for_user(self, store, scope):
        assert store.count_for_user(scope.user_id) == 0
        store.append(_event(scope.user_id))
        assert store.count_for_user(scope.user_id) == 1

    def test_query_by_user_limit_returns_most_recent(self, store, scope):
        for i in range(5):
            store.append(_event(scope.user_id, content=f"e{i}"))
        events = store.query_by_user(scope.user_id, query_embedding=None, limit=2)
        assert [e.content for e in events] == ["e3", "e4"]

    def test_query_by_user_embedding_ranks_by_cosine(self, store, scope):
        target = [1.0, 0.0, 0.0, 0.0]
        off = [0.0, 1.0, 0.0, 0.0]
        store.append(
            MemoryEvent.fresh(user_id=scope.user_id, content="match", embedding=target)
        )
        store.append(
            MemoryEvent.fresh(user_id=scope.user_id, content="miss", embedding=off)
        )
        events = store.query_by_user(scope.user_id, query_embedding=target, limit=1)
        assert events[0].content == "match"


class TestDeletePaths:
    def test_delete_by_event_id(self, store, scope):
        ev = store.append(_event(scope.user_id))
        assert store.delete(ev.event_id) is True
        assert store.get(ev.event_id) is None
        assert store.delete(ev.event_id) is False

    def test_delete_by_user(self, store, scope):
        store.append(_event(scope.user_id))
        store.append(_event(scope.user_id))
        assert store.delete_by_user(scope.user_id) == 2
        assert store.count_for_user(scope.user_id) == 0

    def test_delete_older_than(self, store, scope, db):
        old = store.append(_event(scope.user_id, content="old"))
        store.append(_event(scope.user_id, content="new"))
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        db.compliance_events.update_one(
            {"event_id": str(old.event_id)},
            {"$set": {"created_at": cutoff - timedelta(days=29)}},
        )
        assert store.delete_older_than(scope.user_id, cutoff) == 1
        assert [e.content for e in store.list_by_user(scope.user_id)] == ["new"]


class TestAppendOnlyAndGaps:
    def test_store_has_no_update_api(self, store):
        # Append-only by construction: the public surface offers no
        # update/replace method (role separation covers raw access).
        assert not any(
            name in type(store).__dict__ for name in ("update", "replace", "upsert")
        )

    def test_verify_contiguity_detects_gap(self, store, scope, db):
        for _ in range(4):
            store.append(_event(scope.user_id))
        db.compliance_events.delete_one({"seq": 2})
        missing = store.verify_contiguity(scope.user_id)
        assert missing == [2]

    def test_verify_contiguity_ok(self, store, scope):
        for _ in range(3):
            store.append(_event(scope.user_id))
        assert store.verify_contiguity(scope.user_id) == []


class TestEngineIntegration:
    def test_compliance_state_appends_on_update(self, db, scope, embedder):
        from semvec import SemvecConfig
        from semvec.compliance import ComplianceState

        provision(db, dimension=32, search_indexes=False)
        store = MongoEventStore(db, scope)
        state = ComplianceState(
            SemvecConfig(dimension=32),
            event_store=store,
            user_id=scope.user_id,
            # Ownership anchors at the license subject on every layer.
            owner_subject=scope.tenant,
            default_meta={"channel": "chat"},
        )
        state.update(embedder.get_embedding("hello"), "hello")
        events = store.list_by_user(scope.user_id)
        assert len(events) == 1
        assert events[0].content == "hello"
        assert events[0].meta["channel"] == "chat"

    def test_replay_reproduces_engine_checksum(self, db, scope, embedder):
        from semvec import SemvecConfig, SemvecState
        from semvec.compliance import ComplianceState
        from semvec.compliance.event_replay import EventReplayService

        provision(db, dimension=32, search_indexes=False)
        store = MongoEventStore(db, scope)
        state = ComplianceState(
            SemvecConfig(dimension=32),
            event_store=store,
            user_id=scope.user_id,
            owner_subject=scope.tenant,
        )
        for i in range(4):
            state.update(embedder.get_embedding(f"turn {i}"), f"turn {i}")

        fresh = SemvecState(SemvecConfig(dimension=32))
        replayed = EventReplayService(store=store).replay(fresh, user_id=scope.user_id)
        assert replayed == 4
        # The engine checksum is the determinism contract.
        assert fresh.to_dict()["checksum"] == state.to_dict()["checksum"]


class TestAppendInSession:
    """The 0.8.8 sibling contract: batch append inside a caller's
    transaction (``MongoStateStore.save(events=...)`` calls exactly
    this), drawing the same per-scope sequence as ``append``."""

    def test_batch_append_assigns_contiguous_sequences(self, store, scope):
        written = store.append_in_session(
            [_event(scope.user_id, content=f"e{i}") for i in range(3)],
            mongo_session=None,
        )
        assert [e.sequence for e in written] == [1, 2, 3]
        assert store.last_appended_seq == 3
        assert [e.content for e in store.list_by_user(scope.user_id)] == [
            "e0", "e1", "e2",
        ]

    def test_sequences_continue_after_single_appends(self, store, scope):
        store.append(_event(scope.user_id, content="solo"))
        written = store.append_in_session(
            [_event(scope.user_id, content="batch")], mongo_session=None
        )
        assert written[0].sequence == 2

    def test_foreign_user_id_refused_before_any_write(self, store, scope, db):
        events = [
            _event(scope.user_id, content="ok"),
            _event("evil/agent", content="smuggled"),
        ]
        with pytest.raises(TenantGuardViolation):
            store.append_in_session(events, mongo_session=None)
        assert db.compliance_events.count_documents({}) == 0

    def test_enlists_in_a_real_transaction(self, store, scope, db):
        client = db.client
        with client.start_session() as ms, ms.start_transaction():
            store.append_in_session(
                [_event(scope.user_id, content="txn")], mongo_session=ms
            )
        events = store.list_by_user(scope.user_id)
        assert [e.content for e in events] == ["txn"]
