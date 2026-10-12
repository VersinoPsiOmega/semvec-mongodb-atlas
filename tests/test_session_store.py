"""MongoStateStore.

Duck-type sibling of the wheel's SQL ``StateStore`` and
``RedisStateStore`` (0.8.8 surface: ``save`` with ``scope``/``events``
+ ``pending_message``, ``load_full`` → ``LoadedState``,
``current_version``), so a ``semvec serve`` deployment can select
MongoDB as its durable session backend. Contract points mirrored, not
reinvented:

* CAS semantics — ``expected_version=None`` inserts at version 1 and
  refuses to overwrite; a mismatched version raises the wheel's own
  ``StateVersionConflict`` (**reused, not forked** — callers catch one
  exception type regardless of backend).
* Ownership — ``scope=OwnerScope(...)`` is written on save, scopes the
  CAS (a cross-scope update is refused as a conflict, never a silent
  overwrite), and filters ``load_full(scope=...)``.
* Atomic turn commit — ``events=[...]`` writes the state doc and the
  audit rows in one multi-document transaction through the store's
  event-writer seam; semvec-atlas wires its own tenant-scoped
  ``MongoEventStore`` into that seam so the rows keep the Atlas layout
  (literal ``tenant``/``agent`` columns, per-scope ``seq``) and pass
  the ``compliance_events`` validator.
* ``serialize_state``/``deserialize_state`` stay wheel functions: the
  blob is identical bytes on every backend, so sessions migrate
  losslessly between SQL, Redis, and MongoDB stores.
* ``drift_exempt`` is owned by its setter and survives blob writes.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
from semvec.api.mongo_state_store import SnapshotIntegrityError
from semvec.api.owner_scope import OwnerScope
from semvec.api.state_store import LoadedState, StateVersionConflict
from semvec.compliance.event_store import MemoryEvent

from semvec_atlas.event_store import MongoEventStore
from semvec_atlas.guard import TenantScope
from semvec_atlas.indexes import provision
from semvec_atlas.session_store import MongoStateStore

pytestmark = pytest.mark.mongo


@pytest.fixture()
def store(db):
    return MongoStateStore(db)


class TestSaveCas:
    def test_insert_at_version_1(self, store):
        assert store.save("s1", b"blob-a", expected_version=None) == 1

    def test_insert_refuses_existing_row(self, store):
        store.save("s1", b"blob-a", expected_version=None)
        with pytest.raises(StateVersionConflict):
            store.save("s1", b"blob-b", expected_version=None)

    def test_update_bumps_version_on_match(self, store):
        store.save("s1", b"v1", expected_version=None)
        assert store.save("s1", b"v2", expected_version=1) == 2
        assert store.load("s1") == (b"v2", 2)

    def test_update_conflicts_on_stale_version(self, store):
        store.save("s1", b"v1", expected_version=None)
        store.save("s1", b"v2", expected_version=1)
        with pytest.raises(StateVersionConflict):
            store.save("s1", b"v2-stale", expected_version=1)

    def test_update_missing_row_conflicts(self, store):
        with pytest.raises(StateVersionConflict):
            store.save("ghost", b"x", expected_version=3)


class TestLoadSurface:
    def test_load_returns_blob_and_version(self, store):
        store.save("s1", b"data", expected_version=None)
        assert store.load("s1") == (b"data", 1)

    def test_load_absent_returns_none(self, store):
        assert store.load("nope") is None

    def test_load_full_returns_loaded_state(self, store):
        store.save(
            "s1", b"data", expected_version=None,
            scope=OwnerScope("alice"), pending_message="hello",
        )
        full = store.load_full("s1")
        assert isinstance(full, LoadedState)
        assert full.blob == b"data"
        assert full.version == 1
        assert full.owner_subject == "alice"
        assert full.pending_message == "hello"

    def test_load_owner_and_current_version(self, store):
        store.save("s1", b"data", expected_version=None, scope=OwnerScope("alice"))
        assert store.load_owner("s1") == "alice"
        assert store.current_version("s1") == 1
        assert store.current_version("ghost") is None
        assert store.load_owner("ghost") is None

    def test_owner_refreshed_on_update(self, store):
        store.save("s1", b"v1", expected_version=None, scope=OwnerScope("alice"))
        store.save("s1", b"v2", expected_version=1, scope=OwnerScope("alice"))
        assert store.load_owner("s1") == "alice"


class TestOwnerScopeIsolation:
    """The 0.8.8 scoped-CAS surface, inherited and live on MongoDB."""

    def test_cross_scope_update_is_refused_as_conflict(self, store):
        store.save("s1", b"v1", expected_version=None, scope=OwnerScope("alice"))
        with pytest.raises(StateVersionConflict):
            store.save("s1", b"v2", expected_version=1, scope=OwnerScope("mallory"))
        assert store.load("s1") == (b"v1", 1)

    def test_load_full_under_foreign_scope_reads_none(self, store):
        store.save("s1", b"v1", expected_version=None, scope=OwnerScope("alice"))
        assert store.load_full("s1", scope=OwnerScope("mallory")) is None
        assert store.load_full("s1", scope=OwnerScope("alice")) is not None
        assert store.load_full("s1") is not None  # unscoped read stays permitted

    def test_unowned_row_may_be_adopted(self, store):
        store.save("s1", b"v1", expected_version=None)  # no owner
        store.save("s1", b"v2", expected_version=1, scope=OwnerScope("alice"))
        assert store.load_owner("s1") == "alice"


class TestBlobIntegrity:
    def test_tampered_blob_is_refused_on_load(self, store, db):
        store.save("s1", b"honest bytes", expected_version=None)
        db.semvec_session_state.update_one(
            {"_id": "s1"}, {"$set": {"blob": b"tampered bytes"}}
        )
        with pytest.raises(SnapshotIntegrityError):
            store.load_full("s1")


class TestAtomicEventsCommit:
    """``save(events=...)`` through the wired Atlas event store.

    The upstream store builds its event writer lazily from
    ``SEMVEC_MONGO_URI``; semvec-atlas composes from a ``Database``
    handle instead, so the subclass accepts the tenant-scoped
    ``MongoEventStore`` at construction and pre-populates the seam.
    State doc and audit rows then land — or roll back — together.
    """

    @pytest.fixture()
    def scope(self):
        return TenantScope(tenant="acme", agent="advisor-01")

    @pytest.fixture()
    def wired(self, db, scope):
        provision(db, dimension=4, search_indexes=False)
        events = MongoEventStore(db, scope)
        events.init_schema()
        return MongoStateStore(db, event_store=events), events, scope

    @staticmethod
    def _event(scope: TenantScope, content: str = "hello") -> MemoryEvent:
        return MemoryEvent.fresh(
            user_id=scope.user_id,
            content=content,
            embedding=[0.1, 0.2, 0.3, 0.4],
            meta={"channel": "test"},
        )

    def test_state_and_events_commit_together(self, wired, db):
        store, _, scope = wired
        version = store.save(
            "s1",
            b"turn-1",
            expected_version=None,
            scope=OwnerScope("acme", "advisor-01"),
            events=[self._event(scope)],
        )
        assert version == 1
        rows = list(db.compliance_events.find({"tenant": "acme", "agent": "advisor-01"}))
        assert len(rows) == 1
        assert rows[0]["seq"] == 1
        assert rows[0]["content"] == "hello"

    def test_lost_cas_rolls_back_the_events(self, wired, db):
        store, _, scope = wired
        store.save("s1", b"v1", expected_version=None)
        with pytest.raises(StateVersionConflict):
            store.save(
                "s1",
                b"v2",
                expected_version=99,
                scope=OwnerScope("acme", "advisor-01"),
                events=[self._event(scope, content="rolled back")],
            )
        assert db.compliance_events.count_documents({}) == 0
        assert store.load("s1") == (b"v1", 1)

    def test_event_sequences_continue_across_commits(self, wired, db):
        store, events, scope = wired
        events.append(self._event(scope, content="standalone"))
        store.save(
            "s1",
            b"turn-2",
            expected_version=None,
            scope=OwnerScope("acme", "advisor-01"),
            events=[self._event(scope, content="in txn")],
        )
        seqs = sorted(d["seq"] for d in db.compliance_events.find({}))
        assert seqs == [1, 2]


class TestDeleteAndDriftExempt:
    def test_delete_returns_whether_removed(self, store):
        store.save("s1", b"x", expected_version=None)
        assert store.delete("s1") is True
        assert store.delete("s1") is False
        assert store.load("s1") is None

    def test_drift_exempt_defaults_false_and_survives_blob_writes(self, store):
        store.save("s1", b"v1", expected_version=None)
        assert store.is_drift_exempt("s1") is False
        store.set_drift_exempt("s1", True)
        store.save("s1", b"v2", expected_version=1)  # CAS write
        assert store.is_drift_exempt("s1") is True

    def test_drift_exempt_absent_row_is_false(self, store):
        assert store.is_drift_exempt("ghost") is False


class TestBlobMigrationParity:
    def test_wheel_serialize_roundtrip_through_mongo(self, store, fresh_state, embedder):
        # The blob is the wheel's own codec — identical bytes on every
        # backend, so a session can migrate SQL <-> Redis <-> MongoDB.
        from semvec.api.state_store import deserialize_state, serialize_state

        state = fresh_state()
        state.update(embedder.get_embedding("fact"), "fact")
        blob = serialize_state(state)
        store.save("s1", blob, expected_version=None)
        loaded, version = store.load("s1")
        assert loaded == blob
        restored = deserialize_state(loaded)
        assert restored.to_dict()["checksum"] == state.to_dict()["checksum"]

class TestUpstreamLineage:
    """One source of truth for CAS semantics.

    The wheel ships ``semvec.api.mongo_state_store``; the
    semvec-atlas store must be a *subclass* of it — CAS/save/load logic
    lives upstream, this package only adds the ``drift_exempt`` pair
    (which the upstream class deliberately omits) and the
    Database-object constructor used throughout semvec-atlas.
    """

    def test_is_subclass_of_engine_store(self):
        from semvec.api.mongo_state_store import (
            MongoStateStore as EngineMongoStateStore,
        )

        assert issubclass(MongoStateStore, EngineMongoStateStore)

    def test_cas_methods_are_inherited_not_overridden(self):
        from semvec.api.mongo_state_store import (
            MongoStateStore as EngineMongoStateStore,
        )

        for name in ("save", "load", "load_full", "load_owner",
                     "current_version", "delete",
                     "_write_state_doc", "_commit_state_and_events",
                     "_event_writer", "_supports_transactions"):
            assert name not in MongoStateStore.__dict__, (
                f"{name} must be inherited from the engine store, not forked"
            )
            assert getattr(MongoStateStore, name) is getattr(
                EngineMongoStateStore, name
            )

    def test_drift_exempt_pair_is_the_local_addition(self):
        assert "set_drift_exempt" in MongoStateStore.__dict__
        assert "is_drift_exempt" in MongoStateStore.__dict__


def _self_attributes_assigned_in(func) -> set[str]:
    """Every ``self.<name>`` a function body assigns — an oracle
    independent of the production helper under test."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names: set[str] = set()
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for target in targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                names.add(target.attr)
    return names


class TestUpstreamConstructorDrift:
    """The Database-object constructor bypasses the upstream one (which
    would replace the wheel's process-wide client singleton), so every
    attribute the upstream constructor initialises must still exist on
    our instances — across wheel versions, not just the one pinned
    today."""

    def test_instance_carries_every_attribute_the_upstream_constructor_sets(self, store):
        from semvec.api.mongo_state_store import (
            MongoStateStore as EngineMongoStateStore,
        )

        # Arrange
        upstream = _self_attributes_assigned_in(EngineMongoStateStore.__init__)

        # Act
        missing = sorted(name for name in upstream if not hasattr(store, name))

        # Assert
        assert not missing, (
            f"upstream constructor initialises {missing}, the Database-object "
            "constructor does not — map them in session_store.MongoStateStore"
        )

    def test_literal_defaults_of_unknown_upstream_attributes_are_adopted(self):
        from semvec_atlas.session_store import literal_constructor_defaults

        # Arrange
        class FutureEngineStore:
            def __init__(self, *, uri=None, db=None, collection="c"):
                client = object()
                self._collection = client
                self._future_flag = False
                self._future_cache: dict | None = None
                self._future_limit = -1

        # Act
        defaults = literal_constructor_defaults(FutureEngineStore)

        # Assert
        assert defaults == {
            "_future_flag": False,
            "_future_cache": None,
            "_future_limit": -1,
        }

    def test_only_unconditional_top_level_assignments_count_as_defaults(self):
        from semvec_atlas.session_store import literal_constructor_defaults

        # Arrange
        class EngineStoreWithBranches:
            def __init__(self, fast=False):
                self._mode = "slow"
                if fast:
                    self._mode = "fast"
                    self._fast_only = True

                def _helper():
                    self._nested = 1

        # Act
        defaults = literal_constructor_defaults(EngineStoreWithBranches)

        # Assert
        assert defaults == {"_mode": "slow"}

    def test_mutable_literal_defaults_are_not_shared_between_calls(self):
        from semvec_atlas.session_store import literal_constructor_defaults

        # Arrange
        class EngineStoreWithCache:
            def __init__(self):
                self._cache = {}

        # Act
        first = literal_constructor_defaults(EngineStoreWithCache)["_cache"]
        second = literal_constructor_defaults(EngineStoreWithCache)["_cache"]

        # Assert
        assert first == {} and first is not second
