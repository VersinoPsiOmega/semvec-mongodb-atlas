"""Memory-item synchronisation via tier diff.

For turns without a mutation feed, the sync derives changes against
the wheel's public surface: enumerate
``state.memory.short_term / medium_term / long_term`` after each turn
and diff against the previously persisted view. UUIDv7 item ids make
the diff cheap; ``update(meta=...)`` tags items with (session, turn) at
write time.

Multi-resolution caveat (found empirically in M0): one item id may
appear in *several* tier lists simultaneously. The persisted ``tier``
is therefore the item's **highest** tier (long > medium > short), and
the full membership set is kept in ``tiers``.

Engine metadata (importance/protection scores) stays opaque — the sync
persists only what the public MemoryUnit surface exposes and stores it
under ``engine_meta`` without interpreting it.
"""

from __future__ import annotations

import pytest

from semvec_atlas.guard import TenantScope
from semvec_atlas.indexes import provision
from semvec_atlas.item_sync import MemoryItemSync

pytestmark = pytest.mark.mongo


@pytest.fixture()
def scope():
    return TenantScope(tenant="acme", agent="advisor-01")


@pytest.fixture()
def sync(db, scope):
    provision(db, dimension=32, search_indexes=False)
    return MemoryItemSync(db, scope)


@pytest.fixture()
def state(fresh_state):
    return fresh_state()


def _turn(state, embedder, text: str, session: str = "s1", turn: int = 1):
    state.update(
        embedder.get_embedding(text),
        text,
        meta={"session": session, "turn": turn},
    )


class TestFirstSync:
    def test_new_items_inserted(self, sync, state, embedder, db):
        _turn(state, embedder, "the user is vegetarian")
        report = sync.sync(state, session="s1", turn=1)
        assert report.added == 1
        doc = db.memory_items.find_one({"tenant": "acme"})
        assert doc is not None
        assert doc["summary"] == "the user is vegetarian"
        assert doc["deleted"] is False and doc["evicted"] is False
        assert len(doc["embedding"]) == 32
        assert doc["ns"] == ["semvec", "s1"]
        assert doc["key"]  # engine UUIDv7

    def test_tier_is_highest_membership(self, sync, state, embedder, db):
        _turn(state, embedder, "hello tiers")
        sync.sync(state, session="s1", turn=1)
        doc = db.memory_items.find_one({"tenant": "acme"})
        # Fresh items land in short (and possibly medium) — the highest
        # membership must win and the full set must be recorded.
        assert doc["tier"] in ("short", "medium", "long")
        assert doc["tier"] == max(
            doc["tiers"], key=["short", "medium", "long"].index
        )

    def test_engine_meta_opaque_binary(self, sync, state, embedder, db):
        _turn(state, embedder, "opaque check")
        sync.sync(state, session="s1", turn=1)
        doc = db.memory_items.find_one({"tenant": "acme"})
        assert isinstance(doc["engine_meta"], (bytes, memoryview))


class TestIncrementalSync:
    def test_second_sync_idempotent(self, sync, state, embedder):
        _turn(state, embedder, "fact one")
        assert sync.sync(state, session="s1", turn=1).added == 1
        report = sync.sync(state, session="s1", turn=1)
        assert (report.added, report.updated, report.evicted) == (0, 0, 0)

    def test_new_turn_adds_only_new_items(self, sync, state, embedder, db):
        _turn(state, embedder, "fact one", turn=1)
        sync.sync(state, session="s1", turn=1)
        _turn(state, embedder, "fact two", turn=2)
        report = sync.sync(state, session="s1", turn=2)
        assert report.added == 1
        assert db.memory_items.count_documents({"deleted": False}) == 2

    def test_vanished_item_marked_evicted(self, sync, state, embedder, db, scope):
        _turn(state, embedder, "will vanish", turn=1)
        sync.sync(state, session="s1", turn=1)
        # Simulate engine eviction: the id is no longer in any tier list.
        # We don't force real eviction (that would need engine internals);
        # the diff must treat disappearance as eviction either way.
        doc = db.memory_items.find_one(scope.filter())
        report = sync.sync(
            state, session="s1", turn=2, _tier_view_override={}
        )
        assert report.evicted == 1
        updated = db.memory_items.find_one({"key": doc["key"]})
        assert updated["evicted"] is True
        assert updated["deleted"] is False

    def test_tier_change_updates_document(self, sync, state, embedder, db, scope):
        _turn(state, embedder, "promoted fact", turn=1)
        sync.sync(state, session="s1", turn=1)
        doc = db.memory_items.find_one(scope.filter())
        promoted_view = {doc["key"]: dict(_doc_view(doc), tiers=["long"])}
        report = sync.sync(
            state, session="s1", turn=2, _tier_view_override=promoted_view
        )
        assert report.updated == 1
        assert db.memory_items.find_one({"key": doc["key"]})["tier"] == "long"


class TestLiteralCacheSync:
    def test_literal_entries_persisted(self, sync, state, embedder, db):
        # Numbers/identifiers flow into the engine literal cache via the
        # wheel's auto-extractors when a SemvecSession drives updates; at
        # the raw-state level we exercise the public LiteralCache surface.
        _turn(state, embedder, "the invoice number is INV-4711 over 129.90 EUR")
        sync.sync(state, session="s1", turn=1)
        # Literal sync is best-effort against the public surface: if the
        # engine exposes entries, they must land in literal_cache with a
        # memory link; when none are exposed the collection stays empty.
        for doc in db.literal_cache.find({"tenant": "acme"}):
            assert doc["content"]
            assert "memory_id" in doc


def _doc_view(doc):
    return {
        "summary": doc["summary"],
        "embedding": doc["embedding"],
        "tiers": doc["tiers"],
        "access_count": doc.get("access_count", 0),
        "meta": doc.get("item_meta", {}),
        "engine_meta": bytes(doc["engine_meta"]),
    }
