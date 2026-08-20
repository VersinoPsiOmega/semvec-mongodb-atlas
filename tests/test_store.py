"""SemvecStore — LangGraph ``BaseStore`` adapter.

Signatures are pinned against langgraph 1.2.x (verified in M0):
``aput/aget/adelete/asearch/alist_namespaces`` delegate to
``batch``/``abatch``. Behavior contract: put/get/delete/list are
semantically identical to the native ``MongoDBStore`` (see
test_store_compat.py); ``asearch`` is the added-value path.

``adelete`` is a plain delete by design — the Art. 17 forget path runs
exclusively through the compliance API, never
implicitly through the store adapter.
"""

from __future__ import annotations

import pytest

from semvec_atlas.guard import TenantScope
from semvec_atlas.indexes import provision
from semvec_atlas.store import SemvecStore

pytestmark = pytest.mark.mongo


@pytest.fixture()
def store(db, embedder):
    provision(db, dimension=32, search_indexes=False)
    return SemvecStore(
        db,
        scope=TenantScope(tenant="acme", agent="advisor-01"),
        embedder=embedder,
    )


class TestPutGet:
    async def test_put_then_get_roundtrips_value(self, store):
        await store.aput(("users", "u1"), "prefs", {"diet": "vegetarian"})
        item = await store.aget(("users", "u1"), "prefs")
        assert item is not None
        assert item.value == {"diet": "vegetarian"}
        assert item.namespace == ("users", "u1")
        assert item.key == "prefs"
        assert item.created_at is not None

    async def test_get_missing_returns_none(self, store):
        assert await store.aget(("users", "u1"), "nope") is None

    async def test_put_overwrites_value_and_bumps_updated_at(self, store):
        await store.aput(("users", "u1"), "prefs", {"diet": "vegan"})
        first = await store.aget(("users", "u1"), "prefs")
        await store.aput(("users", "u1"), "prefs", {"diet": "omnivore"})
        second = await store.aget(("users", "u1"), "prefs")
        assert second.value == {"diet": "omnivore"}
        assert second.updated_at >= first.updated_at

    async def test_put_index_false_skips_embedding(self, store, db):
        await store.aput(("users", "u1"), "raw", {"blob": "x"}, index=False)
        doc = db.memory_items.find_one({"key": "raw"})
        assert doc.get("embedding") is None

    async def test_put_embeds_value_by_default(self, store, db):
        await store.aput(("users", "u1"), "fact", {"text": "likes hiking"})
        doc = db.memory_items.find_one({"key": "fact"})
        assert doc["embedding"] is not None and len(doc["embedding"]) == 32

    async def test_ttl_not_supported_yet(self, store):
        with pytest.raises(NotImplementedError):
            await store.aput(("users", "u1"), "k", {"v": 1}, ttl=60.0)


class TestDelete:
    async def test_delete_removes_item(self, store):
        await store.aput(("users", "u1"), "prefs", {"a": 1})
        await store.adelete(("users", "u1"), "prefs")
        assert await store.aget(("users", "u1"), "prefs") is None

    async def test_delete_missing_is_noop(self, store):
        await store.adelete(("users", "u1"), "ghost")  # must not raise

    async def test_delete_is_tombstone_not_hard_delete(self, store, db):
        # Plain adelete tombstones (deleted:true) so the vector index
        # filter can hide it until purge; hard deletion is the forget
        # path's job.
        await store.aput(("users", "u1"), "prefs", {"a": 1})
        await store.adelete(("users", "u1"), "prefs")
        raw = db.memory_items.find_one({"key": "prefs"})
        assert raw is not None and raw["deleted"] is True


class TestSearch:
    async def test_search_by_namespace_prefix(self, store):
        await store.aput(("users", "u1"), "a", {"t": "alpha"})
        await store.aput(("users", "u2"), "b", {"t": "beta"})
        hits = await store.asearch(("users", "u1"))
        assert [i.key for i in hits] == ["a"]

    async def test_search_filter_on_value_fields(self, store):
        await store.aput(("users", "u1"), "a", {"kind": "pref", "v": 1})
        await store.aput(("users", "u1"), "b", {"kind": "note", "v": 2})
        hits = await store.asearch(("users",), filter={"kind": "pref"})
        assert [i.key for i in hits] == ["a"]

    async def test_search_semantic_query_ranks_matching_item_first(self, store):
        await store.aput(("kb",), "hit", {"text": "the quarterly revenue target"})
        await store.aput(("kb",), "miss", {"text": "a completely different topic"})
        hits = await store.asearch(("kb",), query="the quarterly revenue target")
        assert hits[0].key == "hit"
        assert hits[0].score is not None

    async def test_search_respects_limit_and_offset(self, store):
        for i in range(5):
            await store.aput(("kb",), f"k{i}", {"i": i})
        page1 = await store.asearch(("kb",), limit=2)
        page2 = await store.asearch(("kb",), limit=2, offset=2)
        assert len(page1) == 2 and len(page2) == 2
        assert {i.key for i in page1}.isdisjoint({i.key for i in page2})

    async def test_search_excludes_deleted(self, store):
        await store.aput(("kb",), "gone", {"t": "x"})
        await store.adelete(("kb",), "gone")
        assert await store.asearch(("kb",)) == []


class TestListNamespaces:
    async def test_lists_distinct_namespaces(self, store):
        await store.aput(("users", "u1"), "a", {"v": 1})
        await store.aput(("users", "u2"), "b", {"v": 2})
        await store.aput(("kb",), "c", {"v": 3})
        spaces = await store.alist_namespaces()
        assert ("users", "u1") in spaces and ("kb",) in spaces

    async def test_prefix_filter(self, store):
        await store.aput(("users", "u1"), "a", {"v": 1})
        await store.aput(("kb",), "c", {"v": 3})
        spaces = await store.alist_namespaces(prefix=("users",))
        assert spaces == [("users", "u1")]

    async def test_max_depth_truncates(self, store):
        await store.aput(("users", "u1", "prefs"), "a", {"v": 1})
        spaces = await store.alist_namespaces(max_depth=1)
        assert ("users",) in spaces


class TestTenantIsolation:
    async def test_stores_of_different_tenants_do_not_leak(self, db, embedder):
        provision(db, dimension=32, search_indexes=False)
        s1 = SemvecStore(db, scope=TenantScope(tenant="acme", agent="a"), embedder=embedder)
        s2 = SemvecStore(db, scope=TenantScope(tenant="rival", agent="a"), embedder=embedder)
        await s1.aput(("kb",), "secret", {"t": "acme only"})
        assert await s2.aget(("kb",), "secret") is None
        assert await s2.asearch(("kb",)) == []

    def test_sync_facade_works(self, db, embedder):
        # BaseStore's sync methods delegate to batch() — must work too.
        provision(db, dimension=32, search_indexes=False)
        s = SemvecStore(db, scope=TenantScope(tenant="acme", agent="a"), embedder=embedder)
        s.put(("kb",), "k", {"v": 1})
        assert s.get(("kb",), "k").value == {"v": 1}
