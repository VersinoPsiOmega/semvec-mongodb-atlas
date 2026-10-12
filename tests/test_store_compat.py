"""Compatibility suite — CRUD parity with the native LangGraph store.

The same scenario runs against the native ``langgraph-store-mongodb``
``MongoDBStore`` and against ``SemvecStore``; observable CRUD semantics
must match. Search-with-query is intentionally *not* part of the parity
contract (the hybrid path is SemvecStore's added value); namespace
listing and filter-search are.

The native store manages its own collection and TTL indexes, so each
store gets a private collection in the shared test database.
"""

from __future__ import annotations

import pytest

from semvec_atlas.guard import TenantScope
from semvec_atlas.indexes import provision
from semvec_atlas.store import SemvecStore

pytestmark = pytest.mark.mongo


def _native_store(db):
    from langgraph.store.mongodb import MongoDBStore

    return MongoDBStore(collection=db["native_store_items"])


def _semvec_store(db, embedder):
    provision(db, dimension=32, search_indexes=False)
    return SemvecStore(
        db, scope=TenantScope(tenant="acme", agent="compat"), embedder=embedder
    )


@pytest.fixture(params=["native", "semvec"])
def any_store(request, db, embedder):
    if request.param == "native":
        return _native_store(db)
    return _semvec_store(db, embedder)


class TestCrudParity:
    def test_put_get_roundtrip(self, any_store):
        any_store.put(("users", "u1"), "prefs", {"diet": "vegetarian"})
        item = any_store.get(("users", "u1"), "prefs")
        assert item is not None
        assert item.value == {"diet": "vegetarian"}
        assert item.namespace == ("users", "u1")
        assert item.key == "prefs"

    def test_get_missing_returns_none(self, any_store):
        assert any_store.get(("nowhere",), "nope") is None

    def test_overwrite_replaces_value(self, any_store):
        any_store.put(("users", "u1"), "prefs", {"v": 1})
        any_store.put(("users", "u1"), "prefs", {"v": 2})
        assert any_store.get(("users", "u1"), "prefs").value == {"v": 2}

    def test_delete_then_get_returns_none(self, any_store):
        any_store.put(("users", "u1"), "prefs", {"v": 1})
        any_store.delete(("users", "u1"), "prefs")
        assert any_store.get(("users", "u1"), "prefs") is None

    def test_delete_missing_is_noop(self, any_store):
        any_store.delete(("users", "u1"), "ghost")

    def test_search_by_prefix_and_filter(self, any_store):
        any_store.put(("users", "u1"), "a", {"kind": "pref"})
        any_store.put(("users", "u1"), "b", {"kind": "note"})
        any_store.put(("other",), "c", {"kind": "pref"})
        hits = any_store.search(("users",), filter={"kind": "pref"})
        assert [i.key for i in hits] == ["a"]

    def test_list_namespaces_with_prefix(self, any_store):
        any_store.put(("users", "u1"), "a", {"v": 1})
        any_store.put(("kb",), "b", {"v": 2})
        spaces = any_store.list_namespaces(prefix=("users",))
        assert spaces == [("users", "u1")]
