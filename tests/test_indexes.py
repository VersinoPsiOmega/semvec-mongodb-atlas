"""Collection/index provisioning.

``provision()`` is idempotent: calling it twice must not fail or
duplicate anything. Validators enforce the *shape* of compliance
events (insert form); append-only is enforced operationally via
role separation and by the store implementation, which is
tested in test_event_store.py.
"""

from __future__ import annotations

import pymongo
import pytest

from semvec_atlas.indexes import COLLECTIONS, provision

pytestmark = pytest.mark.mongo


class TestProvision:
    def test_creates_all_collections(self, db):
        provision(db, dimension=32, search_indexes=False)
        names = set(db.list_collection_names())
        assert set(COLLECTIONS) <= names

    def test_idempotent(self, db):
        provision(db, dimension=32, search_indexes=False)
        provision(db, dimension=32, search_indexes=False)  # must not raise
        assert set(COLLECTIONS) <= set(db.list_collection_names())

    def test_ns_index_unique_on_live_items(self, db):
        provision(db, dimension=32, search_indexes=False)
        doc = {
            "tenant": "acme",
            "agent": "a1",
            "ns": ["user", "prefs"],
            "key": "diet",
            "deleted": False,
            "evicted": False,
        }
        db.memory_items.insert_one(dict(doc))
        with pytest.raises(pymongo.errors.DuplicateKeyError):
            db.memory_items.insert_one(dict(doc))

    def test_ns_index_partial_allows_reinsert_after_delete_flag(self, db):
        provision(db, dimension=32, search_indexes=False)
        doc = {
            "tenant": "acme",
            "agent": "a1",
            "ns": ["user", "prefs"],
            "key": "diet",
            "deleted": False,
            "evicted": False,
        }
        db.memory_items.insert_one(dict(doc))
        db.memory_items.update_one(
            {"tenant": "acme", "key": "diet"}, {"$set": {"deleted": True}}
        )
        # Same (tenant, agent, ns, key) may exist again once the old row
        # is tombstoned — the unique index is partial on deleted:false.
        db.memory_items.insert_one(dict(doc))

    def test_event_seq_unique_per_tenant_agent(self, db):
        provision(db, dimension=32, search_indexes=False)
        ev = {
            "tenant": "acme",
            "agent": "a1",
            "seq": 1,
            "event_id": "e-1",
            "created_at": "2026-07-30T00:00:00Z",
            "content": "x",
            "embedding": [0.0],
            "meta": {},
        }
        db.compliance_events.insert_one(dict(ev))
        with pytest.raises(pymongo.errors.DuplicateKeyError):
            db.compliance_events.insert_one(dict(ev, event_id="e-2"))
        # Same seq for a different agent is fine.
        db.compliance_events.insert_one(dict(ev, agent="a2", event_id="e-3"))

    def test_event_validator_rejects_malformed_event(self, db):
        provision(db, dimension=32, search_indexes=False)
        with pytest.raises(pymongo.errors.WriteError):
            db.compliance_events.insert_one({"tenant": "acme"})  # missing fields

    def test_snapshot_index_exists(self, db):
        provision(db, dimension=32, search_indexes=False)
        info = db.state_snapshots.index_information()
        keys = [tuple(v["key"]) for v in info.values()]
        assert (("tenant", 1), ("agent", 1), ("session", 1), ("turn", -1)) in keys

    def test_search_indexes_created_when_requested(self, db):
        provision(db, dimension=32, search_indexes=True)
        names = {ix["name"] for ix in db.memory_items.list_search_indexes()}
        assert {"ix_vec", "ix_lex"} <= names

    def test_search_index_provisioning_idempotent(self, db):
        provision(db, dimension=32, search_indexes=True)
        provision(db, dimension=32, search_indexes=True)  # must not raise
        names = [ix["name"] for ix in db.memory_items.list_search_indexes()]
        assert names.count("ix_vec") == 1
