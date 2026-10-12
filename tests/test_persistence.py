"""Snapshot persistence.

``AtlasPersistence`` stores ``to_bytes()`` blobs with metadata and
restores the *latest* valid snapshot. Design mirrors the semvec wheel's
``StateStore``: optimistic concurrency (monotonic turn), loud conflicts,
never silent overwrites.

Determinism note (verified empirically): the engine's own
``to_dict()['checksum']`` is stable across independently built states
with the same update sequence, and ``to_bytes -> from_bytes -> to_bytes``
round-trips byte-identically. Snapshot integrity checks therefore use
the engine checksum plus a transport-level SHA-256 of the blob.
"""

from __future__ import annotations

import pytest

from semvec_atlas.guard import TenantScope
from semvec_atlas.indexes import provision
from semvec_atlas.persistence import AtlasPersistence, SnapshotConflict

pytestmark = pytest.mark.mongo


@pytest.fixture()
def persistence(db):
    provision(db, dimension=32, search_indexes=False)
    return AtlasPersistence(db, TenantScope(tenant="acme", agent="advisor-01"))


def _updated_state(fresh_state, embedder, n: int = 3):
    state = fresh_state()
    for i in range(n):
        state.update(embedder.get_embedding(f"turn {i}"), f"turn {i}")
    return state


class TestSnapshotSaveLoad:
    def test_save_returns_metadata(self, persistence, fresh_state, embedder):
        state = _updated_state(fresh_state, embedder)
        meta = persistence.save_snapshot(state, session="s1", turn=3)
        assert meta.turn == 3
        assert meta.blob_sha256
        assert meta.engine_checksum
        assert meta.engine_version
        assert meta.schema_version == 1

    def test_load_latest_restores_engine_state(self, persistence, fresh_state, embedder):
        state = _updated_state(fresh_state, embedder)
        persistence.save_snapshot(state, session="s1", turn=3)
        restored, meta = persistence.load_latest_snapshot(session="s1")
        assert meta.turn == 3
        # Engine checksum is the determinism contract.
        assert restored.to_dict()["checksum"] == state.to_dict()["checksum"]

    def test_load_latest_returns_none_when_empty(self, persistence):
        assert persistence.load_latest_snapshot(session="nope") is None

    def test_latest_wins_across_turns(self, persistence, fresh_state, embedder):
        state = _updated_state(fresh_state, embedder, n=1)
        persistence.save_snapshot(state, session="s1", turn=1)
        state.update(embedder.get_embedding("more"), "more")
        persistence.save_snapshot(state, session="s1", turn=2)
        _, meta = persistence.load_latest_snapshot(session="s1")
        assert meta.turn == 2

    def test_non_monotonic_turn_raises(self, persistence, fresh_state, embedder):
        state = _updated_state(fresh_state, embedder)
        persistence.save_snapshot(state, session="s1", turn=5)
        with pytest.raises(SnapshotConflict):
            persistence.save_snapshot(state, session="s1", turn=5)
        with pytest.raises(SnapshotConflict):
            persistence.save_snapshot(state, session="s1", turn=4)

    def test_corrupted_blob_raises_loudly(self, persistence, fresh_state, embedder, db):
        state = _updated_state(fresh_state, embedder)
        persistence.save_snapshot(state, session="s1", turn=1)
        db.state_snapshots.update_one({}, {"$set": {"blob_sha256": "deadbeef"}})
        with pytest.raises(ValueError, match="integrity"):
            persistence.load_latest_snapshot(session="s1")

    def test_retention_keeps_last_n(self, db, fresh_state, embedder):
        provision(db, dimension=32, search_indexes=False)
        p = AtlasPersistence(
            db, TenantScope(tenant="acme", agent="a1"), snapshot_retention=2
        )
        state = _updated_state(fresh_state, embedder, n=1)
        for turn in (1, 2, 3, 4):
            p.save_snapshot(state, session="s1", turn=turn)
        turns = sorted(
            d["turn"] for d in db.state_snapshots.find({"session": "s1"})
        )
        assert turns == [3, 4]

    def test_tenant_isolation(self, db, fresh_state, embedder):
        provision(db, dimension=32, search_indexes=False)
        pa = AtlasPersistence(db, TenantScope(tenant="acme", agent="a1"))
        pb = AtlasPersistence(db, TenantScope(tenant="other", agent="a1"))
        state = _updated_state(fresh_state, embedder, n=1)
        pa.save_snapshot(state, session="s1", turn=1)
        assert pb.load_latest_snapshot(session="s1") is None

    def test_privacy_flags_persisted(self, persistence, fresh_state, embedder, db):
        state = _updated_state(fresh_state, embedder, n=1)
        persistence.save_snapshot(
            state,
            session="s1",
            turn=1,
            include_memory_text=False,
            include_literal_cache_text=False,
        )
        doc = db.state_snapshots.find_one({"session": "s1"})
        assert doc["privacy_flags"] == {
            "include_memory_text": False,
            "include_literal_cache_text": False,
            "include_adaptive_params": True,
        }
