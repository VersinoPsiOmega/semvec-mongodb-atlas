"""Mutation-feed integration.

The engine reports per-turn item mutations
(``update(include_mutations=True)`` → ``metrics["mutations"]`` with
kinds ``added``/``promoted``/``demoted``/``merged`` + ``sources``),
verified mirror-grade against the tier enumeration. ``MemoryItemSync``
uses the feed in two ways:

* **fast path** — a turn whose feed reports no mutations skips the
  tier diff entirely (no reads, no writes), except on periodic
  safety-net turns;
* **merge enrichment** — ``merged`` events set ``superseded_by`` on
  the source documents, turning silent displacement into an auditable
  consolidation link.

The tier diff remains the write mechanism (documents are built from
the authoritative tier enumeration); the feed steers *when* it runs
and enriches *why* items disappeared.
"""

from __future__ import annotations

import pytest

from semvec_atlas.commit import TurnCommitter
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


class TestFeedFastPath:
    def test_empty_mutations_skips_diff(self, sync, fresh_state, embedder):
        state = fresh_state()
        state.update(embedder.get_embedding("fact"), "fact")
        sync.sync(state, session="s1", turn=1)  # baseline mirror
        # A no-mutation turn must not touch the collection at all.
        report = sync.sync(state, session="s1", turn=2, mutations=[])
        assert (report.added, report.updated, report.evicted) == (0, 0, 0)
        assert report.skipped is True

    def test_nonempty_mutations_runs_diff(self, sync, fresh_state, embedder):
        state = fresh_state()
        state.update(embedder.get_embedding("fact"), "fact")
        report = sync.sync(
            state, session="s1", turn=1,
            mutations=[{"kind": "added", "id": "irrelevant"}],
        )
        assert report.added == 1
        assert report.skipped is False

    def test_none_mutations_means_no_feed_and_runs_diff(self, sync, fresh_state, embedder):
        # Backwards compatibility: callers without feed keep full diffs.
        state = fresh_state()
        state.update(embedder.get_embedding("fact"), "fact")
        report = sync.sync(state, session="s1", turn=1, mutations=None)
        assert report.added == 1 and report.skipped is False

    def test_safety_net_forces_diff_despite_empty_feed(self, sync, fresh_state, embedder):
        state = fresh_state()
        state.update(embedder.get_embedding("fact"), "fact")
        report = sync.sync(
            state, session="s1", turn=1, mutations=[], force_diff=True
        )
        assert report.added == 1 and report.skipped is False


class TestMergeEnrichment:
    def test_merged_sources_get_superseded_by(self, db, scope, mongo_client, embedder, fresh_state):
        # Drive a real engine at tiny capacities until merges occur;
        # committer wires the feed through.
        from semvec import SemvecConfig, SemvecState

        provision(db, dimension=32, search_indexes=False)
        committer = TurnCommitter(
            mongo_client, db, scope, embedder=embedder, snapshot_every=0
        )
        state = SemvecState(
            SemvecConfig(
                dimension=32, short_term_size=3, medium_term_size=3, long_term_size=3
            )
        )
        merged_seen = 0
        for turn in range(1, 41):
            result = committer.run_turn(
                state, session="s1", turn=turn, text=f"pressure fact {turn}"
            )
            merged_seen += sum(
                1 for m in (result.mutations or []) if m["kind"] == "merged"
            )
        assert merged_seen > 0, "engine produced no merges — capacities too large?"

        superseded = list(
            db.memory_items.find(scope.filter({"superseded_by": {"$exists": True}}))
        )
        assert superseded, "merged sources must carry superseded_by"
        # Every superseded_by target must reference an existing live item.
        for doc in superseded:
            target = db.memory_items.find_one(
                scope.filter({"key": doc["superseded_by"]})
            )
            assert target is not None

    def test_mirror_matches_tier_enumeration_after_pressure(
        self, db, scope, mongo_client, embedder
    ):
        from semvec import SemvecConfig, SemvecState

        provision(db, dimension=32, search_indexes=False)
        committer = TurnCommitter(
            mongo_client, db, scope, embedder=embedder, snapshot_every=0
        )
        state = SemvecState(
            SemvecConfig(
                dimension=32, short_term_size=3, medium_term_size=3, long_term_size=3
            )
        )
        for turn in range(1, 41):
            committer.run_turn(state, session="s1", turn=turn, text=f"p {turn}")

        live_engine = {
            u.id
            for tier in (state.memory.short_term, state.memory.medium_term, state.memory.long_term)
            for u in tier
        }
        live_mirror = {
            d["key"]
            for d in db.memory_items.find(
                scope.filter({"deleted": False, "evicted": False, "kind": "engine"})
            )
        }
        assert live_mirror == live_engine


class TestCommitterWiring:
    def test_run_turn_exposes_mutations(self, db, scope, mongo_client, embedder, fresh_state):
        provision(db, dimension=32, search_indexes=False)
        committer = TurnCommitter(mongo_client, db, scope, embedder=embedder)
        state = fresh_state()
        result = committer.run_turn(state, session="s1", turn=1, text="hello")
        assert result.mutations is not None
        assert any(m["kind"] == "added" for m in result.mutations)
