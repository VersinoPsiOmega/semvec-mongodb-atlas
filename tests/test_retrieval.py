"""Hybrid retrieval.

``RetrievalEngine`` fuses three sub-pipelines (semantic ``$vectorSearch``,
lexical ``$search``, recency) via Atlas ``$rankFusion`` when available,
with a client-side fallback using the semvec wheel's ``rrf_fuse``
(identical weights). Feature detection runs at startup;
``force_fallback=True`` pins the client-side path.

The mandatory tenant/deleted/evicted filters sit inside *every*
sub-pipeline — a deleted item must be invisible on both
paths before the index purge.

These tests build real Atlas Search indexes on atlas-local and wait for
them to become queryable — the slowest tests of the suite (marked
``slow``).
"""

from __future__ import annotations

import time

import pytest

from semvec_atlas.guard import TenantScope
from semvec_atlas.indexes import provision
from semvec_atlas.retrieval import RetrievalEngine, detect_rank_fusion

pytestmark = [pytest.mark.mongo, pytest.mark.slow]

DIM = 32


def _wait_for_search_indexes(col, names, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready = {
            ix["name"]
            for ix in col.list_search_indexes()
            if ix.get("queryable") or ix.get("status") == "READY"
        }
        if set(names) <= ready:
            return
        time.sleep(1.0)
    raise TimeoutError(f"search indexes {names} not queryable after {timeout}s")


def _wait_for_indexed_count(col, expected, timeout=60):
    """Atlas Search indexing is eventually consistent — wait for docs."""
    import pymongo.errors

    deadline = time.monotonic() + timeout
    pipeline = [
        {"$search": {"index": "ix_lex", "exists": {"path": "tenant"}}},
        {"$count": "n"},
    ]
    while time.monotonic() < deadline:
        try:
            res = list(col.aggregate(pipeline))
            if res and res[0]["n"] >= expected:
                return
        except pymongo.errors.OperationFailure:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"search index did not reach {expected} docs in {timeout}s")


@pytest.fixture(scope="module")
def corpus_db(mongo_client, embedder):
    """One module-scoped database with ready search indexes and a corpus."""
    import uuid as _uuid
    from datetime import datetime, timedelta, timezone

    db = mongo_client[f"svaretr_{_uuid.uuid4().hex[:10]}"]
    provision(db, dimension=DIM, search_indexes=True)

    scope = TenantScope(tenant="acme", agent="advisor-01")
    now = datetime.now(timezone.utc)
    docs = [
        # (key, summary, age_days, tier, deleted, evicted)
        ("vegetarian", "the customer follows a strict vegetarian diet", 30, "long", False, False),
        ("retirement", "retirement portfolio rebalancing towards bonds", 20, "long", False, False),
        ("quarterly", "quarterly revenue target for the sales team", 10, "medium", False, False),
        ("recent-note", "short administrative note about scheduling", 0, "short", False, False),
        ("deleted-item", "vegetarian preferences that were deleted", 5, "long", True, False),
        ("evicted-item", "vegetarian cooking classes evicted from memory", 5, "short", False, True),
    ]
    for key, summary, age, tier, deleted, evicted in docs:
        db.memory_items.insert_one(
            scope.filter(
                {
                    "ns": ["semvec", "s1"],
                    "key": key,
                    "kind": "engine",
                    "summary": summary,
                    "embedding": [float(x) for x in embedder.get_embedding(summary)],
                    "tier": tier,
                    "tiers": [tier],
                    "deleted": deleted,
                    "evicted": evicted,
                    "access": {"count": 1, "last": now - timedelta(days=age)},
                    "created_at": now - timedelta(days=age),
                    "updated_at": now - timedelta(days=age),
                }
            )
        )
    # Foreign tenant doc that must never surface.
    db.memory_items.insert_one(
        {
            "tenant": "rival",
            "agent": "advisor-01",
            "ns": ["semvec", "s1"],
            "key": "foreign",
            "kind": "engine",
            "summary": "the customer follows a strict vegetarian diet",
            "embedding": [float(x) for x in embedder.get_embedding("the customer follows a strict vegetarian diet")],
            "tier": "long",
            "tiers": ["long"],
            "deleted": False,
            "evicted": False,
            "access": {"count": 1, "last": now},
            "created_at": now,
            "updated_at": now,
        }
    )
    _wait_for_search_indexes(db.memory_items, ["ix_vec", "ix_lex"])
    _wait_for_indexed_count(db.memory_items, 7)
    yield db, scope
    mongo_client.drop_database(db.name)


class TestFeatureDetection:
    def test_rank_fusion_detected_on_atlas_local(self, mongo_client, db):
        assert detect_rank_fusion(db) is True


@pytest.fixture(params=["native", "fallback"], scope="module")
def engine(request, corpus_db, embedder):
    db, scope = corpus_db
    return RetrievalEngine(
        db,
        scope,
        embedder=embedder,
        force_fallback=(request.param == "fallback"),
    )


class TestHybridSearch:
    def test_semantic_match_ranks_first(self, engine):
        hits = engine.search("the customer follows a strict vegetarian diet", k=3)
        assert hits[0].key == "vegetarian"
        assert hits[0].score > 0

    def test_lexical_exact_term_found(self, engine):
        hits = engine.search("quarterly revenue", k=5)
        assert "quarterly" in [h.key for h in hits]

    def test_deleted_and_evicted_never_surface(self, engine):
        hits = engine.search("vegetarian", k=10)
        keys = {h.key for h in hits}
        assert "deleted-item" not in keys
        assert "evicted-item" not in keys

    def test_tenant_isolation(self, engine):
        hits = engine.search("the customer follows a strict vegetarian diet", k=10)
        assert "foreign" not in {h.key for h in hits}

    def test_k_limits_results(self, engine):
        assert len(engine.search("note", k=2)) <= 2

    def test_hits_expose_summary_and_tier(self, engine):
        hits = engine.search("retirement bonds", k=1)
        assert hits[0].summary
        assert hits[0].tier in ("short", "medium", "long")


class TestPathParity:
    def test_both_paths_agree_on_clear_top_hit(self, corpus_db, embedder):
        db, scope = corpus_db
        native = RetrievalEngine(db, scope, embedder=embedder, force_fallback=False)
        fallback = RetrievalEngine(db, scope, embedder=embedder, force_fallback=True)
        q = "the customer follows a strict vegetarian diet"
        assert native.search(q, k=1)[0].key == fallback.search(q, k=1)[0].key

    def test_engine_reports_active_path(self, corpus_db, embedder):
        db, scope = corpus_db
        native = RetrievalEngine(db, scope, embedder=embedder)
        fallback = RetrievalEngine(db, scope, embedder=embedder, force_fallback=True)
        assert native.uses_rank_fusion is True
        assert fallback.uses_rank_fusion is False


class TestTenantWeights:
    def test_weights_loaded_from_tenants_collection(self, corpus_db, embedder):
        db, scope = corpus_db
        db.tenants.update_one(
            {"tenant": "acme"},
            {"$set": {"fusion_weights": {"semantic": 0.7, "lexical": 0.2, "recency": 0.1}}},
            upsert=True,
        )
        engine = RetrievalEngine.for_tenant(db, scope, embedder=embedder)
        assert engine.weights == {"semantic": 0.7, "lexical": 0.2, "recency": 0.1}

    def test_default_weights_match_spec(self, corpus_db, embedder):
        db, scope = corpus_db
        engine = RetrievalEngine(db, scope, embedder=embedder)
        assert engine.weights == {"semantic": 0.5, "lexical": 0.3, "recency": 0.2}


class TestPrecomputedQueryEmbedding:
    def test_precomputed_embedding_skips_the_embedder(self, corpus_db, embedder):
        # A counting wrapper proves search() never re-embeds when the
        # caller supplies the query embedding.
        class Counting:
            def __init__(self, inner):
                self._inner = inner
                self.calls = 0
                self.model_name = type(inner).__name__  # same fingerprint

            def get_dimension(self):
                return self._inner.get_dimension()

            def get_embedding(self, text):
                self.calls += 1
                return self._inner.get_embedding(text)

        db, scope = corpus_db
        counting = Counting(embedder)
        engine = RetrievalEngine(db, scope, embedder=counting, force_fallback=True)
        query = "retirement portfolio rebalancing towards bonds"
        vec = embedder.get_embedding(query)
        hits = engine.search(query, k=3, query_embedding=vec)
        assert counting.calls == 0
        # Same result as the self-embedding path.
        hits_default = engine.search(query, k=3)
        assert counting.calls == 1
        assert [h.key for h in hits] == [h.key for h in hits_default]
