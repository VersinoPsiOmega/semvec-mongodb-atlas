"""Hybrid retrieval: ``$rankFusion`` with client-side RRF fallback.

Three sub-pipelines feed the fusion:

* **semantic** — ``$vectorSearch`` on the client-side embeddings
  (index ``ix_vec``);
* **lexical** — ``$search`` on ``summary`` (index ``ix_lex``,
  ``lucene.german`` primary + ``lucene.standard`` multi);
* **recency** — ``$match`` + ``$sort access.last`` (deliberately *not*
  engine-internal scores).

When the server offers ``$rankFusion`` (MongoDB 8.1+, preview) the
fusion runs server-side with per-tenant weights. Otherwise — or when
``force_fallback=True`` — the sub-pipelines run separately and the
semvec wheel's ``rrf_fuse`` computes the reciprocal-rank fusion
client-side with identical weights (the fallback ships in the
wheel's Python layer, not the Rust core).

The tenant guard plus ``deleted:false / evicted:false`` are embedded in
**every** sub-pipeline: a tombstoned item is invisible on both paths
before the index purge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pymongo.database import Database
from pymongo.errors import OperationFailure
from semvec.api.hybrid_retrieval import rrf_fuse

# Promised by the wheel since semvec 0.8.8 — re-exported so callers
# and the engine share one probe implementation.
from semvec.api.mongo_retrieval_backend import detect_rank_fusion

from semvec_atlas.embedders import ensure_tenant_embedder
from semvec_atlas.guard import TenantScope

__all__ = ["Hit", "RetrievalEngine", "detect_rank_fusion"]

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"semantic": 0.5, "lexical": 0.3, "recency": 0.2}
_LIVE = {"deleted": False, "evicted": False}


@dataclass(frozen=True)
class Hit:
    """One retrieval candidate handed to the engine for curation."""

    key: str
    summary: str
    tier: str | None
    ns: list[str]
    score: float
    value: dict[str, Any] | None = None


class RetrievalEngine:
    """Hybrid retrieval bound to one :class:`TenantScope`."""

    def __init__(
        self,
        db: Database,
        scope: TenantScope,
        *,
        embedder: Any,
        weights: dict[str, float] | None = None,
        force_fallback: bool = False,
    ) -> None:
        self._db = db
        self._scope = scope
        self._embedder = embedder
        # Query-time embeddings must come from the tenant's registered
        # model — fail closed instead of degrading silently.
        ensure_tenant_embedder(db, scope, embedder)
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self._col = db.memory_items
        self.uses_rank_fusion = (not force_fallback) and detect_rank_fusion(db)

    @classmethod
    def for_tenant(
        cls,
        db: Database,
        scope: TenantScope,
        *,
        embedder: Any,
        force_fallback: bool = False,
    ) -> RetrievalEngine:
        """Build an engine with fusion weights from the tenant config."""
        doc = db.tenants.find_one({"tenant": scope.tenant}) or {}
        weights = doc.get("fusion_weights") or DEFAULT_WEIGHTS
        return cls(
            db, scope, embedder=embedder, weights=weights, force_fallback=force_fallback
        )

    # ------------------------------------------------------------------
    # public search

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        query_embedding: Any | None = None,
    ) -> list[Hit]:
        """Hybrid candidate retrieval.

        The fusion ranks; curation (what enters the context package and
        in which order) stays with the engine's serializer
        — this method returns candidates, it does not build prompts.

        ``query_embedding`` lets callers that already embedded the
        query (or want to time embedding separately) skip the embedder
        call; it must come from the tenant's registered model.
        """
        if query_embedding is None:
            query_embedding = self._embedder.get_embedding(query)
        query_vector = [float(x) for x in query_embedding]
        if self.uses_rank_fusion:
            try:
                return self._search_rank_fusion(query, query_vector, k)
            except OperationFailure as exc:  # preview drift
                logger.warning(
                    "$rankFusion failed at query time, degrading to client RRF: %s", exc
                )
                self.uses_rank_fusion = False
        return self._search_client_rrf(query, query_vector, k)

    # ------------------------------------------------------------------
    # sub-pipelines (shared by both paths)

    def _semantic_stage(self, query_vector: list[float], limit: int) -> dict[str, Any]:
        return {
            "$vectorSearch": {
                "index": "ix_vec",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(limit * 10, 100),
                "limit": limit,
                "filter": self._scope.filter(dict(_LIVE)),
            }
        }

    def _lexical_stages(self, query: str, limit: int) -> list[dict[str, Any]]:
        return [
            {
                "$search": {
                    "index": "ix_lex",
                    "compound": {
                        "must": [
                            {
                                "text": {
                                    "query": query,
                                    "path": [
                                        "summary",
                                        {"value": "summary", "multi": "standard"},
                                    ],
                                }
                            }
                        ],
                        "filter": [
                            {"equals": {"path": "tenant", "value": self._scope.tenant}},
                            {"equals": {"path": "agent", "value": self._scope.agent}},
                            {"equals": {"path": "deleted", "value": False}},
                            {"equals": {"path": "evicted", "value": False}},
                        ],
                    },
                }
            },
            {"$limit": limit},
        ]

    def _recency_stages(self, limit: int) -> list[dict[str, Any]]:
        return [
            {"$match": self._scope.filter(dict(_LIVE))},
            {"$sort": {"access.last": -1, "_id": 1}},
            {"$limit": limit},
        ]

    # ------------------------------------------------------------------
    # native path

    def _search_rank_fusion(
        self, query: str, query_vector: list[float], k: int
    ) -> list[Hit]:
        sub_limit = max(k * 3, 20)
        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "semantic": [self._semantic_stage(query_vector, sub_limit)],
                            "lexical": self._lexical_stages(query, sub_limit),
                            "recency": self._recency_stages(sub_limit),
                        }
                    },
                    "combination": {"weights": self.weights},
                    "scoreDetails": True,
                }
            },
            {"$limit": k},
            {
                "$project": {
                    "key": 1,
                    "summary": 1,
                    "tier": 1,
                    "ns": 1,
                    "value": 1,
                    "scoreDetails": {"$meta": "scoreDetails"},
                }
            },
        ]
        hits = []
        for doc in self._col.aggregate(pipeline):
            score = float(doc.get("scoreDetails", {}).get("value", 0.0))
            hits.append(self._to_hit(doc, score))
        return hits

    # ------------------------------------------------------------------
    # fallback path (identical weights, wheel-provided fusion)

    def _search_client_rrf(
        self, query: str, query_vector: list[float], k: int
    ) -> list[Hit]:
        sub_limit = max(k * 3, 20)
        rankings: list[list[str]] = []
        weights: list[float] = []
        docs_by_id: dict[str, dict[str, Any]] = {}

        for name, stages in (
            ("semantic", [self._semantic_stage(query_vector, sub_limit)]),
            ("lexical", self._lexical_stages(query, sub_limit)),
            ("recency", self._recency_stages(sub_limit)),
        ):
            try:
                docs = list(self._col.aggregate(stages))
            except OperationFailure as exc:
                # A sub-pipeline may be unavailable (e.g. search index
                # still building); degrade to the remaining signals
                # rather than failing retrieval outright.
                logger.warning("sub-pipeline %s failed: %s", name, exc)
                continue
            ranking = []
            for doc in docs:
                doc_id = str(doc["_id"])
                docs_by_id[doc_id] = doc
                ranking.append(doc_id)
            rankings.append(ranking)
            weights.append(self.weights.get(name, 0.0))

        if not rankings:
            return []
        fused = rrf_fuse(rankings, weights=weights)
        return [self._to_hit(docs_by_id[doc_id], score) for doc_id, score in fused[:k]]

    # ------------------------------------------------------------------

    @staticmethod
    def _to_hit(doc: dict[str, Any], score: float) -> Hit:
        return Hit(
            key=doc["key"],
            summary=doc.get("summary", ""),
            tier=doc.get("tier"),
            ns=list(doc.get("ns", [])),
            score=score,
            value=doc.get("value"),
        )
