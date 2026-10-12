"""Hybrid retrieval: ``$rankFusion`` with client-side RRF fallback.

Three sub-pipelines feed the fusion:

* **semantic** — ``$vectorSearch`` on the client-side embeddings
  (index ``ix_vec``; guardrail L2);
* **lexical** — ``$search`` on ``summary`` (index ``ix_lex``,
  ``lucene.german`` primary + ``lucene.standard`` multi);
* **recency** — ``$match`` + ``$sort access.last`` (deliberately *not*
  engine-internal scores — guardrail L3).

When the server offers ``$rankFusion`` (generally available; requires
MongoDB 8.1+ in combination with ``$vectorSearch``) the fusion runs
server-side with per-tenant weights. Otherwise — a server below that
floor, or ``force_fallback=True`` — the sub-pipelines run separately
and the semvec wheel's ``rrf_fuse`` computes the reciprocal-rank
fusion client-side with identical weights (the fallback ships in the
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

# The wheel's probe is the one source of truth, re-exported for callers.
from semvec.api.mongo_retrieval_backend import detect_rank_fusion

from semvec_atlas.client_metadata import announce
from semvec_atlas.embedders import ensure_tenant_embedder
from semvec_atlas.guard import TenantScope

__all__ = ["DomainJoin", "Hit", "RetrievalEngine", "detect_rank_fusion"]

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"semantic": 0.5, "lexical": 0.3, "recency": 0.2}
_LIVE = {"deleted": False, "evicted": False}


@dataclass(frozen=True)
class DomainJoin:
    """Join ranked memory hits to the business records they concern.

    The memory item carries the identifying field (stamped at commit
    time via ``domain_link``); this pulls the matching document from a
    domain collection in the *same* aggregation. The join is
    tenant-filtered inside the lookup pipeline rather than afterwards,
    so a record belonging to another tenant is never a candidate.
    """

    collection: str
    local_field: str
    foreign_field: str | None = None

    @property
    def target_field(self) -> str:
        """Field name on the domain document; defaults to the local one."""
        return self.foreign_field or self.local_field


@dataclass(frozen=True)
class Hit:
    """One retrieval candidate handed to the engine for curation."""

    key: str
    summary: str
    tier: str | None
    ns: list[str]
    score: float
    value: dict[str, Any] | None = None
    # The joined business record, when the query asked for one.
    domain: dict[str, Any] | None = None


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
        announce(db.client)
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
        announce(db.client)
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
        join: DomainJoin | None = None,
    ) -> list[Hit]:
        """Hybrid candidate retrieval.

        The fusion ranks; curation (what enters the context package and
        in which order) stays with the engine's serializer
        — this method returns candidates, it does not build prompts.

        ``query_embedding`` lets callers that already embedded the
        query (or want to time embedding separately) skip the embedder
        call; it must come from the tenant's registered model.

        ``join`` attaches the business record each hit concerns. On the
        server-side path it is a ``$lookup`` inside the same
        aggregation — one round trip for memory and domain data. On the
        client-side fallback the records are fetched in one additional
        batch query, because there is no server-side pipeline to hang
        the lookup on.
        """
        if query_embedding is None:
            query_embedding = self._embedder.get_embedding(query)
        query_vector = [float(x) for x in query_embedding]
        if self.uses_rank_fusion:
            try:
                return self._search_rank_fusion(query, query_vector, k, join)
            except OperationFailure as exc:  # capability lost mid-flight
                logger.warning(
                    "$rankFusion failed at query time, degrading to client RRF: %s", exc
                )
                self.uses_rank_fusion = False
        return self._search_client_rrf(query, query_vector, k, join)

    # ------------------------------------------------------------------
    # domain join

    def _domain_lookup_stages(self, join: DomainJoin) -> list[dict[str, Any]]:
        """``$lookup`` stages that cannot leave the tenant.

        The tenant equality lives inside the lookup's own pipeline, so
        it constrains the *joined* collection rather than filtering
        afterwards — a foreign-tenant record with the same business key
        is never matched in the first place.
        """
        return [
            {
                "$lookup": {
                    "from": join.collection,
                    "let": {"local_value": f"${join.local_field}"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$tenant", self._scope.tenant]},
                                        {
                                            "$eq": [
                                                f"${join.target_field}",
                                                "$$local_value",
                                            ]
                                        },
                                    ]
                                }
                            }
                        },
                        {"$limit": 1},
                    ],
                    "as": "_domain",
                }
            },
            {"$set": {"_domain": {"$first": "$_domain"}}},
        ]

    def _fetch_domain(
        self, docs: list[dict[str, Any]], join: DomainJoin
    ) -> dict[Any, dict[str, Any]]:
        """Fallback-path join: one batch query for all hits, never one per hit.

        Returns ``{business key: record}``. Tenant-scoped through the
        guard, so the batch cannot pull a foreign tenant's record even
        when the business keys collide.
        """
        keys = {
            doc.get(join.local_field)
            for doc in docs
            if doc.get(join.local_field) is not None
        }
        if not keys:
            return {}
        tenant_wide = TenantScope(tenant=self._scope.tenant)
        return {
            record[join.target_field]: record
            for record in self._db[join.collection].find(
                tenant_wide.filter({join.target_field: {"$in": sorted(keys)}})
            )
        }

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
        self,
        query: str,
        query_vector: list[float],
        k: int,
        join: DomainJoin | None = None,
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
                    # The join's local field must survive the projection,
                    # or the lookup below has nothing to match on.
                    **({join.local_field: 1} if join else {}),
                    "scoreDetails": {"$meta": "scoreDetails"},
                }
            },
            # Memory and business record in one aggregation: the reason
            # colocating them in one database is worth anything.
            *(self._domain_lookup_stages(join) if join else []),
        ]
        hits = []
        for doc in self._col.aggregate(pipeline):
            score = float(doc.get("scoreDetails", {}).get("value", 0.0))
            hits.append(self._to_hit(doc, score))
        return hits

    # ------------------------------------------------------------------
    # fallback path (identical weights, wheel-provided fusion)

    def _search_client_rrf(
        self,
        query: str,
        query_vector: list[float],
        k: int,
        join: DomainJoin | None = None,
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
        top = [(docs_by_id[doc_id], score) for doc_id, score in fused[:k]]
        records = self._fetch_domain([d for d, _ in top], join) if join else {}
        return [
            self._to_hit(
                doc,
                score,
                domain=records.get(doc.get(join.local_field)) if join else None,
            )
            for doc, score in top
        ]

    # ------------------------------------------------------------------

    @staticmethod
    def _to_hit(
        doc: dict[str, Any], score: float, *, domain: dict[str, Any] | None = None
    ) -> Hit:
        return Hit(
            key=doc["key"],
            summary=doc.get("summary", ""),
            tier=doc.get("tier"),
            ns=list(doc.get("ns", [])),
            score=score,
            value=doc.get("value"),
            # ``_domain`` is what the server-side $lookup writes; the
            # fallback path passes its batch-fetched record explicitly.
            domain=domain if domain is not None else doc.get("_domain"),
        )
