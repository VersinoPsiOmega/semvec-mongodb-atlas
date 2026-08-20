"""``SemvecStore`` — LangGraph ``BaseStore`` adapter.

Implements the two abstract entry points (``batch``/``abatch``) of the
pinned langgraph 1.2.x interface; every public method
(``put/get/delete/search/list_namespaces`` and async twins) delegates
there. CRUD semantics mirror the native ``MongoDBStore``
(test_store_compat.py enforces parity); ``asearch`` is the added-value
path and is upgraded to hybrid ``$rankFusion`` retrieval in M3 without
changing this interface.

Deletion semantics: ``adelete`` **tombstones** (``deleted: true``) so
vector/search indexes can hide the item until purge. Hard deletion —
including the tombstones — is exclusively the Art. 17 forget path,
never an implicit store operation.

Embeddings are produced client-side by the tenant's embedder;
``index=False`` skips embedding, a field list embeds
only those value fields.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from pymongo.database import Database

from semvec_atlas.guard import TenantScope

__all__ = ["SemvecStore"]

_STORE_MARKER = "basestore"  # discriminates adapter items from engine-synced items


class SemvecStore(BaseStore):
    """LangGraph store on the semvec-atlas ``memory_items`` collection.

    Migration from the native store is one line::

        store = SemvecStore(db, scope=TenantScope("acme", "advisor-01"),
                            embedder=my_embedder)
    """

    supports_ttl: bool = False

    def __init__(
        self,
        db: Database,
        *,
        scope: TenantScope,
        embedder: Any,
        retrieval: Any | None = None,
    ) -> None:
        if scope.agent is None:
            raise ValueError("SemvecStore requires an agent-bound TenantScope")
        self._db = db
        self._scope = scope
        self._embedder = embedder
        self._retrieval = retrieval  # RetrievalEngine, wired in M3
        self._col = db.memory_items

    @property
    def scope(self) -> TenantScope:
        return self._scope

    # ------------------------------------------------------------------
    # BaseStore abstract surface

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                results.append(self._put(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:  # pragma: no cover - future op types
                raise NotImplementedError(f"unsupported op: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # PyMongo is synchronous; run the batch off the event loop so
        # async graph nodes never block (mirrors the semvec wheel's
        # update_state_async pattern).
        materialized = list(ops)
        return await asyncio.to_thread(self.batch, materialized)

    # ------------------------------------------------------------------
    # op handlers

    def _item_filter(self, namespace: tuple[str, ...], key: str) -> dict[str, Any]:
        return self._scope.filter(
            {"ns": list(namespace), "key": key, "kind": _STORE_MARKER, "deleted": False}
        )

    def _get(self, op: GetOp) -> Item | None:
        doc = self._col.find_one(self._item_filter(op.namespace, op.key))
        return self._to_item(doc) if doc else None

    def _put(self, op: PutOp) -> None:
        if op.ttl is not None:
            raise NotImplementedError(
                "TTL is not supported by SemvecStore yet (the short-tier TTL "
                "safety net is collection-level; per-item TTL parity is "
                "on the roadmap)"
            )
        if op.value is None:  # PutOp with value=None is a delete
            self._col.update_one(
                self._item_filter(op.namespace, op.key),
                {"$set": {"deleted": True, "updated_at": datetime.now(timezone.utc)}},
            )
            return None

        now = datetime.now(timezone.utc)
        embedding = self._embed_value(op.value, op.index)
        update = {
            "$set": {
                "value": op.value,
                "summary": self._text_repr(op.value, op.index),
                "embedding": embedding,
                "evicted": False,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        }
        self._col.update_one(
            self._item_filter(op.namespace, op.key), update, upsert=True
        )
        return None

    def _search(self, op: SearchOp) -> list[SearchItem]:
        cond = {"kind": _STORE_MARKER, "deleted": False}
        if op.namespace_prefix:
            for i, part in enumerate(op.namespace_prefix):
                cond[f"ns.{i}"] = part
        if op.filter:
            for field, expected in op.filter.items():
                cond[f"value.{field}"] = expected

        if op.query is None:
            docs = list(
                self._col.find(self._scope.filter(cond))
                .sort([("updated_at", -1), ("key", 1)])
                .skip(op.offset)
                .limit(op.limit)
            )
            return [self._to_search_item(d) for d in docs]

        if self._retrieval is not None:  # hybrid path
            return self._retrieval.search_store_items(
                query=op.query, condition=cond, limit=op.limit, offset=op.offset
            )

        # Fallback ranking until the RetrievalEngine is wired: client-side
        # cosine over the candidate set (same embedder as ingest).
        import numpy as np

        q = np.asarray(self._embedder.get_embedding(op.query), dtype=float)
        qn = float(np.linalg.norm(q))
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self._col.find(self._scope.filter(cond)):
            emb = doc.get("embedding")
            if emb is None:
                continue
            v = np.asarray(emb, dtype=float)
            vn = float(np.linalg.norm(v))
            score = float(q @ v / (qn * vn)) if qn > 0 and vn > 0 else 0.0
            scored.append((score, doc))
        scored.sort(key=lambda t: (-t[0], t[1]["key"]))
        page = scored[op.offset : op.offset + op.limit]
        return [self._to_search_item(d, score=s) for s, d in page]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": self._scope.filter({"kind": _STORE_MARKER, "deleted": False})},
            {"$group": {"_id": "$ns"}},
        ]
        namespaces = {tuple(d["_id"]) for d in self._col.aggregate(pipeline)}

        for cond in op.match_conditions or ():
            namespaces = {ns for ns in namespaces if _matches(ns, cond)}
        if op.max_depth is not None:
            namespaces = {ns[: op.max_depth] for ns in namespaces}
        ordered = sorted(namespaces)
        return ordered[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------
    # mapping helpers

    def _embed_value(
        self, value: dict[str, Any], index: Any
    ) -> list[float] | None:
        if index is False:
            return None
        text = self._text_repr(value, index)
        if not text:
            return None
        vec = self._embedder.get_embedding(text)
        return [float(x) for x in vec]

    @staticmethod
    def _text_repr(value: dict[str, Any], index: Any) -> str:
        if isinstance(index, list):
            parts = [str(value[f]) for f in index if f in value]
        else:
            parts = [str(v) for v in value.values()]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _timestamps(doc: dict[str, Any]) -> tuple[datetime, datetime]:
        created, updated = doc["created_at"], doc["updated_at"]
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return created, updated

    def _to_item(self, doc: dict[str, Any]) -> Item:
        created, updated = self._timestamps(doc)
        return Item(
            value=doc["value"],
            key=doc["key"],
            namespace=tuple(doc["ns"]),
            created_at=created,
            updated_at=updated,
        )

    def _to_search_item(
        self, doc: dict[str, Any], score: float | None = None
    ) -> SearchItem:
        created, updated = self._timestamps(doc)
        return SearchItem(
            namespace=tuple(doc["ns"]),
            key=doc["key"],
            value=doc["value"],
            created_at=created,
            updated_at=updated,
            score=score,
        )


def _matches(ns: tuple[str, ...], cond: MatchCondition) -> bool:
    path = tuple(cond.path)
    if cond.match_type == "prefix":
        head = ns[: len(path)]
        return all(p == "*" or p == h for p, h in zip(path, head, strict=False)) and len(head) == len(path)
    if cond.match_type == "suffix":
        tail = ns[-len(path):] if len(ns) >= len(path) else ns
        return all(p == "*" or p == t for p, t in zip(path, tail, strict=False)) and len(tail) == len(path)
    return True  # pragma: no cover - unknown match types pass through
