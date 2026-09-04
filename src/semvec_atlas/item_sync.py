"""Memory-item synchronisation via tier diff.

For turns without a mutation feed, this
module derives mutations by diffing the public tier enumeration
(``state.memory.short_term / medium_term / long_term``) against the
persisted view:

* id present in engine, absent in DB          → **add**
* id present in both, tier/access changed     → **update**
* id absent in engine, present live in DB     → **evict**

UUIDv7 item ids (time-ordered, stable across ``to_bytes`` round-trips)
make the diff cheap. One id may sit in several tier lists at once
(multi-resolution memory); the persisted ``tier`` is
the highest membership, the full set is kept in ``tiers``.

Engine internals stay opaque: importance/protection/semantic-hash values are persisted
only inside the opaque ``engine_meta`` blob — never as queryable
fields. Queries rank on ``access`` and recency, not on engine scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from bson.binary import Binary
from pymongo import UpdateOne
from pymongo.database import Database

from semvec_atlas.guard import TenantGuardViolation, TenantScope

__all__ = ["MemoryItemSync", "SyncReport"]

#: Item-document fields the engine mirror owns. A ``domain_link`` that
#: touched one of these would let a caller write engine state through a
#: parameter meant for business identifiers.
_MIRROR_FIELDS = frozenset(
    {
        "kind",
        "summary",
        "embedding",
        "tier",
        "tiers",
        "access",
        "item_meta",
        "engine_meta",
        "evicted",
        "deleted",
        "updated_at",
        "created_at",
        "turn_added",
        "key",
        "ns",
        "tenant",
        "agent",
    }
)

# Order matters: index == rank, highest wins.
TIER_ORDER = ("short", "medium", "long")


@dataclass(frozen=True)
class SyncReport:
    """Outcome of one sync pass.

    ``skipped`` marks the feed fast path: the turn's mutation feed was
    empty, so no reads or writes ran at all.
    """

    added: int
    updated: int
    evicted: int
    skipped: bool = False

    @property
    def changed(self) -> int:
        return self.added + self.updated + self.evicted


def _tier_view(state: Any) -> dict[str, dict[str, Any]]:
    """Collapse the engine's tier lists into ``{item_id: view}``.

    The view holds exactly what the public MemoryUnit surface exposes;
    engine-internal scores go into the opaque ``engine_meta`` payload.
    """
    view: dict[str, dict[str, Any]] = {}
    mem = state.memory
    for tier_name, units in (
        ("short", mem.short_term),
        ("medium", mem.medium_term),
        ("long", mem.long_term),
    ):
        for unit in units:
            entry = view.get(unit.id)
            if entry is None:
                entry = {
                    "summary": unit.text,
                    "embedding": [float(x) for x in unit.embedding],
                    "tiers": [],
                    "access_count": int(unit.access_count),
                    "meta": dict(unit.meta) if unit.meta else {},
                    "engine_meta": _opaque_engine_meta(unit),
                }
                view[unit.id] = entry
            entry["tiers"].append(tier_name)
    return view


def _opaque_engine_meta(unit: Any) -> bytes:
    """Serialize engine-internal values as an opaque blob.

    Content is intentionally undocumented and unqueryable; it exists so
    a future engine version can rehydrate without recomputation.
    """
    payload = {
        "importance": float(unit.importance),
        "protection_score": float(unit.protection_score),
        "timestamp": float(unit.timestamp),
        "semantic_hash": str(unit.semantic_hash),
    }
    return json.dumps(payload, sort_keys=True).encode()


def _highest_tier(tiers: list[str]) -> str:
    return max(tiers, key=TIER_ORDER.index)


class MemoryItemSync:
    """Tier-diff synchroniser bound to one :class:`TenantScope`."""

    def __init__(self, db: Database, scope: TenantScope) -> None:
        self._db = db
        self._scope = scope
        self._items = db.memory_items
        self._literals = db.literal_cache

    @property
    def scope(self) -> TenantScope:
        return self._scope

    def sync(
        self,
        state: Any,
        *,
        session: str,
        turn: int,
        mongo_session: Any = None,
        mutations: list[dict[str, Any]] | None = None,
        domain_link: dict[str, Any] | None = None,
        force_diff: bool = False,
        _tier_view_override: dict[str, dict[str, Any]] | None = None,
    ) -> SyncReport:
        """Diff engine tiers against the persisted view and apply changes.

        ``mutations`` is the engine's per-turn feed
        (``update(include_mutations=True)``, semvec >= 0.8.5). Three
        modes:

        * ``None`` — no feed available: always run the full diff
          (pre-0.8.5 behavior, and the periodic safety net's view).
        * ``[]`` — feed reports an unchanged turn: skip everything
          (fast path) unless ``force_diff`` is set.
        * non-empty — run the diff, then enrich ``merged`` events by
          stamping ``superseded_by`` on the source documents so
          consolidations stay auditable instead of reading as plain
          evictions.

        ``_tier_view_override`` exists for tests only: real eviction and
        promotion decisions belong to the engine and cannot be
        forced through public API without volume.
        """
        if domain_link:
            clashing = sorted(set(domain_link) & _MIRROR_FIELDS)
            if clashing:
                raise TenantGuardViolation(
                    f"domain_link may not set mirror field(s) {clashing} — "
                    "those belong to the engine's item state"
                )
        if mutations is not None and not mutations and not force_diff:
            return SyncReport(added=0, updated=0, evicted=0, skipped=True)
        engine_view = (
            _tier_view(state) if _tier_view_override is None else _tier_view_override
        )
        ns = ["semvec", session]
        now = datetime.now(timezone.utc)

        persisted: dict[str, dict[str, Any]] = {
            d["key"]: d
            for d in self._items.find(
                self._scope.filter({"ns": ns, "deleted": False}),
                session=mongo_session,
            )
        }

        ops: list[UpdateOne] = []
        added = updated = evicted = 0

        for item_id, view in engine_view.items():
            doc_update = {
                "kind": "engine",  # discriminator vs BaseStore items
                "summary": view["summary"],
                "embedding": view["embedding"],
                "tier": _highest_tier(view["tiers"]),
                "tiers": sorted(set(view["tiers"]), key=TIER_ORDER.index),
                "access": {"count": view["access_count"], "last": now},
                "item_meta": view["meta"],
                "engine_meta": Binary(view["engine_meta"]),
                "evicted": False,
                "updated_at": now,
            }
            if domain_link:
                # Identifying fields of the business record this turn
                # concerns, so retrieval can join a ranked hit to it.
                # Persisted on the document only — never handed to the
                # engine, so replay stays checksum-faithful.
                #
                # Merged last but checked first: the link names business
                # keys, and a link that reached ``tier`` or ``embedding``
                # would let a caller write engine-owned mirror state.
                doc_update.update(domain_link)
            existing = persisted.get(item_id)
            if existing is None:
                added += 1
                ops.append(
                    UpdateOne(
                        self._scope.filter({"ns": ns, "key": item_id, "deleted": False}),
                        {
                            "$set": doc_update,
                            "$setOnInsert": {"created_at": now, "turn_added": turn},
                        },
                        upsert=True,
                    )
                )
            elif self._differs(existing, doc_update):
                updated += 1
                ops.append(
                    UpdateOne(
                        self._scope.filter({"ns": ns, "key": item_id, "deleted": False}),
                        {"$set": doc_update},
                    )
                )

        for item_id, existing in persisted.items():
            if item_id not in engine_view and not existing.get("evicted", False):
                evicted += 1
                ops.append(
                    UpdateOne(
                        self._scope.filter({"ns": ns, "key": item_id, "deleted": False}),
                        {"$set": {"evicted": True, "updated_at": now, "turn_evicted": turn}},
                    )
                )

        if ops:
            self._items.bulk_write(ops, ordered=False, session=mongo_session)

        if mutations:
            self._apply_merge_enrichment(
                mutations, ns=ns, now=now, mongo_session=mongo_session
            )

        self._sync_literals(state, session=session, mongo_session=mongo_session)
        return SyncReport(added=added, updated=updated, evicted=evicted)

    def _apply_merge_enrichment(
        self,
        mutations: list[dict[str, Any]],
        *,
        ns: list[str],
        now: datetime,
        mongo_session: Any = None,
    ) -> None:
        """Stamp ``superseded_by`` on the sources of ``merged`` events.

        The diff already marked vanished sources ``evicted``; the feed
        tells us *why* — they were consolidated into a new unit. The
        link makes the lineage queryable (audit: "where did this
        memory go?").
        """
        ops = [
            UpdateOne(
                self._scope.filter({"ns": ns, "key": src}),
                {"$set": {"superseded_by": m["id"], "updated_at": now}},
            )
            for m in mutations
            if m.get("kind") == "merged"
            for src in (m.get("sources") or [])
        ]
        if ops:
            self._items.bulk_write(ops, ordered=False, session=mongo_session)

    @staticmethod
    def _differs(existing: dict[str, Any], update: dict[str, Any]) -> bool:
        if existing.get("tiers") != update["tiers"]:
            return True
        if existing.get("access", {}).get("count") != update["access"]["count"]:
            return True
        if bytes(existing.get("engine_meta", b"")) != bytes(update["engine_meta"]):
            return True
        # True when the item re-appeared after eviction.
        return bool(existing.get("evicted", False))

    # ------------------------------------------------------------------
    # literal cache

    def _sync_literals(
        self, state: Any, *, session: str, mongo_session: Any = None
    ) -> None:
        """Best-effort persistence of verbatim literal-cache entities.

        Reads the *serialized* public view (``literal_cache.to_dict()``)
        so the sync never depends on unstable object shapes. Entries are
        keyed by their serialized identity; content is stored verbatim
        (Queryable Encryption candidate).
        """
        cache = getattr(state, "literal_cache", None)
        if cache is None:
            return
        try:
            entities = cache.to_dict().get("entities") or []
        except Exception:
            return
        if not entities:
            return
        now = datetime.now(timezone.utc)
        ops = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            key = ent.get("key") or ent.get("name") or json.dumps(ent, sort_keys=True)
            ops.append(
                UpdateOne(
                    self._scope.filter({"session": session, "entity_key": key}),
                    {
                        "$set": {
                            "content": ent.get("value") or ent.get("content") or "",
                            "kind": ent.get("kind"),
                            "memory_id": ent.get("memory_id"),
                            "raw": ent,
                            "updated_at": now,
                        },
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            )
        if ops:
            self._literals.bulk_write(ops, ordered=False, session=mongo_session)
