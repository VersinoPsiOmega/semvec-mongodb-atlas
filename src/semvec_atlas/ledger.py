# ABOUTME: Reconciles a scope's counts: event log into memory turns and markers, items into engine and basestore.
# ABOUTME: Lets a reader check events_wiped against replayed_events and item totals against turns.
"""Ledgers over a scope's event log and memory items.

An audited event stream holds two kinds of rows: memory turns, which
recovery replays into the engine, and content-free marker events
(firewall ``quarantine`` and ``inoculation``, Art. 17 ``forget``),
which prove *that* something happened without carrying *what* and are
excluded from replay. ``memory_items`` likewise holds engine-synced
turns next to LangGraph ``BaseStore`` items. The ledgers split those
counts so ``events_wiped``, ``replayed_events`` and the item total
reconcile at a glance.

Both read through :meth:`TenantScope.filter`, like every other query.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymongo.database import Database

from semvec_atlas.guard import TenantScope

__all__ = ["EventLedger", "ItemLedger", "event_ledger", "item_ledger"]

#: Where marker events carry their kind; memory turns have no such field.
_MARKER_PATH = "meta.semvec_atlas.marker"


@dataclass(frozen=True)
class EventLedger:
    """``total`` events = ``memory`` replayable turns + the ``markers`` by kind."""

    total: int
    memory: int
    markers: dict[str, int]

    @property
    def marker_total(self) -> int:
        return sum(self.markers.values())


@dataclass(frozen=True)
class ItemLedger:
    """``total`` live items = ``engine``-synced turns + ``basestore`` items."""

    total: int
    engine: int
    basestore: int


def event_ledger(db: Database, scope: TenantScope) -> EventLedger:
    """Count the scope's events, split into memory turns and markers by kind."""
    rows = db.compliance_events.aggregate([
        {"$match": scope.filter()},
        {"$group": {"_id": f"${_MARKER_PATH}", "n": {"$sum": 1}}},
    ])
    counts = {row["_id"]: int(row["n"]) for row in rows}
    memory = counts.pop(None, 0)
    return EventLedger(total=memory + sum(counts.values()), memory=memory, markers=counts)


def item_ledger(db: Database, scope: TenantScope) -> ItemLedger:
    """Count the scope's live items, split into engine-synced and basestore."""
    rows = db.memory_items.aggregate([
        {"$match": scope.filter({"deleted": False})},
        {"$group": {"_id": "$kind", "n": {"$sum": 1}}},
    ])
    counts = {row["_id"]: int(row["n"]) for row in rows}
    return ItemLedger(
        total=sum(counts.values()),
        engine=counts.get("engine", 0),
        basestore=counts.get("basestore", 0),
    )
