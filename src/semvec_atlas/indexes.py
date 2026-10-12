"""Idempotent collection, index, and validator provisioning.

Collections follow the specification's data model:

* ``state_snapshots`` — serialized engine states (``to_bytes`` blobs).
* ``memory_items`` — structured memory (engine-synced items and
  LangGraph ``BaseStore`` items share one collection and one guard).
* ``literal_cache`` — verbatim entries for exact text recall.
* ``compliance_events`` — append-only event store (shape enforced by a
  JSON-schema validator; append-only is enforced by role separation in
  Atlas and by ``MongoEventStore``, which never updates in place).
* ``deletion_certificates`` — signed Art. 17 forget certificates.
* ``tenants`` — per-tenant configuration (embedder model/dimension,
  fusion weights, retention).

``provision()`` may be called at every service start: existing
collections, indexes, and search indexes are left untouched.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from pymongo.database import Database
from pymongo.errors import CollectionInvalid, OperationFailure

from semvec_atlas.client_metadata import announce

logger = logging.getLogger(__name__)

#: Business records that live beside the memory collections and may be
#: written inside a turn's transaction (see ``commit.DomainWrite``).
#:
#: This tuple is an allowlist, not an inventory. ``domain_writes`` is a
#: generic upsert primitive driven by caller-supplied names, so without
#: a closed set it would reach ``compliance_events`` — a log whose whole
#: value is that only the event store appends to it.
#:
#: Domain records are scoped to the **tenant**, not to (tenant, agent):
#: several specialist agents work the same case, and binding a claim to
#: whichever agent opened it would hide it from the rest.
DOMAIN_COLLECTIONS = (
    "claims",
    "policies",
)

#: The multi-agent coordination layer's mirror (see ``cortex_store``).
CORTEX_COLLECTIONS = (
    "cortex_agents",
    "cortex_observations",
    "cortex_proposals",
)

COLLECTIONS = (
    "state_snapshots",
    "memory_items",
    "literal_cache",
    "compliance_events",
    "deletion_certificates",
    "tenants",
    "quarantined_inputs",
    "firewall_anchors",
    *DOMAIN_COLLECTIONS,
    *CORTEX_COLLECTIONS,
)

# Insert *shape* of one compliance event. Mirrors semvec's MemoryEvent
# (event_id, created_at, content, embedding, meta, source_event_id) plus
# the Atlas-side partition (tenant, agent) and the monotonic seq the
# adapter assigns.
_EVENT_VALIDATOR: dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "tenant",
            "agent",
            "seq",
            "event_id",
            "created_at",
            "content",
            "embedding",
            "meta",
        ],
        "properties": {
            "tenant": {"bsonType": "string", "minLength": 1},
            "agent": {"bsonType": "string", "minLength": 1},
            "seq": {"bsonType": ["int", "long"], "minimum": 1},
            "event_id": {"bsonType": "string", "minLength": 1},
            "created_at": {"bsonType": ["date", "string"]},
            "content": {"bsonType": "string"},
            "embedding": {"bsonType": "array"},
            "meta": {"bsonType": "object"},
            "source_event_id": {"bsonType": ["string", "null"]},
        },
    }
}


def _ensure_collection(db: Database, name: str, **opts: Any) -> None:
    # Idempotent: an existing collection is the expected steady state.
    with contextlib.suppress(CollectionInvalid):
        db.create_collection(name, **opts)


def _ensure_search_index(collection, model: dict[str, Any]) -> None:
    existing = {ix["name"] for ix in collection.list_search_indexes()}
    if model["name"] in existing:
        return
    try:
        collection.create_search_index(model)
    except OperationFailure as exc:  # pragma: no cover - race on parallel start
        if "already exists" not in str(exc).lower():
            raise


def provision(
    db: Database,
    *,
    dimension: int,
    search_indexes: bool = True,
    ttl_short_seconds: int | None = None,
) -> None:
    """Create all collections, validators, and indexes. Safe to re-run.

    ``dimension`` is the tenant's embedder dimension (the vector
    index runs on client-side embeddings). ``ttl_short_seconds``
    arms the TTL safety net on evicted short-tier items;
    ``None`` leaves TTL disarmed.
    """
    announce(db.client)
    for name in COLLECTIONS:
        if name == "compliance_events":
            _ensure_collection(db, name, validator=_EVENT_VALIDATOR)
        else:
            _ensure_collection(db, name)

    # --- standard indexes (create_index is idempotent by definition) ---
    db.memory_items.create_index(
        [("tenant", 1), ("agent", 1), ("ns", 1), ("key", 1)],
        name="ix_ns",
        unique=True,
        partialFilterExpression={"deleted": False},
    )
    db.state_snapshots.create_index(
        [("tenant", 1), ("agent", 1), ("session", 1), ("turn", -1)],
        name="ix_snap",
        unique=True,
    )
    db.compliance_events.create_index(
        [("tenant", 1), ("agent", 1), ("seq", 1)],
        name="ix_evt",
        unique=True,
    )
    db.literal_cache.create_index(
        [("tenant", 1), ("agent", 1), ("memory_id", 1)],
        name="ix_lit",
    )
    db.deletion_certificates.create_index(
        [("tenant", 1), ("agent", 1), ("created_at", -1)],
        name="ix_cert",
    )
    db.quarantined_inputs.create_index(
        [("tenant", 1), ("agent", 1), ("status", 1), ("created_at", -1)],
        name="ix_qi",
    )
    # Anchors are tenant-wide by design (inoculation) — no agent key.
    db.firewall_anchors.create_index(
        [("tenant", 1), ("created_at", -1)],
        name="ix_fwa",
    )
    # Domain records: tenant-wide, unique per business key, so every
    # agent of the tenant reaches the same document and a repeated
    # upsert updates rather than duplicates.
    db.claims.create_index(
        [("tenant", 1), ("claim_id", 1)],
        name="ix_claim",
        unique=True,
    )
    db.policies.create_index(
        [("tenant", 1), ("policy_number", 1)],
        name="ix_policy",
        unique=True,
    )
    # Cortex mirror: participation is one row per (case, agent);
    # readings and decisions are series, read oldest-first per case.
    db.cortex_agents.create_index(
        [("tenant", 1), ("case", 1), ("instance_id", 1)],
        name="ix_cortex_agent",
        unique=True,
    )
    db.cortex_observations.create_index(
        [("tenant", 1), ("case", 1), ("recorded_at", 1)],
        name="ix_cortex_obs",
    )
    # Unique: a recorded decision must not be rewritten under its own id.
    db.cortex_proposals.create_index(
        [("tenant", 1), ("proposal_id", 1)],
        name="ix_cortex_prop",
        unique=True,
    )
    if ttl_short_seconds is not None:
        db.memory_items.create_index(
            [("updated_at", 1)],
            name="ix_ttl",
            expireAfterSeconds=ttl_short_seconds,
            partialFilterExpression={"tier": "short", "evicted": True},
        )

    # --- Atlas Search / Vector Search indexes ---
    if search_indexes:
        _ensure_search_index(
            db.memory_items,
            {
                "name": "ix_vec",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": dimension,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "tenant"},
                        {"type": "filter", "path": "agent"},
                        {"type": "filter", "path": "tier"},
                        {"type": "filter", "path": "evicted"},
                        {"type": "filter", "path": "deleted"},
                    ]
                },
            },
        )
        _ensure_search_index(
            db.memory_items,
            {
                "name": "ix_lex",
                "type": "search",
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "summary": {
                                "type": "string",
                                "analyzer": "lucene.german",
                                "multi": {
                                    "standard": {
                                        "type": "string",
                                        "analyzer": "lucene.standard",
                                    }
                                },
                            },
                            "tenant": {"type": "token"},
                            "agent": {"type": "token"},
                            "deleted": {"type": "boolean"},
                            "evicted": {"type": "boolean"},
                        },
                    }
                },
            },
        )
        logger.info("search indexes ensured on memory_items (dim=%d)", dimension)
