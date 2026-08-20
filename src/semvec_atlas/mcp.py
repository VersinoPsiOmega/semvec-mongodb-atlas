"""MCP tools for the Atlas backend.

Structure mirrors the wheel's ``semvec.coding.mcp_server``: plain
``handle_*`` functions carry the logic and are unit-testable without a
transport; ``create_atlas_mcp_server`` wraps them as FastMCP tools.

Tool names use underscores per MCP identifier rules; the specification's
dotted names map 1:1: ``memory.search`` → ``memory_search``,
``memory.forget`` → ``memory_forget``, ``memory.audit`` → ``memory_audit``.

``memory_forget`` deliberately requires an explicit ``reason`` — an
Art. 17 wipe is an audited business action, never a side effect.
"""

from __future__ import annotations

import json
from typing import Any

from pymongo.database import Database

from semvec_atlas.compliance import AtlasForget
from semvec_atlas.firewall import QuarantineVault
from semvec_atlas.guard import TenantScope
from semvec_atlas.retrieval import RetrievalEngine

__all__ = [
    "create_atlas_mcp_server",
    "handle_audit",
    "handle_firewall_status",
    "handle_forget",
    "handle_quarantine_list",
    "handle_quarantine_release",
    "handle_search",
]


def handle_search(
    db: Database,
    scope: TenantScope,
    embedder: Any,
    *,
    query: str,
    k: int = 8,
    force_fallback: bool = False,
) -> str:
    """Hybrid retrieval over the scope's memory items (JSON result)."""
    engine = RetrievalEngine.for_tenant(
        db, scope, embedder=embedder, force_fallback=force_fallback
    )
    hits = engine.search(query, k=k)
    return json.dumps(
        {
            "path": "rankFusion" if engine.uses_rank_fusion else "clientRRF",
            "weights": engine.weights,
            "hits": [
                {
                    "key": h.key,
                    "summary": h.summary,
                    "tier": h.tier,
                    "ns": h.ns,
                    "score": h.score,
                }
                for h in hits
            ],
        }
    )


def handle_forget(
    db: Database,
    scope: TenantScope,
    *,
    reason: str,
    priv_pem: bytes | None = None,
) -> str:
    """Execute an Art. 17 forget for the scope; returns the certificate summary."""
    cert = AtlasForget(db, scope, priv_pem=priv_pem).forget_all(reason=reason)
    return json.dumps(
        {
            "user_id": cert.user_id,
            "event_count": cert.event_count,
            "deleted_at": cert.deleted_at.isoformat(),
            "reason": cert.reason,
            "issuer": cert.issuer,
            "signature_b64": cert.signature_b64,
        }
    )


def handle_audit(
    db: Database, scope: TenantScope, *, pub_pem: bytes | None = None
) -> str:
    """Return the verified deletion-certificate chain (read-only)."""
    chain = AtlasForget(db, scope).audit(pub_pem=pub_pem)
    return json.dumps(
        {
            "user_id": scope.user_id,
            "certificates": [
                {
                    "reason": e.reason,
                    "request_reason": e.request_reason,
                    "issuer": e.issuer,
                    "event_count": e.event_count,
                    "deleted_at": e.deleted_at.isoformat(),
                    "signature_b64": e.signature_b64,
                    "valid": e.valid,
                }
                for e in chain
            ],
        }
    )


def handle_firewall_status(db: Database, scope: TenantScope) -> str:
    """Content-free firewall observability (JSON result).

    Reports the tenant's anchor catalog with provenance (who learned
    what, correlated via ``text_hash`` — never text, never vectors)
    plus the held-quarantine count, so an operator can see the fleet's
    current immunity at a glance.
    """
    tenant_scope = TenantScope(tenant=scope.tenant)
    anchors = [
        {
            "anchor_id": doc["_id"],
            "kind": doc["kind"],
            "agent": (doc.get("provenance") or {}).get("agent"),
            "text_hash": doc.get("text_hash"),
            "created_at": doc["created_at"].isoformat(),
        }
        for doc in db.firewall_anchors.find(
            tenant_scope.filter(), projection={"embedding": False}
        ).sort("created_at", 1)
    ]
    counts: dict[str, int] = {}
    for a in anchors:
        counts[a["kind"]] = counts.get(a["kind"], 0) + 1
    held = db.quarantined_inputs.count_documents(
        scope.filter({"status": "held"})
    )
    return json.dumps(
        {
            "quarantine": {"held": held},
            "anchor_counts": counts,
            "anchors": anchors,
        }
    )


def handle_quarantine_list(db: Database, scope: TenantScope) -> str:
    """Content-free listing of firewall-held inputs (JSON result).

    Never exposes the held text or embedding — preview is redaction-
    capped, correlation runs over the SHA-256 hash.
    """
    entries = QuarantineVault(db, scope).list()
    return json.dumps(
        {
            "entries": [
                {
                    "quarantine_id": e["_id"],
                    "text_preview": e["text_preview"],
                    "text_hash": e["text_hash"],
                    "score": e["score"],
                    "reason": e["reason"],
                    "level": e["level"],
                    "session": e["session"],
                    "turn": e["turn"],
                    "status": e["status"],
                }
                for e in entries
            ]
        }
    )


def handle_quarantine_release(
    db: Database, scope: TenantScope, *, quarantine_id: str, reason: str
) -> str:
    """Release a held input; returns the full text for re-commit.

    A release is an audited human decision (``reason`` is mandatory,
    like ``memory_forget``); the caller re-ingests the returned text
    as a regular turn. Unknown/double releases report an error.
    """
    try:
        doc = QuarantineVault(db, scope).release(quarantine_id, reason=reason)
    except KeyError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {
            "quarantine_id": quarantine_id,
            "text": doc["text"],
            "status": "released",
            "release_reason": reason,
        }
    )


def create_atlas_mcp_server(
    db: Database,
    scope: TenantScope,
    *,
    embedder: Any,
    priv_pem: bytes | None = None,
    pub_pem: bytes | None = None,
):
    """Build the FastMCP server exposing the three Atlas memory tools."""
    from fastmcp import FastMCP

    mcp = FastMCP(
        "semvec-atlas",
        instructions=(
            "Semvec Atlas Edition memory tools. memory_search runs hybrid "
            "retrieval; memory_forget executes a signed GDPR Art. 17 wipe "
            "for this agent's memory stream; memory_audit returns the "
            "verified deletion-certificate chain."
        ),
    )

    @mcp.tool()
    def memory_search(query: str, k: int = 8) -> str:
        """Hybrid search (semantic + lexical + recency) over agent memory."""
        return handle_search(db, scope, embedder, query=query, k=k)

    @mcp.tool()
    def memory_forget(reason: str) -> str:
        """GDPR Art. 17: wipe this agent's memory stream, return signed certificate."""
        return handle_forget(db, scope, reason=reason, priv_pem=priv_pem)

    @mcp.tool()
    def memory_audit() -> str:
        """Return the verified deletion-certificate chain for this agent."""
        return handle_audit(db, scope, pub_pem=pub_pem)

    @mcp.tool()
    def memory_quarantine_list() -> str:
        """List firewall-held inputs (content-free: preview + hash only)."""
        return handle_quarantine_list(db, scope)

    @mcp.tool()
    def memory_quarantine_release(quarantine_id: str, reason: str) -> str:
        """Release a held input (audited decision); returns its text for re-commit."""
        return handle_quarantine_release(
            db, scope, quarantine_id=quarantine_id, reason=reason
        )

    @mcp.tool()
    def memory_firewall_status() -> str:
        """Tenant firewall status: anchor catalog with provenance, held count (content-free)."""
        return handle_firewall_status(db, scope)

    return mcp
