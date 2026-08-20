"""Art. 17 forget and audit on Atlas.

The forget *procedure* is the semvec wheel's ``forget_user`` — event
wipe plus RS256-signed :class:`DeletionCertificate`, unchanged
(no new evidence format). This module adds the
Atlas-side orchestration:

* hard-delete of ``memory_items`` / ``literal_cache`` / snapshots for
  the scope (snapshots embed the forgotten content — they must go;
  the event log stays authoritative for recovery);
* certificate persistence (``deletion_certificates``) with timestamps
  stored as ISO strings so signature verification round-trips
  byte-exactly (BSON dates truncate to milliseconds);
* a content-free **forget marker event**: append-only holds, the log
  proves the forgetting without carrying the forgotten;
* :meth:`AtlasForget.audit` — the verified certificate chain behind
  the ``memory.audit`` tool.

Key handling: RS256 private key via ``priv_pem`` or the wheel's env
resolution (``SEMVEC_COMPLIANCE_PRIVKEY_FILE`` / ``_PEM``). Key
resolution failures abort **before** any deletion (wheel guarantee,
held for the Atlas stores as well).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from pymongo.database import Database
from semvec.compliance.certificates import (
    DeletionCertificate,
    resolve_priv_pem,
    verify_certificate,
)
from semvec.compliance.event_store import MemoryEvent
from semvec.compliance.retention import AuditLogEntry, forget_user

from semvec_atlas.event_store import MongoEventStore
from semvec_atlas.firewall import AnchorStore, QuarantineVault
from semvec_atlas.guard import TenantScope

__all__ = ["AtlasForget", "AuditEntry", "MongoAuditLog", "FORGET_MARKER"]

FORGET_MARKER = {"semvec_atlas": {"marker": "forget"}}


class MongoAuditLog:
    """Durable audit log fulfilling the wheel's ``AuditLog`` duck type."""

    def __init__(self, db: Database, scope: TenantScope) -> None:
        self._scope = scope
        self._col = db["compliance_audit_log"]

    def write(self, entry: AuditLogEntry) -> None:
        self._col.insert_one(
            self._scope.filter(
                {
                    "user_id": entry.user_id,
                    "action": entry.action,
                    "reason": entry.reason,
                    "event_count": entry.event_count,
                    "issued_at": entry.issued_at.isoformat(),
                    "cert_signature_b64": entry.cert_signature_b64,
                    "issuer": entry.issuer,
                }
            )
        )

    def entries_for_user(self, user_id: str) -> list[AuditLogEntry]:
        return [
            AuditLogEntry(
                user_id=d["user_id"],
                action=d["action"],
                reason=d["reason"],
                event_count=d["event_count"],
                issued_at=datetime.fromisoformat(d["issued_at"]),
                cert_signature_b64=d["cert_signature_b64"],
                issuer=d["issuer"],
            )
            for d in self._col.find({"user_id": user_id}).sort("issued_at", 1)
        ]

    def all(self) -> list[AuditLogEntry]:
        return self.entries_for_user(self._scope.user_id)


@dataclass(frozen=True)
class AuditEntry:
    """One verified link of the deletion-certificate chain.

    ``reason`` is the *signed* value (the wheel's ``forget_user`` sets
    it — currently always ``"user_request"``); ``request_reason`` is
    the operator-supplied context stored alongside, outside the
    signature.
    """

    reason: str
    request_reason: str | None
    issuer: str
    event_count: int
    deleted_at: datetime
    signature_b64: str
    valid: bool


class AtlasForget:
    """Forget/audit orchestration bound to one :class:`TenantScope`."""

    def __init__(
        self,
        db: Database,
        scope: TenantScope,
        *,
        priv_pem: bytes | None = None,
        issuer: str = "semvec-atlas",
    ) -> None:
        self._db = db
        self._scope = scope
        self._priv_pem = priv_pem
        self._issuer = issuer
        self._events = MongoEventStore(db, scope)
        self._audit_log = MongoAuditLog(db, scope)

    # ------------------------------------------------------------------

    def forget_all(self, *, reason: str) -> DeletionCertificate:
        """Execute an Art. 17 request for this scope's memory stream.

        Order matters: key resolution first (a misconfiguration must
        never eat data without leaving a verifiable receipt), then the
        wheel's event wipe + certificate, then the Atlas hard deletes,
        then the marker append.
        """
        resolve_priv_pem(self._priv_pem)  # fail fast, before any delete

        cert = forget_user(
            user_id=self._scope.user_id,
            store=self._events,
            audit_log=self._audit_log,
            priv_pem=self._priv_pem,
            issuer=self._issuer,
        )

        # Atlas hard deletes — including tombstoned/evicted rows: an
        # Art. 17 wipe leaves nothing behind.
        self._db.memory_items.delete_many(self._scope.filter())
        self._db.literal_cache.delete_many(self._scope.filter())
        self._db.state_snapshots.delete_many(self._scope.filter())

        # Firewall surfaces: the vault keeps full text and
        # learned anchors derive from user content — both go with the
        # scope. Seed anchors (operator catalog) survive.
        QuarantineVault(self._db, self._scope).wipe()
        AnchorStore(self._db, self._scope).wipe_learned()

        self._persist_certificate(cert, request_reason=reason)
        self._append_marker(cert)
        return cert

    def _persist_certificate(
        self, cert: DeletionCertificate, *, request_reason: str
    ) -> None:
        self._db.deletion_certificates.insert_one(
            self._scope.filter(
                {
                    "user_id": cert.user_id,
                    "event_count": cert.event_count,
                    # ISO string, not BSON date: verification re-canonicalises
                    # the payload and must see the exact signed value.
                    "deleted_at": cert.deleted_at.isoformat(),
                    "reason": cert.reason,
                    "request_reason": request_reason,
                    "issuer": cert.issuer,
                    "signature_b64": cert.signature_b64,
                    "created_at": datetime.now(timezone.utc),
                }
            )
        )

    def _append_marker(self, cert: DeletionCertificate) -> None:
        meta = {k: dict(v) for k, v in FORGET_MARKER.items()}
        meta["semvec_atlas"]["certificate_sig"] = cert.signature_b64[:24]
        self._events.append(
            MemoryEvent.fresh(
                user_id=self._scope.user_id,
                content="",  # content-free by design
                embedding=[],
                meta=meta,
            )
        )

    # ------------------------------------------------------------------

    def audit(self, *, pub_pem: bytes | None = None) -> list[AuditEntry]:
        """Return the certificate chain, each entry re-verified."""
        chain: list[AuditEntry] = []
        for doc in self._db.deletion_certificates.find(self._scope.filter()).sort(
            "created_at", 1
        ):
            cert = DeletionCertificate(
                user_id=doc["user_id"],
                event_count=doc["event_count"],
                deleted_at=datetime.fromisoformat(doc["deleted_at"]),
                reason=doc["reason"],
                issuer=doc["issuer"],
                signature_b64=doc["signature_b64"],
            )
            try:
                valid = verify_certificate(cert, pub_pem=pub_pem)
            except Exception:
                valid = False
            chain.append(
                AuditEntry(
                    reason=cert.reason,
                    request_reason=doc.get("request_reason"),
                    issuer=cert.issuer,
                    event_count=cert.event_count,
                    deleted_at=cert.deleted_at,
                    signature_b64=cert.signature_b64,
                    valid=valid,
                )
            )
        return chain
