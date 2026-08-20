"""Durable snapshot persistence for ``SemvecState``.

Pure orchestration, mirroring the semvec wheel's ``StateStore``: the
blob is the engine's own checksummed binary format
(``to_bytes``/``from_bytes``); no semantic math lives here. Writes are
optimistic-concurrency guarded (monotonic ``turn`` per session, backed
by a unique index) and never silently overwrite.

Integrity is checked on two levels:

* transport — SHA-256 of the stored blob, verified on load;
* engine — the Rust core re-verifies its embedded checksum inside
  ``from_bytes`` (tamper or corruption raises loudly).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from bson.binary import Binary
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from semvec_atlas.guard import TenantScope

__all__ = [
    "AtlasPersistence",
    "SnapshotConflict",
    "SnapshotMeta",
    "blob_codec_version",
    "checksums_comparable",
]

SCHEMA_VERSION = 1

_SNAPSHOT_MAGIC = b"SVB"


def blob_codec_version(blob: bytes) -> int:
    """Return the engine snapshot codec version from the blob header.

    Every semvec snapshot starts with the ``SVB`` magic followed by one
    codec-version byte (verified across 0.7.6 → 0.8.5). The codec
    version is recorded per snapshot so a mixed-fleet
    operator can audit which engine versions can read which snapshots
    — newer codecs are not readable by older wheels.
    """
    if len(blob) < 4 or blob[:3] != _SNAPSHOT_MAGIC:
        raise ValueError(
            "not a semvec snapshot blob — missing SVB magic header"
        )
    return blob[3]


def checksums_comparable(engine_version_a: str, engine_version_b: str) -> bool:
    """Whether two engine checksums may be compared at all.

    Codifies a verified engine property: the
    engine checksum is an intra-version integrity value. State
    corrections between releases legitimately change the value without
    any algorithm change, so cross-version equality carries no signal.
    Cross-version state identity requires full replay in one target
    version instead.
    """
    return engine_version_a == engine_version_b


class SnapshotConflict(RuntimeError):
    """A snapshot write lost the monotonic-turn race (never overwrite)."""


@dataclass(frozen=True)
class SnapshotMeta:
    """Metadata of one persisted snapshot."""

    session: str
    turn: int
    blob_sha256: str
    engine_checksum: str
    engine_version: str
    schema_version: int
    created_at: datetime
    privacy_flags: dict[str, bool]
    # Sequence number of the last compliance event folded into this
    # snapshot — the replay boundary for recovery. ``None``
    # for snapshots taken outside the committed turn path.
    last_event_seq: int | None = None
    # Engine snapshot codec version from the blob header;
    # newer codecs are unreadable by older wheels — mixed fleets audit
    # on this field. 0 only for legacy rows written before the field.
    codec_version: int = 0


class AtlasPersistence:
    """Snapshot store bound to one :class:`TenantScope`.

    ``snapshot_retention`` keeps the newest N snapshots per session
   ; ``None`` keeps everything (archival is an Atlas
    Online-Archive concern, not a delete).
    """

    def __init__(
        self,
        db: Database,
        scope: TenantScope,
        *,
        snapshot_retention: int | None = None,
    ) -> None:
        self._db = db
        self._scope = scope
        self._retention = snapshot_retention
        self._col = db.state_snapshots

    @property
    def scope(self) -> TenantScope:
        return self._scope

    # ------------------------------------------------------------------
    # write path

    def save_snapshot(
        self,
        state: Any,
        *,
        session: str,
        turn: int,
        include_memory_text: bool = True,
        include_literal_cache_text: bool = True,
        include_adaptive_params: bool = True,
        last_event_seq: int | None = None,
        mongo_session: Any = None,
    ) -> SnapshotMeta:
        """Persist ``state`` for ``(session, turn)`` and return metadata.

        ``turn`` must be strictly greater than the latest persisted turn
        of the session — a stale or duplicate writer gets
        :class:`SnapshotConflict`, never a silent overwrite. The
        ``include_*`` flags are the engine's privacy toggles
        and are persisted alongside the blob so a reader knows what the
        snapshot contains.
        """
        latest = self._col.find_one(
            self._scope.filter({"session": session}),
            sort=[("turn", -1)],
            projection={"turn": 1},
            session=mongo_session,
        )
        if latest is not None and turn <= latest["turn"]:
            raise SnapshotConflict(
                f"snapshot turn {turn} is not newer than persisted turn "
                f"{latest['turn']} for session {session!r}"
            )

        privacy_flags = {
            "include_memory_text": include_memory_text,
            "include_literal_cache_text": include_literal_cache_text,
            "include_adaptive_params": include_adaptive_params,
        }
        blob = state.to_bytes(compress=True, **privacy_flags)
        codec_version = blob_codec_version(blob)
        engine_checksum = state.to_dict()["checksum"]
        import semvec

        now = datetime.now(timezone.utc)
        doc = self._scope.filter(
            {
                "session": session,
                "turn": turn,
                "blob": Binary(blob),
                "blob_sha256": hashlib.sha256(blob).hexdigest(),
                "engine_checksum": engine_checksum,
                "engine_version": semvec.__version__,
                "schema_version": SCHEMA_VERSION,
                "privacy_flags": privacy_flags,
                "last_event_seq": last_event_seq,
                "codec_version": codec_version,
                "created_at": now,
            }
        )
        try:
            self._col.insert_one(doc, session=mongo_session)
        except DuplicateKeyError as exc:
            raise SnapshotConflict(
                f"concurrent writer already persisted turn {turn} for session {session!r}"
            ) from exc

        if self._retention is not None:
            self._enforce_retention(session, mongo_session)

        return SnapshotMeta(
            session=session,
            turn=turn,
            blob_sha256=doc["blob_sha256"],
            engine_checksum=engine_checksum,
            engine_version=semvec.__version__,
            schema_version=SCHEMA_VERSION,
            created_at=now,
            privacy_flags=privacy_flags,
            last_event_seq=last_event_seq,
            codec_version=codec_version,
        )

    def _enforce_retention(self, session: str, mongo_session: Any = None) -> None:
        keep = [
            d["turn"]
            for d in self._col.find(
                self._scope.filter({"session": session}),
                sort=[("turn", -1)],
                limit=self._retention,
                projection={"turn": 1},
                session=mongo_session,
            )
        ]
        if keep:
            self._col.delete_many(
                self._scope.filter({"session": session, "turn": {"$lt": min(keep)}}),
                session=mongo_session,
            )

    # ------------------------------------------------------------------
    # read path

    def load_latest_snapshot(self, *, session: str) -> tuple[Any, SnapshotMeta] | None:
        """Restore the newest snapshot of ``session`` or ``None``.

        Verifies the transport SHA-256 before handing the blob to the
        engine; the engine then verifies its own embedded checksum in
        ``from_bytes``. Corruption raises — never a degraded state.
        """
        doc = self._col.find_one(
            self._scope.filter({"session": session}), sort=[("turn", -1)]
        )
        if doc is None:
            return None
        blob = bytes(doc["blob"])
        if hashlib.sha256(blob).hexdigest() != doc["blob_sha256"]:
            raise ValueError(
                f"snapshot integrity check failed for session {session!r} "
                f"turn {doc['turn']} — stored SHA-256 does not match blob"
            )
        from semvec import SemvecState

        state = SemvecState.from_bytes(blob)
        created = doc["created_at"]
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        meta = SnapshotMeta(
            session=doc["session"],
            turn=doc["turn"],
            blob_sha256=doc["blob_sha256"],
            engine_checksum=doc["engine_checksum"],
            engine_version=doc["engine_version"],
            schema_version=doc["schema_version"],
            created_at=created,
            privacy_flags=doc["privacy_flags"],
            last_event_seq=doc.get("last_event_seq"),
            codec_version=doc.get("codec_version", 0),
        )
        return state, meta
