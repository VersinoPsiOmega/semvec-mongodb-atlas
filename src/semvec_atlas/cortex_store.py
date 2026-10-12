"""Persistence for the multi-agent coordination layer.

The engine's cortex (``semvec.cortex``) coordinates several agents
in process: each keeps its own semantic state, a shared observer
aggregates them into one view, and a consensus engine turns the shared
view into recorded decisions. All of it lives in memory, which is
exactly right while a case is being worked and useless once the shift
ends.

This module mirrors that layer into the same database that already
holds the memory items, the compliance events and the business records.
Three collections, each append-mostly:

``cortex_agents``
    One document per (case, agent). Upserted, because participation is
    a running fact — the latest interaction count is the interesting
    one, a row per turn is not.

``cortex_observations``
    One document per observer reading. Insert-only: the series *is* the
    signal. Convergence shows up as resonance rising across rows, and
    collapsing that to a current value would throw away the moment it
    happened.

``cortex_proposals``
    One document per decision, with its rationale, every member's vote
    and the outcome. Insert-only and refused on re-use of an id: a
    decision that can be rewritten afterwards is not a record.

:meth:`AtlasCortexStore.case_timeline` reads all three in one round
trip, which is the point of putting them in one database.

The class also satisfies the engine's persistence seam — one duck-typed
async ``list_active_states()`` — so a ``SemvecCortexService`` can pull
live sessions from Atlas instead of holding them only in memory.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from semvec_atlas.client_metadata import announce
from semvec_atlas.guard import TenantScope
from semvec_atlas.persistence import AtlasPersistence

__all__ = ["AtlasCortexStore", "ObservationRecord", "ProposalConflict"]

logger = logging.getLogger(__name__)


class ProposalConflict(RuntimeError):
    """A decision was recorded twice under one id (never overwrite).

    Mirrors ``persistence.SnapshotConflict``: a write that lost a
    uniqueness race is a conflict, not a scope breach, and callers that
    retry on one must not swallow the other.
    """

AGENTS = "cortex_agents"
OBSERVATIONS = "cortex_observations"
PROPOSALS = "cortex_proposals"

#: Collections this store owns. Provisioned by ``indexes.provision``.
CORTEX_COLLECTIONS = (AGENTS, OBSERVATIONS, PROPOSALS)


@dataclass(frozen=True)
class ObservationRecord:
    """One observer reading, as persisted.

    ``coherence`` and ``resonance`` are the engine's own aggregate
    measures; ``participants`` is how many agents the reading covered,
    which is what makes two readings comparable.
    """

    coherence: float
    resonance: float
    participants: int
    update_count: int
    recorded_at: datetime


class AtlasCortexStore:
    """Cortex persistence bound to one :class:`TenantScope`.

    Records are scoped to the **tenant** rather than to (tenant, agent):
    a coordination layer whose rows belonged to one agent would record
    the opposite of what it exists to show.
    """

    def __init__(self, db: Database, scope: TenantScope) -> None:
        announce(db.client)
        self._db = db
        self._scope = scope
        # Tenant-wide view: the network spans the tenant's agents.
        self._tenant = TenantScope(tenant=scope.tenant)
        self._persistence = AtlasPersistence(db, scope)

    @property
    def scope(self) -> TenantScope:
        return self._scope

    # ------------------------------------------------------------------
    # writes

    def record_agents(
        self,
        agents: Iterable[Any],
        *,
        case: str,
        mongo_session: Any = None,
    ) -> int:
        """Upsert participation for each agent. Returns the count written.

        Accepts anything exposing ``instance_id``, ``interaction_count``
        and ``last_update_time`` — the engine's ``SemvecAgent`` does, and
        duck-typing keeps this module free of a hard import from the
        compiled core.
        """
        now = datetime.now(timezone.utc)
        written = 0
        for agent in agents:
            self._db[AGENTS].update_one(
                self._tenant.filter(
                    {"case": case, "instance_id": agent.instance_id}
                ),
                {
                    "$set": {
                        "interaction_count": int(agent.interaction_count),
                        "last_update_time": float(agent.last_update_time),
                        "updated_at": now,
                    },
                    "$setOnInsert": {"first_seen": now},
                },
                upsert=True,
                session=mongo_session,
            )
            written += 1
        return written

    def record_observation(
        self,
        observer: Any,
        *,
        agents: Sequence[str],
        case: str,
        mongo_session: Any = None,
    ) -> ObservationRecord:
        """Append one observer reading to the case's series."""
        record = ObservationRecord(
            coherence=float(observer.global_coherence),
            resonance=float(observer.network_resonance),
            participants=len(agents),
            update_count=int(observer.update_count),
            recorded_at=datetime.now(timezone.utc),
        )
        self._db[OBSERVATIONS].insert_one(
            self._tenant.filter(
                {
                    "case": case,
                    "coherence": record.coherence,
                    "resonance": record.resonance,
                    "participants": record.participants,
                    "agents": list(agents),
                    "update_count": record.update_count,
                    "recorded_at": record.recorded_at,
                }
            ),
            session=mongo_session,
        )
        return record

    def record_proposal(
        self,
        *,
        proposal_id: str,
        proposal_type: str,
        rationale: str,
        consensus_level: str,
        votes: Mapping[str, bool],
        accepted: bool,
        case: str,
        mongo_session: Any = None,
    ) -> str:
        """Record one decision. Refuses to overwrite an existing id.

        The unique index on (tenant, proposal_id) is what enforces this;
        the raise translates the driver's error into the guard's
        vocabulary so callers handle one exception type.
        """
        try:
            self._db[PROPOSALS].insert_one(
                self._tenant.filter(
                    {
                        "case": case,
                        "proposal_id": proposal_id,
                        "proposal_type": proposal_type,
                        "rationale": rationale,
                        "consensus_level": consensus_level,
                        "votes": dict(votes),
                        "accepted": bool(accepted),
                        "recorded_at": datetime.now(timezone.utc),
                    }
                ),
                session=mongo_session,
            )
        except DuplicateKeyError as exc:
            raise ProposalConflict(
                f"proposal {proposal_id!r} is already recorded — a decision "
                "is append-only and cannot be rewritten"
            ) from exc
        return proposal_id

    # ------------------------------------------------------------------
    # reads

    def case_timeline(self, case: str) -> dict[str, Any]:
        """Everything the coordination layer knows about one case.

        Returns participants, the observation series oldest-first, the
        peak resonance reached, and the decisions taken. ``None`` peak
        means no reading was recorded, which is different from a peak of
        zero.
        """
        agents = list(
            self._db[AGENTS].find(
                self._tenant.filter({"case": case}),
                {"_id": 0, "tenant": 0},
                sort=[("instance_id", ASCENDING)],
            )
        )
        observations = list(
            self._db[OBSERVATIONS].find(
                self._tenant.filter({"case": case}),
                {"_id": 0, "tenant": 0},
                sort=[("recorded_at", ASCENDING)],
            )
        )
        proposals = list(
            self._db[PROPOSALS].find(
                self._tenant.filter({"case": case}),
                {"_id": 0, "tenant": 0},
                sort=[("recorded_at", ASCENDING)],
            )
        )
        peak = max((o["resonance"] for o in observations), default=None)
        return {
            "case": case,
            "agents": agents,
            "observations": observations,
            "proposals": proposals,
            "peak_resonance": peak,
        }

    # ------------------------------------------------------------------
    # engine seam

    async def list_active_states(self) -> AsyncIterator[tuple[str, Any]]:
        """Sessions with a persisted snapshot, as ``(session_id, state)``.

        The engine's ``SemvecCortexService`` accepts any object with this
        method and uses it to pull live states instead of relying purely
        on what its in-memory cache happens to hold. Async because the
        engine's protocol is; the pymongo calls underneath are
        synchronous and cheap.

        A snapshot that fails to decode is skipped rather than raising:
        one unreadable session must not blind the observer to the rest
        of the network. ``AtlasPersistence`` already verifies integrity,
        so a failure here means a codec the running engine cannot read.
        """
        sessions = self._db.state_snapshots.distinct(
            "session", self._scope.filter()
        )
        for session in sorted(sessions):
            try:
                loaded = self._persistence.load_latest_snapshot(session=session)
            except Exception as exc:  # noqa: BLE001 - one bad snapshot, not the fleet
                # Logged, never silent: a session missing from the
                # observer's view is exactly the kind of gap that looks
                # like "the network was quiet" if nobody says otherwise.
                logger.warning(
                    "skipping session %r in the active-state listing, its "
                    "snapshot could not be loaded: %s",
                    session,
                    exc,
                )
                continue
            if loaded is not None:
                state, _meta = loaded
                yield session, state
