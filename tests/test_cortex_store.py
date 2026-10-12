"""Persisting the multi-agent coordination layer.

The engine's cortex — an agent network, a shared observer, a consensus
engine — runs in process and keeps its state in memory. That is fine
for one shift and useless afterwards: when someone asks in six months
why a case was escalated, "the agents converged" is not an answer
unless the convergence was written down.

``AtlasCortexStore`` mirrors three things into the same database that
already holds the memory and the business records:

* **agents** — who participated, how much, and when they last spoke;
* **observations** — the observer's coherence/resonance readings, which
  is where convergence becomes a timestamped fact;
* **proposals** — what was proposed, by whom, how each member voted,
  and how it resolved.

With all three colocated, :meth:`case_timeline` answers the question
the coordination layer exists for in one round trip.

The store also satisfies the engine's persistence seam
(``list_active_states``), so a ``SemvecCortexService`` can be handed an
Atlas-backed store instead of running purely in memory.
"""

from __future__ import annotations

import pytest

from semvec_atlas.cortex_store import (
    AtlasCortexStore,
    ObservationRecord,
    ProposalConflict,
)
from semvec_atlas.guard import TenantScope
from semvec_atlas.indexes import provision

pytestmark = pytest.mark.mongo

DIM = 32
CASE = "CL-88371"


@pytest.fixture()
def scope():
    return TenantScope(tenant="northwind", agent="claims-intake")


@pytest.fixture()
def store(db, scope):
    provision(db, dimension=DIM, search_indexes=False)
    return AtlasCortexStore(db, scope)


class _FakeAgent:
    """Duck-type of the engine's SemvecAgent, for the store's contract."""

    def __init__(self, instance_id: str, interactions: int) -> None:
        self.instance_id = instance_id
        self.interaction_count = interactions
        self.last_update_time = 1_764_000_000.0


class _FakeObserver:
    def __init__(self, coherence: float, resonance: float, updates: int) -> None:
        self.global_coherence = coherence
        self.network_resonance = resonance
        self.update_count = updates


class TestAgentRecording:
    def test_agents_are_recorded_once_per_id(self, store, db):
        agents = [_FakeAgent("coverage", 2), _FakeAgent("fraud", 3)]
        assert store.record_agents(agents, case=CASE) == 2
        store.record_agents([_FakeAgent("coverage", 5)], case=CASE)
        docs = list(db.cortex_agents.find({}))
        assert len(docs) == 2
        coverage = next(d for d in docs if d["instance_id"] == "coverage")
        assert coverage["interaction_count"] == 5  # latest wins, no duplicate

    def test_records_are_tenant_scoped(self, store, db, scope):
        store.record_agents([_FakeAgent("coverage", 1)], case=CASE)
        doc = db.cortex_agents.find_one({})
        assert doc["tenant"] == scope.tenant
        assert doc["case"] == CASE

    def test_a_foreign_tenant_cannot_be_read(self, db, scope):
        provision(db, dimension=DIM, search_indexes=False)
        db.cortex_agents.insert_one(
            {"tenant": "competitor", "instance_id": "spy", "case": CASE}
        )
        ours = AtlasCortexStore(db, scope)
        assert ours.case_timeline(CASE)["agents"] == []


class TestObservationRecording:
    def test_observation_captures_the_convergence_reading(self, store):
        rec = store.record_observation(
            _FakeObserver(0.335, 0.597, 2),
            agents=["coverage", "fraud", "assessment"],
            case=CASE,
        )
        assert isinstance(rec, ObservationRecord)
        assert rec.resonance == pytest.approx(0.597)
        assert rec.participants == 3

    def test_observations_accumulate_as_a_series(self, store):
        store.record_observation(
            _FakeObserver(0.265, 0.317, 1), agents=["coverage"], case=CASE
        )
        store.record_observation(
            _FakeObserver(0.335, 0.597, 2), agents=["coverage", "fraud"], case=CASE
        )
        timeline = store.case_timeline(CASE)
        readings = [o["resonance"] for o in timeline["observations"]]
        assert readings == [pytest.approx(0.317), pytest.approx(0.597)]


class TestProposalRecording:
    def test_proposal_records_votes_and_outcome(self, store, db):
        pid = store.record_proposal(
            proposal_id="prop-1",
            proposal_type="escalation",
            rationale="damage pattern matches three prior claims",
            consensus_level="simple_majority",
            votes={"coverage": True, "fraud": True, "assessment": False},
            accepted=True,
            case=CASE,
        )
        doc = db.cortex_proposals.find_one({"proposal_id": pid})
        assert doc["accepted"] is True
        assert doc["votes"]["assessment"] is False
        assert doc["rationale"].startswith("damage pattern")

    def test_proposal_is_immutable_once_recorded(self, store):
        store.record_proposal(
            proposal_id="prop-1",
            proposal_type="escalation",
            rationale="first",
            consensus_level="simple_majority",
            votes={"coverage": True},
            accepted=True,
            case=CASE,
        )
        # A decision that could be rewritten afterwards is not a record.
        with pytest.raises(ProposalConflict):
            store.record_proposal(
                proposal_id="prop-1",
                proposal_type="escalation",
                rationale="rewritten",
                consensus_level="unanimous",
                votes={"coverage": False},
                accepted=False,
                case=CASE,
            )


class TestCaseTimeline:
    def test_one_query_answers_who_converged_and_how_it_was_decided(self, store):
        store.record_agents(
            [_FakeAgent("coverage", 2), _FakeAgent("fraud", 3)], case=CASE
        )
        store.record_observation(
            _FakeObserver(0.265, 0.317, 1), agents=["coverage"], case=CASE
        )
        store.record_observation(
            _FakeObserver(0.335, 0.597, 2),
            agents=["coverage", "fraud"],
            case=CASE,
        )
        store.record_proposal(
            proposal_id="prop-1",
            proposal_type="escalation",
            rationale="damage pattern matches three prior claims",
            consensus_level="simple_majority",
            votes={"coverage": True, "fraud": True},
            accepted=True,
            case=CASE,
        )

        timeline = store.case_timeline(CASE)
        assert {a["instance_id"] for a in timeline["agents"]} == {"coverage", "fraud"}
        assert timeline["peak_resonance"] == pytest.approx(0.597)
        assert len(timeline["proposals"]) == 1
        assert timeline["proposals"][0]["accepted"] is True

    def test_unknown_case_is_empty_not_an_error(self, store):
        timeline = store.case_timeline("CL-NOBODY")
        assert timeline["agents"] == []
        assert timeline["peak_resonance"] is None


class TestEngineSeam:
    """``list_active_states`` — the one method the cortex service needs."""

    def test_store_satisfies_the_engines_protocol(self, db, scope):
        """Pinned against the wheel, not asserted in prose.

        The seam is a ``runtime_checkable`` Protocol, so conformance is
        checkable here — and if the engine ever widens it, this fails
        instead of the store silently ceasing to be accepted.
        """
        from semvec.cortex.service import SemvecCortexService, _StateStore

        provision(db, dimension=DIM, search_indexes=False)
        store = AtlasCortexStore(db, scope)
        assert isinstance(store, _StateStore)
        # And the service takes it without complaint.
        assert SemvecCortexService(pss_store=store, dimension=DIM) is not None

    async def test_returns_persisted_sessions_as_states(
        self, mongo_client, db, scope, embedder, fresh_state
    ):
        from semvec_atlas.commit import TurnCommitter

        provision(db, dimension=DIM, search_indexes=False)
        committer = TurnCommitter(
            mongo_client, db, scope, embedder=embedder, snapshot_every=1
        )
        state = fresh_state(DIM)
        committer.run_turn(state, session="claim-88371", turn=1, text="a fact")

        store = AtlasCortexStore(db, scope)
        active = [pair async for pair in store.list_active_states()]
        assert [sid for sid, _ in active] == ["claim-88371"]
        restored = active[0][1]
        assert restored.to_dict()["checksum"] == state.to_dict()["checksum"]

    async def test_no_snapshots_yields_nothing(self, db, scope):
        provision(db, dimension=DIM, search_indexes=False)
        store = AtlasCortexStore(db, scope)
        assert [p async for p in store.list_active_states()] == []
