"""Claims platform — the scenario the MongoDB article is written around.

Four specialist agents work one motor claim for one insurer on a
multi-tenant platform. The script shows the three properties that need
a database underneath them, and prints what each one actually produced:

1. **One transaction.** The adjuster's note and the agent's memory of
   writing it commit together, or neither does.
2. **One query.** Ranked memory hits come back joined to the claim
   record they concern.
3. **One record of the decision.** The agent network's convergence and
   the escalation vote are persisted and queryable months later.

Run with the local container::

    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local:latest
    python demo/claims_scenario.py

No licence key, no LLM, no API calls: the engine, the coordination
layer and the firewall all run in process. The only download is the
sentence-transformer model on first use.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import pymongo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semvec import SemvecConfig, SemvecState  # noqa: E402
from semvec.cortex import (  # noqa: E402
    AttentionAggregation,
    ConsensusEngine,
    ConsensusLevel,
    SemvecAgentNetwork,
)

from semvec_atlas.commit import DomainWrite, TurnCommitter  # noqa: E402
from semvec_atlas.cortex_store import AtlasCortexStore  # noqa: E402
from semvec_atlas.deployment import probe_deployment  # noqa: E402
from semvec_atlas.guard import TenantScope  # noqa: E402
from semvec_atlas.indexes import provision  # noqa: E402
from semvec_atlas.retrieval import DomainJoin, RetrievalEngine  # noqa: E402

TENANT = "northwind-insurance"
CASE = "CL-88371"
POLICY = "LV-2044-88371-X"
DIM = 768
MODEL = "sentence-transformers/all-mpnet-base-v2"

#: The four specialists and the note each contributes. Phase one keeps
#: them on separate matters; phase two moves all four onto this claim,
#: which is where the observer's resonance reading earns its keep.
SEPARATE_WORK = [
    ("intake", "Routine change-of-address request filed for another policyholder."),
    ("coverage", "Annual renewal terms reviewed for a commercial fleet account."),
    ("fraud", "Quarterly pattern report closed with no findings."),
    ("assessment", "Training refresher on bumper repair cost tables completed."),
]
SHARED_CASE = [
    ("intake", f"Claim {CASE}: rear-end collision on the A5, bumper damage, no injuries."),
    ("coverage", f"Policy {POLICY} covers collision damage; excess 500 EUR applies."),
    ("fraud", f"Third collision claim on policy {POLICY} within fourteen months."),
    ("assessment", "Bumper and sensor replacement quoted at 4,200 EUR by the garage."),
]


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--uri", default="mongodb://localhost:27017/?directConnection=true"
    )
    ap.add_argument("--keep-db", action="store_true", help="do not drop the database")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    client = pymongo.MongoClient(args.uri, serverSelectionTimeoutMS=5000)

    banner("DEPLOYMENT")
    report = probe_deployment(client)
    print(report.summary())
    if not report.ready:
        return 1

    db_name = f"claims_demo_{uuid.uuid4().hex[:8]}"
    db = client[db_name]
    model = SentenceTransformer(MODEL, device="cpu")

    class Embedder:
        def get_dimension(self) -> int:
            return DIM

        def get_embedding(self, text: str) -> np.ndarray:
            return np.asarray(
                model.encode(text, normalize_embeddings=True), dtype=np.float64
            )

    embedder = Embedder()
    provision(db, dimension=DIM, search_indexes=False)

    # ------------------------------------------------------------------
    banner("ONE TRANSACTION — the claim record and the memory of it")

    scope = TenantScope(tenant=TENANT, agent="intake")
    committer = TurnCommitter(client, db, scope, embedder=embedder, snapshot_every=2)
    state = SemvecState(SemvecConfig(dimension=DIM))

    result = committer.run_turn(
        state,
        session=CASE,
        turn=1,
        text=SHARED_CASE[0][1],
        domain_writes=[
            DomainWrite(
                collection="claims",
                key={"claim_id": CASE},
                doc={
                    "policy_number": POLICY,
                    "line_of_business": "motor",
                    "status": "open",
                    "reserve_eur": 4200,
                },
            )
        ],
        domain_link={"claim_id": CASE},
    )
    print(f"  turn committed      | event_seq={result.event_seq} "
          f"| domain_writes={result.domain_writes} | txn={result.txn_ms:.1f} ms")
    claim = db.claims.find_one({"claim_id": CASE}, {"_id": 0})
    print(f"  claim record        | {claim}")
    print(f"  compliance events   | {db.compliance_events.count_documents({})}")

    for turn, (_agent, text) in enumerate(SHARED_CASE[1:], start=2):
        committer.run_turn(
            state, session=CASE, turn=turn, text=text, domain_link={"claim_id": CASE}
        )
    print(f"  memory items        | {db.memory_items.count_documents({})}")

    # ------------------------------------------------------------------
    banner("ONE QUERY — ranked memory joined to the claim it concerns")

    # force_fallback keeps this runnable in seconds: the server-side path
    # needs Atlas Search indexes, which take a minute or two to build.
    # The join works identically on both paths — one $lookup inside the
    # aggregation there, one batch query here.
    retrieval = RetrievalEngine(db, scope, embedder=embedder, force_fallback=True)
    hits = retrieval.search(
        "collision damage on this policy",
        k=3,
        join=DomainJoin(collection="claims", local_field="claim_id"),
    )
    print("  ranking: client-side reciprocal-rank fusion "
          "(scores are RRF weights, not cosine similarities)")
    for hit in hits:
        record = hit.domain or {}
        print(f"  {hit.score:.4f}  {hit.summary[:58]:<58}")
        print(f"          └─ claim {record.get('claim_id')} · policy "
              f"{record.get('policy_number')} · reserve "
              f"{record.get('reserve_eur')} EUR · {record.get('status')}")

    # ------------------------------------------------------------------
    banner("ONE RECORD OF THE DECISION — four agents, one claim")

    net = SemvecAgentNetwork(
        aggregation_strategy=AttentionAggregation(),
        enable_feedback=True,
        dimension=DIM,
    )
    for role, _ in SEPARATE_WORK:
        net.add_local_instance(role)
    agents = net.local_instances
    cortex = AtlasCortexStore(db, scope)

    for role, text in SEPARATE_WORK:
        agents[role].process_input_embedding(embedder.get_embedding(text), text)
    net.meta_pss.update_global_state(list(agents.values()))
    first = cortex.record_observation(
        net.meta_pss, agents=sorted(agents), case=CASE
    )
    print(f"  separate matters    | coherence={first.coherence:.4f} "
          f"resonance={first.resonance:.4f}")

    for role, text in SHARED_CASE:
        agents[role].process_input_embedding(embedder.get_embedding(text), text)
    net.meta_pss.update_global_state(list(agents.values()))
    second = cortex.record_observation(
        net.meta_pss, agents=sorted(agents), case=CASE
    )
    print(f"  all on {CASE}    | coherence={second.coherence:.4f} "
          f"resonance={second.resonance:.4f}")
    print(f"  resonance change    | {first.resonance:.4f} → {second.resonance:.4f} "
          f"({(second.resonance / first.resonance - 1) * 100:+.0f} %)")

    cortex.record_agents(agents.values(), case=CASE)

    consensus = ConsensusEngine(local_instance_id="fraud", network_id=CASE)
    for role, weight in (
        ("intake", 0.5), ("coverage", 1.0), ("fraud", 1.0), ("assessment", 1.0)
    ):
        consensus.register_instance(role, weight)
    proposal = consensus.create_proposal(
        "escalation",
        net.meta_pss.global_state,
        f"third collision claim on {POLICY} in fourteen months",
        ConsensusLevel.SIMPLE_MAJORITY,
    )
    votes = {"fraud": True, "coverage": True, "assessment": True, "intake": False}
    for role, vote in votes.items():
        consensus.vote_on_proposal(proposal.proposal_id, vote, role)
    stats = consensus.get_statistics()["statistics"]
    accepted = stats["accepted_proposals"] == 1
    cortex.record_proposal(
        proposal_id=proposal.proposal_id,
        proposal_type="escalation",
        rationale=f"third collision claim on {POLICY} in fourteen months",
        consensus_level=str(proposal.consensus_level),
        votes=votes,
        accepted=accepted,
        case=CASE,
    )
    print(f"  escalation vote     | {sum(votes.values())}/{len(votes)} in favour "
          f"→ {'accepted' if accepted else 'rejected'}")

    # ------------------------------------------------------------------
    banner("THE QUESTION, SIX MONTHS LATER")

    timeline = cortex.case_timeline(CASE)
    print(f"  who worked {CASE} | "
          f"{', '.join(a['instance_id'] for a in timeline['agents'])}")
    print(f"  peak resonance      | {timeline['peak_resonance']:.4f} "
          f"across {len(timeline['observations'])} readings")
    for prop in timeline["proposals"]:
        infavour = sum(1 for v in prop["votes"].values() if v)
        print(f"  decision            | {prop['proposal_type']} · "
              f"{infavour}/{len(prop['votes'])} in favour · "
              f"{'accepted' if prop['accepted'] else 'rejected'}")
        print(f"  rationale           | {prop['rationale']}")

    if args.keep_db:
        print(f"\ndatabase kept: {db_name}")
    else:
        client.drop_database(db_name)
        print(f"\ndatabase {db_name} dropped (--keep-db to inspect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
