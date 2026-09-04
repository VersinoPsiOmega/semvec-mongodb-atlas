"""Domain documents beside agent memory, in one database.

The claim (or order, or case) an agent is working on is business data;
what the agent remembers about working on it is memory. Splitting them
across two systems buys a synchronisation problem and a window in which
the record exists and the memory of it does not.

This module pins the two properties that make colocation worth having:

* **One transaction.** A domain write handed to ``run_turn`` commits
  inside the same multi-document transaction as the memory event, the
  item mutations and the snapshot. A lost CAS or a failing write rolls
  back *both* halves — there is no state in which the adjuster's note
  survives without the memory of the turn that produced it.
* **One query.** Retrieval joins the ranked memory hits to the domain
  documents they concern through ``$lookup``, inside the same
  aggregation. A join is unavoidable; a join *across systems* is not.

The tenant guard extends over both halves, and the writable-collection
allowlist is part of it: without one, ``domain_writes`` would be a
generic write primitive pointing at ``compliance_events``.
"""

from __future__ import annotations

import pytest
from pymongo.errors import PyMongoError

from semvec_atlas.commit import DomainWrite, TurnCommitter
from semvec_atlas.guard import TenantGuardViolation, TenantScope
from semvec_atlas.indexes import DOMAIN_COLLECTIONS, provision
from semvec_atlas.retrieval import DomainJoin, RetrievalEngine

pytestmark = pytest.mark.mongo

DIM = 32


@pytest.fixture()
def scope():
    return TenantScope(tenant="northwind", agent="claims-intake")


@pytest.fixture()
def committer(mongo_client, db, scope, embedder):
    provision(db, dimension=DIM, search_indexes=False)
    return TurnCommitter(
        mongo_client, db, scope, embedder=embedder, snapshot_every=0
    )


def _claim(claim_id: str = "CL-88371") -> DomainWrite:
    return DomainWrite(
        collection="claims",
        key={"claim_id": claim_id},
        doc={
            "policy_number": "LV-2044-88371-X",
            "line_of_business": "motor",
            "status": "open",
        },
    )


class TestProvisioning:
    def test_domain_collections_are_created(self, db):
        provision(db, dimension=DIM, search_indexes=False)
        names = set(db.list_collection_names())
        assert set(DOMAIN_COLLECTIONS) <= names

    def test_claims_are_indexed_per_tenant(self, db):
        provision(db, dimension=DIM, search_indexes=False)
        keys = {ix["name"] for ix in db.claims.list_indexes()}
        assert "ix_claim" in keys

    def test_allowlist_excludes_the_engine_collections(self):
        # domain_writes must never be able to reach the audit log or the
        # memory mirror — those have their own writers and invariants.
        for protected in ("compliance_events", "memory_items", "state_snapshots"):
            assert protected not in DOMAIN_COLLECTIONS


class TestAtomicDomainWrite:
    def test_domain_document_lands_with_the_turn(self, committer, db, fresh_state, scope):
        state = fresh_state(DIM)
        result = committer.run_turn(
            state,
            session="claim-88371",
            turn=1,
            text="Claimant reports rear-end collision on the A5.",
            domain_writes=[_claim()],
        )
        assert result.metrics is not None
        doc = db.claims.find_one({"claim_id": "CL-88371"})
        assert doc is not None
        assert doc["policy_number"] == "LV-2044-88371-X"
        # Scoped by the guard, not by the caller — and tenant-wide, not
        # per agent: four specialists work the same claim, so binding the
        # record to the agent that happened to open it would hide it from
        # the other three.
        assert doc["tenant"] == scope.tenant
        assert "agent" not in doc
        assert db.compliance_events.count_documents({}) == 1

    def test_upsert_updates_an_existing_claim(self, committer, db, fresh_state):
        state = fresh_state(DIM)
        committer.run_turn(
            state, session="s", turn=1, text="first note", domain_writes=[_claim()]
        )
        committer.run_turn(
            state,
            session="s",
            turn=2,
            text="second note",
            domain_writes=[
                DomainWrite(
                    collection="claims",
                    key={"claim_id": "CL-88371"},
                    doc={"status": "escalated"},
                )
            ],
        )
        docs = list(db.claims.find({"claim_id": "CL-88371"}))
        assert len(docs) == 1
        assert docs[0]["status"] == "escalated"
        assert docs[0]["policy_number"] == "LV-2044-88371-X"  # untouched field survives

    def test_a_failing_domain_write_rolls_back_the_whole_turn(
        self, committer, db, fresh_state
    ):
        state = fresh_state(DIM)
        # A record whose identifier is already taken. The realistic shape
        # of this failure is an importer that carries ids in from a source
        # system and hits one that exists.
        db.claims.insert_one({"_id": "CL-TAKEN", "tenant": "northwind"})
        bad = DomainWrite(
            collection="claims",
            key={"claim_id": "CL-NEW"},
            doc={"_id": "CL-TAKEN"},
        )
        with pytest.raises(PyMongoError):
            committer.run_turn(
                state, session="s", turn=1, text="note", domain_writes=[bad]
            )
        # The write failed after the memory event had already been
        # appended inside the same transaction — that ordering is exactly
        # what the rollback has to undo.
        assert db.claims.count_documents({"claim_id": "CL-NEW"}) == 0
        assert db.compliance_events.count_documents({}) == 0
        assert db.memory_items.count_documents({}) == 0

    def test_turn_without_domain_writes_is_unchanged(self, committer, db, fresh_state):
        state = fresh_state(DIM)
        result = committer.run_turn(state, session="s", turn=1, text="plain turn")
        assert result.metrics is not None
        assert db.compliance_events.count_documents({}) == 1


class TestDomainGuard:
    def test_collection_outside_the_allowlist_is_refused(
        self, committer, db, fresh_state
    ):
        state = fresh_state(DIM)
        with pytest.raises(TenantGuardViolation):
            committer.run_turn(
                state,
                session="s",
                turn=1,
                text="note",
                domain_writes=[
                    DomainWrite(
                        collection="compliance_events",
                        key={"seq": 1},
                        doc={"content": "forged"},
                    )
                ],
            )
        assert db.compliance_events.count_documents({}) == 0

    def test_key_cannot_smuggle_a_foreign_tenant(self, committer, fresh_state):
        state = fresh_state(DIM)
        with pytest.raises(TenantGuardViolation):
            committer.run_turn(
                state,
                session="s",
                turn=1,
                text="note",
                domain_writes=[
                    DomainWrite(
                        collection="claims",
                        key={"claim_id": "CL-1", "tenant": "competitor"},
                        doc={"status": "open"},
                    )
                ],
            )

    def test_key_cannot_carry_a_query_operator(self, committer, db, fresh_state):
        """A key is an identifier, never a query.

        ``{"claim_id": {"$exists": True}}`` passes the tenant guard —
        it names no scope field — and would then match some *other*
        claim of the same tenant and overwrite it. Keys must be scalar.
        """
        state = fresh_state(DIM)
        db.claims.insert_one(
            {"tenant": "northwind", "claim_id": "CL-OTHER", "status": "open"}
        )
        with pytest.raises(TenantGuardViolation):
            committer.run_turn(
                state,
                session="s",
                turn=1,
                text="note",
                domain_writes=[
                    DomainWrite(
                        collection="claims",
                        key={"claim_id": {"$exists": True}},
                        doc={"status": "hijacked"},
                    )
                ],
            )
        assert db.claims.find_one({"claim_id": "CL-OTHER"})["status"] == "open"

    def test_domain_link_cannot_overwrite_mirror_fields(
        self, committer, db, fresh_state
    ):
        """The link names business keys, not memory-item internals.

        Letting it through would corrupt the mirror: a link that sets
        ``tier`` or ``embedding`` writes engine-owned state from the
        caller's side.
        """
        state = fresh_state(DIM)
        with pytest.raises(TenantGuardViolation):
            committer.run_turn(
                state,
                session="s",
                turn=1,
                text="note",
                domain_link={"tier": "long"},
            )
        assert db.memory_items.count_documents({}) == 0

    def test_doc_cannot_overwrite_the_scope_fields(self, committer, db, fresh_state):
        state = fresh_state(DIM)
        with pytest.raises(TenantGuardViolation):
            committer.run_turn(
                state,
                session="s",
                turn=1,
                text="note",
                domain_writes=[
                    DomainWrite(
                        collection="claims",
                        key={"claim_id": "CL-1"},
                        doc={"tenant": "competitor", "status": "open"},
                    )
                ],
            )
        assert db.claims.count_documents({}) == 0


class TestJoinedRetrieval:
    """One aggregation returns the memory hits and their claim records."""

    @pytest.fixture()
    def seeded(self, committer, db, fresh_state, scope, embedder):
        state = fresh_state(DIM)
        committer.run_turn(
            state,
            session="claim-88371",
            turn=1,
            text="Rear-end collision on the A5, bumper damage, no injuries.",
            domain_writes=[_claim()],
            domain_link={"claim_id": "CL-88371"},
        )
        return RetrievalEngine(
            db, scope, embedder=embedder, force_fallback=True
        )

    def test_hits_carry_the_joined_domain_document(self, seeded):
        hits = seeded.search(
            "collision damage",
            k=5,
            join=DomainJoin(collection="claims", local_field="claim_id"),
        )
        assert hits, "expected at least one memory hit"
        joined = [h for h in hits if h.domain]
        assert joined, "no hit carried its claim record"
        assert joined[0].domain["policy_number"] == "LV-2044-88371-X"

    def test_search_without_join_leaves_domain_empty(self, seeded):
        hits = seeded.search("collision damage", k=5)
        assert hits
        assert all(h.domain is None for h in hits)

    def test_join_does_not_cross_the_tenant_boundary(self, seeded, db, embedder, scope):
        # A claim of another tenant carrying the same claim_id must not be
        # pulled in by the lookup.
        db.claims.insert_one(
            {
                "tenant": "competitor",
                "claim_id": "CL-88371",
                "policy_number": "LEAKED",
            }
        )
        hits = seeded.search(
            "collision damage",
            k=5,
            join=DomainJoin(collection="claims", local_field="claim_id"),
        )
        for h in hits:
            if h.domain:
                assert h.domain["policy_number"] != "LEAKED"
