"""Transactional turn commit + recovery.

One agent turn persists atomically in the order **event → item
mutations → optional snapshot** (multi-document transaction on the
atlas-local single-node replica set). Crash consistency: recovery =
newest valid snapshot + deterministic event replay of the tail
(events with seq greater than the snapshot's recorded boundary).

The determinism gate (a release blocker): the recovered
state's engine checksum equals the live state's checksum.
"""

from __future__ import annotations

import numpy as np
import pytest

from semvec_atlas.commit import TurnCommitter
from semvec_atlas.guard import TenantScope
from semvec_atlas.indexes import provision

pytestmark = pytest.mark.mongo


@pytest.fixture()
def scope():
    return TenantScope(tenant="acme", agent="advisor-01")


@pytest.fixture()
def committer(mongo_client, db, scope, embedder):
    provision(db, dimension=32, search_indexes=False)
    return TurnCommitter(
        mongo_client, db, scope, embedder=embedder, snapshot_every=4
    )


class TestCommitTurn:
    def test_commit_persists_event_items_snapshot(self, committer, fresh_state, db):
        state = fresh_state()
        for turn in range(1, 5):
            result = committer.run_turn(
                state, session="s1", turn=turn, text=f"fact number {turn}"
            )
            assert result.metrics is not None
            assert result.event_seq == turn
        assert db.compliance_events.count_documents({}) == 4
        assert db.memory_items.count_documents({"deleted": False}) >= 1
        # snapshot_every=4 → exactly one snapshot, at turn 4
        snaps = list(db.state_snapshots.find())
        assert [s["turn"] for s in snaps] == [4]
        assert snaps[0]["last_event_seq"] == 4

    def test_snapshot_records_event_boundary(self, committer, fresh_state, db):
        state = fresh_state()
        for turn in range(1, 9):
            committer.run_turn(state, session="s1", turn=turn, text=f"t{turn}")
        boundaries = sorted(
            s["last_event_seq"] for s in db.state_snapshots.find()
        )
        assert boundaries == [4, 8]

    def test_commit_is_atomic_on_failure(self, committer, fresh_state, db, monkeypatch):
        state = fresh_state()
        committer.run_turn(state, session="s1", turn=1, text="pre-existing")
        events_before = db.compliance_events.count_documents({})

        def boom(*a, **k):
            raise RuntimeError("induced sync failure")

        monkeypatch.setattr(committer._item_sync, "sync", boom)
        with pytest.raises(RuntimeError, match="induced"):
            committer.run_turn(state, session="s1", turn=2, text="doomed turn")
        # The event append ran before the failure — the transaction
        # must have rolled it back.
        assert db.compliance_events.count_documents({}) == events_before

    def test_filtered_update_commits_nothing(self, committer, fresh_state, db):
        state = fresh_state()
        zero = np.zeros(32)
        result = committer.run_turn(
            state, session="s1", turn=1, text="ignored", embedding=zero
        )
        assert result.metrics is None
        assert db.compliance_events.count_documents({}) == 0


class TestRecovery:
    """Recovery modes and the determinism gate.

    * ``mode="replay"`` (default) — full event-log replay. Checksum-
      **identical** to the live state (the log is the source of truth).
      This is the determinism gate.
    * ``mode="snapshot_tail"`` — fast path: snapshot + tail replay.
      Checksum-identical as well (the snapshot codec is full-precision);
      it requires the snapshot codec to be readable by the running
      engine.
    """

    def test_full_replay_recovery_is_checksum_identical(self, committer, fresh_state):
        state = fresh_state()
        for turn in range(1, 11):
            committer.run_turn(state, session="s1", turn=turn, text=f"unique fact {turn}")
        live_checksum = state.to_dict()["checksum"]

        recovered, info = committer.recover(session="s1")
        assert info.snapshot_turn is None  # full replay ignores snapshots
        assert info.replayed_events == 10
        # Determinism gate: checksum-identical recovery.
        assert recovered.to_dict()["checksum"] == live_checksum

    def test_snapshot_tail_recovery_is_reproducible(self, committer, fresh_state):
        state = fresh_state()
        for turn in range(1, 11):
            committer.run_turn(state, session="s1", turn=turn, text=f"unique fact {turn}")

        r1, info1 = committer.recover(session="s1", mode="snapshot_tail")
        r2, info2 = committer.recover(session="s1", mode="snapshot_tail")
        assert info1.snapshot_turn == info2.snapshot_turn == 8
        assert info1.replayed_events == info2.replayed_events == 2
        # Deterministic fast path: same inputs, same result — always.
        assert r1.to_dict()["checksum"] == r2.to_dict()["checksum"]

    def test_snapshot_tail_recovery_is_checksum_identical_to_live(
        self, committer, fresh_state
    ):
        # The engine's full-precision snapshot codec keeps a restored
        # snapshot evolution-faithful, so snapshot + tail replay must
        # equal the live state exactly: the fast path is a second
        # checksum-identical recovery mode.
        state = fresh_state()
        for turn in range(1, 11):
            committer.run_turn(state, session="s1", turn=turn, text=f"unique fact {turn}")
        live = state.to_dict()["checksum"]

        recovered, info = committer.recover(session="s1", mode="snapshot_tail")
        assert info.snapshot_turn == 8 and info.replayed_events == 2
        assert recovered.to_dict()["checksum"] == live

    def test_snapshot_restore_matches_recorded_checksum(self, committer, fresh_state, db):
        state = fresh_state()
        for turn in range(1, 5):
            committer.run_turn(state, session="s1", turn=turn, text=f"fact {turn}")
        doc = db.state_snapshots.find_one({"turn": 4})
        recovered, info = committer.recover(session="s1", mode="snapshot_tail")
        assert info.snapshot_turn == 4 and info.replayed_events == 0
        # Snapshot integrity: restored state == state at save time.
        assert recovered.to_dict()["checksum"] == doc["engine_checksum"]

    def test_recover_without_snapshot_replays_everything(self, mongo_client, db, scope, embedder, fresh_state):
        provision(db, dimension=32, search_indexes=False)
        c = TurnCommitter(mongo_client, db, scope, embedder=embedder, snapshot_every=0)
        state = fresh_state()
        for turn in range(1, 4):
            c.run_turn(state, session="s1", turn=turn, text=f"fact {turn}")
        recovered, info = c.recover(session="s1")
        assert info.snapshot_turn is None
        assert info.replayed_events == 3
        assert recovered.to_dict()["checksum"] == state.to_dict()["checksum"]

    def test_recover_empty_session_returns_fresh_state(self, committer):
        recovered, info = committer.recover(session="never-used")
        assert info.snapshot_turn is None
        assert info.replayed_events == 0
        assert recovered.to_dict()["interaction_count"] == 0

    def test_recover_refuses_gapped_log(self, committer, fresh_state, db):
        state = fresh_state()
        for turn in range(1, 4):
            committer.run_turn(state, session="s1", turn=turn, text=f"fact {turn}")
        db.compliance_events.delete_one({"seq": 2})  # simulate tampering
        with pytest.raises(RuntimeError, match="gap"):
            committer.recover(session="s1")


class TestTurnTiming:
    def test_committed_turn_reports_transaction_latency(self, committer, fresh_state):
        result = committer.run_turn(
            fresh_state(), session="s1", turn=1, text="a fact"
        )
        assert result.txn_ms is not None
        assert 0.0 < result.txn_ms < 60_000.0  # measured, plausible

    def test_zero_norm_skip_has_no_transaction(self, committer, fresh_state):
        import numpy as np

        result = committer.run_turn(
            fresh_state(),
            session="s1",
            turn=1,
            text="ignored",
            embedding=np.zeros(32),
        )
        assert result.metrics is None
        assert result.txn_ms is None  # nothing was persisted
