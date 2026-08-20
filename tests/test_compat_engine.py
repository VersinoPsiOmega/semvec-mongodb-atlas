"""Engine compatibility gates — snapshot blobs and update math.

Frozen fixtures guard three contracts against future semvec releases:

* **Backward blob compatibility** — a 0.7.6 snapshot blob must load in
  every supported engine version and restore the *identical state*
  (compared on the public ``semantic_state`` vector against a frozen
  dump — deliberately **not** on the engine checksum, which is an
  intra-version integrity value and not comparable across versions).
* **Engine-math stability** — replaying the fixture's update sequence
  on the current engine must reproduce the frozen state vector. If a
  release changes the update equation, this fails loudly.
* **Blob header discipline** — snapshots start with the ``SVB`` magic
  plus a codec version byte; unknown magic is rejected loudly, and the
  persistence layer records the codec version per snapshot (T-3) so a
  mixed-fleet operator can audit which pods can read which snapshots.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_DIM = 32


def fixture_embedding(text: str) -> np.ndarray:
    """The exact generator used to build the frozen fixture."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(FIXTURE_DIM)
    return v / np.linalg.norm(v)


FIXTURE_TEXTS = [f"unique fact {i}" for i in range(8)]


@pytest.fixture(scope="module")
def expected_state() -> np.ndarray:
    data = json.loads((FIXTURES / "state_v076_expected.json").read_text())
    return np.asarray(data["semantic_state"], dtype=float)


class TestBackwardBlobCompatibility:
    def test_v076_blob_loads_and_restores_identical_state(self, expected_state):
        from semvec import SemvecState

        blob = (FIXTURES / "state_v076.bin").read_bytes()
        state = SemvecState.from_bytes(blob)
        restored = np.asarray(state.semantic_state, dtype=float)
        assert float(np.max(np.abs(restored - expected_state))) == 0.0

    def test_restored_state_continues_deterministically(self, expected_state):
        # A restored legacy state must accept updates; two restores that
        # apply the same next update must agree with each other (the
        # evolution-fidelity contract, fixed in semvec 0.8.2).
        from semvec import SemvecState

        blob = (FIXTURES / "state_v076.bin").read_bytes()
        a, b = SemvecState.from_bytes(blob), SemvecState.from_bytes(blob)
        nxt = fixture_embedding("unique fact 9")
        a.update(nxt, "unique fact 9")
        b.update(nxt, "unique fact 9")
        assert a.to_dict()["checksum"] == b.to_dict()["checksum"]


class TestEngineMathStability:
    def test_fresh_build_reproduces_frozen_state_vector(self, expected_state):
        from semvec import SemvecConfig, SemvecState

        state = SemvecState(SemvecConfig(dimension=FIXTURE_DIM))
        for text in FIXTURE_TEXTS:
            state.update(fixture_embedding(text), text)
        built = np.asarray(state.semantic_state, dtype=float)
        assert float(np.max(np.abs(built - expected_state))) == 0.0, (
            "engine update math changed — persisted trajectories are no "
            "longer reproducible; this blocks the release (determinism gate)"
        )


class TestBlobHeaderDiscipline:
    def test_fixture_blob_carries_svb_magic(self):
        blob = (FIXTURES / "state_v076.bin").read_bytes()
        assert blob[:3] == b"SVB"
        assert blob[3] >= 1  # codec version byte

    def test_current_engine_blob_carries_svb_magic(self, fresh_state, embedder):
        state = fresh_state()
        state.update(embedder.get_embedding("x"), "x")
        blob = state.to_bytes(compress=True)
        assert blob[:3] == b"SVB"

    def test_unknown_magic_rejected_loudly(self):
        from semvec import SemvecState

        with pytest.raises(ValueError):
            SemvecState.from_bytes(b"NOPE" + b"\x00" * 64)

    def test_codec_version_helper_parses_header(self):
        from semvec_atlas.persistence import blob_codec_version

        blob = (FIXTURES / "state_v076.bin").read_bytes()
        assert blob_codec_version(blob) == 1
        assert blob_codec_version(b"SVB\x0a" + b"x") == 10

    def test_codec_version_helper_rejects_foreign_bytes(self):
        from semvec_atlas.persistence import blob_codec_version

        with pytest.raises(ValueError, match="SVB"):
            blob_codec_version(b"garbage")


@pytest.mark.mongo
class TestSnapshotCodecVersionPersisted:
    def test_snapshot_metadata_records_codec_version(self, db, fresh_state, embedder):
        from semvec_atlas.guard import TenantScope
        from semvec_atlas.indexes import provision
        from semvec_atlas.persistence import AtlasPersistence

        provision(db, dimension=32, search_indexes=False)
        p = AtlasPersistence(db, TenantScope(tenant="acme", agent="a1"))
        state = fresh_state()
        state.update(embedder.get_embedding("x"), "x")
        meta = p.save_snapshot(state, session="s1", turn=1)
        assert isinstance(meta.codec_version, int) and meta.codec_version >= 1
        doc = db.state_snapshots.find_one({"session": "s1"})
        assert doc["codec_version"] == meta.codec_version

    def test_checksum_comparisons_are_version_scoped(self, db, fresh_state, embedder):
        # The comparability helper codifies the rule: engine
        # checksums are intra-version integrity values only.
        import semvec

        from semvec_atlas.persistence import checksums_comparable

        assert checksums_comparable(semvec.__version__, semvec.__version__) is True
        assert checksums_comparable("0.8.4", "0.8.5") is False
