"""Shared test fixtures for semvec-atlas.

Tests marked ``mongo`` require the local ``mongodb/mongodb-atlas-local``
container (MongoDB 8.3+, single-node replica set, Atlas Search enabled)::

    docker run -d --name semvec-atlas-mongo -p 27017:27017 \
        mongodb/mongodb-atlas-local:latest

Each test gets a private database (UUID-suffixed) that is dropped on
teardown — tests never share state.

The semvec license key is read from the repository root ``.env``
(``SEMVEC_KEY``) and exported as ``SEMVEC_LICENSE_KEY`` before the first
``semvec`` import, mirroring production deployment (key injected via
environment, never hard-coded).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# License key: map .env SEMVEC_KEY -> SEMVEC_LICENSE_KEY before importing
# semvec anywhere. Walk up from this file so the fixture works from any cwd.


def _load_license_key() -> None:
    if os.environ.get("SEMVEC_LICENSE_KEY"):
        return
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                if line.startswith("SEMVEC_KEY="):
                    os.environ["SEMVEC_LICENSE_KEY"] = line.split("=", 1)[1].strip()
                    return


_load_license_key()

MONGO_URI = os.environ.get(
    "SEMVEC_ATLAS_TEST_URI", "mongodb://localhost:27017/?directConnection=true"
)

EMBED_DIM = 32


class DeterministicEmbedder:
    """Test embedder fulfilling the semvec ``EmbedderProtocol``.

    Vectors are a pure function of the input text (SHA-256 seeded RNG,
    L2-normalised), so persistence and retrieval tests are reproducible
    without a model download. Semantic *quality* is irrelevant here —
    these tests exercise storage, transactions, and query plumbing.
    """

    def get_dimension(self) -> int:
        return EMBED_DIM

    def get_embedding(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBED_DIM)
        return v / np.linalg.norm(v)


@pytest.fixture(scope="session")
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder()


@pytest.fixture(scope="session")
def mongo_client():
    pymongo = pytest.importorskip("pymongo")
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception:  # pragma: no cover - environment guard
        pytest.skip(f"MongoDB not reachable at {MONGO_URI}")
    yield client
    client.close()


@pytest.fixture()
def db(mongo_client):
    """A private, empty database per test, dropped on teardown."""
    name = f"svatest_{uuid.uuid4().hex[:12]}"
    database = mongo_client[name]
    yield database
    mongo_client.drop_database(name)


@pytest.fixture()
def fresh_state():
    """A small-dimension SemvecState factory (engine from the wheel)."""
    from semvec import SemvecConfig, SemvecState

    def make(dimension: int = EMBED_DIM) -> SemvecState:
        return SemvecState(SemvecConfig(dimension=dimension))

    return make
