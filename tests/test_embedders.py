"""Embedding providers and the tenant embedder-fingerprint guard.

The Voyage tests run against an injected fake transport — no network,
no API key, no SDK. The guard tests pin the fail-closed contract: a
tenant's persisted embedder identity must match the embedder every
write/read path is constructed with, so a silent model switch cannot
degrade similarity search.
"""

from __future__ import annotations

import numpy as np
import pytest

from semvec_atlas.embedders import (
    EmbedderMismatch,
    VoyageEmbedder,
    embedder_fingerprint,
    ensure_tenant_embedder,
)
from semvec_atlas.guard import TenantScope


class FakeTransport:
    """Records requests; returns configurable embeddings (status, body)."""

    def __init__(self, dimension: int = 4, status: int = 200):
        self.dimension = dimension
        self.status = status
        self.calls: list[dict] = []

    def __call__(self, url: str, headers: dict, payload: dict):
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        if self.status != 200:
            return self.status, {"detail": "rate limited"}
        data = []
        for i, _text in enumerate(payload["input"]):
            # Scaled one-hot at the position of the request index,
            # deliberately NOT normalized and reverse-ordered: the
            # client must sort by index AND L2-normalize, and the
            # direction proves which slot each vector came from.
            vec = [0.0] * self.dimension
            vec[i % self.dimension] = 3.0
            data.append({"embedding": vec, "index": i})
        return 200, {"data": list(reversed(data)), "model": payload["model"]}


@pytest.fixture()
def transport():
    return FakeTransport()


@pytest.fixture()
def voyage(transport, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    return VoyageEmbedder("voyage-3.5", transport=transport)


class TestVoyageEmbedder:
    def test_request_carries_model_auth_and_input(self, voyage, transport):
        voyage.get_embedding("hello world")
        call = transport.calls[0]
        assert call["url"].endswith("/embeddings")
        assert call["headers"]["Authorization"] == "Bearer test-key"
        assert call["payload"]["model"] == "voyage-3.5"
        assert call["payload"]["input"] == ["hello world"]

    def test_vectors_come_back_float64_and_l2_normalized(self, voyage):
        vec = voyage.get_embedding("hello")
        assert vec.dtype == np.float64
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0)

    def test_batch_preserves_order_across_chunks(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        transport = FakeTransport(dimension=4)
        emb = VoyageEmbedder("voyage-3.5", transport=transport, batch_size=2)
        out = emb.get_embeddings([f"t{i}" for i in range(5)])
        assert out.shape == (5, 4)
        assert len(transport.calls) == 3  # 2 + 2 + 1
        assert transport.calls[0]["payload"]["input"] == ["t0", "t1"]
        # The fake one-hot direction encodes the within-chunk index and
        # its responses arrive reversed — row i must still be the
        # normalized one-hot for its own slot (index-based reassembly).
        expected_slots = [0, 1, 0, 1, 0]
        for row, slot in zip(out, expected_slots, strict=True):
            onehot = np.zeros(4)
            onehot[slot] = 1.0
            np.testing.assert_allclose(row, onehot)

    def test_output_dimension_is_sent_and_answers_get_dimension(
        self, transport, monkeypatch
    ):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        emb = VoyageEmbedder("voyage-3.5", transport=transport, output_dimension=512)
        assert emb.get_dimension() == 512
        assert transport.calls == []  # no probe needed
        emb.get_embedding("x")
        assert transport.calls[0]["payload"]["output_dimension"] == 512

    def test_get_dimension_probes_once_and_caches(self, voyage, transport):
        assert voyage.get_dimension() == 4
        assert voyage.get_dimension() == 4
        assert len(transport.calls) == 1  # probe exactly once

    def test_missing_api_key_raises(self, transport, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
            VoyageEmbedder("voyage-3.5", transport=transport)

    def test_http_error_surfaces_status(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        emb = VoyageEmbedder(
            "voyage-3.5", transport=FakeTransport(status=429)
        )
        with pytest.raises(RuntimeError, match="429"):
            emb.get_embedding("x")

    def test_exposes_model_name(self, voyage):
        assert voyage.model_name == "voyage-3.5"


class _NamedStub:
    """Minimal embedder with an explicit identity, for guard tests."""

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self._dim = dimension

    def get_dimension(self) -> int:
        return self._dim

    def get_embedding(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self._dim)
        return v / np.linalg.norm(v)


class TestFingerprint:
    def test_model_name_and_dimension(self):
        assert embedder_fingerprint(_NamedStub("voyage-3.5", 1024)) == "voyage-3.5/1024"

    def test_falls_back_to_class_name(self, embedder):
        assert embedder_fingerprint(embedder) == "DeterministicEmbedder/32"


@pytest.mark.mongo
class TestTenantEmbedderGuard:
    @pytest.fixture()
    def scope(self):
        return TenantScope(tenant="acme", agent="advisor-01")

    def test_first_use_registers_the_fingerprint(self, db, scope, embedder):
        fp = ensure_tenant_embedder(db, scope, embedder)
        doc = db.tenants.find_one({"tenant": "acme"})
        assert doc["embedder"]["fingerprint"] == fp == "DeterministicEmbedder/32"
        assert doc["embedder"]["dimension"] == 32

    def test_same_fingerprint_passes(self, db, scope, embedder):
        ensure_tenant_embedder(db, scope, embedder)
        ensure_tenant_embedder(db, scope, embedder)  # must not raise

    def test_model_switch_fails_closed(self, db, scope):
        ensure_tenant_embedder(db, scope, _NamedStub("voyage-3.5", 32))
        with pytest.raises(EmbedderMismatch, match="voyage-3.5/32"):
            ensure_tenant_embedder(db, scope, _NamedStub("voyage-3.5-lite", 32))

    def test_dimension_switch_fails_closed(self, db, scope):
        ensure_tenant_embedder(db, scope, _NamedStub("voyage-3.5", 1024))
        with pytest.raises(EmbedderMismatch):
            ensure_tenant_embedder(db, scope, _NamedStub("voyage-3.5", 512))

    def test_committer_and_retrieval_are_guarded(
        self, mongo_client, db, scope, embedder
    ):
        from semvec_atlas.commit import TurnCommitter
        from semvec_atlas.indexes import provision
        from semvec_atlas.retrieval import RetrievalEngine

        provision(db, dimension=32, search_indexes=False)
        TurnCommitter(mongo_client, db, scope, embedder=embedder, snapshot_every=0)
        assert db.tenants.find_one({"tenant": "acme"})["embedder"]["fingerprint"] == (
            "DeterministicEmbedder/32"
        )
        # Reading with a different embedder identity must fail closed.
        with pytest.raises(EmbedderMismatch):
            RetrievalEngine(db, scope, embedder=_NamedStub("voyage-3.5", 32))
        # The matching embedder passes.
        RetrievalEngine(db, scope, embedder=embedder)
