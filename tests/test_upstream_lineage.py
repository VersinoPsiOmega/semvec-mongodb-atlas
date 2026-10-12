"""Upstream identity pins.

The wheel promises several pieces this package relies on. Each pin
below asserts the semvec-atlas name IS the upstream object — one
source of truth, so a copy that could drift cannot exist. Mirrors the
lineage tests in ``test_session_store.py``.
"""

from __future__ import annotations

import pytest


class TestGuardLineage:
    def test_violation_is_the_upstream_exception(self):
        from semvec.api.owner_scope import TenantGuardViolation as Upstream

        from semvec_atlas.guard import TenantGuardViolation

        assert TenantGuardViolation is Upstream

    def test_scope_validation_is_delegated_slash_refused(self):
        # OwnerScope encodes (tenant, agent) into one "tenant/agent"
        # string; a "/" inside either component would make two scopes
        # share rows. TenantScope delegates validation, so the same
        # refusal holds here.
        from semvec_atlas.guard import TenantGuardViolation, TenantScope

        with pytest.raises(TenantGuardViolation):
            TenantScope(tenant="a/b", agent="c")
        with pytest.raises(TenantGuardViolation):
            TenantScope(tenant="a", agent="b/c")

    def test_to_owner_scope_encodes_like_user_id(self):
        from semvec.api.owner_scope import OwnerScope

        from semvec_atlas.guard import TenantScope

        two = TenantScope(tenant="acme", agent="advisor-01").to_owner_scope()
        assert isinstance(two, OwnerScope)
        assert two.owner_subject == "acme/advisor-01"
        assert two.owner_subject == TenantScope("acme", "advisor-01").user_id

        wide = TenantScope(tenant="acme").to_owner_scope()
        assert wide.owner_subject == "acme"
        assert wide.agent is None


class TestRetrievalLineage:
    def test_detect_rank_fusion_is_the_upstream_probe(self):
        from semvec.api.mongo_retrieval_backend import (
            detect_rank_fusion as upstream_probe,
        )

        from semvec_atlas.retrieval import detect_rank_fusion

        assert detect_rank_fusion is upstream_probe


class TestComplianceLineage:
    def test_priv_pem_resolver_is_the_promised_public_function(self):
        # Promised public (in __all__); no private fallback import.
        import semvec.compliance.certificates as certs

        import semvec_atlas.compliance as compliance

        assert compliance.resolve_priv_pem is certs.resolve_priv_pem
        assert "resolve_priv_pem" in certs.__all__


class TestVersionPolicy:
    def test_semvec_requirements_have_no_upper_bound(self):
        # The wheel's newer minors must stay installable; compatibility
        # is guarded by the lineage and drift suites, not by a cap.
        tomllib = pytest.importorskip("tomllib")
        from pathlib import Path

        # Arrange
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        requirements = list(project["dependencies"])
        for extra in project["optional-dependencies"].values():
            requirements.extend(extra)

        # Act
        capped = [r for r in requirements if r.startswith("semvec") and "<" in r]

        # Assert
        assert not capped, f"semvec requirement capped: {capped}"

    def test_pymongo_floor_provides_append_metadata(self):
        # ``MongoClient.append_metadata`` (the handshake spec's wrapping-
        # library API) ships since pymongo 4.14, but only 4.14.1 drops
        # duplicate driver info; client_metadata calls it unguarded from
        # every entry point, so 4.14.0 would grow the handshake per store.
        tomllib = pytest.importorskip("tomllib")
        from pathlib import Path

        # Arrange
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

        # Act
        pymongo_requirements = [
            r for r in project["dependencies"] if r.startswith("pymongo")
        ]

        # Assert
        assert pymongo_requirements == ["pymongo>=4.14.1"]
