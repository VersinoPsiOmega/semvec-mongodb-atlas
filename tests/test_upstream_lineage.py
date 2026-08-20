"""Upstream dedup pins (semvec >= 0.8.8).

The wheel (>= 0.8.8) promises these pieces directly. Each pin below
asserts the semvec-atlas name IS the upstream object — one source of truth, so the copy that would drift cannot
exist. Mirrors the lineage tests in ``test_session_store.py``.
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
        # Promised public (in __all__) since semvec 0.8.5.
        import semvec.compliance.certificates as certs

        import semvec_atlas.compliance as compliance

        assert compliance.resolve_priv_pem is certs.resolve_priv_pem
        assert "resolve_priv_pem" in certs.__all__
