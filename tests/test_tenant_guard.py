"""Tenant guard.

Every query path in semvec-atlas must carry a tenant filter. The guard
is a central query builder, not a convention: filters are constructed
through ``TenantScope.filter()`` and attempts to smuggle a different
tenant (or drop it) raise ``TenantGuardViolation``.
"""

from __future__ import annotations

import pytest

from semvec_atlas.guard import TenantGuardViolation, TenantScope


class TestTenantScope:
    def test_filter_always_contains_tenant(self):
        scope = TenantScope(tenant="acme", agent="advisor-01")
        f = scope.filter()
        assert f["tenant"] == "acme"
        assert f["agent"] == "advisor-01"

    def test_filter_merges_extra_conditions(self):
        scope = TenantScope(tenant="acme", agent="advisor-01")
        f = scope.filter({"tier": "long", "deleted": False})
        assert f["tenant"] == "acme"
        assert f["tier"] == "long"
        assert f["deleted"] is False

    def test_extra_cannot_override_tenant(self):
        scope = TenantScope(tenant="acme", agent="advisor-01")
        with pytest.raises(TenantGuardViolation):
            scope.filter({"tenant": "other"})

    def test_extra_cannot_override_agent(self):
        scope = TenantScope(tenant="acme", agent="advisor-01")
        with pytest.raises(TenantGuardViolation):
            scope.filter({"agent": "other-agent"})

    def test_empty_tenant_rejected_at_construction(self):
        with pytest.raises(TenantGuardViolation):
            TenantScope(tenant="", agent="a")

    def test_none_tenant_rejected_at_construction(self):
        with pytest.raises(TenantGuardViolation):
            TenantScope(tenant=None, agent="a")  # type: ignore[arg-type]

    def test_agent_optional_filter_scopes_tenant_wide(self):
        # Some paths (tenant-wide audit) legitimately span all agents.
        scope = TenantScope(tenant="acme")
        f = scope.filter()
        assert f["tenant"] == "acme"
        assert "agent" not in f

    def test_dollar_operators_rejected_in_tenant_position(self):
        # Injection guard: a tenant value must be a plain string, never
        # an operator document.
        with pytest.raises(TenantGuardViolation):
            TenantScope(tenant={"$ne": ""}, agent="a")  # type: ignore[arg-type]

    def test_filter_returns_fresh_dict(self):
        # Mutating a returned filter must not poison later calls.
        scope = TenantScope(tenant="acme", agent="a")
        f1 = scope.filter()
        f1["tenant"] = "evil"
        assert scope.filter()["tenant"] == "acme"
