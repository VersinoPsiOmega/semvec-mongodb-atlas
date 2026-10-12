"""Mandatory tenant scoping for every query path.

The guard is a *builder*, not a convention: repository code never
assembles raw filters. It asks a :class:`TenantScope` for one, and the
scope guarantees the tenant (and, when bound, the agent) is present and
cannot be overridden by caller-supplied conditions.

Since semvec 0.8.8 the wheel promises this idea itself as
:class:`semvec.api.owner_scope.OwnerScope` — a frozen ``(tenant,
agent?)`` value type whose encoding into one ``owner_subject`` string
equals this package's ``user_id`` convention. The two differ only in
the persisted shape: semvec writes the single encoded string,
semvec-atlas writes two literal ``tenant``/``agent`` columns (they
index and validate independently). So validation and the exception are
**delegated, not forked** — :class:`TenantGuardViolation` IS the
upstream exception, and construction runs through ``OwnerScope`` so
both sides refuse the same inputs (empty components, and ``/`` inside
either component, which would make ``("a/b", "c")`` and ``("a",
"b/c")`` encode identically). ``filter()`` stays local because the
two-column shape is this package's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semvec.api.owner_scope import OwnerScope, TenantGuardViolation

__all__ = ["TenantGuardViolation", "TenantScope"]


_PROTECTED_KEYS = ("tenant", "agent")


@dataclass(frozen=True)
class TenantScope:
    """An immutable (tenant, agent) binding all queries run under.

    ``agent`` is optional: tenant-wide paths (audit, retention sweeps)
    legitimately span all agents of one tenant. The tenant itself is
    never optional.
    """

    tenant: str
    agent: str | None = None

    def __post_init__(self) -> None:
        # One validator for both scope types: OwnerScope refuses empty
        # or non-string components and the reserved "/" separator, and
        # raises the same TenantGuardViolation re-exported above.
        OwnerScope(self.tenant, self.agent)

    def to_owner_scope(self) -> OwnerScope:
        """The upstream twin of this scope.

        ``to_owner_scope().owner_subject`` equals :attr:`user_id` for an
        agent-bound scope (both encode to ``"{tenant}/{agent}"``), so a
        caller wiring semvec-atlas stores into upstream seams — e.g.
        ``MongoStateStore.save(scope=...)`` next to an event store bound
        to this scope — cannot hand the two sides different owners.
        """
        return OwnerScope(self.tenant, self.agent)

    @property
    def user_id(self) -> str:
        """The event-store partition key: ``"{tenant}/{agent}"``.

        The semvec ``EventStore`` ABC knows a single ``user_id`` string;
        this property is the canonical encoding of the scope into that
        field (identical to ``OwnerScope.owner_subject`` by
        construction). Requires a bound agent — event streams are
        per-agent.
        """
        if self.agent is None:
            raise TenantGuardViolation(
                "user_id requires an agent-bound scope (event streams are per-agent)"
            )
        return self.to_owner_scope().owner_subject

    def filter(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a fresh MongoDB filter guaranteed to carry this scope.

        ``extra`` conditions are merged in; any attempt to set a
        protected key (``tenant``/``agent``) raises
        :class:`TenantGuardViolation` — even to the same value, so a
        smuggled override never passes silently.
        """
        if extra:
            for key in _PROTECTED_KEYS:
                if key in extra:
                    raise TenantGuardViolation(
                        f"filter may not override {key!r} (scope is {self.tenant!r}/{self.agent!r})"
                    )
        out: dict[str, Any] = {"tenant": self.tenant}
        if self.agent is not None:
            out["agent"] = self.agent
        if extra:
            out.update(extra)
        return out
