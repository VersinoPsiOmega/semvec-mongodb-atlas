"""MongoDB session-state store for ``semvec serve``.

The CAS store ships in the engine package
(``semvec.api.mongo_state_store``, with ``scope=OwnerScope`` ownership
on save/load and the atomic ``save(..., events=...)`` turn commit).
This class is a thin
**subclass** of it — one source of truth for the CAS and transaction
semantics upstream — plus exactly the local additions the upstream
class deliberately omits:

* ``set_drift_exempt`` / ``is_drift_exempt`` — the sibling SQL and
  Redis stores carry this pair; upstream's ``cluster_manager`` holds
  the flag in-memory and persists best-effort to SQL, so the engine's
  MongoDB store does not need it. semvec-atlas keeps the pair so all
  three duck-type surfaces stay interchangeable for our callers.
* a ``Database``-object constructor — semvec-atlas composes stores from
  an existing ``pymongo`` database handle (one client per process, one
  tenant-guarded database), while the upstream constructor resolves its
  own client from ``SEMVEC_MONGO_URI``/``SEMVEC_MONGO_DB``.
* an optional ``event_store`` — the writer behind the inherited
  ``save(..., events=...)`` atomic turn commit. Upstream builds its
  writer lazily from the environment into ``self._event_store``; this
  constructor pre-populates that same seam with the tenant-scoped
  :class:`~semvec_atlas.event_store.MongoEventStore`, so the audit rows
  keep the Atlas layout (literal ``tenant``/``agent`` columns,
  per-scope ``seq``) and pass the ``compliance_events`` validator.
  Without one, the inherited env-resolved writer applies unchanged.

Everything else — ``save`` (CAS insert/update, scoped CAS, the atomic
state+events transaction), ``load``, ``load_full`` (with the SHA-256
integrity check), ``load_owner``, ``current_version``, ``delete``, the
reused ``StateVersionConflict`` — is inherited, never forked. The
lineage is enforced by tests
(tests/test_session_store.py::TestUpstreamLineage).
"""

from __future__ import annotations

import ast
import functools
import inspect
import logging
import textwrap
from typing import Any

from pymongo.database import Database
from semvec.api.mongo_state_store import DEFAULT_COLLECTION
from semvec.api.mongo_state_store import MongoStateStore as _EngineMongoStateStore

from semvec_atlas.client_metadata import announce

__all__ = ["MongoStateStore", "literal_constructor_defaults"]

logger = logging.getLogger(__name__)


@functools.cache
def _literal_assignments(cls: type) -> tuple[tuple[str, ast.expr], ...]:
    """``(name, value node)`` for every unconditional top-level
    ``self.<name> = <literal>`` in ``cls.__init__``, in source order;
    empty when the source is not shipped."""
    try:
        source = textwrap.dedent(inspect.getsource(cls.__init__))
    except (OSError, TypeError):
        logger.warning(
            "source of %s.__init__ unavailable; its literal attribute "
            "defaults are not adopted", cls.__qualname__,
        )
        return ()
    function = ast.parse(source).body[0]
    if not isinstance(function, ast.FunctionDef):
        return ()
    found: list[tuple[str, ast.expr]] = []
    for node in function.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if not (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                continue
            try:
                ast.literal_eval(value)
            except ValueError:
                continue
            found.append((target.attr, value))
    return tuple(found)


def literal_constructor_defaults(cls: type) -> dict[str, Any]:
    """The literal attribute defaults ``cls.__init__`` assigns.

    Evaluated afresh on every call, so a mutable default (``{}``,
    ``[]``) is never shared between instances.
    """
    return {name: ast.literal_eval(value) for name, value in _literal_assignments(cls)}


class MongoStateStore(_EngineMongoStateStore):
    """Engine CAS store bound to an existing database handle,
    plus the ``drift_exempt`` pair and an optional wired event store."""

    def __init__(
        self,
        db: Database,
        *,
        collection: str = DEFAULT_COLLECTION,
        event_store: Any | None = None,
    ) -> None:
        # Deliberately no super().__init__(): the upstream constructor
        # resolves a client from the environment; semvec-atlas already
        # holds a tenant-scoped Database (and calling it would replace
        # the wheel's process-wide client singleton). Literal defaults
        # the upstream constructor assigns — lazy flags and caches such
        # as ``_ttl_index_ensured`` — are adopted from its source, so
        # wheel releases that add one keep working unchanged. The
        # attributes below are the derived surface the inherited
        # methods read (contract pinned by the lineage and drift tests):
        # ``_collection`` for every CAS read/write, ``_event_store`` /
        # ``_uri`` / ``_db_name`` for the lazily built event writer
        # behind ``save(events=...)``, and ``_txn_supported`` for the
        # cached transaction-capability probe.
        announce(db.client)
        for name, value in literal_constructor_defaults(_EngineMongoStateStore).items():
            setattr(self, name, value)
        self._collection = db[collection]
        self._uri = None
        self._db_name = db.name
        self._event_store = event_store
        self._txn_supported: bool | None = None

    # ------------------------------------------------------------------
    # local additions — the flag is owned by its setter and survives
    # CAS blob writes (same rule as the SQL/Redis siblings).

    def set_drift_exempt(self, session_id: str, value: bool) -> None:
        self._collection.update_one(
            {"_id": session_id}, {"$set": {"drift_exempt": bool(value)}}
        )

    def is_drift_exempt(self, session_id: str) -> bool:
        doc = self._collection.find_one({"_id": session_id}, {"drift_exempt": 1})
        return bool(doc.get("drift_exempt", False)) if doc else False
