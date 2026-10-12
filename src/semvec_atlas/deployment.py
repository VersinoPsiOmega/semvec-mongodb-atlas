"""Capability probe for a MongoDB deployment.

This package makes promises that depend on the server it runs against:
a turn is atomic *because* transactions exist, retrieval is server-side
*because* the Search Index service is there, fusion happens in the
database *because* the version supports ``$rankFusion``. Each of those
can be absent, and each absence changes the guarantee rather than
producing an obvious error at startup.

So ask the server. :func:`probe_deployment` reports what it found and
:meth:`DeploymentReport.summary` turns that into something an operator
can act on. Run it before assuming a cluster behaves like the local
container this package is developed against::

    python -m semvec_atlas.deployment "mongodb+srv://…"

One distinction runs through the report. Transactions are a hard
requirement: without them the write path cannot keep its promise, and
``ready`` is false. Search indexes and ``$rankFusion`` are
optimisations with working fallbacks — their absence is reported, and
the deployment still counts as usable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from semvec_atlas.client_metadata import announce

logger = logging.getLogger(__name__)

__all__ = ["DeploymentReport", "probe_deployment"]

_PROBE_DB = "semvec_atlas_probe"
_PROBE_COLLECTION = "_capability_probe"


@dataclass(frozen=True)
class DeploymentReport:
    """What one deployment can do, as measured rather than assumed."""

    reachable: bool
    server_version: str
    topology: str
    transactions: bool
    search_indexes: bool
    rank_fusion: bool
    notes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        """Whether the package's hard requirements are met.

        Only reachability and transactions. Everything else degrades to
        a documented fallback, and calling a deployment unusable because
        it lacks an optimisation would be wrong.
        """
        return self.reachable and self.transactions

    def summary(self) -> str:
        """A short operator-facing report, one finding per line."""
        if not self.reachable:
            return (
                "Deployment unreachable — server selection failed. Check the "
                "connection string, network access list and credentials."
            )
        lines = [
            f"MongoDB {self.server_version} ({self.topology})",
            f"  transactions   : {'yes' if self.transactions else 'NO — required'}",
            f"  search indexes : {'yes' if self.search_indexes else 'no'}",
            f"  $rankFusion    : {'yes' if self.rank_fusion else 'no'}",
        ]
        if not self.transactions:
            lines.append(
                "  → A turn cannot be committed atomically here. Run a replica "
                "set (one member is enough: mongod --replSet rs0, then "
                "rs.initiate())."
            )
        if not self.search_indexes:
            lines.append(
                "  → No Search Index service: retrieval falls back to "
                "in-process ranking. Item sync and the write path are "
                "unaffected."
            )
        if not self.rank_fusion:
            lines.append(
                "  → Fusion runs client-side (reciprocal-rank fusion at "
                "identical weights). Needs MongoDB 8.1+ alongside "
                "$vectorSearch to run in the server."
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        lines.append(f"  verdict        : {'ready' if self.ready else 'NOT ready'}")
        return "\n".join(lines)


def probe_deployment(client: Any, *, database: str = _PROBE_DB) -> DeploymentReport:
    """Ask a deployment what it supports. Read-only but for one scratch doc.

    Never raises for a server-side answer: an unreachable or limited
    deployment is a *result*, and a probe that crashed would tell the
    caller less than one that reports.
    """
    from pymongo.errors import PyMongoError

    announce(client)
    try:
        hello = client.admin.command("hello")
    except PyMongoError as exc:
        logger.warning("deployment probe could not reach the server: %s", exc)
        return DeploymentReport(
            reachable=False,
            server_version="unknown",
            topology="unknown",
            transactions=False,
            search_indexes=False,
            rank_fusion=False,
            notes=(str(exc)[:200],),
        )

    # A replica set advertises setName; mongos answers isdbgrid. A
    # standalone does neither, and cannot open a transaction.
    if "setName" in hello:
        topology = "replicaSet"
    elif hello.get("msg") == "isdbgrid":
        topology = "sharded"
    else:
        topology = "standalone"

    try:
        version = str(client.server_info().get("version", "unknown"))
    except PyMongoError:  # pragma: no cover - hello succeeded, this rarely fails
        version = "unknown"

    collection = client[database][_PROBE_COLLECTION]
    return DeploymentReport(
        reachable=True,
        server_version=version,
        topology=topology,
        transactions=topology in {"replicaSet", "sharded"},
        search_indexes=_probe_search_indexes(collection),
        rank_fusion=_probe_rank_fusion(collection),
    )


def _probe_search_indexes(collection: Any) -> bool:
    """Is the Search Index Management service present?"""
    from pymongo.errors import PyMongoError

    try:
        list(collection.list_search_indexes())
    except PyMongoError as exc:
        logger.info("no Search Index Management service: %s", exc)
        return False
    return True


def _probe_rank_fusion(collection: Any) -> bool:
    """Can the server run a ``$rankFusion`` stage?

    Uses a scratch document so the probe does not depend on any real
    collection existing, and drops it again.
    """
    from pymongo.errors import PyMongoError

    try:
        collection.insert_one({"probe": 1})
        collection.aggregate(
            [
                {
                    "$rankFusion": {
                        "input": {
                            "pipelines": {
                                "p": [
                                    {"$match": {"probe": 1}},
                                    {"$sort": {"probe": 1}},
                                    {"$limit": 1},
                                ]
                            }
                        }
                    }
                }
            ]
        )
        return True
    except PyMongoError as exc:
        logger.info("$rankFusion unavailable: %s", exc)
        return False
    finally:
        try:
            collection.drop()
        except PyMongoError:  # pragma: no cover - cleanup is best-effort
            logger.debug("could not drop the probe collection")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    """``python -m semvec_atlas.deployment <uri>``."""
    import argparse

    import pymongo

    parser = argparse.ArgumentParser(
        description="Report what a MongoDB deployment supports."
    )
    parser.add_argument("uri", help="connection string")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="server selection timeout (default: 5000)",
    )
    args = parser.parse_args(argv)

    client = pymongo.MongoClient(args.uri, serverSelectionTimeoutMS=args.timeout_ms)
    report = probe_deployment(client)
    print(report.summary())
    return 0 if report.ready else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
