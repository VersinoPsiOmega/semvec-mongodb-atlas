"""What can this deployment actually do?

Everything in this package has been verified against a local
``mongodb/mongodb-atlas-local`` container. That is a faithful stand-in
for most of the surface and not a substitute for the real thing: a
community server has no Search Index service, a standalone cannot open
a transaction, and ``$rankFusion`` needs a version floor. Each of those
changes what the software can promise, and each is invisible until the
first query fails.

:func:`probe_deployment` asks the server directly and reports what it
found, so an operator (or a reviewer with a cluster we do not have) can
answer the capability questions in one run instead of by reading
release notes.

The probe is read-only apart from one scratch collection it creates and
drops for the ``$rankFusion`` check — the same technique the engine's
own capability probe uses.
"""

from __future__ import annotations

import pytest

from semvec_atlas.deployment import DeploymentReport, probe_deployment

pytestmark = pytest.mark.mongo


class TestProbe:
    def test_reports_every_capability_the_package_depends_on(self, mongo_client):
        report = probe_deployment(mongo_client)
        assert isinstance(report, DeploymentReport)
        for field in (
            "server_version",
            "topology",
            "transactions",
            "search_indexes",
            "rank_fusion",
        ):
            assert getattr(report, field) is not None, f"{field} not probed"

    def test_atlas_local_supports_transactions(self, mongo_client):
        """The container is a single-member replica set, so this must hold.

        It is also the property the whole write path rests on: without
        transactions a turn cannot be atomic, and the package would be
        making a promise the server cannot keep.
        """
        report = probe_deployment(mongo_client)
        assert report.transactions is True
        assert report.topology in {"replicaSet", "sharded"}

    def test_report_names_what_is_missing(self, mongo_client):
        """A report is only useful if it says what to do about a gap."""
        report = probe_deployment(mongo_client)
        summary = report.summary()
        assert report.server_version in summary
        if not report.search_indexes:
            assert "search" in summary.lower()

    def test_ready_reflects_the_hard_requirements(self, mongo_client):
        """``ready`` is about what the package cannot work without.

        Transactions are non-negotiable. Server-side search and fusion
        are optimisations with working fallbacks, so their absence is
        reported without failing the deployment.
        """
        report = probe_deployment(mongo_client)
        assert report.ready is report.transactions

    def test_unreachable_server_reports_instead_of_hanging(self):
        import pymongo

        dead = pymongo.MongoClient(
            "mongodb://127.0.0.1:1/?directConnection=true",
            serverSelectionTimeoutMS=250,
        )
        report = probe_deployment(dead)
        assert report.reachable is False
        assert report.ready is False
        assert "unreachable" in report.summary().lower()
