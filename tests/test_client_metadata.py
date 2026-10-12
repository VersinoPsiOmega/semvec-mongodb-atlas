# ABOUTME: Tests that semvec-atlas announces itself in the MongoDB connection handshake.
# ABOUTME: Observes server-side via $currentOp that every public entry point appends its driver info.
"""Client metadata: semvec-atlas identifies itself in the handshake.

The MongoDB handshake spec ("Supporting Wrapping Libraries") lets a
library that wraps the driver append its own name and version to the
client metadata every new connection sends. MongoDB reads that field to
see which integrations run against Atlas, so every public entry point
that receives a customer's client announces ``semvec-atlas`` on it —
before its first operation, because metadata only reaches connections
opened after the append.

Asserted server-side via ``$currentOp``: what the server records is the
contract, not the driver's internal bookkeeping. Each test runs on its
own client, tagged with a unique ``appname``, so one test's announcement
cannot satisfy another's.
"""

from __future__ import annotations

import importlib.metadata
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pymongo
import pytest
from pymongo.driver_info import DriverInfo

from semvec_atlas.guard import TenantScope

LIBRARY = "semvec-atlas"
SCOPE = TenantScope(tenant="acme", agent="advisor-01")


@pytest.fixture(autouse=True)
def _fresh_version_cache() -> Iterator[None]:
    """Each test resolves the installed version itself."""
    from semvec_atlas.client_metadata import _installed_version

    _installed_version.cache_clear()
    yield
    _installed_version.cache_clear()


class TestDriverInfo:
    def test_installed_version_is_resolved_once_per_process(self, monkeypatch) -> None:
        # Entry points announce on every construction (MCP handlers build
        # one per tool call); the distribution lookup costs ~0.6 ms.
        import semvec_atlas.client_metadata as client_metadata

        # Arrange
        lookups: list[str] = []

        def counting_version(name: str) -> str:
            lookups.append(name)
            return "1.0"

        monkeypatch.setattr(client_metadata, "version", counting_version)

        # Act
        for _ in range(3):
            client_metadata.driver_info()

        # Assert
        assert lookups == [LIBRARY]

    def test_names_the_library_and_its_installed_version(self) -> None:
        from semvec_atlas.client_metadata import driver_info

        # Act
        info = driver_info()

        # Assert
        assert info == DriverInfo(LIBRARY, importlib.metadata.version(LIBRARY), None)

    def test_uninstalled_package_reports_no_version(self, monkeypatch) -> None:
        import semvec_atlas.client_metadata as client_metadata

        # Arrange
        def not_installed(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(client_metadata, "version", not_installed)

        # Act
        info = client_metadata.driver_info()

        # Assert
        assert info == DriverInfo(LIBRARY, None, None)

    def test_metadata_delimiter_in_the_version_is_refused(self, monkeypatch) -> None:
        # The spec makes the driver reject "|"; pymongo does not, and a
        # stray delimiter would shift every name/version pair after it.
        import semvec_atlas.client_metadata as client_metadata

        # Arrange
        monkeypatch.setattr(client_metadata, "version", lambda name: "1.0|2.0")

        # Act / Assert
        with pytest.raises(ValueError, match=r"\|"):
            client_metadata.driver_info()


# ---------------------------------------------------------------------------
# server-side observation


@pytest.fixture()
def tagged_client(mongo_client: Any) -> Iterator[tuple[pymongo.MongoClient, str]]:
    """A private client whose connections carry a unique appname.

    Deliberately unused before the test acts: an operation here would
    pool a connection opened without the library's metadata.
    """
    host, port = mongo_client.address
    app = f"svatest-{uuid.uuid4().hex[:12]}"
    client: pymongo.MongoClient = pymongo.MongoClient(
        host, port, directConnection=True, appname=app, serverSelectionTimeoutMS=5000
    )
    yield client, app
    client.close()


@pytest.fixture()
def tagged_db(tagged_client: tuple[pymongo.MongoClient, str]) -> Iterator[Any]:
    client, _ = tagged_client
    name = f"svatest_{uuid.uuid4().hex[:12]}"
    yield client[name]
    client.drop_database(name)


def _driver_names(client: pymongo.MongoClient, app: str) -> list[list[str]]:
    """``client.driver.name`` of every server connection tagged ``app``,
    split into its per-library entries."""
    ops = client.admin.aggregate([
        {"$currentOp": {"allUsers": True, "idleConnections": True}},
        {"$match": {"appName": app}},
        {"$project": {"_id": 0, "name": "$clientMetadata.driver.name"}},
    ])
    return [op["name"].split("|") for op in ops if "name" in op]


@pytest.mark.mongo
class TestAnnounce:
    def test_server_sees_the_library_on_connections_opened_after(
        self, tagged_client: tuple[pymongo.MongoClient, str]
    ) -> None:
        from semvec_atlas.client_metadata import announce

        # Arrange
        client, app = tagged_client

        # Act
        announce(client)
        client.admin.command("ping")
        names = _driver_names(client, app)

        # Assert
        assert any(LIBRARY in entries for entries in names), names

    def test_repeated_announcements_add_one_entry(
        self, tagged_client: tuple[pymongo.MongoClient, str]
    ) -> None:
        from semvec_atlas.client_metadata import announce

        # Arrange
        client, app = tagged_client

        # Act
        for _ in range(3):
            announce(client)
        client.admin.command("ping")
        names = _driver_names(client, app)

        # Assert
        assert all(entries.count(LIBRARY) <= 1 for entries in names), names


# ---------------------------------------------------------------------------
# every public entry point announces before its first operation


def _turn_committer(client: Any, db: Any, embedder: Any) -> Any:
    from semvec_atlas.commit import TurnCommitter

    return TurnCommitter(client, db, SCOPE, embedder=embedder)


def _retrieval_engine(client: Any, db: Any, embedder: Any) -> Any:
    from semvec_atlas.retrieval import RetrievalEngine

    return RetrievalEngine(db, SCOPE, embedder=embedder)


def _retrieval_engine_for_tenant(client: Any, db: Any, embedder: Any) -> Any:
    from semvec_atlas.retrieval import RetrievalEngine

    return RetrievalEngine.for_tenant(db, SCOPE, embedder=embedder)


def _semvec_store(client: Any, db: Any, embedder: Any) -> Any:
    from semvec_atlas.store import SemvecStore

    return SemvecStore(db, scope=SCOPE, embedder=embedder)


def _session_store(client: Any, db: Any, embedder: Any) -> Any:
    from semvec_atlas.session_store import MongoStateStore

    return MongoStateStore(db)


def _provision(client: Any, db: Any, embedder: Any) -> Any:
    from semvec_atlas.indexes import provision

    return provision(db, dimension=embedder.get_dimension(), search_indexes=False)


def _atlas_forget(client: Any, db: Any, embedder: Any) -> Any:
    from semvec_atlas.compliance import AtlasForget

    return AtlasForget(db, SCOPE)


def _cortex_store(client: Any, db: Any, embedder: Any) -> Any:
    from semvec_atlas.cortex_store import AtlasCortexStore

    return AtlasCortexStore(db, SCOPE)


def _mcp_server(client: Any, db: Any, embedder: Any) -> Any:
    pytest.importorskip("fastmcp")
    from semvec_atlas.mcp import create_atlas_mcp_server

    return create_atlas_mcp_server(db, SCOPE, embedder=embedder)


def _deployment_probe(client: Any, db: Any, embedder: Any) -> Any:
    from semvec_atlas.deployment import probe_deployment

    return probe_deployment(client)


ENTRY_POINTS: dict[str, Callable[[Any, Any, Any], Any]] = {
    "TurnCommitter": _turn_committer,
    "RetrievalEngine": _retrieval_engine,
    "RetrievalEngine.for_tenant": _retrieval_engine_for_tenant,
    "SemvecStore": _semvec_store,
    "MongoStateStore": _session_store,
    "provision": _provision,
    "AtlasForget": _atlas_forget,
    "AtlasCortexStore": _cortex_store,
    "create_atlas_mcp_server": _mcp_server,
    "probe_deployment": _deployment_probe,
}


@pytest.mark.mongo
class TestEntryPointsAnnounce:
    @pytest.mark.parametrize("entry", ENTRY_POINTS, ids=str)
    def test_entry_point_announces_the_library(
        self,
        entry: str,
        tagged_client: tuple[pymongo.MongoClient, str],
        tagged_db: Any,
        embedder: Any,
    ) -> None:
        # Arrange
        client, app = tagged_client

        # Act
        ENTRY_POINTS[entry](client, tagged_db, embedder)
        tagged_db.command("ping")
        names = _driver_names(client, app)

        # Assert
        assert any(LIBRARY in entries for entries in names), (
            f"{entry} did not announce {LIBRARY} before its first operation: {names}"
        )
