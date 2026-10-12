# ABOUTME: Identifies semvec-atlas in the MongoDB handshake of the customer's client.
# ABOUTME: Appends the library name and installed version as wrapping-library driver info.
"""Handshake client metadata for semvec-atlas.

The MongoDB handshake spec lets a library that wraps the driver append
its own name and version to the client metadata (``client.driver``)
that every new connection sends; the server records it, and on Atlas
MongoDB uses it to see which integrations run against the platform.

semvec-atlas never creates the client — the caller passes one in — so
every public entry point calls :func:`announce` on that client before
its first operation: metadata only reaches connections opened after
the append. The driver de-duplicates identical announcements, so many
stores on one client still yield a single ``semvec-atlas`` entry.

Only the library name and its version are sent: no tenant, agent,
host, or content.
"""

from __future__ import annotations

import functools
from importlib.metadata import PackageNotFoundError, version

from pymongo import MongoClient
from pymongo.driver_info import DriverInfo

__all__ = ["LIBRARY_NAME", "announce", "driver_info"]

#: The name semvec-atlas reports in ``client.driver.name``.
LIBRARY_NAME = "semvec-atlas"

#: The spec's delimiter between the per-library entries of one field.
_METADATA_DELIMITER = "|"


@functools.cache
def _installed_version() -> str | None:
    """The installed distribution's version, resolved once per process
    (the lookup scans the environment; entry points announce on every
    construction); ``None`` for a source tree that was never installed."""
    try:
        return version(LIBRARY_NAME)
    except PackageNotFoundError:
        return None


def driver_info() -> DriverInfo:
    """The wrapping-library driver info: name plus installed version.

    The version comes from the installed distribution, so it always
    matches what the customer runs.
    """
    library_version = _installed_version()
    for field in (LIBRARY_NAME, library_version):
        # The spec obliges the driver to reject the delimiter; pymongo
        # does not, and a stray one would shift every later name/version pair.
        if field is not None and _METADATA_DELIMITER in field:
            raise ValueError(
                f"driver info {field!r} contains the metadata delimiter "
                f"{_METADATA_DELIMITER!r}"
            )
    return DriverInfo(LIBRARY_NAME, library_version, None)


def announce(client: MongoClient) -> None:
    """Append the semvec-atlas driver info to ``client``'s handshake metadata."""
    client.append_metadata(driver_info())
