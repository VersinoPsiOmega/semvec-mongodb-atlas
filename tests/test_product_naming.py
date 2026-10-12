# ABOUTME: Tests that the product is always called "MongoDB" in documentation and comments.
# ABOUTME: Code names such as MongoClient, pymongo, mongod or SEMVEC_MONGO_URI are not affected.
"""Product naming.

The database is called MongoDB. A bare "Mongo" in a README, a docstring or a
comment reads as a nickname for someone else's product, so the repository
never uses it. Identifiers that merely contain the letters — ``MongoClient``,
``MongoStateStore``, ``pymongo``, ``mongod``, ``SEMVEC_MONGO_URI``, the
``mongo`` pytest marker — are code, not prose, and stay as they are: the
check looks for the standalone word only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The standalone word, in any casing the product name could be mistyped in
#: prose ("Mongo store", "Mongo's"), but not as part of a longer name.
STANDALONE_MONGO = re.compile(r"\bMongo\b")

SUFFIXES = {".md", ".py", ".toml", ".sh", ".txt", ".yml", ".yaml"}
SKIPPED_DIRS = {
    ".git",
    ".claude",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "docs-internal",
    "results",
    "build",
}


def standalone_mongo_lines(text: str) -> list[int]:
    """1-based numbers of the lines that use the bare word "Mongo"."""
    return [n for n, line in enumerate(text.splitlines(), start=1) if STANDALONE_MONGO.search(line)]


def _repository_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix in SUFFIXES
        and not SKIPPED_DIRS.intersection(path.relative_to(ROOT).parts)
        and path != Path(__file__).resolve()
    )


class TestStandaloneMongoCheck:
    def test_flags_the_bare_word(self) -> None:
        # Arrange
        text = "the Mongo store\nan unrelated line\nrunning on Mongo, as well\n"

        # Act
        lines = standalone_mongo_lines(text)

        # Assert
        assert lines == [1, 3]

    @pytest.mark.parametrize(
        "line",
        [
            "MongoDB Atlas",
            "pymongo.MongoClient",
            "class MongoStateStore",
            "MongoEventStore and MongoAuditLog",
            "SEMVEC_MONGO_URI",
            "SEMVEC_SESSION_BACKEND=mongo",
            "pytest.mark.mongo",
            "mongod aborts mid-run",
            "mongodb/mongodb-atlas-local",
            "docker start semvec-atlas-mongo",
        ],
    )
    def test_leaves_product_name_and_code_names_alone(self, line: str) -> None:
        # Act
        lines = standalone_mongo_lines(line)

        # Assert
        assert lines == []


class TestRepository:
    def test_the_check_reads_files(self) -> None:
        # The scan must not pass vacuously on an empty file list.
        assert len(_repository_files()) > 30

    def test_no_file_uses_the_bare_word_mongo(self) -> None:
        # Arrange
        offenders = []
        for path in _repository_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            offenders.extend(
                f"{path.relative_to(ROOT).as_posix()}:{n}" for n in standalone_mongo_lines(text)
            )

        # Assert
        assert not offenders, "write MongoDB, not Mongo: " + ", ".join(offenders)
