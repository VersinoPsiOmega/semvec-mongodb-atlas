# ABOUTME: Tests that the advisor demo reads its .env only when it talks to an LLM.
# ABOUTME: A clean checkout without any .env must run `--no-llm` end to end.
"""The demo's environment loading.

``demo/advisor_demo.py --no-llm`` is the documented way to try the
package (README, article). It needs MongoDB and a local embedder — no
LLM, so no ``OPENAI_API_KEY`` and no ``.env``. ``load_env`` refuses to
continue when it finds no ``.env`` with an API key, so the demo must
not call it on the ``--no-llm`` path; a developer machine with a
``.env`` in some parent directory would never notice if it did.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

DEMO = Path(__file__).resolve().parents[1] / "demo" / "advisor_demo.py"


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("advisor_demo_under_test", DEMO)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLoadDemoEnv:
    def test_without_llm_no_env_file_is_read(self, demo, monkeypatch) -> None:
        # Arrange
        def refuse(*_args, **_kwargs):
            raise SystemExit("no .env with OPENAI_API_KEY found")

        monkeypatch.setattr(demo, "load_env", refuse)

        # Act
        env = demo.load_demo_env(use_llm=False)

        # Assert
        assert env == {}

    def test_with_llm_the_env_file_is_required(self, demo, monkeypatch) -> None:
        # Arrange
        def refuse(*_args, **_kwargs):
            raise SystemExit("no .env with OPENAI_API_KEY found")

        monkeypatch.setattr(demo, "load_env", refuse)

        # Act / Assert
        with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
            demo.load_demo_env(use_llm=True)

    def test_with_llm_the_loaded_values_are_returned(self, demo, monkeypatch) -> None:
        # Arrange
        loaded = {"OPENAI_API_KEY": "k", "OPENAI_MODEL": "m"}
        monkeypatch.setattr(demo, "load_env", lambda: loaded)

        # Act
        env = demo.load_demo_env(use_llm=True)

        # Assert
        assert env == loaded
