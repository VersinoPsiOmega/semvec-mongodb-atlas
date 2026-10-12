"""Documentation claims that must stay true about the server surface.

These are cheap file assertions, not behaviour tests, and they exist
because a stale capability claim is the kind of error no runtime test
catches: the code keeps working while the prose tells an operator
something false about the database they are running on.

``$rankFusion`` shipped as a preview in MongoDB 8.0/8.1 and is now
generally available. The client-side RRF fallback is still correct and
still shipped — but the reason changed. It covers servers *below* the
feature's availability, not a preview that might be withdrawn, and the
comments must say so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Files that describe the fusion path and its fallback.
FUSION_DOCS = (
    ROOT / "src" / "semvec_atlas" / "retrieval.py",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "PRODUCT_OVERVIEW.md",
)


def test_retrieval_module_makes_no_preview_claim() -> None:
    """Every use of "preview" in ``retrieval.py`` is an availability claim.

    Unlike the firewall modules — where a *text* preview is the
    redaction cap and entirely legitimate — this module has no other
    reason to say the word, so the check can be absolute here.
    """
    text = (ROOT / "src" / "semvec_atlas" / "retrieval.py").read_text(encoding="utf-8")
    offenders = [ln.strip() for ln in text.splitlines() if "preview" in ln.lower()]
    assert not offenders, f"retrieval.py still calls $rankFusion a preview: {offenders}"


@pytest.mark.parametrize("path", FUSION_DOCS, ids=lambda p: p.name)
def test_no_preview_drift_justification(path: Path) -> None:
    """The fallback is not justified by "the preview might be withdrawn".

    Matched as a phrase so the firewall's redaction preview — a
    different concept that shares the word — is untouched.
    """
    text = path.read_text(encoding="utf-8").lower()
    assert "preview drift" not in text, (
        f"{path.name} still justifies the RRF fallback with preview drift"
    )
    assert "preview)" not in text, (
        f"{path.name} still labels a feature as preview"
    )


def test_fallback_reason_names_the_version_floor() -> None:
    """The fallback must be justified by server version, not by risk.

    An operator reading the module needs to know *when* the client-side
    path is taken — on a server below the feature's availability — so
    they can decide whether to upgrade instead of tuning around it.
    """
    text = (ROOT / "src" / "semvec_atlas" / "retrieval.py").read_text(encoding="utf-8")
    assert "8.1" in text, "retrieval.py must name the MongoDB version floor"
