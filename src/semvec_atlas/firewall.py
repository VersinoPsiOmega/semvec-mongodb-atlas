"""Semantic ingest firewall against memory poisoning.

A durable memory store turns a transient prompt injection into a
persistent one: whatever passes ingest is retrieved into future
contexts and survives snapshot/replay. This module gates the write
path *before* the engine — an input whose embedding is too close to
any *exclusion anchor* (a text or vector describing an attack /
off-domain surface) is held instead of committed.

Design notes:

* Built only on the public wheel surface — the engine's
  ``InputIsolationFilter`` lives in the private compiled
  ``semvec._core`` module (not part of the public namespace) and is not
  needed: this package is the gatekeeper *upstream* of
  ``state.update()``, so plain cosine scoring against the anchor set
  suffices.
* The protection is exactly as strong as the anchor catalog. This is
  a similarity gate, not a classifier: callers must seed meaningful
  exclusion texts, and false positives are handled by quarantine +
  release (see ``commit.TurnCommitter``), never silent drops.
* Obfuscated attacks (leetspeak, Cyrillic/Greek homoglyphs, combining
  marks) are folded back to their plain form and re-checked, so a
  disguised variant of a seeded anchor is still caught. Tables follow
  semvec-polaris (``quarantine.normalize_for_firewall``), where they
  are battle-tested.
"""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from semvec_atlas.guard import TenantScope

__all__ = [
    "AnchorStore",
    "FirewallVerdict",
    "QuarantineVault",
    "SemanticFirewall",
    "load_tenant_firewall",
    "normalize_for_firewall",
    "redact",
]

# Hard cap for content-free surfaces (listings, audit events). Short
# enough that even a compact PII string (e.g. a card number) cannot be
# reconstructed; correlation runs over the SHA-256 hash instead.
_PREVIEW_MAX = 16

_VALID_LEVELS = frozenset(("open", "filter", "quarantine", "lockdown"))

# Common confusable homoglyphs (Cyrillic/Greek look-alikes -> Latin).
# Folding these before embedding defeats homoglyph substitution that
# keyword filters and raw embeddings both miss.
_HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "м": "m",
    "т": "t", "к": "k", "в": "b", "н": "h",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "У": "Y",
    "Х": "X", "М": "M", "Т": "T", "К": "K", "В": "B", "Н": "H",
    "ο": "o", "α": "a", "ρ": "p", "ν": "v", "ι": "i", "κ": "k",
    "μ": "m", "τ": "t",
}
# Leetspeak digit/symbol -> letter (hashcat-style sa4 se3 si1 so0 ss5 st7).
_LEET = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"}


def normalize_for_firewall(text: str) -> str:
    """Fold homoglyphs, NFKD-strip combining marks, and de-leet.

    Maps obfuscated attack text back near its plain form before
    embedding, so similarity to a plain-text anchor survives the
    disguise.
    """
    text = "".join(_HOMOGLYPHS.get(ch, ch) for ch in text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(_LEET.get(ch, ch) for ch in text)


@dataclass(frozen=True)
class FirewallVerdict:
    """Outcome of one firewall inspection.

    ``score`` is the max cosine similarity to any exclusion anchor
    (0.0 when no anchors are configured).
    """

    allowed: bool
    level: str
    score: float
    reason: str


class SemanticFirewall:
    """Similarity gate over a set of exclusion anchors.

    Args:
        embedder: Object satisfying the semvec embedder protocol
            (``get_embedding`` / ``get_dimension``); embeddings must be
            L2-normalised so cosine == dot product.
        exclusion: Seed anchor texts (the initial attack catalog).
        threshold: Block when max cosine to any anchor exceeds this.
        level: ``open`` (never block), ``filter``/``quarantine``
            (block on anchor match), ``lockdown`` (block everything).
            The distinction between ``filter`` and ``quarantine`` is
            made by the caller (drop vs. hold), not here.
        normalize: Re-check the de-obfuscated form of inputs that
            passed the raw check (default True).
        anchor_store: Optional :class:`AnchorStore`. When set, every
            inspection also scores against the store's **current**
            anchors — a live view, so tenant-wide learning (inoculation)
            and un-learning (release) take effect on all firewall
            instances immediately, without rebuilds. Costs one anchor
            read per inspection; anchor sets are small by design.
    """

    def __init__(
        self,
        embedder,
        *,
        exclusion: list[str] | None = None,
        threshold: float = 0.7,
        level: str = "quarantine",
        normalize: bool = True,
        anchor_store: AnchorStore | None = None,
    ) -> None:
        _check_level(level)
        self._embedder = embedder
        self._threshold = float(threshold)
        self._level = level
        self._normalize = normalize
        self._anchor_store = anchor_store
        self._exclusion_vecs: list[np.ndarray] = [
            np.asarray(embedder.get_embedding(text), dtype=np.float64)
            for text in exclusion or []
        ]

    @property
    def level(self) -> str:
        return self._level

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def anchor_store(self) -> AnchorStore | None:
        return self._anchor_store

    def inspect(
        self, text: str, embedding: np.ndarray | None = None
    ) -> FirewallVerdict:
        """Score ``text`` against the anchor set.

        ``embedding`` (when the caller already embedded the text, e.g.
        the commit path) is used as-is for the primary check. If the
        raw form passes and normalization changes the text, the folded
        form is embedded and re-checked so obfuscated attacks are
        still caught.
        """
        verdict = self._verdict_for(text, embedding)
        if verdict.allowed and self._normalize:
            folded = normalize_for_firewall(text)
            if folded and folded != text:
                normalized = self._verdict_for(folded, None)
                if not normalized.allowed:
                    return FirewallVerdict(
                        allowed=False,
                        level=normalized.level,
                        score=normalized.score,
                        reason=normalized.reason + " (normalized)",
                    )
        return verdict

    def add_exclusion(self, text: str) -> np.ndarray:
        """Register a new anchor text; returns its embedding.

        This is the runtime learning hook: a blocked attack (or an
        operator-supplied new surface) becomes an anchor, so
        near-variants are caught from now on.
        """
        vec = np.asarray(self._embedder.get_embedding(text), dtype=np.float64)
        self._exclusion_vecs.append(vec)
        return vec

    def add_exclusion_vector(self, vec: np.ndarray) -> None:
        """Register a pre-computed anchor *embedding* (no text).

        The privacy-preserving transfer path: anchors learned in one
        scope can immunize another as vectors only.
        """
        self._exclusion_vecs.append(np.asarray(vec, dtype=np.float64))

    def exclusion_vectors(self) -> list[np.ndarray]:
        """Current anchor embeddings (seeds + learned + live store)."""
        vecs = list(self._exclusion_vecs)
        if self._anchor_store is not None:
            vecs.extend(self._anchor_store.load())
        return vecs

    def set_level(self, level: str) -> None:
        _check_level(level)
        self._level = level

    # ------------------------------------------------------------------

    def _verdict_for(
        self, text: str, embedding: np.ndarray | None
    ) -> FirewallVerdict:
        if self._level == "open":
            return FirewallVerdict(
                allowed=True, level="open", score=0.0, reason="open: firewall disabled"
            )
        if self._level == "lockdown":
            return FirewallVerdict(
                allowed=False,
                level="lockdown",
                score=0.0,
                reason="lockdown: all inputs blocked",
            )
        if embedding is None:
            embedding = np.asarray(
                self._embedder.get_embedding(text), dtype=np.float64
            )
        score = self._max_anchor_similarity(embedding)
        if score > self._threshold:
            return FirewallVerdict(
                allowed=False,
                level=self._level,
                score=score,
                reason=(
                    f"off-domain: exclusion similarity {score:.3f} "
                    f"> threshold {self._threshold:.3f}"
                ),
            )
        return FirewallVerdict(
            allowed=True,
            level=self._level,
            score=score,
            reason="in-domain: no exclusion anchor exceeded",
        )

    def _max_anchor_similarity(self, embedding: np.ndarray) -> float:
        vecs = self.exclusion_vectors()
        if not vecs:
            return 0.0
        return max(float(np.dot(embedding, vec)) for vec in vecs)


def _check_level(level: str) -> None:
    if level not in _VALID_LEVELS:
        raise ValueError(
            f"unknown firewall level {level!r}; valid: {sorted(_VALID_LEVELS)}"
        )


def redact(text: str) -> tuple[str, str]:
    """Content-free rendering of an untrusted text: (preview, sha256).

    The preview is hard-capped at ``_PREVIEW_MAX`` characters (the
    ellipsis counts), so it never reconstructs the input; the hash
    correlates entries across surfaces without exposing content.
    """
    digest = hashlib.sha256(text.encode()).hexdigest()
    if len(text) > _PREVIEW_MAX - 1:
        return text[: _PREVIEW_MAX - 1] + "…", digest
    return text, digest


class QuarantineVault:
    """Tenant-scoped store for firewall-held inputs.

    The vault keeps the **full text and embedding** — unlike audit
    surfaces, which are content-free — because release re-commits the
    original input (this package is the system of record; there is no
    upstream producer to re-ask). Consequence, enforced in
    ``compliance.AtlasForget``: an Art. 17 wipe must clear the vault.
    """

    def __init__(self, db, scope) -> None:
        self._db = db
        self._col = db.quarantined_inputs
        self._scope = scope

    def hold(
        self,
        text: str,
        embedding: np.ndarray,
        verdict: FirewallVerdict,
        *,
        session: str,
        turn: int,
        mongo_session=None,
    ) -> str:
        """Persist a blocked input as ``held``; returns the vault id."""
        preview, digest = redact(text)
        qid = str(uuid.uuid4())
        self._col.insert_one(
            self._scope.filter(
                {
                    "_id": qid,
                    "session": session,
                    "turn": turn,
                    "text": text,
                    "embedding": [float(x) for x in embedding],
                    "text_preview": preview,
                    "text_hash": digest,
                    "score": verdict.score,
                    "reason": verdict.reason,
                    "level": verdict.level,
                    "status": "held",
                    "created_at": datetime.now(timezone.utc),
                }
            ),
            session=mongo_session,
        )
        return qid

    def list(self, status: str = "held") -> list[dict[str, Any]]:
        """Content-free listing (preview + hash, never text/embedding)."""
        cursor = self._col.find(
            self._scope.filter({"status": status}),
            projection={"text": False, "embedding": False},
        ).sort("created_at", -1)
        return list(cursor)

    def release(self, qid: str, *, reason: str | None = None) -> dict[str, Any]:
        """Mark a held entry ``released`` and return the full document.

        The caller re-commits ``doc["text"]`` as a regular turn (a
        release is a human decision that the block was a false
        positive; ``reason`` records that decision). Raises
        ``KeyError`` when the id is unknown, out of scope, or not
        currently held — a double release must be loud.
        """
        doc = self._col.find_one_and_update(
            self._scope.filter({"_id": qid, "status": "held"}),
            {"$set": {"status": "released", "release_reason": reason}},
        )
        if doc is None:
            raise KeyError(f"no held quarantine entry {qid!r} in scope")
        # Un-learn: a release declares the block a false positive, so a
        # learned anchor for this content (inoculation) must go too —
        # tenant-wide, or every sibling agent stays immune against
        # legitimate content. Seed anchors are never touched.
        self._db.firewall_anchors.delete_many(
            TenantScope(tenant=self._scope.tenant).filter(
                {"kind": "learned", "text_hash": doc["text_hash"]}
            )
        )
        return doc

    def discard(self, qid: str) -> bool:
        """Hard-delete a vault entry (malicious content is not retained)."""
        result = self._col.delete_one(self._scope.filter({"_id": qid}))
        return result.deleted_count == 1

    def wipe(self) -> int:
        """Delete every vault entry of the scope (Art. 17 path)."""
        return self._col.delete_many(self._scope.filter()).deleted_count

class AnchorStore:
    """Tenant-shared exclusion anchors (the inoculation surface).

    Anchors are deliberately **tenant-wide**: when one agent's
    firewall blocks an attack, persisting its embedding immunizes
    every agent of the tenant against near-variants (semvec-polaris'
    ``FleetFirewall`` idea, mapped onto the Atlas tenant model).
    Reads therefore drop the agent binding; writes record the
    originating agent as provenance so learned anchors stay
    attributable — and wipeable (Art. 17).
    """

    def __init__(self, db, scope) -> None:
        self._col = db.firewall_anchors
        self._scope = scope
        self._tenant_scope = TenantScope(tenant=scope.tenant)

    def add(
        self,
        vec: np.ndarray,
        *,
        kind: str = "learned",
        text_hash: str | None = None,
        mongo_session=None,
    ) -> str:
        """Persist one anchor embedding; returns its id.

        ``kind`` is ``"seed"`` (curated catalog, no user provenance)
        or ``"learned"`` (promoted from a block). ``text_hash``
        optionally correlates a learned anchor with its quarantine
        entry — never the text itself.
        """
        if kind not in ("seed", "learned"):
            raise ValueError(f"unknown anchor kind {kind!r}; valid: seed, learned")
        anchor_id = str(uuid.uuid4())
        self._col.insert_one(
            self._tenant_scope.filter(
                {
                    "_id": anchor_id,
                    "embedding": [float(x) for x in vec],
                    "kind": kind,
                    "provenance": {"agent": self._scope.agent, "kind": kind},
                    "text_hash": text_hash,
                    "created_at": datetime.now(timezone.utc),
                }
            ),
            session=mongo_session,
        )
        return anchor_id

    def load(self) -> list[np.ndarray]:
        """All anchor embeddings of the tenant (seed + learned)."""
        return [
            np.asarray(doc["embedding"], dtype=np.float64)
            for doc in self._col.find(
                self._tenant_scope.filter(), projection={"embedding": True}
            )
        ]

    def wipe_learned(self) -> int:
        """Delete the *learned* anchors this scope's agent contributed.

        Art. 17 support: learned anchors derive from user content;
        seed anchors are operator-curated and carry no user provenance,
        so they survive.
        """
        return self._col.delete_many(
            self._tenant_scope.filter(
                {"kind": "learned", "provenance.agent": self._scope.agent}
            )
        ).deleted_count


def load_tenant_firewall(
    db,
    scope,
    embedder,
    *,
    exclusion: list[str] | None = None,
    threshold: float = 0.7,
    level: str = "quarantine",
    normalize: bool = True,
) -> SemanticFirewall:
    """Build a firewall with a **live view** on the tenant's anchors.

    ``exclusion`` seeds additional catalog texts on top; the persisted
    anchors (seed + learned, tenant-wide) are read through
    :class:`AnchorStore` on every inspection, so an attack blocked by
    any sibling agent is covered immediately — and a released false
    positive stops blocking immediately as well.
    """
    return SemanticFirewall(
        embedder,
        exclusion=exclusion,
        threshold=threshold,
        level=level,
        normalize=normalize,
        anchor_store=AnchorStore(db, scope),
    )
