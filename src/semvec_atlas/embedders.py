"""Embedding providers and the tenant embedder-fingerprint guard.

The package is embedder-agnostic — anything with
``get_embedding``/``get_dimension`` works. This module provides:

* :class:`VoyageEmbedder` — Voyage AI (MongoDB's embedding-model
  family) over its REST API, with batching and defensive L2
  normalization. Zero new dependencies: the default transport is
  stdlib ``urllib``; tests inject a fake transport.
* :func:`embedder_fingerprint` / :func:`ensure_tenant_embedder` — the
  fail-closed identity guard. Embeddings are produced client-side and
  persisted, so a silent model or dimension switch would degrade
  similarity search without any error. The tenant document therefore
  records the embedder identity (``<model>/<dimension>``); every
  committer and retrieval engine verifies it at construction.

Recovery note: the event log persists the embeddings themselves and
replay folds ``(embedding, content)`` pairs — recovery never calls an
embedding API and is independent of provider availability. Only
*query-time* embeddings must come from the registered model, which is
exactly what the guard enforces.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import numpy as np

from semvec_atlas.guard import TenantScope

__all__ = [
    "EmbedderMismatch",
    "VoyageEmbedder",
    "embedder_fingerprint",
    "ensure_tenant_embedder",
]

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

# transport(url, headers, payload) -> (status_code, parsed_json_body)
Transport = Callable[[str, dict, dict], tuple[int, dict]]


def _urllib_transport(url: str, headers: dict, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:  # pragma: no cover - live API only
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return exc.code, body


class VoyageEmbedder:
    """Voyage AI embedder satisfying the semvec embedder protocol.

    Args:
        model: Voyage model name (e.g. ``voyage-3.5``,
            ``voyage-3.5-lite``).
        api_key: Voyage API key; defaults to ``VOYAGE_API_KEY`` from the
            environment. Missing key fails loudly at construction, not
            on the first turn.
        output_dimension: Optional reduced output dimension for models
            that support it; when unset, the dimension is probed once
            and cached.
        input_type: Optional Voyage retrieval hint (``"document"`` /
            ``"query"``); unset sends none.
        batch_size: Max texts per request; larger inputs are chunked
            and reassembled in request order via the response indexes.
        transport: Injectable request function (tests); defaults to a
            stdlib ``urllib`` POST.

    Vectors are returned as float64 and L2-normalized defensively —
    the package's cosine paths (engine, retrieval, firewall) assume
    unit length regardless of what the API guarantees.
    """

    def __init__(
        self,
        model: str = "voyage-3.5",
        *,
        api_key: str | None = None,
        output_dimension: int | None = None,
        input_type: str | None = None,
        batch_size: int = 128,
        transport: Transport | None = None,
    ) -> None:
        self.model_name = model
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Voyage API key missing — pass api_key= or set VOYAGE_API_KEY"
            )
        self._output_dimension = output_dimension
        self._input_type = input_type
        self._batch_size = max(1, batch_size)
        self._transport: Transport = transport or _urllib_transport
        self._dim: int | None = output_dimension

    # ------------------------------------------------------------------
    # embedder protocol

    def get_dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self._embed_batch(["dimension probe"]).shape[1])
        return self._dim

    def get_embedding(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        chunks = [
            self._embed_batch(texts[i : i + self._batch_size])
            for i in range(0, len(texts), self._batch_size)
        ]
        return np.vstack(chunks)

    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        payload: dict[str, Any] = {"model": self.model_name, "input": texts}
        if self._output_dimension is not None:
            payload["output_dimension"] = self._output_dimension
        if self._input_type is not None:
            payload["input_type"] = self._input_type
        status, body = self._transport(
            _VOYAGE_URL, {"Authorization": f"Bearer {self._api_key}"}, payload
        )
        if status != 200:
            raise RuntimeError(
                f"Voyage embeddings request failed with HTTP {status}: "
                f"{str(body)[:200]}"
            )
        rows = sorted(body["data"], key=lambda item: item["index"])
        matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # zero vectors stay zero, no division error
        return matrix / norms


# ----------------------------------------------------------------------
# tenant embedder-fingerprint guard


class EmbedderMismatch(RuntimeError):
    """The embedder does not match the tenant's registered identity."""


def embedder_fingerprint(embedder: Any) -> str:
    """Identity of an embedder: ``<model_name or class name>/<dimension>``.

    A model switch or finetune shifts the embedding space even at the
    same dimension — the fingerprint makes that DETECTED instead of a
    silent similarity degradation.
    """
    name = getattr(embedder, "model_name", None) or type(embedder).__name__
    return f"{name}/{embedder.get_dimension()}"


def ensure_tenant_embedder(db: Any, scope: TenantScope, embedder: Any) -> str:
    """Register or verify the tenant's embedder identity (fail-closed).

    First use records ``{fingerprint, model, dimension}`` on the tenant
    document; every later use must match exactly or
    :class:`EmbedderMismatch` is raised. Returns the fingerprint.
    """
    fingerprint = embedder_fingerprint(embedder)
    tenant_filter = TenantScope(tenant=scope.tenant).filter()
    doc = db.tenants.find_one(tenant_filter) or {}
    registered = (doc.get("embedder") or {}).get("fingerprint")
    if registered is not None and registered != fingerprint:
        raise EmbedderMismatch(
            f"tenant {scope.tenant!r} is registered for embedder "
            f"{registered!r} but got {fingerprint!r} — refusing to mix "
            "embedding spaces (re-embed and re-register to migrate)"
        )
    if registered is None:
        db.tenants.update_one(
            tenant_filter,
            {
                "$set": {
                    "embedder": {
                        "fingerprint": fingerprint,
                        "model": getattr(
                            embedder, "model_name", type(embedder).__name__
                        ),
                        "dimension": embedder.get_dimension(),
                    }
                }
            },
            upsert=True,
        )
    return fingerprint
