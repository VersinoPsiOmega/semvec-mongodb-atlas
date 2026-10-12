"""semvec-atlas — MongoDB Atlas persistence backend for the Semvec engine.

Layering (guardrails L1/L3 of the specification):

* The **engine** — state update math, retention tiers, eviction,
  dedup — lives in the ``semvec`` wheel and is never re-implemented
  here.
* This package persists engine *results* (snapshots, memory items,
  compliance events, deletion certificates) and provides retrieval
  (Atlas ``$rankFusion`` with a client-side RRF fallback), the
  LangGraph ``BaseStore`` adapter, and the Art. 17 forget path.
"""

from __future__ import annotations

__version__ = "0.1.0"
