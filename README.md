```text
   _____ ________  ____    ______________     ___  ________    ___   _____
  / ___// ____/  |/  / |  / / ____/ ____/    /   |/_  __/ /   /   | / ___/
  \__ \/ __/ / /|_/ /| | / / __/ / /        / /| | / / / /   / /| | \__ \
 ___/ / /___/ /  / / | |/ / /___/ /___     / ___ |/ / / /___/ ___ |___/ /
/____/_____/_/  /_/  |___/_____/\____/    /_/  |_/_/ /_____/_/  |_/____/

        d u r a b l e   ·   g u a r d e d   ·   p r o v a b l e
                     agent memory on MongoDB Atlas

  agent turn ──▶ semantic firewall ──▶ engine ──▶ ⟨event → items → snapshot⟩ ──▶ Atlas
```

<p align="center">
  <img alt="tests" src="https://img.shields.io/badge/tests-297%20passing-brightgreen">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="mongodb" src="https://img.shields.io/badge/MongoDB%20Atlas-8.3%20%C2%B7%20%24rankFusion-47A248?logo=mongodb&logoColor=white">
  <img alt="engine" src="https://img.shields.io/badge/engine-semvec%20%E2%89%A5%200.8.8-6E4AFF">
  <img alt="voyage" src="https://img.shields.io/badge/embeddings-Voyage%20AI%20ready-0AA8A7">
  <img alt="gdpr" src="https://img.shields.io/badge/GDPR-Art.%2017%20certified%20forget-005EA8">
  <img alt="status" src="https://img.shields.io/badge/status-alpha-orange">
  <img alt="license" src="https://img.shields.io/badge/license-proprietary-lightgrey">
</p>

**Semvec Atlas Edition** — a durable, multi-tenant MongoDB Atlas
persistence and retrieval backend for the
[Semvec](https://semvec-docs.pages.dev) memory engine.

> Status: alpha. Verified by 297 tests against the public
> API of the `semvec` PyPI wheel (`>=0.8.8,<0.9`) and a local
> `mongodb/mongodb-atlas-local` deployment (MongoDB 8.3.4 — Atlas
> Search, Vector Search, transactions, native `$rankFusion`).

The engine (state math, retention tiers, eviction, dedup) stays in the
`semvec` wheel — this package is pure orchestration:

* **Snapshots** — engine `to_bytes()` blobs, CAS-versioned, integrity
  checked twice (SHA-256 + the engine's own embedded checksum).
* **Event store** — `MongoEventStore` implementing the wheel's
  `EventStore` ABC: monotonic per-(tenant, agent) sequence, gap
  detection, append-only surface.
* **Transactional turn commit** — event → item mutations → snapshot in
  one multi-document transaction (`TurnCommitter`).
* **Recovery** — full event replay (checksum-identical, the
  determinism gate) or snapshot + tail (fast path).
* **Hybrid retrieval** — server-side `$rankFusion`
  (semantic + lexical + recency, per-tenant weights) with automatic
  client-side RRF fallback via the wheel's own `rrf_fuse`.
* **LangGraph adapter** — `SemvecStore(BaseStore)`, CRUD-parity with
  the native `MongoDBStore` (enforced by a shared test suite);
  migration is one line.
* **GDPR Art. 17** — the wheel's signed deletion certificates (RS256)
  plus Atlas hard-deletes, a content-free forget-marker event, and a
  re-verified audit chain. MCP tools: `memory_search`,
  `memory_forget`, `memory_audit`.
* **Semantic firewall** — opt-in ingest gate against memory
  poisoning: exclusion-anchor cosine scoring with leetspeak/homoglyph
  de-obfuscation, quarantine + audited release instead of silent
  drops, tenant-wide inoculation (one agent's block immunizes its
  siblings, each learning event auditable via its own marker event),
  Art.-17-integrated. MCP tools: `memory_quarantine_list`,
  `memory_quarantine_release`, `memory_firewall_status`.
* **Tenant guard** — every query path is scope-bound by construction;
  enforced by tests, not convention.
* **Voyage AI embeddings** — first-class `VoyageEmbedder`
  (MongoDB's embedding models) with a fail-closed per-tenant
  fingerprint guard; any embedder with
  `get_embedding`/`get_dimension` plugs in identically.

## Quickstart

```bash
docker run -d --name semvec-atlas-mongo -p 27017:27017 \
    mongodb/mongodb-atlas-local:latest

python -m venv .venv && . .venv/bin/activate
pip install -e ".[langgraph,compliance,mcp,dev]"
export SEMVEC_LICENSE_KEY=...        # or SEMVEC_KEY in the repo .env
```

```python
import pymongo
from semvec import SemvecConfig, SemvecState
from semvec_atlas.commit import TurnCommitter
from semvec_atlas.guard import TenantScope
from semvec_atlas.indexes import provision
from semvec_atlas.retrieval import RetrievalEngine

client = pymongo.MongoClient("mongodb://localhost:27017/?directConnection=true")
db = client["agents"]
provision(db, dimension=my_embedder.get_dimension())

scope = TenantScope(tenant="acme", agent="advisor-01")
committer = TurnCommitter(client, db, scope, embedder=my_embedder, snapshot_every=10)

state = SemvecState(SemvecConfig(dimension=my_embedder.get_dimension()))
committer.run_turn(state, session="s1", turn=1, text="customer prefers index funds")

hits = RetrievalEngine(db, scope, embedder=my_embedder).search("what does the customer prefer?")
state, info = committer.recover(session="s1")   # crash? full-replay recovery
```

LangGraph migration (one line):

```python
from semvec_atlas.store import SemvecStore
store = SemvecStore(db, scope=scope, embedder=my_embedder)  # BaseStore-compatible
```

## Demo

A six-act financial-advisory walkthrough (onboarding → semantic
firewall: injection + leetspeak blocked, tenant-wide inoculation,
audited false-positive release → crash+resume → verbatim recall →
Art. 17 forget with verified certificate → audit):

```bash
demo/run.sh                            # picks the right venv, starts MongoDB, uses the LLM from the repo .env
demo/run.sh --no-llm                   # without LLM calls
demo/run.sh --embedder voyage          # Voyage AI embeddings (needs VOYAGE_API_KEY)
```

`run.sh` resolves the virtualenv automatically (`$SEMVEC_ATLAS_VENV`
override, else the newest `~/.venvs/semvec-atlas*` with the package
installed) — the system Python won't work (PEP 668, and the demo needs
`pymongo`/`sentence-transformers`/`openai`).

## Tests

```bash
python -m pytest tests/ -q -m "not slow"   # 279 tests, ~70 s
python -m pytest tests/ -q                 # + 18 search-index tests
```

## LOCOMO benchmark

```bash
python benchmarks/run_locomo_atlas.py \
    --data <locomo10.json> --configs atlas,inprocess,fullcontext
```

Configurations: (a) Atlas Edition, (d) in-process parity reference —
identical engine, embedder, prompts (mem0 reader/judge verbatim), fixed
inputs; the run aborts on any engine-checksum mismatch between (a) and
(d). See `docs/BENCHMARK_REPORT.md` for results and for why benchmarks
run with tier capacities sized above the corpus.

## Documentation

* **`docs/PRODUCT_OVERVIEW.md` — product & architecture overview**
  (mermaid diagrams: layering, write path, firewall, recovery,
  retrieval, compliance; benchmark evidence) — start here
* `docs/ARCHITECTURE.md` — layering, write path, recovery, retrieval
* `docs/BENCHMARK_REPORT.md` — LOCOMO results (generated per run)
