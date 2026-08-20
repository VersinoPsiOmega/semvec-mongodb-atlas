# Changelog

## 0.1.0a5

First public evaluation release of the **Semvec Atlas Edition**: a
durable, multi-tenant MongoDB Atlas persistence and retrieval backend
for the Semvec memory engine, built against the public API of the
`semvec` PyPI wheel (`>=0.8.8,<0.9`) and
`mongodb/mongodb-atlas-local` (MongoDB 8.3). Verified by 253 tests.

### Persistence & recovery

- **Transactional turn commit** (`TurnCommitter`) — memory event
  (monotonic per-scope sequence, gap detection), item mutations, and
  periodic engine snapshots (`to_bytes` blobs, CAS-versioned,
  integrity-checked twice) in one multi-document transaction.
- **Checksum-identical recovery** — full event replay (the determinism
  gate) or snapshot + tail replay; gapped logs refuse recovery.
- **Mutation-feed item sync** — the engine's per-turn mutation feed
  drives the item mirror; quiet turns skip the tier diff, `merged`
  events stamp `superseded_by` lineage, a periodic full diff guards
  consistency.

### Retrieval & integration

- **Voyage AI embedding provider** — `embedders.VoyageEmbedder`
  (`voyage-3.5` family, batching, defensive L2 normalization, zero
  extra dependencies) plus a **fail-closed tenant embedder-fingerprint
  guard**: committers and retrieval engines verify the registered
  model/dimension at construction, so a silent embedder switch cannot
  degrade similarity search. `TurnCommitResult.txn_ms` reports the
  transaction-only commit latency; `search(query_embedding=…)` lets
  callers time or supply query embeddings separately.
- **Hybrid retrieval** — server-side `$rankFusion`
  (semantic + lexical + recency, per-tenant weights) with an automatic
  client-side RRF fallback of identical contract.
- **LangGraph adapter** — `SemvecStore(BaseStore)` with CRUD parity to
  the native `MongoDBStore`, enforced by a shared test suite.
- **MCP tools** — `memory_search`, `memory_forget`, `memory_audit`,
  `memory_quarantine_list`, `memory_quarantine_release`,
  `memory_firewall_status`; each a transport-free, unit-tested handler.
- **`MongoStateStore`** — CAS session store for `semvec serve`: a thin
  subclass of the engine's `semvec.api.mongo_state_store` (scoped CAS,
  SHA-256 blob integrity on load, atomic `save(events=…)` turn commit)
  adding a `Database`-handle constructor, the `drift_exempt` pair, and
  an optional wired tenant-scoped event store so audit rows keep the
  Atlas layout.

### Semantic firewall

- **Ingest gate against memory poisoning** — exclusion-anchor cosine
  scoring with leetspeak/homoglyph/NFKD de-obfuscation re-check and
  `open`/`filter`/`quarantine`/`lockdown` levels; blocked turns reach
  neither the engine nor the event log (content-free marker events
  only — replay determinism holds by construction).
- **Quarantine vault + audited release** — held inputs keep full text
  for release; every listing/audit surface is content-free (16-char
  preview + SHA-256). A release un-learns the anchor learned from that
  input, tenant-wide.
- **Tenant-wide inoculation with a live anchor view** — one agent's
  block immunizes every agent of the tenant immediately; each learning
  event appends its own audit marker.

### Compliance

- **GDPR Art. 17** — the engine's signed deletion certificates (RS256)
  plus Atlas hard-deletes across all stores (items, literal cache,
  snapshots, quarantine vault, learned anchors), a content-free
  forget-marker event, and a re-verified audit chain.
- **Tenant guard** — every query path is scope-bound by construction
  (`TenantScope`, sharing the engine's `OwnerScope` validation and
  encoding); enforced by tests with real foreign-tenant writes, not
  convention.

### Evidence & docs

- **LOCOMO benchmark harness** — Atlas vs. in-process parity
  configuration; engine-checksum parity 10/10 conversations, 96.4 %
  token reduction vs. full context (see `docs/BENCHMARK_REPORT.md`).
- **Six-act demo** (`demo/run.sh`) — onboarding → semantic firewall →
  crash/resume → verbatim recall → Art. 17 forget → audit, against
  atlas-local and a live LLM; closes with measured numbers (Atlas
  transaction, `$rankFusion` query, and embedding latency reported
  separately) next to the LOCOMO benchmark evidence. `--embedder
  voyage` runs the same demo on Voyage AI embeddings.
- **Documentation** — product overview with architecture diagrams
  (`docs/PRODUCT_OVERVIEW.md`), engineering architecture
  (`docs/ARCHITECTURE.md`), benchmark report
  (`docs/BENCHMARK_REPORT.md`).
