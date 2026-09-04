# Changelog

## 0.1.0a6

Business records beside agent memory, the coordination layer persisted,
and two pieces of operational honesty: a threshold registry that refuses
an unmeasured embedder, and a probe that reports what a deployment can
actually do. Verified by 297 tests.

### Domain records in the turn transaction

- **`DomainWrite`** — a turn can carry business records (`claims`,
  `policies`) that commit inside the *same* multi-document transaction as
  the memory event, the item mutations and the snapshot. A failing domain
  write rolls the turn back whole, so a record can never exist without
  the memory of the turn that produced it.
- **Scoped tenant-wide, not per agent** — several specialist agents work
  the same case, so binding a record to whichever agent opened it would
  hide it from the rest.
- **Guarded twice**: the writable collections are an allowlist (the
  audit log has exactly one writer), and a write key must be a scalar
  identifier — a query operator there would match and overwrite another
  record of the same tenant.
- **`DomainJoin`** — retrieval returns ranked memory hits already joined
  to the record they concern: a `$lookup` inside the same aggregation on
  the server-side path, one batch query on the client-side fallback. The
  tenant equality lives inside the lookup pipeline, so a foreign record
  with a colliding business key is never a candidate.

### Coordination layer as queryable data

- **`AtlasCortexStore`** mirrors the engine's multi-agent cortex into
  three collections: participation per (case, agent), the observer's
  coherence/resonance readings as an append-only series, and one document
  per decision with its rationale, votes and outcome.
- **`case_timeline()`** answers "which agents worked this case, when did
  they converge, and how was it decided" in one round trip.
- A recorded decision cannot be rewritten under its own id
  (`ProposalConflict`), and the store satisfies the engine's persistence
  seam, so a `SemvecCortexService` can be handed an Atlas-backed store.

### Operational honesty

- **Calibrated threshold registry** — a firewall threshold is a property
  of the embedding model, so `threshold_for()` serves only measured
  profiles and refuses an unknown model rather than lending it another
  model's number. `calibrate_threshold()` reproduces the shipped profile
  and raises when a corpus is not separable.
- **Deployment probe** — `python -m semvec_atlas.deployment <uri>`
  reports server version, topology, transaction support, Search Index
  availability and `$rankFusion`, separating a hard requirement
  (transactions) from an optimisation with a working fallback.
- **`$rankFusion` is documented as generally available**, and the
  client-side RRF fallback is explained by the version floor it covers
  rather than by a preview that might be withdrawn.

### Demo and documentation

- **`demo/claims_scenario.py`** — a claims platform serving several
  insurers, four specialist agents on one motor claim. Runs in seconds
  against the local container with no licence key, no LLM and no API
  calls; `demo/run.sh --claims` launches it.
- **`docs/blog/agent-memory-on-atlas-draft.md`** — a full technical
  article built on this release, with six diagrams and every figure
  measured on the reference implementation.

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
