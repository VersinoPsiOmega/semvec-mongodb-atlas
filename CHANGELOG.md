# Changelog

## 0.1.0

First release of the **Semvec Atlas Edition**: a durable, multi-tenant
MongoDB Atlas persistence and retrieval backend for the Semvec memory
engine, built test-first (344 tests) against the public API of the
`semvec` PyPI wheel (`>=0.8.8`) and `mongodb/mongodb-atlas-local`
(MongoDB 8.3 — Atlas Search, Vector Search, transactions, native
`$rankFusion`).

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
- **Session store** — `MongoStateStore` subclasses the engine's CAS
  store (scoped CAS, SHA-256 blob integrity on load, atomic
  `save(events=…)` turn commit) and binds it to an existing database
  handle; the tenant-scoped `MongoEventStore` plugs into the event-writer
  seam so audit rows keep the Atlas layout.
- **Business records in the turn** — `DomainWrite` commits records
  (`claims`, `policies`) inside the same transaction as the memory
  event; a failing domain write rolls the turn back whole. Tenant-wide
  scope, collection allowlist and a scalar-key rule guard the write.
- **Coordination layer persisted** — `AtlasCortexStore` records
  participation, observer readings as an append-only series and one
  immutable document per decision; `case_timeline()` answers who
  converged and how it was decided in one round trip.

### Retrieval & integration

- **Hybrid retrieval** — server-side `$rankFusion` (semantic + lexical
  + recency, per-tenant weights) with automatic client-side RRF fallback
  via the wheel's own `rrf_fuse`; `DomainJoin` returns ranked hits
  already joined to the record they concern, tenant equality enforced
  inside the lookup pipeline.
- **LangGraph adapter** — `SemvecStore(BaseStore)`, CRUD-parity with the
  native `MongoDBStore`, enforced by a shared test suite.
- **Voyage AI embeddings** — first-class `VoyageEmbedder` with batching
  and a fail-closed per-tenant embedder-fingerprint guard; any embedder
  with `get_embedding`/`get_dimension` plugs in identically.
- **Deployment probe** — `python -m semvec_atlas.deployment <uri>`
  reports transactions, Search indexes and `$rankFusion`, separating a
  hard requirement from an optimisation with a fallback.
- **Client metadata** — every public entry point announces
  `semvec-atlas` and its version in the MongoDB connection handshake of
  the caller's client (`MongoClient.append_metadata`); only name and
  version are sent.

### Semantic firewall

- Opt-in ingest gate against memory poisoning: exclusion-anchor cosine
  scoring with leetspeak/homoglyph de-obfuscation, quarantine + audited
  release instead of silent drops, tenant-wide inoculation with a
  content-free marker event per learning moment, Art.-17-integrated.
- **Calibrated threshold registry** — `threshold_for()` serves only
  measured profiles and refuses an unknown embedder;
  `calibrate_threshold()` reproduces the shipped profile and raises on
  an inseparable corpus.
- MCP tools: `memory_quarantine_list`, `memory_quarantine_release`,
  `memory_firewall_status`.

### Compliance

- **GDPR Art. 17** — signed deletion certificates (RS256) plus Atlas
  hard-deletes across memory items, literal cache, snapshots, the
  quarantine vault and learned anchors; a content-free forget-marker
  event and a re-verified audit chain. MCP tools: `memory_search`,
  `memory_forget`, `memory_audit`.
- **Tenant guard** — every query path is scope-bound by construction,
  enforced by tests with real foreign-tenant writes.
- **Ledgers** — `event_ledger` and `item_ledger` reconcile a scope's
  counts (memory turns vs. markers, engine-synced vs. BaseStore items);
  the demo prints and asserts the split.

### Evidence & demos

- LOCOMO benchmark harness with an in-process parity reference; the run
  aborts on any engine-checksum mismatch (`docs/BENCHMARK_REPORT.md`).
- Six-act financial-advisory demo (`demo/run.sh`) and the claims
  scenario (`demo/claims_scenario.py`).
