# semvec-atlas — Architecture

Engineering architecture of the *Semvec Atlas Edition*, built
test-first against the **public API of the `semvec` PyPI wheel**
(`>=0.8.8`) and a local `mongodb/mongodb-atlas-local`
deployment (MongoDB 8.3.4 — Atlas Search, Vector Search, transactions,
native `$rankFusion`). For the product-level view with diagrams, see
[`PRODUCT_OVERVIEW.md`](PRODUCT_OVERVIEW.md).

## Layering (guardrails L1/L3)

```
Agent frameworks: LangGraph (SemvecStore) · MCP (memory_* tools)
────────────────────────────────────────────────────────────────
semvec wheel (unchanged): SemvecState.update · tiers · literal cache
compliance pack (EventStore ABC · replay · forget_user · RS256 certs)
────────────────────────────────────────────────────────────────
semvec-atlas (this package, pure orchestration):
  guard.py        TenantScope — mandatory tenant filter builder
  client_metadata.py  announce() — "semvec-atlas" + version in the
                  handshake of the caller's client (every entry point)
  ledger.py       event/item ledgers — memory turns vs markers,
                  engine-synced vs BaseStore items (scoped counts)
  embedders.py    VoyageEmbedder · embedder-fingerprint guard
                  (fail-closed per tenant)
  indexes.py      idempotent collections/validators/search indexes
  persistence.py  AtlasPersistence — to_bytes snapshots, CAS, retention
  event_store.py  MongoEventStore(EventStore) — seq, gaps, append-only
  item_sync.py    item mirror — mutation feed + safety-net tier diff
  commit.py       TurnCommitter — event→items→snapshot in one txn;
                  recovery (full replay | snapshot+tail)
  retrieval.py    RetrievalEngine — $rankFusion | rrf_fuse fallback
  store.py        SemvecStore(BaseStore) — LangGraph adapter
  firewall.py     SemanticFirewall — ingest gate (anchor cosine,
                  de-leet/homoglyph fold) · QuarantineVault · AnchorStore
  compliance.py   AtlasForget — Art. 17 orchestration + audit chain
                  (incl. firewall surfaces)
  mcp.py          FastMCP tools memory_search/forget/audit/
                  quarantine_list/quarantine_release/firewall_status
────────────────────────────────────────────────────────────────
MongoDB Atlas: state_snapshots · memory_items · literal_cache ·
compliance_events (validator, append-only by role) ·
deletion_certificates · tenants · quarantined_inputs ·
firewall_anchors · ix_vec/ix_lex/ix_ns/ix_evt/ix_snap/ix_qi/ix_fwa
```

No engine logic is re-implemented here. The engine decides *what* is
remembered; this package decides *where it lives* and *how it is
found, audited, and forgotten*.

## Write path (one agent turn)

0. Optional ingest gate (`firewall=` on the committer): the turn's
   embedding is scored against the tenant's exclusion anchors; inputs
   that passed raw are re-checked in de-obfuscated form (leetspeak /
   homoglyph fold). A held input reaches **neither** the engine nor
   the event log — it lands in `quarantined_inputs` (full text, for
   release) plus a content-free marker event (16-char preview +
   SHA-256), and with `inoculate=True` its embedding becomes a
   tenant-shared anchor so sibling agents are immune on first sight —
   recorded by its own content-free inoculation marker event
   (`anchor_id` + `text_hash`), so the learning moment is auditable.
   Replay determinism is therefore unaffected by construction (marker
   events are excluded from replay, like forget markers).
1. `state.update(embedding, text)` — in-process, deterministic, no LLM.
2. One MongoDB transaction (single-node replica set / Atlas):
   `MemoryEvent` append (monotonic per-scope `seq`) → tier-diff item
   upserts → every Nth turn a `to_bytes` snapshot recording
   `last_event_seq`.
3. Zero-norm inputs follow the wheel's None-means-skip contract and
   persist nothing.

Replay fidelity rule: the committer calls `update()` **without**
per-memory meta, because the wheel's replay folds `(embedding,
content)` pairs only. Session/turn tagging lives in document fields
(`ns`, `turn_added`), not in engine memory.

## Recovery

* `recover(mode="replay")` (default): fresh state + full event-log
  replay through the wheel's `EventReplayService`. **Checksum-identical
  to the live state** — the determinism gate. Marker events (forget,
  quarantine, inoculation) are excluded (content-free bookkeeping).
* `recover(mode="snapshot_tail")`: newest verified snapshot + tail
  replay from `last_event_seq`. With the engine's full-precision codec
  this fast path is
  checksum-identical to the live state as well; it requires the
  snapshot codec to be readable by the running engine
  (`SnapshotMeta.codec_version` — newer codecs are not readable by
  older wheels, so mixed fleets upgrade fully before writing new
  snapshots, or recover via full replay, which is version-independent).
* A gapped sequence (tampering / out-of-band deletion) refuses
  recovery.

### Item sync: mutation feed + tier diff

The engine reports per-turn item mutations
(`update(include_mutations=True)`, kinds
`added`/`promoted`/`demoted`/`merged` + `sources`; verified
mirror-grade and trajectory-neutral). The committer feeds
this into `MemoryItemSync`: quiet turns skip the diff (fast path),
`merged` events stamp `superseded_by` lineage on source documents, and
a periodic safety-net diff (default every 25 turns) guards
consistency. Engine maintenance is deterministic and
observation-invariant at default capacities.

## Retrieval

Three sub-pipelines, each carrying the tenant guard plus
`deleted:false, evicted:false`:

* semantic — `$vectorSearch` over client-side embeddings (L2),
* lexical — `$search` on `summary` (lucene.german + standard multi),
* recency — `$sort access.last` (never engine scores — L3).

Fusion: server-side `$rankFusion` (generally available; needs MongoDB
8.1+ alongside `$vectorSearch`) is detected at startup and runs with
per-tenant weights, default 0.5/0.3/0.2, `scoreDetails`. Automatic
degradation to the wheel's `rrf_fuse` at identical weights covers a
server below that floor and a capability that disappears mid-flight.
Both paths are covered by the same test suite and agree on clear-cut
top hits.

## Multi-tenancy

`TenantScope` is the only filter factory; protected keys cannot be
overridden, even to the same value. Event streams are per
`(tenant, agent)`; `user_id = "tenant/agent"` maps the wheel's
single-field contract. Enforced by tests, not convention.

## Compliance

* Append-only: schema validator enforces insert shape; the store
  exposes no update path; Atlas role separation covers raw access.
* Forget: the wheel's `forget_user` (event wipe + RS256
  `DeletionCertificate`) plus Atlas hard-deletes of items, literals and
  snapshots, certificate persistence (ISO timestamps for byte-exact
  re-verification), and a content-free forget-marker event.
* Audit: the certificate chain, each entry re-verified with the public
  key. Honest limit: the proof covers the store, not past LLM outputs.
* Forget covers every store the scope touches: memory items (including
  tombstones), literal cache, snapshots, the quarantine vault, and the
  scope's learned firewall anchors.

## Scope boundaries (this release)

| Area | Status |
|---|---|
| TTL safety net | Collection-level TTL index shipped (`ttl_short_seconds`); per-item TTL parity with the native LangGraph store is on the roadmap |
| Queryable Encryption | Atlas Enterprise feature — outside the local verification target; the data model does not preclude it |
| OpenTelemetry | Timings are measured and reported by the benchmark harness and demo; OTel exporters are on the roadmap |
| JS/TS adapter | Python-first release; a JS/TS `BaseStore` sibling is on the roadmap |
