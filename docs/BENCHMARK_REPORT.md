# LOCOMO benchmark report — Semvec Atlas Edition

Run `31d983e3`, 2026-07-30T21:46:19.850766+00:00 → 2026-07-30T22:08:51.721202+00:00  
Reader/judge model: `Qwen/Qwen3.6-27B` (self-hosted, thinking disabled) · Embedder: `sentence-transformers/all-mpnet-base-v2 (cpu)` · k=10  
Backend: mongodb-atlas-local (MongoDB 8.3.4, $rankFusion native)

## Configurations

* **(a) atlas** — semvec engine + semvec-atlas: transactional turn
  commits (event → items → snapshot) into MongoDB, QA context via
  hybrid `$rankFusion` retrieval.
* **(d) inprocess** — parity reference: identical engine, embedder,
  prompts; in-process `get_relevant_memories`, no database.
* **(c) fullcontext** — token baseline (whole conversation as context).

## Determinism / parity gate

Engine checksums (a) == (d) held for **10/10** conversations — durable persistence did not alter the engine
trajectory. Gate aborts the run on first mismatch.

## Results (categories 1–4, adversarial excluded)

| Metric | (a) atlas | (d) inprocess |
|---|---|---|
| QAs judged | 1540 | 1539 |
| **J (LLM-judge accuracy)** | **0.5773** | **0.5322** |
| J cat 1 (multi-hop) | 0.4007 | 0.4057 |
| J cat 2 (temporal) | 0.4984 | 0.405 |
| J cat 3 (open-domain) | 0.4375 | 0.3958 |
| J cat 4 (single-hop) | 0.6825 | 0.6385 |
| recall@k (approx.) | 0.6917 | 0.6381 |
| reader input tokens (mean) | 987.1 | 1001.9 |
| token reduction vs full context | 0.9639 | 0.9634 |
| retrieval ms p50 / p95 | 187.4 / 258.9 | 165.2 / 230.9 |

Full-context reader prompt: mean **27352.9** tokens over 1540 QAs.

## Ingest

* 5882 dialogue turns over 10 conversations
* (a) atlas ingest wall-clock: **377.1s** (incl. transactions, item sync, snapshots; commit p50 ≈ 40.1 ms, worst conv p95 ≈ 85.2 ms)
* (d) in-process ingest wall-clock: **3.9s**
* durable-persistence overhead: **96.0×** in-process

## Methodology & honest caveats

* **reader_prompt**: mem0 ANSWER_PROMPT verbatim
* **judge_prompt**: mem0 LLM-judge prompt verbatim
* **scored_categories**: [1, 2, 3, 4]
* **excluded**: category 5 (adversarial), per published semvec methodology
* **recall_at_k**: approximate — evidence-turn 60-char prefix containment in context
* **note**: Absolute J is NOT comparable to the published semvec LOCOMO pipeline (prompt routing, BM25 hybrid, entity blocks not replicated). The claim under test is (a) vs (d) parity at identical conditions, plus token/latency deltas.
* **fullcontext**: token counts on all QAs; LLM on sample of 150
* **engine_config**: tier capacities 1000/1000/1000 (eviction disabled — capacity >= corpus). Parity checksums were recorded in the eviction-free regime.

## Reproducibility

* per-conversation engine checksums: see `parity` in the summary JSON
* raw per-QA records: `locomo_31d983e3_atlas.jsonl` / `_inprocess.jsonl`
* summary: `locomo_atlas_31d983e3_summary.json`
* index config: `semvec_atlas.indexes.provision` (ix_vec cosine, ix_lex german+standard)
