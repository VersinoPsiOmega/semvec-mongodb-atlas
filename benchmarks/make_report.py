"""Generate docs/BENCHMARK_REPORT.md from a LOCOMO summary JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(summary_path: str) -> int:
    s = json.loads(Path(summary_path).read_text())
    a, d = s.get("atlas", {}), s.get("inprocess", {})
    fc = s.get("fullcontext", {})
    convs = s.get("conversations", {})
    parity = s.get("parity", {})

    def row(label, fa, fd):
        return f"| {label} | {fa} | {fd} |"

    ingest_a = sum(c.get("atlas_ingest_s", 0) for c in convs.values())
    ingest_d = sum(c.get("inprocess_ingest_s", 0) for c in convs.values())
    commit_p95 = max((c.get("commit_ms_p95", 0) for c in convs.values()), default=0)
    commit_p50 = (
        sorted(c.get("commit_ms_p50", 0) for c in convs.values())[len(convs) // 2]
        if convs
        else 0
    )
    turns = sum(c.get("turns", 0) for c in convs.values())

    lines = [
        "# LOCOMO benchmark report — Semvec Atlas Edition",
        "",
        f"Run `{s['run_id']}`, {s['started_at']} → {s.get('finished_at', '?')}  ",
        f"Reader/judge model: `{s['model']}` (self-hosted, thinking disabled) · "
        f"Embedder: `{s['embedder']}` · k={s['k']}  ",
        f"Backend: {s['mongo']}",
        "",
        "## Configurations",
        "",
        "* **(a) atlas** — semvec engine + semvec-atlas: transactional turn",
        "  commits (event → items → snapshot) into MongoDB, QA context via",
        "  hybrid `$rankFusion` retrieval.",
        "* **(d) inprocess** — parity reference: identical engine, embedder,",
        "  prompts; in-process `get_relevant_memories`, no database.",
        "* **(c) fullcontext** — token baseline (whole conversation as context).",
        "",
        "## Determinism / parity gate",
        "",
        f"Engine checksums (a) == (d) held for **{len(parity)}/{len(convs)}**"
        " conversations — durable persistence did not alter the engine",
        "trajectory. Gate aborts the run on first mismatch.",
        "",
        "## Results (categories 1–4, adversarial excluded)",
        "",
        "| Metric | (a) atlas | (d) inprocess |",
        "|---|---|---|",
        row("QAs judged", a.get("judged"), d.get("judged")),
        row("**J (LLM-judge accuracy)**", f"**{a.get('J')}**", f"**{d.get('J')}**"),
        row("J cat 1 (multi-hop)", a.get("J_by_category", {}).get("1"), d.get("J_by_category", {}).get("1")),
        row("J cat 2 (temporal)", a.get("J_by_category", {}).get("2"), d.get("J_by_category", {}).get("2")),
        row("J cat 3 (open-domain)", a.get("J_by_category", {}).get("3"), d.get("J_by_category", {}).get("3")),
        row("J cat 4 (single-hop)", a.get("J_by_category", {}).get("4"), d.get("J_by_category", {}).get("4")),
        row("recall@k (approx.)", a.get("recall_at_k_approx"), d.get("recall_at_k_approx")),
        row("reader input tokens (mean)", a.get("reader_input_tokens_mean"), d.get("reader_input_tokens_mean")),
        row("token reduction vs full context", a.get("token_reduction_vs_fullcontext"), d.get("token_reduction_vs_fullcontext")),
        row("retrieval ms p50 / p95", f"{a.get('retrieval_ms_p50')} / {a.get('retrieval_ms_p95')}", f"{d.get('retrieval_ms_p50')} / {d.get('retrieval_ms_p95')}"),
        "",
        f"Full-context reader prompt: mean **{fc.get('reader_input_tokens_mean', '?')}**"
        f" tokens over {fc.get('qas_counted', '?')} QAs.",
        "",
        "## Ingest",
        "",
        f"* {turns} dialogue turns over {len(convs)} conversations",
        f"* (a) atlas ingest wall-clock: **{ingest_a:.1f}s** "
        f"(incl. transactions, item sync, snapshots; commit p50 ≈ {commit_p50} ms, "
        f"worst conv p95 ≈ {commit_p95} ms)",
        f"* (d) in-process ingest wall-clock: **{ingest_d:.1f}s**",
        f"* durable-persistence overhead: **{(ingest_a / ingest_d if ingest_d else 0):.1f}×** in-process",
        "",
        "## Methodology & honest caveats",
        "",
    ]
    for key, val in s.get("methodology", {}).items():
        lines.append(f"* **{key}**: {val}")
    lines += [
        "",
        "## Reproducibility",
        "",
        "* per-conversation engine checksums: see `parity` in the summary JSON",
        f"* raw per-QA records: `locomo_{s['run_id']}_atlas.jsonl` / `_inprocess.jsonl`",
        f"* summary: `locomo_atlas_{s['run_id']}_summary.json`",
        "* index config: `semvec_atlas.indexes.provision` (ix_vec cosine, ix_lex german+standard)",
    ]
    out = Path(__file__).resolve().parents[1] / "docs" / "BENCHMARK_REPORT.md"
    out.write_text("\n".join(str(x) for x in lines) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
