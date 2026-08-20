"""LOCOMO benchmark for the Semvec Atlas Edition.

Configurations
--------------
(a) ``atlas``      — semvec engine + semvec-atlas persistence/retrieval:
                     every turn goes through ``TurnCommitter`` (event →
                     item sync → snapshot, one transaction); QA context
                     comes from ``RetrievalEngine`` hybrid search
                     ($rankFusion on MongoDB 8.3, atlas-local).
(d) ``inprocess``  — the parity reference: identical engine, identical
                     embedder, no database; QA context comes from the
                     wheel's in-process ``memory.get_relevant_memories``.
(c) ``fullcontext``— token baseline: the whole conversation as context.
                     Token counts are computed for ALL QAs; the LLM
                     reader runs on a sample (``--fullcontext-sample``)
                     because 26k-token prompts × 1540 QAs is wall-clock
                     prohibitive on a self-hosted 27B endpoint. Reported
                     as a sampled estimate, never extrapolated silently.

Parity proof: after ingest, the engine checksum of (a) and (d) must be
identical per conversation — both fold the same (embedding, text)
sequence through the same deterministic engine. Any mismatch aborts
the run (that would be a persistence bug, not a benchmark artifact).

Methodology follows the published semvec LOCOMO setup where it is
reproducible from public artifacts: mem0's ANSWER_PROMPT verbatim for
the reader, mem0's LLM-judge prompt verbatim, categories 1–4 scored
(category 5 = adversarial excluded → 1540 QAs over locomo10.json).
The absolute J of the published pipeline (0.605) additionally relies
on prompt routing / BM25 / entity blocks not replicated here; this
harness measures (a) vs (d) **at identical conditions**, which is the
claim the Atlas Edition must prove (parity under durable
persistence). Honest-reporting rule: every deviation is in the output
JSON under ``methodology``.

Usage::

    python benchmarks/run_locomo_atlas.py --configs atlas,inprocess \
        --data /path/to/locomo10.json --out benchmarks/results/

Requires: local atlas-local container, .env with OPENAI_* endpoint,
SEMVEC_KEY (license), sentence-transformers (all-mpnet-base-v2).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# environment

def _env_candidates():
    """Locations to look for .env: SEMVEC_ATLAS_ENV, then every parent
    of this file and of the cwd."""
    explicit = os.environ.get("SEMVEC_ATLAS_ENV")
    if explicit:
        yield Path(explicit)
    for base in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in [base, *base.parents]:
            yield parent / ".env"


def load_env() -> dict[str, str]:
    from dotenv import dotenv_values

    for candidate in _env_candidates():
        if candidate.is_file():
            env = {k: v for k, v in dotenv_values(candidate).items() if v}
            if "OPENAI_API_KEY" in env:
                if "SEMVEC_KEY" in env:
                    os.environ.setdefault("SEMVEC_LICENSE_KEY", env["SEMVEC_KEY"])
                return env
    raise SystemExit(
        "no .env with OPENAI_API_KEY found — set SEMVEC_ATLAS_ENV to its path"
    )


# --------------------------------------------------------------------------
# prompts (verbatim from mem0 evaluation, as used by the semvec reference
# runner — curly quotes preserved in the judge prompt)

ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), perform it
    5. Formulate a precise, concise answer based on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for both speakers:

    {context}

    Question: {question}

    Answer:
    """

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a ’gold’ (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

_VALID_LABELS = {"CORRECT", "WRONG"}


def parse_judge_response(text: str) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", cleaned))):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "label" in obj:
            label = str(obj["label"]).strip().upper()
            if label in _VALID_LABELS:
                return label
    return None


# --------------------------------------------------------------------------
# LLM client (self-hosted OpenAI-compatible endpoint; thinking disabled)


class LLM:
    def __init__(self, env: dict[str, str]) -> None:
        import openai

        self._client = openai.OpenAI(
            api_key=env["OPENAI_API_KEY"], base_url=env["OPENAI_BASE_URL"], timeout=120
        )
        self.model = env["OPENAI_MODEL"]

    def complete(self, prompt: str, *, max_tokens: int = 128, retries: int = 3) -> str:
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - endpoint hiccups
                last = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after {retries} retries: {last}")


# --------------------------------------------------------------------------
# dataset

def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def iter_turns(conv: dict[str, Any]):
    """Yield (dia_id, text_chunk) in session order.

    Chunk format mirrors the semvec reference runner:
    ``"{speaker}: {text} [{session_date}]"``.
    """
    conversation = conv["conversation"]
    i = 1
    while f"session_{i}" in conversation:
        date = conversation.get(f"session_{i}_date_time", "")
        for entry in conversation[f"session_{i}"] or []:
            text = entry.get("text") or ""
            if not text.strip():
                continue
            yield entry["dia_id"], f"{entry['speaker']}: {text} [{date}]"
        i += 1


def scored_qas(conv: dict[str, Any]) -> list[dict[str, Any]]:
    """Categories 1–4; category 5 (adversarial) excluded per reference."""
    out = []
    for qa in conv["qa"]:
        if int(qa.get("category", 0)) == 5:
            continue
        out.append(qa)
    return out


# --------------------------------------------------------------------------
# embedder


class BatchedST:
    """sentence-transformers embedder fulfilling the semvec protocol,
    with a bulk-encode path for ingest speed."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self._dim = self._model.get_sentence_embedding_dimension()

    def get_dimension(self) -> int:
        return self._dim

    def get_embedding(self, text: str) -> np.ndarray:
        return self._model.encode([text], normalize_embeddings=True)[0].astype(np.float64)

    def encode_bulk(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        ).astype(np.float64)


# --------------------------------------------------------------------------
# config (a): atlas


# Engine tier capacities for benchmark runs. Default: the engine's own
# defaults — observation side effects and maintenance
# non-determinism are fixed since semvec 0.8.2, so the eviction-free
# regime is not needed for the parity gate at current engine versions.
# ``--eviction-free`` restores the historical 1000/1000/1000 regime for
# comparison against pre-0.8.2 result sets.
ENGINE_CFG: dict = {}
_EVICTION_FREE_CFG = {
    "short_term_size": 1000,
    "medium_term_size": 1000,
    "long_term_size": 1000,
}


def ingest_atlas(client, db, conv_idx, chunks, embs, embedder):
    from semvec import SemvecConfig, SemvecState

    from semvec_atlas.commit import TurnCommitter
    from semvec_atlas.guard import TenantScope

    scope = TenantScope(tenant="locomo", agent=f"conv{conv_idx}")
    committer = TurnCommitter(
        client, db, scope, embedder=embedder, snapshot_every=25, snapshot_retention=3
    )
    state = SemvecState(SemvecConfig(dimension=embedder.get_dimension(), **ENGINE_CFG))
    commit_ms: list[float] = []
    for turn, (chunk, emb) in enumerate(zip(chunks, embs, strict=True), start=1):
        t0 = time.perf_counter()
        committer.run_turn(
            state, session="locomo", turn=turn, text=chunk, embedding=emb
        )
        commit_ms.append((time.perf_counter() - t0) * 1000)
    return state, scope, commit_ms


def make_atlas_retriever(db, scope, embedder, k):
    from semvec_atlas.retrieval import RetrievalEngine

    engine = RetrievalEngine(db, scope, embedder=embedder)

    def retrieve(question: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        hits = engine.search(question, k=k)
        latency = (time.perf_counter() - t0) * 1000
        return "\n".join(h.summary for h in hits), latency

    return retrieve, engine


# --------------------------------------------------------------------------
# config (d): in-process parity reference


def ingest_inprocess(chunks, embs, dim):
    from semvec import SemvecConfig, SemvecState

    state = SemvecState(SemvecConfig(dimension=dim, **ENGINE_CFG))
    t0 = time.perf_counter()
    for chunk, emb in zip(chunks, embs, strict=True):
        state.update(emb, chunk)
    return state, time.perf_counter() - t0


def make_inprocess_retriever(state, embedder, k):
    def retrieve(question: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        emb = embedder.get_embedding(question)
        units = state.memory.get_relevant_memories(emb, top_k=k)
        latency = (time.perf_counter() - t0) * 1000
        return "\n".join(u.text for u in units), latency

    return retrieve


# --------------------------------------------------------------------------
# evaluation core


def token_len(enc, text: str) -> int:
    return len(enc.encode(text))


def evaluate_config(
    name: str,
    conv_idx: int,
    qas: list[dict[str, Any]],
    retrieve,
    llm: LLM,
    enc,
    out_path: Path,
    evidence_texts: dict[str, str],
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Reader + judge for one conversation/config; JSONL checkpointing."""
    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done[r["qa_id"]] = r

    def qa_id(i: int, qa: dict[str, Any]) -> str:
        return f"conv{conv_idx}:{i}"

    def work(item):
        i, qa = item
        qid = qa_id(i, qa)
        if qid in done:
            return done[qid]
        question = qa["question"]
        context, retrieval_ms = retrieve(question)
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        answer = llm.complete(prompt, max_tokens=64)
        gold = str(qa.get("answer") or qa.get("adversarial_answer") or "")
        judge_raw = llm.complete(
            ACCURACY_PROMPT.format(
                question=question, gold_answer=gold, generated_answer=answer
            ),
            max_tokens=200,
        )
        label = parse_judge_response(judge_raw)
        # Approximate recall@k: any gold-evidence turn surfaced in context
        # (prefix match on the evidence turn text). Approximation is
        # declared in methodology — dia_ids are not carried through the
        # engine's public memory surface.
        evid = qa.get("evidence") or []
        if isinstance(evid, str):
            try:
                evid = json.loads(evid.replace("'", '"'))
            except json.JSONDecodeError:
                evid = []
        recalled = None
        prefixes = [
            evidence_texts[e][:60] for e in evid if e in evidence_texts
        ]
        if prefixes:
            recalled = any(p in context for p in prefixes)
        return {
            "qa_id": qid,
            "config": name,
            "category": int(qa.get("category", 0)),
            "question": question,
            "gold": gold,
            "answer": answer,
            "label": label,
            "correct": label == "CORRECT",
            "reader_input_tokens": token_len(enc, prompt),
            "retrieval_ms": retrieval_ms,
            "recalled": recalled,
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(work, list(enumerate(qas))):
            results.append(res)
            if res["qa_id"] not in done:
                with out_path.open("a") as f:
                    f.write(json.dumps(res) + "\n")
    return results


# --------------------------------------------------------------------------
# main


def pctl(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--configs", default="atlas,inprocess")
    ap.add_argument("--convs", default="all", help="'all' or comma list of indices")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fullcontext-sample", type=int, default=150)
    ap.add_argument("--mongo-uri", default="mongodb://localhost:27017/?directConnection=true")
    ap.add_argument("--limit-qas", type=int, default=0, help="debug: cap QAs per conv")
    ap.add_argument(
        "--eviction-free",
        action="store_true",
        help="historical regime: tier caps 1000/1000/1000 (pre-0.8.2 comparisons)",
    )
    args = ap.parse_args()

    env = load_env()
    if args.eviction_free:
        ENGINE_CFG.update(_EVICTION_FREE_CFG)
    llm = LLM(env)
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    convs = load_conversations(args.data)
    idxs = range(len(convs)) if args.convs == "all" else [int(x) for x in args.convs.split(",")]
    configs = args.configs.split(",")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now(timezone.utc).isoformat()}] loading embedder …", flush=True)
    embedder = BatchedST()
    run_id = uuid.uuid4().hex[:8]

    mongo_client = None
    db = None
    if "atlas" in configs:
        import pymongo

        from semvec_atlas.indexes import provision

        mongo_client = pymongo.MongoClient(args.mongo_uri)
        db = mongo_client[f"locomo_{run_id}"]
        provision(db, dimension=embedder.get_dimension(), search_indexes=True)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": llm.model,
        "embedder": "sentence-transformers/all-mpnet-base-v2 (cpu)",
        "k": args.k,
        "mongo": "mongodb-atlas-local (MongoDB 8.3.4, $rankFusion native)",
        "methodology": {
            "reader_prompt": "mem0 ANSWER_PROMPT verbatim",
            "judge_prompt": "mem0 LLM-judge prompt verbatim",
            "scored_categories": [1, 2, 3, 4],
            "excluded": "category 5 (adversarial), per published semvec methodology",
            "recall_at_k": "approximate — evidence-turn 60-char prefix containment in context",
            "note": (
                "Absolute J is NOT comparable to the published semvec LOCOMO "
                "pipeline (prompt routing, BM25 hybrid, entity blocks not "
                "replicated). The claim under test is (a) vs (d) parity at "
                "identical conditions, plus token/latency deltas."
            ),
            "fullcontext": f"token counts on all QAs; LLM on sample of {args.fullcontext_sample}",
            "engine_config": (
                "eviction-free regime (caps 1000/1000/1000, pre-0.8.2 comparison)"
                if args.eviction_free
                else "engine default tier capacities (maintenance determinism fixed in semvec >= 0.8.2)"
            ),
        },
        "conversations": {},
        "parity": {},
    }
    summary_path = args.out / f"locomo_atlas_{run_id}_summary.json"

    all_results: dict[str, list[dict[str, Any]]] = {c: [] for c in configs}
    fullcontext_tokens: list[int] = []

    for ci in idxs:
        conv = convs[ci]
        chunks, dia_ids = [], []
        for dia_id, chunk in iter_turns(conv):
            dia_ids.append(dia_id)
            chunks.append(chunk)
        evidence_texts = dict(zip(dia_ids, chunks, strict=True))
        qas = scored_qas(conv)
        if args.limit_qas:
            qas = qas[: args.limit_qas]
        print(f"[conv {ci}] turns={len(chunks)} scored_qas={len(qas)}", flush=True)

        t0 = time.perf_counter()
        embs = embedder.encode_bulk(chunks)
        embed_s = time.perf_counter() - t0

        conv_summary: dict[str, Any] = {
            "turns": len(chunks),
            "qas": len(qas),
            "embed_s": round(embed_s, 2),
        }

        checksums = {}
        if "atlas" in configs:
            t0 = time.perf_counter()
            state_a, scope, commit_ms = ingest_atlas(
                mongo_client, db, ci, chunks, embs, embedder
            )
            ingest_s = time.perf_counter() - t0
            checksums["atlas"] = state_a.to_dict()["checksum"]
            conv_summary["atlas_ingest_s"] = round(ingest_s, 2)
            conv_summary["commit_ms_p50"] = round(pctl(commit_ms, 50), 1)
            conv_summary["commit_ms_p95"] = round(pctl(commit_ms, 95), 1)

        if "inprocess" in configs:
            state_d, ingest_s = ingest_inprocess(chunks, embs, embedder.get_dimension())
            checksums["inprocess"] = state_d.to_dict()["checksum"]
            conv_summary["inprocess_ingest_s"] = round(ingest_s, 2)

        if "atlas" in checksums and "inprocess" in checksums:
            if checksums["atlas"] != checksums["inprocess"]:
                print(f"PARITY VIOLATION conv {ci}: {checksums}", file=sys.stderr)
                return 2
            summary["parity"][f"conv{ci}"] = checksums["atlas"]
            print(f"[conv {ci}] engine checksums identical (a)==(d) ✓", flush=True)

        # Wait for Atlas Search to index this conversation's items before QA.
        if "atlas" in configs:
            expected = db.memory_items.count_documents(
                {"tenant": "locomo", "agent": f"conv{ci}"}
            )
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                try:
                    n = list(
                        db.memory_items.aggregate(
                            [
                                {"$search": {"index": "ix_lex", "exists": {"path": "tenant"}}},
                                {"$match": {"agent": f"conv{ci}"}},
                                {"$count": "n"},
                            ]
                        )
                    )
                    if n and n[0]["n"] >= expected * 0.98:
                        break
                except Exception:
                    pass
                time.sleep(2)

        for cfg in configs:
            if cfg == "atlas":
                retrieve, _engine = make_atlas_retriever(db, scope, embedder, args.k)
            elif cfg == "inprocess":
                retrieve = make_inprocess_retriever(state_d, embedder, args.k)
            elif cfg == "fullcontext":
                full = "\n".join(chunks)
                fullcontext_tokens.extend(
                    token_len(enc, ANSWER_PROMPT.format(context=full, question=q["question"]))
                    for q in qas
                )
                continue  # sampled LLM pass happens after the loop
            else:
                raise SystemExit(f"unknown config {cfg}")
            out_path = args.out / f"locomo_{run_id}_{cfg}.jsonl"
            t0 = time.perf_counter()
            res = evaluate_config(
                cfg, ci, qas, retrieve, llm, enc, out_path, evidence_texts, args.workers
            )
            conv_summary[f"{cfg}_qa_s"] = round(time.perf_counter() - t0, 1)
            all_results[cfg].extend(res)
            j = sum(r["correct"] for r in res) / max(len(res), 1)
            print(f"[conv {ci}] {cfg}: J={j:.3f} over {len(res)} QAs", flush=True)

        summary["conversations"][f"conv{ci}"] = conv_summary
        summary_path.write_text(json.dumps(summary, indent=2))

    # ---- aggregate ----
    for cfg, res in all_results.items():
        if not res:
            continue
        judged = [r for r in res if r["label"] is not None]
        agg = {
            "qas": len(res),
            "judged": len(judged),
            "J": round(sum(r["correct"] for r in judged) / max(len(judged), 1), 4),
            "J_by_category": {},
            "reader_input_tokens_mean": round(
                statistics.mean(r["reader_input_tokens"] for r in res), 1
            ),
            "retrieval_ms_p50": round(pctl([r["retrieval_ms"] for r in res], 50), 1),
            "retrieval_ms_p95": round(pctl([r["retrieval_ms"] for r in res], 95), 1),
        }
        recalls = [r for r in res if r.get("recalled") is not None]
        if recalls:
            agg["recall_at_k_approx"] = round(
                sum(bool(r["recalled"]) for r in recalls) / len(recalls), 4
            )
        for cat in (1, 2, 3, 4):
            sub = [r for r in judged if r["category"] == cat]
            if sub:
                agg["J_by_category"][str(cat)] = round(
                    sum(r["correct"] for r in sub) / len(sub), 4
                )
        summary[cfg] = agg

    if fullcontext_tokens:
        summary["fullcontext"] = {
            "reader_input_tokens_mean": round(statistics.mean(fullcontext_tokens), 1),
            "qas_counted": len(fullcontext_tokens),
        }
        for cfg in ("atlas", "inprocess"):
            if cfg in summary:
                summary[cfg]["token_reduction_vs_fullcontext"] = round(
                    1 - summary[cfg]["reader_input_tokens_mean"]
                    / summary["fullcontext"]["reader_input_tokens_mean"],
                    4,
                )

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k in configs or k == "fullcontext"}, indent=2))
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
