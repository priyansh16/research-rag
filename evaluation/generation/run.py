"""
Generation evaluation, phase 1: generate answers.

Runs every question in questions.jsonl through GenerationService.generate()
(guardrails -> retrieval -> prompt -> Ollama), the same call the API makes,
and writes one JSON row per question to results/<date>_<label>.jsonl as soon
as it is answered. Deterministic metrics (correctness, refusals, gold chunk in
context, error attribution, truncation) are computed here; faithfulness is
judged later by judge.py from the same file.

Settings come from the app config, so experiments are env vars:
    uv run python -m evaluation.generation.run --label gen_k3
    TOP_K_RESULTS=5 uv run python -m evaluation.generation.run --label gen_k5

Needs Ollama running with OLLAMA_MODEL pulled, and the indexed documents.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TextIO

from src.core.config import settings
from src.services.generation.prompt_builder import SYSTEM_INSTRUCTIONS
from evaluation.common import (
    DEFAULT_QUESTIONS,
    git_revision,
    load_questions,
    output_path,
    parse_label,
)
from evaluation.generation.metrics import (
    attribute,
    gold_rank,
    is_correct,
    is_refusal,
    is_truncated,
    soft_refusal_phrases,
    summarize,
)
from evaluation.retrieval.run import configure_logging, load_indexed_chunks


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = EVAL_DIR / "results"
RESULTS_SUFFIX = ".jsonl"
# Answerable-looking but outside the corpus: loads the embedding model,
# reranker and the Ollama model before anything is timed.
WARMUP_QUERY = "What is the capital of France?"
ANSWER_PREVIEW_CHARS = 120


# CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and score answers for questions.jsonl.")
    parser.add_argument("--label", type=parse_label, required=True,
                        help=f"Name for this run; output is results/<date>_<label>{RESULTS_SUFFIX}")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N questions (smoke test; don't compare these runs)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing non-empty results file with the same name")
    parser.add_argument("--verbose", action="store_true",
                        help="Show INFO logs from the services (default: warnings only)")
    return parser.parse_args(argv)


# Services
def build_services():
    """
    Build the app's generation stack the way src/main.py does.
    Imported lazily so tests can import this module without loading models.
    """
    from evaluation.retrieval.run import build_services as build_retrieval_services
    from src.services.generation.generation_service import GenerationService
    from src.services.generation.ollama_client import OllamaClient
    from src.services.guardrails.guardrails_service import GuardrailsService

    chroma_service, retrieval_service = build_retrieval_services()
    ollama_client = OllamaClient()
    generation_service = GenerationService(
        retrieval_service=retrieval_service,
        guardrails_service=GuardrailsService(),
        ollama_client=ollama_client,
    )
    return chroma_service, ollama_client, generation_service


# Rows
def prompt_version() -> str:
    """Short hash of the generation instructions; changes whenever the prompt does."""
    return hashlib.sha256(SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest()[:12]


def run_config(label: str, questions_file: Path, n_questions: int) -> Dict:
    """Everything that decides the answers; stored on every row."""
    return {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "model": settings.OLLAMA_MODEL,
        "temperature": settings.GENERATION_TEMPERATURE,
        "seed": settings.GENERATION_SEED,
        "num_ctx": settings.OLLAMA_NUM_CTX,
        "num_predict": settings.GENERATION_MAX_TOKEN,
        "top_k": settings.TOP_K_RESULTS,
        "max_context_chunks": settings.GENERATION_MAX_CONTEXT_CHUNKS,
        "prompt_version": prompt_version(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker": settings.RETRIEVAL_RERANKER_MODEL or None,
        "rerank_candidates": settings.RETRIEVAL_RERANK_CANDIDATES if settings.RETRIEVAL_RERANKER_MODEL else None,
        "neighbor_window": settings.RETRIEVAL_NEIGHBOR_WINDOW,
        "collection": settings.CHROMA_COLLECTION_NAME,
        "questions_file": str(questions_file),
        "questions": n_questions,
    }


def build_row(
    q: Dict,
    config: Dict,
    latency_ms: float,
    result=None,
    blocked_reason: Optional[str] = None,
) -> Dict:
    """
    One results row. `result` is a GenerationResult, or None when guardrails
    blocked the question (then `blocked_reason` says why).
    """
    answerable = q["type"] != "unanswerable"
    blocked = result is None
    answer = None if blocked else result.answer
    context = [] if blocked else result.context_chunks
    ollama = None if blocked else result.ollama
    prompt_tokens = None if blocked else result.prompt_eval_count

    rank = gold_rank(context, q.get("doc"), q.get("must_contain"))
    correct = is_correct(answer, q.get("must_contain"))

    return {
        "id": q["id"],
        "type": q["type"],
        "subtype": q.get("subtype"),
        "location": q.get("location"),
        "question": q["question"],
        "doc": q.get("doc"),
        "must_contain": q.get("must_contain"),
        "answerable": answerable,
        "status": "blocked" if blocked else "ok",
        "blocked_reason": blocked_reason,
        "answer": answer,
        "refused": is_refusal(answer),
        "soft_refusal_phrases": soft_refusal_phrases(answer),
        "correct": correct,
        "gold_rank": rank,
        "gold_in_context": rank is not None,
        "attribution": attribute(answerable, blocked, correct, rank is not None),
        "latency_ms": round(latency_ms, 2),
        "retrieval_latency_ms": None if blocked else result.retrieval_latency_ms,
        "generation_latency_ms": None if blocked else result.generation_latency_ms,
        "prompt_eval_count": prompt_tokens,
        "eval_count": None if blocked else result.eval_count,
        "done_reason": ollama.done_reason if ollama else None,
        "truncated": is_truncated(prompt_tokens, config["num_ctx"]),
        "context": context,
        "config": config,
    }


# Run
async def generate_rows(
    questions: List[Dict],
    generation_service,
    config: Dict,
    out_file: TextIO,
) -> List[Dict]:
    from src.services.generation.generation_service import GenerationBlockedError

    # The first call pays model-loading costs; keep it out of the latency numbers.
    await generation_service.generate(WARMUP_QUERY)

    rows = []
    for i, q in enumerate(questions, start=1):
        start = time.perf_counter()
        try:
            result = await generation_service.generate(q["question"])
            blocked_reason = None
        except GenerationBlockedError as exc:
            result, blocked_reason = None, exc.reason
        latency_ms = (time.perf_counter() - start) * 1000

        row = build_row(q, config, latency_ms, result=result, blocked_reason=blocked_reason)
        # Written per row so an interrupted run keeps what it has.
        out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_file.flush()
        rows.append(row)
        print(f"\r  generated {i}/{len(questions)}", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    return rows


async def check_ollama(ollama_client) -> Optional[str]:
    """Problem description, or None when Ollama is up with the model pulled."""
    health = await ollama_client.health_check()
    if health.get("status") != "ok":
        return f"Ollama is not reachable at {ollama_client.base_url}: {health.get('error')}"
    if not health.get("model_available"):
        return f"Model '{ollama_client.model}' is not pulled. Run: ollama pull {ollama_client.model}"
    return None


# Report
def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.0f}"
    return str(value)


def _stats_line(stats: Dict) -> str:
    return f"p50={_fmt(stats['p50'])} p95={_fmt(stats['p95'])} max={_fmt(stats['max'])}"


def _preview(text: Optional[str]) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= ANSWER_PREVIEW_CHARS else text[:ANSWER_PREVIEW_CHARS] + "…"


def format_summary(summary: Dict, rows: List[Dict], config: Dict) -> str:
    ans, unans, attr = summary["answerable"], summary["unanswerable"], summary["attribution"]
    by_id = {r["id"]: r for r in rows}
    lines = [
        f"\nGeneration evaluation: {config['label']}  ({config['timestamp']}, git {config['git']})",
        f"  model: {config['model']} | temperature={config['temperature']} seed={config['seed']} "
        f"num_ctx={config['num_ctx']} num_predict={config['num_predict']} | prompt {config['prompt_version']}",
        f"  retrieval: top_k={config['top_k']} (max {config['max_context_chunks']} context chunks) "
        f"| reranker={config['reranker']} | neighbor_window={config['neighbor_window']}",
        f"  questions: {summary['n']} ({ans['n']} answerable, {unans['n']} unanswerable)",
        "",
        f"Answerable ({ans['n']}): correct {ans['correct']} | false refusals {ans['false_refusals']} "
        f"| gold chunk in context {ans['gold_in_context']}",
        f"  wrong answers: retrieval miss {attr['retrieval_miss']} | generation miss {attr['generation_miss']} "
        f"| blocked {attr['blocked']}",
        f"Unanswerable ({unans['n']}): refused {unans['refused']}"
        + "".join(f" | {name} {s['refused']}/{s['n']}" for name, s in unans["by_subtype"].items()),
        "",
        f"Latency ms (after warm-up): {_stats_line(summary['latency_ms'])}",
        f"Prompt tokens: {_stats_line(summary['prompt_tokens'])} (num_ctx {config['num_ctx']}) "
        f"| truncated: {len(summary['truncated_ids'])}",
        f"Output tokens: {_stats_line(summary['output_tokens'])} "
        f"| hit num_predict: {len(summary['hit_num_predict_ids'])}",
    ]

    def section(title: str, ids: List[str], detail) -> None:
        if ids:
            lines.append("")
            lines.append(f"{title} ({len(ids)})")
            lines.extend(f"  {i:<10} {detail(by_id[i])}" for i in ids)

    answered_unanswerable = [r["id"] for r in rows if not r["answerable"] and not r["refused"]]
    false_refusals = [r["id"] for r in rows if r["answerable"] and r["refused"]]
    generation_misses = [r["id"] for r in rows if r["attribution"] == "generation_miss"]

    section("Unanswerable but answered", answered_unanswerable,
            lambda r: f"[{r['subtype']}] {_preview(r['answer'])}")
    section("False refusals", false_refusals,
            lambda r: f"gold_rank={_fmt(r['gold_rank'])}")
    section("Generation misses (gold chunk in context, answer wrong)", generation_misses,
            lambda r: f"gold_rank={r['gold_rank']} want '{r['must_contain']}' | {_preview(r['answer'])}")
    section("Soft refusals to review (not counted as refusals)", summary["soft_refusal_ids"],
            lambda r: f"{r['soft_refusal_phrases']} | {_preview(r['answer'])}")
    section("Truncated prompts", summary["truncated_ids"],
            lambda r: f"prompt_tokens={r['prompt_eval_count']}")
    section("Blocked by guardrails", summary["blocked_ids"],
            lambda r: r["blocked_reason"] or "")
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> int:
    from src.services.generation.ollama_client import OllamaClientError

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    out_path = output_path(args.out_dir, args.label, args.overwrite, suffix=RESULTS_SUFFIX)

    chroma_service, ollama_client, generation_service = build_services()

    problem = await check_ollama(ollama_client)
    if problem:
        print(problem, file=sys.stderr)
        return 1

    # A question whose document isn't indexed would silently score as a retrieval miss.
    indexed = load_indexed_chunks(chroma_service)
    missing = sorted({q["doc"] for q in questions if q.get("doc")} - indexed.keys())
    if missing:
        print(f"Documents referenced by questions but not indexed: {missing}", file=sys.stderr)
        return 1

    config = run_config(args.label, args.questions, len(questions))
    if config["git"].endswith("-dirty"):
        print("Warning: uncommitted changes; this run is tagged "
              f"'{config['git']}'. Commit first for a comparable run.", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        try:
            rows = await generate_rows(questions, generation_service, config, out_file)
        except OllamaClientError as exc:
            print(f"\nOllama failed: {exc}\nPartial results kept in {out_path}; "
                  "rerun with --overwrite.", file=sys.stderr)
            return 1

    print(format_summary(summarize(rows), rows, config))
    print(f"Saved results to {out_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
