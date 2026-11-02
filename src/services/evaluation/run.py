"""
Retrieval evaluation harness.

Runs every question in questions.jsonl through RetrievalService.search()
directly (no HTTP, no LLM) and reports hit@k / MRR overall and broken down
by question type, location and document. Unanswerable questions are scored
separately via their top similarity score, to inform a future
"no relevant context" threshold.

Usage (from the repo root):
    uv run python -m src.services.evaluation.run --label baseline
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from loguru import logger

from src.core.config import settings
from src.services.evaluation.metrics import (
    QuestionResult,
    RetrievedChunk,
    find_answer_chunks,
    group_by,
    is_relevant,
    normalize_for_match,
    percentile,
    reachability,
    score_separation,
    score_separation_by_subtype,
    summarize,
)


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = EVAL_DIR / "questions.jsonl"
DEFAULT_OUT_DIR = EVAL_DIR / "results"
DEFAULT_KS = "1,3,5,10"

VALID_TYPES = {"direct", "paraphrased", "unanswerable"}
VALID_SUBTYPES = {"near_miss", "off_topic"}
WARMUP_QUERY = "warm-up query to load the embedding model"
EXCERPT_CHARS = 160


# Questions file
def validate_question(q: Dict) -> List[str]:
    """Return a list of problems with one question row (empty if valid)."""
    errors = []

    for key in ("id", "type", "question"):
        if not q.get(key):
            errors.append(f"missing '{key}'")

    q_type = q.get("type")
    if q_type and q_type not in VALID_TYPES:
        errors.append(f"unknown type '{q_type}' (expected one of {sorted(VALID_TYPES)})")

    subtype = q.get("subtype")
    if subtype is not None:
        if q_type != "unanswerable":
            errors.append("'subtype' is only for unanswerable questions")
        elif subtype not in VALID_SUBTYPES:
            errors.append(f"unknown subtype '{subtype}' (expected one of {sorted(VALID_SUBTYPES)})")

    if q_type == "unanswerable":
        if q.get("doc") is not None or q.get("must_contain") is not None:
            errors.append("unanswerable questions must have doc and must_contain set to null")
    elif q_type in VALID_TYPES:
        if not q.get("doc"):
            errors.append("answerable questions need a 'doc'")
        must_contain = q.get("must_contain")
        if not must_contain or not normalize_for_match(must_contain):
            # An empty normalized string would match every chunk.
            errors.append("answerable questions need a non-empty 'must_contain'")

    return errors


def load_questions(path: Path) -> List[Dict]:
    """Load and validate questions.jsonl, raising ValueError listing every problem."""
    questions, errors, seen_ids = [], [], set()

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
            continue

        label = f"line {lineno} ({q.get('id', '?')})"
        errors.extend(f"{label}: {e}" for e in validate_question(q))

        if q.get("id") in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(q.get("id"))
        questions.append(q)

    if errors:
        raise ValueError("Invalid questions file:\n  " + "\n  ".join(errors))
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


# CLI helpers
def parse_ks(value: str) -> List[int]:
    try:
        ks = sorted({int(part) for part in value.split(",") if part.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(f"--ks must be comma-separated integers, got '{value}'")
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError("--ks values must be >= 1")
    return ks


def parse_label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("--label may only contain letters, digits, '_' and '-'")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on questions.jsonl.")
    parser.add_argument("--label", type=parse_label, default="run",
                        help="Name for this run; output is results/<date>_<label>.json")
    parser.add_argument("--ks", type=parse_ks, default=parse_ks(DEFAULT_KS),
                        help=f"Comma-separated k values for hit@k (default {DEFAULT_KS})")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing non-empty results file with the same name")
    parser.add_argument("--verbose", action="store_true",
                        help="Show INFO logs from the services (default: warnings only)")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    # Not using setup_logging(): the services log at INFO on every query,
    # which would bury the report. Warnings still come through.
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO" if verbose else "WARNING",
        format="<level>{level:<8}</level> | {message}",
    )


def git_revision() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=EVAL_DIR,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=EVAL_DIR,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Services
def build_services():
    """
    Build services the same way the app lifespan in src/main.py does,
    limited to what retrieval needs: no Ollama, no SQLite.
    Imported lazily so tests can import this module without loading models.
    """
    from src.services.embeddings.embedding_service import EmbeddingService
    from src.services.vectorstore.chroma_service import ChromaService
    from src.services.retrieval.retrieval_service import RetrievalService

    embedding_service = EmbeddingService()
    chroma_service = ChromaService()
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        chroma_service=chroma_service,
    )
    return chroma_service, retrieval_service


def load_indexed_chunks(chroma_service) -> Dict[str, List[Tuple[int, str]]]:
    """Every stored chunk as {document_name: [(chunk_index, text), ...]}."""
    stored = chroma_service.collection.get(include=["documents", "metadatas"])
    chunks: Dict[str, List[Tuple[int, str]]] = {}
    for text, metadata in zip(stored["documents"], stored["metadatas"]):
        chunks.setdefault(metadata["document_name"], []).append((metadata["chunk_index"], text))
    return dict(sorted(chunks.items()))


def to_chunks(raw_results: List[Dict]) -> List[RetrievedChunk]:
    return [
        RetrievedChunk(
            document_name=r["metadata"]["document_name"],
            chunk_index=r["metadata"]["chunk_index"],
            score=float(r["score"]),
            text=r["chunks"],
        )
        for r in raw_results
    ]


def evaluate(
    questions: List[Dict],
    retrieval_service,
    k_max: int,
    indexed_chunks: Dict[str, List[Tuple[int, str]]],
) -> List[QuestionResult]:
    # The first call pays one-off warm-up costs; keep it out of the latency numbers.
    retrieval_service.search(WARMUP_QUERY, top_k=1)

    results = []
    for i, q in enumerate(questions, start=1):
        start = time.perf_counter()
        raw = retrieval_service.search(q["question"], top_k=k_max)
        latency_ms = (time.perf_counter() - start) * 1000

        # Experiment 0: is the answer in the index at all, ranked or not?
        answer_chunks = (
            find_answer_chunks(indexed_chunks.get(q["doc"], []), q["must_contain"])
            if q.get("doc") else None
        )

        results.append(
            QuestionResult(
                id=q["id"],
                type=q["type"],
                location=q.get("location"),
                doc=q.get("doc"),
                must_contain=q.get("must_contain"),
                retrieved=to_chunks(raw),
                latency_ms=latency_ms,
                answer_chunks=answer_chunks,
                subtype=q.get("subtype"),
            )
        )
        print(f"\r  evaluated {i}/{len(questions)}", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    return results


# Report
def build_report(results: List[QuestionResult], ks: List[int], doc_k: int, meta: Dict) -> Dict:
    def _summaries(attr: str) -> Dict:
        return {
            key: summarize(group, ks, doc_k)
            for key, group in group_by([r for r in results if r.answerable], attr).items()
        }

    latencies = [r.latency_ms for r in results]

    return {
        "meta": meta,
        "summary": {
            "overall": summarize(results, ks, doc_k),
            # Same metrics over questions whose answer exists in the index:
            # retrieval quality with ingestion failures taken out.
            "overall_reachable": summarize([r for r in results if r.reachable], ks, doc_k),
            "reachability": reachability(results),
            "by_type": _summaries("type"),
            "by_location": _summaries("location"),
            "by_doc": _summaries("doc"),
            "unanswerable": score_separation(results),
            "unanswerable_by_subtype": score_separation_by_subtype(results),
            "latency_ms": {
                "p50": round(percentile(latencies, 50), 2),
                "p95": round(percentile(latencies, 95), 2),
                "max": round(max(latencies), 2),
            },
        },
        "questions": [question_record(r) for r in results],
    }


def question_record(r: QuestionResult) -> Dict:
    return {
        "id": r.id,
        "type": r.type,
        "subtype": r.subtype,
        "location": r.location,
        "doc": r.doc,
        "must_contain": r.must_contain,
        "rank": r.rank,
        "doc_rank": r.doc_rank,
        "reachable": r.reachable,
        "answer_chunks": r.answer_chunks,
        "top_score": round(r.top_score, 4) if r.top_score is not None else None,
        "latency_ms": round(r.latency_ms, 2),
        "retrieved": [
            {
                "rank": position,
                "document_name": c.document_name,
                "chunk_index": c.chunk_index,
                "score": round(c.score, 4),
                "relevant": r.answerable and is_relevant(c, r.doc, r.must_contain),
                "excerpt": c.text[:EXCERPT_CHARS],
            }
            for position, c in enumerate(r.retrieved, start=1)
        ],
    }


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_table(title: str, rows: Dict[str, Dict]) -> str:
    """Render {group_name: summary_dict} as a fixed-width text table."""
    if not rows:
        return f"{title}\n  (no answerable questions)\n"

    columns = list(next(iter(rows.values())).keys())
    name_width = max(len(name) for name in rows) + 2
    widths = [max(len(c), 5) + 2 for c in columns]

    lines = [title, "  " + "".ljust(name_width) + "".join(c.rjust(w) for c, w in zip(columns, widths))]
    for name, summary in rows.items():
        cells = "".join(_fmt(summary[c]).rjust(w) for c, w in zip(columns, widths))
        lines.append("  " + name.ljust(name_width) + cells)
    return "\n".join(lines) + "\n"


def print_report(report: Dict, results: List[QuestionResult], doc_k: int) -> None:
    meta, summary = report["meta"], report["summary"]
    answerable = [r for r in results if r.answerable]

    print(f"\nRetrieval evaluation: {meta['label']}  ({meta['timestamp']}, git {meta['git']})")
    print(f"  model:      {meta['embedding_model']}")
    print(f"  collection: {meta['collection']} | metric={meta['distance_metric']} | chunks={meta['indexed_chunks']}")
    print(f"  questions:  {meta['questions']} ({len(answerable)} answerable, "
          f"{meta['questions'] - len(answerable)} unanswerable) | app top_k={doc_k}\n")

    reach = summary["reachability"]
    print(f"Answer reachability: {reach['reachable']}/{reach['n']} answers exist in an indexed chunk "
          f"(ceiling for every hit@k: {_fmt(reach['rate'])})")
    if reach["unreachable_ids"]:
        print(f"  unreachable: {', '.join(reach['unreachable_ids'])}")
    print()

    print(format_table("Overall", {
        "all": summary["overall"],
        "reachable only": summary["overall_reachable"],
    }))
    print(format_table("By type", summary["by_type"]))
    print(format_table("By location", summary["by_location"]))
    print(format_table("By document", summary["by_doc"]))

    sep = summary["unanswerable"]
    print("Unanswerable vs relevant scores")
    print(f"  unanswerable top-1 score: mean={_fmt(sep['unanswerable_top1_mean'])} "
          f"max={_fmt(sep['unanswerable_top1_max'])} (n={sep['unanswerable_n']})")
    print(f"  relevant chunk score:     min={_fmt(sep['relevant_chunk_score_min'])} "
          f"median={_fmt(sep['relevant_chunk_score_median'])} (n={sep['relevant_chunk_n']})")
    print(f"  clean threshold possible: {_fmt(sep['clean_threshold_possible'])}")
    for subtype, sub in summary["unanswerable_by_subtype"].items():
        print(f"  {subtype:<10} top-1: mean={_fmt(sub['unanswerable_top1_mean'])} "
              f"max={_fmt(sub['unanswerable_top1_max'])} (n={sub['unanswerable_n']}) "
              f"| below every relevant chunk: {_fmt(sub['clean_threshold_possible'])}")
    print()

    latency = summary["latency_ms"]
    print(f"Latency (ms, after warm-up): p50={latency['p50']} p95={latency['p95']} max={latency['max']}\n")

    misses = [r for r in answerable if r.rank is None or r.rank > doc_k]
    print(f"Misses at k={doc_k} ({len(misses)})")
    for r in misses:
        top = r.retrieved[0] if r.retrieved else None
        top_desc = f"{top.document_name}#{top.chunk_index} ({top.score:.2f})" if top else "-"
        answer_in = (
            "UNREACHABLE" if r.reachable is False
            else ",".join(str(i) for i in r.answer_chunks or []) or "-"
        )
        print(f"  {r.id:<10} rank={_fmt(r.rank):<4} doc_rank={_fmt(r.doc_rank):<4} "
              f"answer_in={answer_in:<12} top={top_desc}")
    print()


def output_path(out_dir: Path, label: str, overwrite: bool) -> Path:
    path = out_dir / f"{date.today().isoformat()}_{label}.json"
    # Zero-byte files are placeholders and safe to fill in.
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite or use a different --label")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    questions = load_questions(args.questions)
    ks = args.ks
    doc_k = settings.TOP_K_RESULTS
    out_path = output_path(args.out_dir, args.label, args.overwrite)

    chroma_service, retrieval_service = build_services()

    # Reconcile before evaluating: a question whose document isn't indexed
    # would silently score 0 (the v0.4.0 chunk-ID bug looked exactly like that).
    indexed_chunks = load_indexed_chunks(chroma_service)
    chunk_counts = {doc: len(chunks) for doc, chunks in indexed_chunks.items()}
    missing = sorted({q["doc"] for q in questions if q.get("doc")} - chunk_counts.keys())
    if missing:
        print(f"Documents referenced by questions but not indexed: {missing}", file=sys.stderr)
        return 1

    collection_metadata = chroma_service.collection.metadata or {}
    meta = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "collection": settings.CHROMA_COLLECTION_NAME,
        "distance_metric": collection_metadata.get("hnsw:space", "l2 (default)"),
        "indexed_chunks": sum(chunk_counts.values()),
        "chunks_per_doc": chunk_counts,
        "questions": len(questions),
        "questions_file": str(args.questions),
        "ks": ks,
        "app_top_k": doc_k,
        "relevance_rule": "expected doc AND normalized must_contain in normalized chunk text",
    }

    # Retrieve deep enough for both the largest k and the app's own top_k.
    results = evaluate(questions, retrieval_service, k_max=max(*ks, doc_k), indexed_chunks=indexed_chunks)
    report = build_report(results, ks, doc_k, meta)

    print_report(report, results, doc_k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
