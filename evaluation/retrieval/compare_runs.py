"""
Compare two evaluation runs question by question.

Aggregate metrics can hide changes that cancel out (two questions fixed,
two broken, hit@3 unchanged). This lists exactly which questions crossed
the hit@k line in each direction, which moved rank without crossing it,
and how unanswerable questions' top scores shifted.

Usage (from the repo root), with labels or file paths:
    uv run python -m evaluation.retrieval.compare_runs baseline bge-prefix
    uv run python -m evaluation.retrieval.compare_runs evaluation/retrieval/results/a.json evaluation/retrieval/results/b.json --k 5
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.retrieval.metrics import hit_at_k
from evaluation.retrieval.summarize_results import (
    DEFAULT_RESULTS_DIR,
    Run,
    parse_run,
    run_label,
)


# Meta keys that always differ between runs and say nothing about the setup.
IGNORED_META_KEYS = {"label", "timestamp", "questions_file"}


@dataclass(frozen=True)
class QuestionChange:
    id: str
    type: str
    location: Optional[str]
    doc: Optional[str]
    rank_a: Optional[int]
    rank_b: Optional[int]
    top_score_a: Optional[float]
    top_score_b: Optional[float]
    # doc or must_contain differs between the runs, so a rank change may
    # come from relabelling rather than retrieval.
    label_changed: bool = False
    subtype: Optional[str] = None


@dataclass
class Comparison:
    k: int
    improved: List[QuestionChange] = field(default_factory=list)
    regressed: List[QuestionChange] = field(default_factory=list)
    rank_changed: List[QuestionChange] = field(default_factory=list)
    unchanged: int = 0
    unanswerable: List[QuestionChange] = field(default_factory=list)
    only_a: List[str] = field(default_factory=list)
    only_b: List[str] = field(default_factory=list)
    # Overall metrics only count answerable questions, so they stay
    # comparable when just unanswerable questions were added or removed.
    answerable_sets_differ: bool = False


# Loading
def resolve_run_path(arg: str, results_dir: Path) -> Path:
    """Accept a file path, or a label matching results/<date>_<label>.json exactly."""
    path = Path(arg)
    if path.is_file():
        return path

    matches = sorted(p for p in results_dir.glob("*.json") if run_label(p) == arg)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No results file or label '{arg}' in {results_dir}")
    names = ", ".join(p.name for p in matches)
    raise ValueError(f"Label '{arg}' matches several files ({names}); pass a file path instead")


def load_run(path: Path) -> Tuple[Run, Dict[str, Dict]]:
    """The run plus its per-question records keyed by id."""
    run = parse_run(path)
    records = json.loads(path.read_text(encoding="utf-8")).get("questions") or []
    return run, {record["id"]: record for record in records if isinstance(record, dict) and "id" in record}


# Comparison
def compare_questions(a: Dict[str, Dict], b: Dict[str, Dict], k: int) -> Comparison:
    def answerable_ids(records: Dict[str, Dict]) -> set:
        return {qid for qid, record in records.items() if record.get("doc") is not None}

    comparison = Comparison(
        k=k,
        only_a=sorted(a.keys() - b.keys()),
        only_b=sorted(b.keys() - a.keys()),
        answerable_sets_differ=answerable_ids(a) != answerable_ids(b),
    )

    for qid in sorted(a.keys() & b.keys()):
        rec_a, rec_b = a[qid], b[qid]
        change = QuestionChange(
            id=qid,
            type=rec_b.get("type"),
            location=rec_b.get("location"),
            doc=rec_b.get("doc"),
            rank_a=rec_a.get("rank"),
            rank_b=rec_b.get("rank"),
            top_score_a=rec_a.get("top_score"),
            top_score_b=rec_b.get("top_score"),
            label_changed=(
                rec_a.get("doc") != rec_b.get("doc")
                or rec_a.get("must_contain") != rec_b.get("must_contain")
            ),
            subtype=rec_b.get("subtype"),
        )

        if rec_b.get("doc") is None and rec_a.get("doc") is None:
            comparison.unanswerable.append(change)
            continue

        hit_a, hit_b = hit_at_k(change.rank_a, k), hit_at_k(change.rank_b, k)
        if not hit_a and hit_b:
            comparison.improved.append(change)
        elif hit_a and not hit_b:
            comparison.regressed.append(change)
        elif change.rank_a != change.rank_b:
            comparison.rank_changed.append(change)
        else:
            comparison.unchanged += 1

    # Biggest movements first; a move to or from "not retrieved" counts as largest.
    def movement(c: QuestionChange) -> float:
        if c.rank_a is None or c.rank_b is None:
            return float("inf")
        return abs(c.rank_a - c.rank_b)

    for bucket in (comparison.improved, comparison.regressed, comparison.rank_changed):
        bucket.sort(key=lambda c: (-movement(c), c.id))

    # Unanswerable: biggest score increases (worse) first, unknown deltas last.
    def increase(c: QuestionChange) -> float:
        delta = score_delta(c)
        return delta if delta is not None else float("-inf")

    comparison.unanswerable.sort(key=lambda c: (-increase(c), c.id))
    return comparison


def score_delta(change: QuestionChange) -> Optional[float]:
    if change.top_score_a is None or change.top_score_b is None:
        return None
    return change.top_score_b - change.top_score_a


def metric_deltas(a: Dict, b: Dict) -> List[Tuple[str, Optional[float], Optional[float], Optional[float]]]:
    """(metric, value_a, value_b, b - a) for every numeric metric in either summary, except n."""
    rows = []
    for key in list(dict.fromkeys([*a.keys(), *b.keys()])):
        if key == "n":
            continue
        value_a, value_b = _number(a.get(key)), _number(b.get(key))
        delta = value_b - value_a if value_a is not None and value_b is not None else None
        rows.append((key, value_a, value_b, delta))
    return rows


def _number(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def config_differences(meta_a: Dict, meta_b: Dict) -> List[Tuple[str, object, object]]:
    """Meta keys whose values differ, i.e. what this experiment changed."""
    keys = sorted((meta_a.keys() | meta_b.keys()) - IGNORED_META_KEYS)
    return [(key, meta_a.get(key), meta_b.get(key)) for key in keys if meta_a.get(key) != meta_b.get(key)]


# Formatting
def _rank(rank: Optional[int]) -> str:
    return "-" if rank is None else str(rank)


def _num(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _short(value, limit: int = 60) -> str:
    text = "(missing)" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _question_line(c: QuestionChange) -> str:
    where = f"{c.type}/{c.location or '-'}"
    marker = " *" if c.label_changed else ""
    return f"  {c.id:<10} {where:<18} {c.doc or '-':<24} rank {_rank(c.rank_a):>2} -> {_rank(c.rank_b)}{marker}"


def format_comparison(
    run_a: Run,
    run_b: Run,
    comparison: Comparison,
) -> str:
    k = comparison.k
    lines = [
        f"Comparing  A: {run_a.label} ({run_a.timestamp[:10]}, git {run_a.meta.get('git', '?')})",
        f"       ->  B: {run_b.label} ({run_b.timestamp[:10]}, git {run_b.meta.get('git', '?')})",
        "",
    ]

    differences = config_differences(run_a.meta, run_b.meta)
    lines.append("Config differences" if differences else "Config differences: none besides label/timestamp")
    lines += [f"  {key}: {_short(va)} -> {_short(vb)}" for key, va, vb in differences]
    lines.append("")

    lines.append("Overall (answerable)")
    lines.append(f"  {'metric':<12}{'A':>8}{'B':>8}{'delta':>9}")
    for key, va, vb, delta in metric_deltas(run_a.summary["overall"], run_b.summary["overall"]):
        lines.append(f"  {key:<12}{_num(va):>8}{_num(vb):>8}{_num(delta, signed=True):>9}")
    lines.append("")

    lines.append(
        f"Outcomes at k={k}: {len(comparison.improved)} improved, {len(comparison.regressed)} regressed, "
        f"{len(comparison.rank_changed)} rank changed, {comparison.unchanged} unchanged"
    )
    lines.append("")

    sections = [
        (f"Improved (miss -> hit at k={k})", comparison.improved),
        (f"Regressed (hit -> miss at k={k})", comparison.regressed),
        (f"Rank changed, same outcome at k={k}", comparison.rank_changed),
    ]
    for title, bucket in sections:
        lines.append(f"{title} ({len(bucket)})")
        lines += [_question_line(c) for c in bucket] or ["  (none)"]
        lines.append("")

    lines.append(f"Unanswerable top-1 score, lower is better ({len(comparison.unanswerable)})")
    for c in comparison.unanswerable:
        lines.append(
            f"  {c.id:<10} {c.subtype or '-':<10} {_num(c.top_score_a)} -> {_num(c.top_score_b)}  "
            f"({_num(score_delta(c), signed=True)})"
        )
    if not comparison.unanswerable:
        lines.append("  (none)")
    lines.append("")

    changed = comparison.improved + comparison.regressed + comparison.rank_changed
    if any(c.label_changed for c in changed):
        lines.append("* doc or must_contain changed between runs: the rank change may come from the label, not retrieval.")
    if comparison.only_a:
        lines.append(f"Only in A ({len(comparison.only_a)}): {', '.join(comparison.only_a)}")
    if comparison.only_b:
        lines.append(f"Only in B ({len(comparison.only_b)}): {', '.join(comparison.only_b)}")
    if comparison.answerable_sets_differ:
        lines.append("The answerable question sets differ, so the overall metrics are not directly comparable.")
    elif comparison.only_a or comparison.only_b:
        lines.append("Only unanswerable questions differ; the overall metrics cover the same answerable questions.")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two evaluation runs question by question.")
    parser.add_argument("run_a", help="Earlier run: label (e.g. baseline) or path to a results JSON")
    parser.add_argument("run_b", help="Later run: label or path")
    parser.add_argument("--k", type=int, default=None,
                        help="Cutoff for hit/miss (default: run B's app_top_k, else 3)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args(argv)

    try:
        run_a, records_a = load_run(resolve_run_path(args.run_a, args.results_dir))
        run_b, records_b = load_run(resolve_run_path(args.run_b, args.results_dir))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    k = args.k if args.k is not None else run_b.meta.get("app_top_k", 3)
    if k < 1:
        print("error: --k must be >= 1", file=sys.stderr)
        return 1

    print(format_comparison(run_a, run_b, compare_questions(records_a, records_b, k)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())