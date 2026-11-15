"""
Compare two generation runs question by question (e.g. k=3 vs k=5).

Each question gets one outcome per run:
- answerable:   correct / partial / incorrect (from the judge), refusal, blocked
- unanswerable: correct_refusal, or answered
When either run isn't fully judged, both fall back to the strict must_contain
match (correct / incorrect), so the two sides are always scored the same way.

Improved / Regressed use a score (correct 1, partial 0.5, everything else 0
for answerable; refusing 1, answering 0 for unanswerable). Same score but a
different outcome (e.g. refusal -> incorrect) is listed as Changed: an
honest refusal turning into a wrong answer isn't a score change but matters.

Usage (from the repo root), with labels or file paths:
    mkdir -p evaluation/generation/comparisons
    uv run python -m evaluation.generation.compare_runs gen_k3 gen_k5 \\
        | tee evaluation/generation/comparisons/$(date +%F)_gen_k3_vs_gen_k5.txt

"correct per 1k prompt tokens" = judged-correct answers / prompt tokens over
all questions * 1000: how much answer quality each token of context buys.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.common import resolve_run_path
from evaluation.generation.judge import (
    JUDGED_SUFFIX,
    judge_prompt_version,
    latest_judgments,
    sibling,
    summarize_judged,
)
from evaluation.generation.metrics import summarize
from evaluation.generation.run import DEFAULT_OUT_DIR, RESULTS_SUFFIX, load_rows
from evaluation.retrieval.metrics import percentile


SCORES = {"correct": 1.0, "partial": 0.5, "correct_refusal": 1.0}
# Config keys that always differ and say nothing about the setup.
IGNORED_CONFIG = {"label", "timestamp", "questions_file"}


@dataclass
class Run:
    label: str
    path: Path
    rows: List[Dict]
    judged: Dict[str, Dict]   # {} when the run has no complete judgments

    @property
    def config(self) -> Dict:
        return self.rows[0]["config"]

    @property
    def fully_judged(self) -> bool:
        return bool(self.rows) and all(
            r["id"] in self.judged and self.judged[r["id"]]["source"] != "error" for r in self.rows
        )

    def judge_identity(self) -> Optional[Tuple[str, str]]:
        """(resolved judge model, judge prompt version) of the judged rows."""
        for j in self.judged.values():
            if j["source"] == "judge":
                return (j.get("judge_model_resolved"), j["judge_prompt_version"])
        return None


def load_run(arg: str, results_dir: Path) -> Run:
    path = resolve_run_path(arg, results_dir, suffix=RESULTS_SUFFIX)
    rows = load_rows(path)
    if not rows:
        raise ValueError(f"{path} has no rows")
    judged_path = sibling(path, JUDGED_SUFFIX)
    judged = (
        latest_judgments(load_rows(judged_path), rows, judge_prompt_version())
        if judged_path.exists() else {}
    )
    return Run(label=path.name[: -len(RESULTS_SUFFIX)], path=path, rows=rows, judged=judged)


# Outcomes
def outcome(row: Dict, judgment: Optional[Dict], use_judge: bool) -> str:
    if row["status"] == "blocked":
        return "blocked"
    if not row["answerable"]:
        return "correct_refusal" if row["refused"] else "answered"
    if row["refused"]:
        return "refusal"
    if use_judge:
        return judgment["correctness"]
    return "correct" if row["correct"] else "incorrect"


def score(result: str) -> float:
    return SCORES.get(result, 0.0)


@dataclass
class Change:
    id: str
    before: str
    after: str
    gold_before: Optional[int]
    gold_after: Optional[int]
    faith_before: Optional[float]
    faith_after: Optional[float]


def compare_questions(a: Run, b: Run, use_judge: bool) -> Dict[str, List[Change]]:
    rows_b = {r["id"]: r for r in b.rows}
    groups: Dict[str, List[Change]] = {"improved": [], "regressed": [], "changed": []}
    for ra in a.rows:
        rb = rows_b[ra["id"]]
        ja, jb = a.judged.get(ra["id"]), b.judged.get(ra["id"])
        before, after = outcome(ra, ja, use_judge), outcome(rb, jb, use_judge)
        if before == after:
            continue
        change = Change(
            id=ra["id"], before=before, after=after,
            gold_before=ra["gold_rank"], gold_after=rb["gold_rank"],
            faith_before=(ja or {}).get("faithfulness"), faith_after=(jb or {}).get("faithfulness"),
        )
        delta = score(after) - score(before)
        groups["improved" if delta > 0 else "regressed" if delta < 0 else "changed"].append(change)
    return groups


def faithfulness_drops(a: Run, b: Run, threshold: float = 0.0) -> List[Tuple[str, float, float]]:
    """Questions judged in both runs whose faithfulness went down."""
    drops = []
    for qid, ja in a.judged.items():
        jb = b.judged.get(qid)
        fa, fb = ja.get("faithfulness"), (jb or {}).get("faithfulness")
        if fa is not None and fb is not None and fb < fa - threshold:
            drops.append((qid, fa, fb))
    return sorted(drops)


# Metrics table
def run_metrics(run: Run, use_judge: bool) -> Dict[str, Optional[float]]:
    s = summarize(run.rows)
    ans, unans = s["answerable"], s["unanswerable"]
    outcomes = [outcome(r, run.judged.get(r["id"]), use_judge) for r in run.rows if r["answerable"]]
    correct = outcomes.count("correct")
    prompt_tokens = [r["prompt_eval_count"] for r in run.rows if r["prompt_eval_count"] is not None]
    latencies = [r["latency_ms"] for r in run.rows]

    metrics: Dict[str, Optional[float]] = {
        "correct": correct,
        "partial": outcomes.count("partial") if use_judge else None,
        "incorrect": outcomes.count("incorrect"),
        "refusal (answerable)": ans["false_refusals"],
        "  with gold in context": ans["false_refusals_gold_in_context"],
        "unanswerable refused": unans["refused"],
        "gold chunk in context": ans["gold_in_context"],
        "strict must_contain": ans["correct"],
    }
    if use_judge:
        f = summarize_judged(run.rows, run.judged)
        metrics.update({
            "faithfulness mean": f["faithfulness"]["mean"],
            "fully supported answers": f["faithfulness"]["fully_supported"],
            "unsupported claims": len(f["unsupported_claims"]),
        })
    metrics.update({
        "prompt tokens p50": percentile(prompt_tokens, 50),
        "prompt tokens p95": percentile(prompt_tokens, 95),
        "prompt tokens total": sum(prompt_tokens),
        "output tokens p50": percentile([r["eval_count"] for r in run.rows if r["eval_count"] is not None], 50),
        "latency ms p50": percentile(latencies, 50),
        "latency ms p95": percentile(latencies, 95),
        "correct per 1k prompt tokens": round(correct / sum(prompt_tokens) * 1000, 3) if prompt_tokens else None,
        "mean latency s per question": round(mean(latencies) / 1000, 2) if latencies else None,
    })
    return metrics


def config_differences(a: Run, b: Run) -> List[Tuple[str, object, object]]:
    keys = sorted((a.config.keys() | b.config.keys()) - IGNORED_CONFIG)
    return [(k, a.config.get(k), b.config.get(k)) for k in keys if a.config.get(k) != b.config.get(k)]


# Report
def _num(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}" if abs(value) < 10 else f"{value:,.0f}"
    return f"{int(value):,}"


def _delta(a, b) -> str:
    if a is None or b is None:
        return "-"
    d = b - a
    if d == 0:
        return "0"
    pct = f" ({d / a * 100:+.0f}%)" if a else ""
    text = f"{d:+.3f}" if isinstance(d, float) and not float(d).is_integer() and abs(d) < 10 else f"{d:+,.0f}"
    return text + pct


def _gold(rank: Optional[int]) -> str:
    return "-" if rank is None else str(rank)


def _faith(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_comparison(a: Run, b: Run) -> str:
    use_judge = a.fully_judged and b.fully_judged
    lines = [
        f"Comparing  A: {a.label} (git {a.config.get('git')})",
        f"       ->  B: {b.label} (git {b.config.get('git')})",
        "Outcomes from: " + ("the judge (correct / partial / incorrect)" if use_judge
                             else "STRICT must_contain match (not both runs fully judged)"),
    ]
    identity = a.judge_identity() if use_judge else None
    if identity:
        lines.append(f"Judge: {identity[0]} | prompt {identity[1]}")

    diffs = config_differences(a, b)
    lines.append("")
    lines.append("Config differences" if diffs else "Config differences: none besides label/timestamp")
    lines += [f"  {k}: {va} -> {vb}" for k, va, vb in diffs]

    ma, mb = run_metrics(a, use_judge), run_metrics(b, use_judge)
    width = max(len(k) for k in ma) + 2
    lines += ["", f"{'Metric':<{width}}{'A':>12}{'B':>12}   Change"]
    for key in ma:
        lines.append(f"{key:<{width}}{_num(ma[key]):>12}{_num(mb[key]):>12}   {_delta(ma[key], mb[key])}")

    groups = compare_questions(a, b, use_judge)
    n_ans = sum(r["answerable"] for r in a.rows)
    lines += ["", f"Outcomes: {len(groups['improved'])} improved, {len(groups['regressed'])} regressed, "
                  f"{len(groups['changed'])} changed without a score change (of {len(a.rows)} questions, "
                  f"{n_ans} answerable)"]
    titles = {
        "improved": "Improved",
        "regressed": "Regressed",
        "changed": "Changed, same score (e.g. honest refusal -> wrong answer)",
    }
    for key, title in titles.items():
        if groups[key]:
            lines += ["", f"{title} ({len(groups[key])})"]
            lines += [f"  {c.id:<10} {c.before:>15} -> {c.after:<15} gold rank {_gold(c.gold_before)} -> "
                      f"{_gold(c.gold_after)} | faithfulness {_faith(c.faith_before)} -> {_faith(c.faith_after)}"
                      for c in groups[key]]

    if use_judge:
        drops = faithfulness_drops(a, b)
        if drops:
            lines += ["", f"Faithfulness dropped ({len(drops)})"]
            lines += [f"  {qid:<10} {fa:.2f} -> {fb:.2f}" for qid, fa, fb in drops]
    return "\n".join(lines) + "\n"


def check_comparable(a: Run, b: Run) -> List[str]:
    """Problems that make the comparison meaningless (empty when fine)."""
    problems = []
    ids_a, ids_b = [r["id"] for r in a.rows], [r["id"] for r in b.rows]
    if sorted(ids_a) != sorted(ids_b):
        problems.append("the runs cover different questions")
    if any(r.get("scoring_version") != b.rows[0].get("scoring_version") for r in a.rows + b.rows):
        problems.append("rows use different scoring versions; run --rescore on both")
    if a.fully_judged and b.fully_judged and a.judge_identity() != b.judge_identity():
        problems.append(f"different judge model or prompt: {a.judge_identity()} vs {b.judge_identity()}")
    return problems


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two generation runs question by question.")
    parser.add_argument("run_a", help="Earlier run: label (e.g. gen_k3) or path to its .jsonl")
    parser.add_argument("run_b", help="Later run: label or path")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        a, b = load_run(args.run_a, args.results_dir), load_run(args.run_b, args.results_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    problems = check_comparable(a, b)
    if problems:
        print("Not comparable: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(format_comparison(a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
