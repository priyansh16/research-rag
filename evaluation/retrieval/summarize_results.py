"""
Regenerate results.md from every run in the results folder.

results.md is a generated file: it is rewritten from scratch on every
call, never appended to. Keep hand-written observations in notes.md.

Usage (from the repo root):
    uv run python -m evaluation.retrieval.summarize_results
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

# run_label lives in evaluation/common.py (shared with the generation
# tooling); imported here so `from summarize_results import run_label` works.
from evaluation.common import run_label


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_OUTPUT = EVAL_DIR / "results.md"

METRIC_COLUMNS = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "doc_hit@3"]
DOC_METRIC = "hit@3"


@dataclass(frozen=True)
class Run:
    label: str
    path: Path
    meta: Dict
    summary: Dict

    @property
    def timestamp(self) -> str:
        return self.meta["timestamp"]



def parse_run(path: Path) -> Run:
    """Parse one results file, raising ValueError if it isn't a usable run."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not valid JSON ({exc})") from exc

    if not isinstance(data, dict):
        raise ValueError("top level is not an object")
    meta, summary = data.get("meta"), data.get("summary")
    if not isinstance(meta, dict) or not isinstance(summary, dict):
        raise ValueError("missing 'meta' or 'summary'")
    if not isinstance(meta.get("timestamp"), str):
        raise ValueError("missing meta.timestamp")
    if not isinstance(summary.get("overall"), dict):
        raise ValueError("missing summary.overall")
    # Optional sections, but if present they must have the right shape.
    for key in ("by_doc", "latency_ms"):
        if summary.get(key) is not None and not isinstance(summary[key], dict):
            raise ValueError(f"summary.{key} is not an object")
    by_doc = summary.get("by_doc") or {}
    if not all(isinstance(value, dict) for value in by_doc.values()):
        raise ValueError("summary.by_doc entries must be objects")

    return Run(label=run_label(path), path=path, meta=meta, summary=summary)


def load_runs(results_dir: Path) -> Tuple[List[Run], List[Tuple[Path, str]]]:
    """Return (runs sorted by timestamp, [(skipped_path, reason), ...])."""
    runs, skipped = [], []

    for path in sorted(results_dir.glob("*.json")):
        try:
            runs.append(parse_run(path))
        except ValueError as exc:
            skipped.append((path, str(exc)))
            logger.warning("Skipping {}: {}", path.name, exc)

    runs.sort(key=lambda r: (r.timestamp, r.label))
    return runs, skipped


# Formatting
def fmt_rate(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"


def fmt_ms(value) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"


def fmt_rerank(meta: Dict) -> str:
    value = meta.get("rerank")
    if value is None or value is False:
        return "off"
    if value is True:
        return "on"
    return str(value)


def fmt_cell(value: Optional[object]) -> str:
    text = "-" if value is None or value == "" else str(value)
    return text.replace("|", "\\|")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(fmt_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


# Tables
def runs_table(runs: Sequence[Run]) -> str:
    headers = ["date", "label", "git", "metric", "rerank", "chunks", "n",
               *METRIC_COLUMNS, "p50 ms", "p95 ms"]
    rows = []
    for run in runs:
        overall = run.summary["overall"]
        latency = run.summary.get("latency_ms") or {}
        rows.append([
            run.timestamp[:10],
            run.label,
            run.meta.get("git"),
            run.meta.get("distance_metric"),
            fmt_rerank(run.meta),
            run.meta.get("indexed_chunks"),
            overall.get("n"),
            *(fmt_rate(overall.get(column)) for column in METRIC_COLUMNS),
            fmt_ms(latency.get("p50")),
            fmt_ms(latency.get("p95")),
        ])
    return markdown_table(headers, rows)


def by_doc_table(runs: Sequence[Run]) -> str:
    by_doc = [run.summary.get("by_doc") or {} for run in runs]
    docs = sorted({doc for summaries in by_doc for doc in summaries})

    rows = [
        [run.timestamp[:10], run.label,
         *(fmt_rate((summaries.get(doc) or {}).get(DOC_METRIC)) for doc in docs)]
        for run, summaries in zip(runs, by_doc)
    ]
    return markdown_table(["date", "label", *docs], rows)


def render_markdown(runs: Sequence[Run], skipped: Sequence[Tuple[Path, str]] = ()) -> str:
    parts = [
        "# Retrieval evaluation results",
        "",
        "_Generated by `summarize_results.py` from `results/*.json`; do not edit by hand, "
        "it is overwritten on every run. Observations go in `notes.md`._",
        "",
    ]

    if not runs:
        parts += ["No runs found.", ""]
    else:
        parts += [
            "## Runs",
            "",
            "Rates are over answerable questions (`n`). `mrr@10` and hit@k use chunk-level "
            "relevance; `doc_hit@3` only requires the right document. Latency is per query, after warm-up.",
            "",
            runs_table(runs),
            "",
            f"## {DOC_METRIC} by document",
            "",
            by_doc_table(runs),
            "",
        ]

    if skipped:
        parts += ["## Skipped files", ""]
        parts += [f"- `{path.name}`: {reason}" for path, reason in skipped]
        parts.append("")

    return "\n".join(parts)


def write_markdown(text: str, output: Path) -> None:
    # Write to a temp file and swap it in, so a crash never leaves a half-written results.md.
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate results.md from results/*.json.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    runs, skipped = load_runs(args.results_dir)
    write_markdown(render_markdown(runs, skipped), args.output)

    print(f"Wrote {args.output} ({len(runs)} runs, {len(skipped)} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
