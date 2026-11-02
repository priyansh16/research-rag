"""
Unit tests for summarize_results.py: pure file handling and rendering,
no evaluation run, no services.
"""

import json

import pytest

from src.services.evaluation.summarize_results import (
    fmt_rerank,
    load_runs,
    main,
    parse_run,
    render_markdown,
    run_label,
)


def make_report(timestamp="2026-11-02T05:42:51+00:00", git="abc1234", by_doc=None, **meta_extra):
    meta = {
        "label": "ignored-in-favour-of-filename",
        "timestamp": timestamp,
        "git": git,
        "distance_metric": "cosine",
        "indexed_chunks": 313,
        **meta_extra,
    }
    summary = {
        "overall": {
            "n": 44, "hit@1": 0.4318, "hit@3": 0.6818, "hit@5": 0.7956,
            "hit@10": 0.8864, "mrr@10": 0.5762, "doc_hit@3": 0.9091,
        },
        "by_doc": by_doc if by_doc is not None else {
            "a.pdf": {"hit@3": 0.75},
            "b.pdf": {"hit@3": 0.4},
        },
        "latency_ms": {"p50": 22.57, "p95": 77.74, "max": 81.6},
    }
    return {"meta": meta, "summary": summary, "questions": []}


def write(dir_path, name, content):
    path = dir_path / name
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def table_rows(markdown, heading):
    """Data rows (as cell lists) of the table under a '## heading'."""
    section = markdown.split(f"## {heading}\n", 1)[1]
    lines = [line for line in section.split("\n## ", 1)[0].splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


# Labels and parsing
@pytest.mark.parametrize(
    "stem, expected",
    [
        ("2026-11-01_pre_label_fix", "pre_label_fix"),
        ("2026-11-02_baseline", "baseline"),
        ("handmade", "handmade"),
    ],
)
def test_run_label_comes_from_filename(tmp_path, stem, expected):
    assert run_label(tmp_path / f"{stem}.json") == expected


@pytest.mark.parametrize(
    "content, reason",
    [
        ("", "not valid JSON"),                      # zero-byte placeholder
        ("{not json", "not valid JSON"),
        ("[1, 2]", "not an object"),
        (json.dumps({"meta": {}}), "missing 'meta' or 'summary'"),
        (json.dumps({"meta": {}, "summary": {"overall": {}}}), "missing meta.timestamp"),
        (json.dumps({"meta": {"timestamp": "t"}, "summary": {}}), "missing summary.overall"),
        (json.dumps({"meta": {"timestamp": "t"}, "summary": {"overall": {}, "by_doc": []}}),
         "summary.by_doc is not an object"),
    ],
)
def test_parse_run_rejects_unusable_files(tmp_path, content, reason):
    path = write(tmp_path, "2026-11-02_bad.json", content)
    with pytest.raises(ValueError, match=reason):
        parse_run(path)


def test_load_runs_skips_bad_files_and_sorts_by_timestamp(tmp_path):
    write(tmp_path, "2026-11-02_later.json", make_report(timestamp="2026-11-02T09:00:00+00:00"))
    write(tmp_path, "2026-11-01_earlier.json", make_report(timestamp="2026-11-01T17:00:00+00:00"))
    write(tmp_path, "2026-11-02_broken.json", "{oops")
    write(tmp_path, "notes.txt", "not a result")  # non-JSON files are ignored entirely

    runs, skipped = load_runs(tmp_path)

    assert [r.label for r in runs] == ["earlier", "later"]
    assert [(p.name, "not valid JSON" in reason) for p, reason in skipped] == [
        ("2026-11-02_broken.json", True)
    ]


def test_load_runs_empty_folder(tmp_path):
    assert load_runs(tmp_path) == ([], [])


# Rendering
@pytest.mark.parametrize(
    "meta, expected",
    [({}, "off"), ({"rerank": None}, "off"), ({"rerank": False}, "off"),
     ({"rerank": True}, "on"), ({"rerank": "bge-reranker-base"}, "bge-reranker-base")],
)
def test_fmt_rerank(meta, expected):
    assert fmt_rerank(meta) == expected


def test_runs_table_columns_and_values(tmp_path):
    write(tmp_path, "2026-11-02_baseline.json", make_report())
    runs, _ = load_runs(tmp_path)

    rows = table_rows(render_markdown(runs), "Runs")

    assert rows == [[
        "2026-11-02", "baseline", "abc1234", "cosine", "off", "313", "44",
        "0.432", "0.682", "0.796", "0.886", "0.576", "0.909", "22.6", "77.7",
    ]]


def test_runs_table_shows_dash_for_missing_metrics(tmp_path):
    report = make_report(rerank=True)
    del report["summary"]["overall"]["hit@10"]
    del report["summary"]["latency_ms"]
    write(tmp_path, "2026-11-02_partial.json", report)
    runs, _ = load_runs(tmp_path)

    row = table_rows(render_markdown(runs), "Runs")[0]

    assert row[4] == "on"
    assert row[10] == "-"            # hit@10
    assert row[-2:] == ["-", "-"]    # p50, p95


def test_by_doc_table_unions_documents_across_runs(tmp_path):
    write(tmp_path, "2026-11-01_one.json",
          make_report(timestamp="2026-11-01T00:00:00+00:00", by_doc={"a.pdf": {"hit@3": 0.5}}))
    write(tmp_path, "2026-11-02_two.json",
          make_report(timestamp="2026-11-02T00:00:00+00:00", by_doc={"b.pdf": {"hit@3": 1.0}}))
    runs, _ = load_runs(tmp_path)

    markdown = render_markdown(runs)
    header = [l for l in markdown.split("## hit@3 by document\n", 1)[1].splitlines() if l.startswith("|")][0]

    assert header == "| date | label | a.pdf | b.pdf |"
    assert table_rows(markdown, "hit@3 by document") == [
        ["2026-11-01", "one", "0.500", "-"],
        ["2026-11-02", "two", "-", "1.000"],
    ]


def test_render_lists_skipped_files(tmp_path):
    write(tmp_path, "2026-11-02_broken.json", "")
    runs, skipped = load_runs(tmp_path)

    markdown = render_markdown(runs, skipped)

    assert "No runs found." in markdown
    assert "`2026-11-02_broken.json`" in markdown


def test_render_escapes_pipes_in_cells(tmp_path):
    write(tmp_path, "2026-11-02_x.json", make_report(git="a|b"))
    runs, _ = load_runs(tmp_path)
    assert "a\\|b" in render_markdown(runs)


# main
def test_main_regenerates_instead_of_appending(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    output = tmp_path / "results.md"
    output.write_text("stale hand-written content\n")
    write(results_dir, "2026-11-02_baseline.json", make_report())

    assert main(["--results-dir", str(results_dir), "--output", str(output)]) == 0
    first = output.read_text()
    assert main(["--results-dir", str(results_dir), "--output", str(output)]) == 0

    assert "stale hand-written content" not in first
    assert output.read_text() == first
    assert first.count("## Runs") == 1
    assert not (tmp_path / "results.md.tmp").exists()


def test_main_does_not_crash_on_bad_files(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    write(results_dir, "2026-11-02_good.json", make_report())
    write(results_dir, "2026-11-02_bad.json", "{")
    output = tmp_path / "results.md"

    assert main(["--results-dir", str(results_dir), "--output", str(output)]) == 0
    assert "| 2026-11-02 | good |" in output.read_text()
