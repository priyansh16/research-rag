"""
Unit tests for compare_runs.py: pure comparison and formatting over
hand-written result files. No evaluation run, no services.
"""

import json

import pytest

from evaluation.retrieval.compare_runs import (
    compare_questions,
    config_differences,
    format_comparison,
    load_run,
    main,
    metric_deltas,
    resolve_run_path,
)


def q(id, rank, doc="a.pdf", must_contain="needle", top_score=0.7, type="direct", location="body"):
    return {"id": id, "type": type, "location": location, "doc": doc,
            "must_contain": must_contain, "rank": rank, "top_score": top_score}


def u(id, top_score):
    return {"id": id, "type": "unanswerable", "location": None, "doc": None,
            "must_contain": None, "rank": None, "top_score": top_score}


def by_id(*records):
    return {r["id"]: r for r in records}


def make_report(questions, timestamp="2026-11-02T05:00:00+00:00", overall=None, **meta_extra):
    return {
        "meta": {"timestamp": timestamp, "git": "abc1234", "app_top_k": 3,
                 "embedding_model": "bge-small", **meta_extra},
        "summary": {"overall": overall or {"n": 2, "hit@3": 0.5}},
        "questions": questions,
    }


def write(dir_path, name, report):
    path = dir_path / name
    path.write_text(json.dumps(report))
    return path


# compare_questions
def test_classifies_improved_regressed_rank_changed_unchanged():
    a = by_id(q("up", 5), q("down", 2), q("moved", 9), q("same", 1), q("never", None))
    b = by_id(q("up", 1), q("down", None), q("moved", 4), q("same", 1), q("never", None))

    c = compare_questions(a, b, k=3)

    assert [x.id for x in c.improved] == ["up"]
    assert [x.id for x in c.regressed] == ["down"]
    assert [x.id for x in c.rank_changed] == ["moved"]
    assert c.unchanged == 2   # "same" and "never"


def test_k_controls_the_hit_line():
    a, b = by_id(q("x", 5)), by_id(q("x", 4))
    assert [x.id for x in compare_questions(a, b, k=3).rank_changed] == ["x"]
    assert [x.id for x in compare_questions(a, b, k=4).improved] == ["x"]


def test_biggest_movements_listed_first():
    a = by_id(q("small", 9), q("big", 10), q("from_none", None))
    b = by_id(q("small", 8), q("big", 4), q("from_none", 5))

    c = compare_questions(a, b, k=3)

    assert [x.id for x in c.rank_changed] == ["from_none", "big", "small"]


def test_unanswerable_tracked_by_top_score_worst_first():
    a = by_id(u("u1", 0.70), u("u2", 0.70), u("u3", 0.60))
    b = by_id(u("u1", 0.60), u("u2", 0.80), u("u3", None))

    c = compare_questions(a, b, k=3)

    assert [x.id for x in c.unanswerable] == ["u2", "u1", "u3"]
    assert c.unchanged == 0 and not c.improved and not c.regressed


def test_question_set_differences():
    c = compare_questions(by_id(q("old", 1), q("both", 1)), by_id(q("both", 1), q("new", 1)), k=3)
    assert c.only_a == ["old"] and c.only_b == ["new"]


def test_answerable_sets_differ_only_for_answerable_changes():
    base = by_id(q("a1", 1), u("u1", 0.5))

    added_unanswerable = compare_questions(base, by_id(q("a1", 1), u("u1", 0.5), u("u2", 0.6)), k=3)
    added_answerable = compare_questions(base, by_id(q("a1", 1), q("a2", 2), u("u1", 0.5)), k=3)

    assert added_unanswerable.only_b == ["u2"] and not added_unanswerable.answerable_sets_differ
    assert added_answerable.only_b == ["a2"] and added_answerable.answerable_sets_differ


def test_question_set_note_wording(tmp_path):
    path_a = write(tmp_path, "2026-11-01_a.json", make_report([q("a1", 1)], timestamp="2026-11-01T00:00:00+00:00"))
    path_b = write(tmp_path, "2026-11-02_b.json", make_report([q("a1", 1), u("u1", 0.5)]))
    run_a, rec_a = load_run(path_a)
    run_b, rec_b = load_run(path_b)

    text = format_comparison(run_a, run_b, compare_questions(rec_a, rec_b, k=3))

    assert "Only unanswerable questions differ" in text
    assert "not directly comparable" not in text


def test_flags_label_changes():
    a = by_id(q("x", None, must_contain="length 13"))
    b = by_id(q("x", 2, must_contain="length 13 for defenders"))

    change = compare_questions(a, b, k=3).improved[0]

    assert change.label_changed


# Metrics and config
def test_metric_deltas_skip_n_and_handle_missing():
    rows = metric_deltas({"n": 44, "hit@3": 0.5, "mrr@10": 0.4}, {"n": 44, "hit@3": 0.75, "hit@10": 0.9})
    assert rows == [
        ("hit@3", 0.5, 0.75, 0.25),
        ("mrr@10", 0.4, None, None),
        ("hit@10", None, 0.9, None),
    ]


def test_config_differences_ignore_label_and_timestamp():
    meta_a = {"label": "a", "timestamp": "t1", "git": "abc", "rerank": None, "questions_file": "/x"}
    meta_b = {"label": "b", "timestamp": "t2", "git": "def", "rerank": "bge-reranker", "questions_file": "/y"}
    assert config_differences(meta_a, meta_b) == [("git", "abc", "def"), ("rerank", None, "bge-reranker")]


# Resolving and loading
def test_resolve_by_label_or_path(tmp_path):
    path = write(tmp_path, "2026-11-02_baseline.json", make_report([]))
    write(tmp_path, "2026-11-02_pre_baseline.json", make_report([]))  # must not match "baseline"

    assert resolve_run_path("baseline", tmp_path) == path
    assert resolve_run_path(str(path), tmp_path) == path


def test_resolve_errors(tmp_path):
    write(tmp_path, "2026-11-01_dup.json", make_report([]))
    write(tmp_path, "2026-11-02_dup.json", make_report([]))

    with pytest.raises(FileNotFoundError):
        resolve_run_path("missing", tmp_path)
    with pytest.raises(ValueError, match="several files"):
        resolve_run_path("dup", tmp_path)


def test_load_run_keys_records_by_id(tmp_path):
    path = write(tmp_path, "2026-11-02_x.json", make_report([q("a1", 1), {"no_id": True}]))
    run, records = load_run(path)
    assert run.label == "x"
    assert list(records) == ["a1"]


def test_load_run_rejects_unparseable_file(tmp_path):
    path = tmp_path / "2026-11-02_bad.json"
    path.write_text("{")
    with pytest.raises(ValueError):
        load_run(path)


# Formatting and main
def test_format_comparison_sections(tmp_path):
    path_a = write(tmp_path, "2026-11-01_baseline.json", make_report(
        [q("up", 5), q("down", 1, must_contain="old label"), u("u1", 0.77)],
        timestamp="2026-11-01T00:00:00+00:00", overall={"n": 2, "hit@3": 0.5},
    ))
    path_b = write(tmp_path, "2026-11-02_prefix.json", make_report(
        [q("up", 2), q("down", None, must_contain="new label"), u("u1", 0.70)],
        overall={"n": 2, "hit@3": 0.5}, query_prefix="Represent this sentence",
    ))
    run_a, rec_a = load_run(path_a)
    run_b, rec_b = load_run(path_b)

    text = format_comparison(run_a, run_b, compare_questions(rec_a, rec_b, k=3))

    assert "A: baseline" in text and "B: prefix" in text
    assert "query_prefix: (missing) -> Represent this sentence" in text
    assert "Outcomes at k=3: 1 improved, 1 regressed, 0 rank changed, 0 unchanged" in text
    assert any(line.split() == ["hit@3", "0.500", "0.500", "+0.000"] for line in text.splitlines())
    assert "rank  5 -> 2" in text
    assert "rank  1 -> - *" in text              # label changed marker
    assert "0.770 -> 0.700  (-0.070)" in text
    assert "* doc or must_contain changed" in text


def test_main_by_label(tmp_path, capsys):
    write(tmp_path, "2026-11-01_a.json", make_report([q("x", 5)], timestamp="2026-11-01T00:00:00+00:00"))
    write(tmp_path, "2026-11-02_b.json", make_report([q("x", 1)]))

    assert main(["a", "b", "--results-dir", str(tmp_path)]) == 0
    assert "1 improved" in capsys.readouterr().out


def test_main_reports_errors_without_crashing(tmp_path, capsys):
    assert main(["nope", "also-nope", "--results-dir", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_main_rejects_bad_k(tmp_path, capsys):
    write(tmp_path, "2026-11-01_a.json", make_report([]))
    write(tmp_path, "2026-11-02_b.json", make_report([]))
    assert main(["a", "b", "--k", "0", "--results-dir", str(tmp_path)]) == 1