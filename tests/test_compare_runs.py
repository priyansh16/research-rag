"""
Core tests for compare_runs.py: the classification behind every adopt/reject
decision, and one end-to-end run of main. Hand-written result files only.

Marked `eval`: evaluation tooling, not app code. Run with `pytest -m eval`.
"""

import json

import pytest

from evaluation.retrieval.compare_runs import (
    compare_questions,
    config_differences,
    main,
)

pytestmark = pytest.mark.eval


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


def test_answerable_sets_differ_only_for_answerable_changes():
    base = by_id(q("a1", 1), u("u1", 0.5))

    added_unanswerable = compare_questions(base, by_id(q("a1", 1), u("u1", 0.5), u("u2", 0.6)), k=3)
    added_answerable = compare_questions(base, by_id(q("a1", 1), q("a2", 2), u("u1", 0.5)), k=3)

    assert added_unanswerable.only_b == ["u2"] and not added_unanswerable.answerable_sets_differ
    assert added_answerable.only_b == ["a2"] and added_answerable.answerable_sets_differ


def test_flags_label_changes():
    a = by_id(q("x", None, must_contain="length 13"))
    b = by_id(q("x", 2, must_contain="length 13 for defenders"))

    assert compare_questions(a, b, k=3).improved[0].label_changed


def test_config_differences_ignore_label_and_timestamp():
    meta_a = {"label": "a", "timestamp": "t1", "git": "abc", "rerank": None, "questions_file": "/x"}
    meta_b = {"label": "b", "timestamp": "t2", "git": "def", "rerank": "bge-reranker", "questions_file": "/y"}
    assert config_differences(meta_a, meta_b) == [("git", "abc", "def"), ("rerank", None, "bge-reranker")]


def test_main_by_label(tmp_path, capsys):
    write(tmp_path, "2026-11-01_a.json", make_report([q("x", 5)], timestamp="2026-11-01T00:00:00+00:00"))
    write(tmp_path, "2026-11-02_b.json", make_report([q("x", 1)]))

    assert main(["a", "b", "--results-dir", str(tmp_path)]) == 0
    assert "Outcomes at k=3: 1 improved, 0 regressed" in capsys.readouterr().out
