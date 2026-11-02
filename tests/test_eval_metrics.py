"""
Unit tests for the retrieval evaluation harness.

Pure logic only: no embedding model, no ChromaDB, no Ollama.
"""

import argparse
import json

import pytest

from src.services.evaluation.metrics import (
    QuestionResult,
    RetrievedChunk,
    find_answer_chunks,
    first_doc_rank,
    first_relevant_rank,
    group_by,
    hit_at_k,
    is_relevant,
    normalize_for_match,
    percentile,
    reachability,
    reciprocal_rank,
    score_separation,
    score_separation_by_subtype,
    summarize,
)
from src.services.evaluation.run import (
    DEFAULT_QUESTIONS,
    format_table,
    load_questions,
    output_path,
    parse_ks,
    parse_label,
    question_record,
    validate_question,
)


def chunk(doc="a.pdf", text="", score=0.5, index=0):
    return RetrievedChunk(document_name=doc, chunk_index=index, score=score, text=text)


def answerable(id="q1", doc="a.pdf", must_contain="needle", retrieved=(), type="direct",
               location="body", answer_chunks=None):
    return QuestionResult(
        id=id, type=type, location=location, doc=doc,
        must_contain=must_contain, retrieved=list(retrieved),
        answer_chunks=answer_chunks,
    )


def unanswerable(id="u1", retrieved=(), subtype=None):
    return QuestionResult(
        id=id, type="unanswerable", location=None, doc=None,
        must_contain=None, retrieved=list(retrieved), subtype=subtype,
    )


# normalize_for_match
@pytest.mark.parametrize(
    "chunk_text, must_contain",
    [
        ("ﬁnite mixture models", "finite mixture"),       # ligature
        ("Matt Caito", "matt caito"),                       # case
        ("the data are right- skewed", "right-skewed"),     # line-break hyphenation
        ("varying degrees of member- ship", "varying degrees of membership"),
        ("the 2021–2022 NHL season", "2021-2022 nhl"),      # en dash vs hyphen
        ("length  13\nfor defenders", "length 13 for defenders"),  # whitespace
        ("a total of 15,360 models", "15,360"),             # punctuation kept
    ],
)
def test_normalize_for_match_handles_pdf_artifacts(chunk_text, must_contain):
    assert normalize_for_match(must_contain) in normalize_for_match(chunk_text)


def test_normalize_for_match_keeps_digits_distinct():
    assert normalize_for_match("15,360") not in normalize_for_match("15,361 models")


# Relevance and ranks
def test_is_relevant_requires_expected_doc():
    c = chunk(doc="other.pdf", text="the needle is here")
    assert not is_relevant(c, "a.pdf", "needle")
    assert is_relevant(chunk(text="the needle is here"), "a.pdf", "needle")


def test_first_relevant_rank_is_one_based():
    chunks = [chunk(text="hay"), chunk(text="hay"), chunk(text="NEEDLE")]
    assert first_relevant_rank(chunks, "a.pdf", "needle") == 3


def test_first_relevant_rank_none_when_missing():
    assert first_relevant_rank([chunk(text="hay")], "a.pdf", "needle") is None


def test_first_doc_rank_ignores_text():
    chunks = [chunk(doc="b.pdf"), chunk(doc="a.pdf", text="hay")]
    assert first_doc_rank(chunks, "a.pdf") == 2


@pytest.mark.parametrize(
    "rank, k, expected",
    [(1, 1, True), (3, 3, True), (4, 3, False), (None, 10, False)],
)
def test_hit_at_k(rank, k, expected):
    assert hit_at_k(rank, k) is expected


@pytest.mark.parametrize("rank, expected", [(1, 1.0), (4, 0.25), (None, 0.0)])
def test_reciprocal_rank(rank, expected):
    assert reciprocal_rank(rank) == expected


def test_unanswerable_result_has_no_rank():
    r = unanswerable(retrieved=[chunk(text="needle", score=0.7)])
    assert not r.answerable
    assert r.rank is None and r.doc_rank is None
    assert r.top_score == 0.7


# summarize
def test_summarize_computes_hits_mrr_and_doc_hits():
    results = [
        answerable(id="r1", retrieved=[chunk(text="needle")]),                     # rank 1
        answerable(id="r3", retrieved=[chunk(doc="b.pdf"), chunk(doc="b.pdf"),
                                       chunk(text="needle")]),                     # rank 3
        answerable(id="miss", retrieved=[chunk(doc="b.pdf"), chunk(text="hay")]),  # doc rank 2
        unanswerable(retrieved=[chunk(text="needle")]),                            # excluded
    ]

    summary = summarize(results, ks=[1, 3], doc_k=1)

    assert summary["n"] == 3
    assert summary["hit@1"] == round(1 / 3, 4)
    assert summary["hit@3"] == round(2 / 3, 4)
    assert summary["mrr@3"] == round((1 + 1 / 3 + 0) / 3, 4)
    assert summary["doc_hit@1"] == round(1 / 3, 4)


def test_summarize_with_no_answerable_results():
    summary = summarize([unanswerable()], ks=[1, 5], doc_k=3)
    assert summary == {"n": 0, "hit@1": None, "hit@5": None, "mrr@5": None, "doc_hit@3": None}


def test_group_by_maps_none_to_key():
    results = [answerable(id="a", location="table"), answerable(id="b", location=None), unanswerable()]
    groups = group_by(results, "location")
    assert list(groups) == ["none", "table"]
    assert [r.id for r in groups["none"]] == ["b", "u1"]


# score_separation
def test_score_separation_clean_threshold():
    results = [
        answerable(retrieved=[chunk(text="hay", score=0.9), chunk(text="needle", score=0.8)]),
        unanswerable(retrieved=[chunk(score=0.5)]),
        unanswerable(id="u2", retrieved=[chunk(score=0.6)]),
    ]
    sep = score_separation(results)

    # Uses the relevant chunk's score (0.8), not the top-1 score (0.9).
    assert sep["relevant_chunk_score_min"] == 0.8
    assert sep["unanswerable_top1_max"] == 0.6
    assert sep["unanswerable_top1_mean"] == 0.55
    assert sep["clean_threshold_possible"] is True


def test_score_separation_overlap():
    results = [
        answerable(retrieved=[chunk(text="needle", score=0.5)]),
        unanswerable(retrieved=[chunk(score=0.7)]),
    ]
    assert score_separation(results)["clean_threshold_possible"] is False


def test_score_separation_without_unanswerable():
    sep = score_separation([answerable(retrieved=[chunk(text="needle")])])
    assert sep["unanswerable_n"] == 0
    assert sep["clean_threshold_possible"] is None


# percentile
def test_percentile_nearest_rank():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 100
    assert percentile([5], 95) == 5
    assert percentile([], 50) is None


# Question validation and loading
def valid_row(**overrides):
    row = {"id": "q1", "type": "direct", "location": "body",
           "question": "How many?", "doc": "a.pdf", "must_contain": "42"}
    row.update(overrides)
    return row


def test_validate_question_accepts_valid_rows():
    assert validate_question(valid_row()) == []
    assert validate_question(valid_row(type="unanswerable", location=None, doc=None, must_contain=None)) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "trick"},
        {"question": ""},
        {"doc": None},
        {"must_contain": None},
        {"must_contain": " - "},  # normalizes to "", would match every chunk
        {"type": "unanswerable", "doc": "a.pdf"},
    ],
)
def test_validate_question_rejects_invalid_rows(overrides):
    assert validate_question(valid_row(**overrides))


def write_jsonl(path, rows):
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def test_load_questions_reports_all_problems(tmp_path):
    path = write_jsonl(tmp_path / "q.jsonl", [valid_row(), valid_row(), "{not json", valid_row(id="q2", doc=None)])

    with pytest.raises(ValueError) as exc:
        load_questions(path)

    message = str(exc.value)
    assert "duplicate id" in message
    assert "invalid JSON" in message
    assert "need a 'doc'" in message


def test_load_questions_skips_blank_lines(tmp_path):
    path = write_jsonl(tmp_path / "q.jsonl", [valid_row(), "", valid_row(id="q2")])
    assert [q["id"] for q in load_questions(path)] == ["q1", "q2"]


def test_load_questions_rejects_empty_file(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError):
        load_questions(path)


def test_project_questions_file_is_valid():
    questions = load_questions(DEFAULT_QUESTIONS)
    assert questions


# CLI helpers and output
def test_parse_ks_sorts_and_dedupes():
    assert parse_ks("5,1,3,3") == [1, 3, 5]


@pytest.mark.parametrize("value", ["", "0,3", "a,b"])
def test_parse_ks_rejects_bad_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_ks(value)


def test_parse_label_rejects_path_characters():
    assert parse_label("bge-prefix_v2") == "bge-prefix_v2"
    with pytest.raises(argparse.ArgumentTypeError):
        parse_label("../oops")


def test_output_path_refuses_to_clobber(tmp_path):
    path = output_path(tmp_path, "baseline", overwrite=False)

    path.write_text("")  # zero-byte placeholder is fine
    assert output_path(tmp_path, "baseline", overwrite=False) == path

    path.write_text("{}")
    with pytest.raises(FileExistsError):
        output_path(tmp_path, "baseline", overwrite=False)
    assert output_path(tmp_path, "baseline", overwrite=True) == path


def test_question_record_marks_every_relevant_chunk():
    r = answerable(retrieved=[chunk(text="hay"), chunk(text="needle"), chunk(text="needle again")])
    record = question_record(r)

    assert record["rank"] == 2
    assert [c["relevant"] for c in record["retrieved"]] == [False, True, True]
    assert [c["rank"] for c in record["retrieved"]] == [1, 2, 3]


def test_format_table_renders_missing_values():
    table = format_table("T", {"all": {"n": 2, "hit@1": None, "hit@3": 0.5}})
    assert "hit@1" in table and "-" in table and "0.50" in table


# Answer reachability (experiment 0)
def test_find_answer_chunks_uses_normalized_matching():
    doc_chunks = [(4, "hay"), (0, "the data are right- skewed"), (7, "RIGHT-SKEWED again")]
    assert find_answer_chunks(doc_chunks, "right-skewed") == [0, 7]


def test_find_answer_chunks_none_found():
    assert find_answer_chunks([(0, "hay")], "needle") == []
    assert find_answer_chunks([], "needle") == []


def test_reachable_property():
    assert answerable(answer_chunks=None).reachable is None   # not checked
    assert answerable(answer_chunks=[]).reachable is False
    assert answerable(answer_chunks=[3]).reachable is True
    assert unanswerable().reachable is None


def test_reachability_summary():
    results = [
        answerable(id="ok", answer_chunks=[1]),
        answerable(id="lost", answer_chunks=[]),
        answerable(id="unchecked", answer_chunks=None),
        unanswerable(),
    ]
    assert reachability(results) == {
        "n": 2, "reachable": 1, "rate": 0.5, "unreachable_ids": ["lost"],
    }


def test_reachable_but_not_retrieved_is_a_ranking_miss():
    # Answer exists in chunk 9 but retrieval returned other chunks.
    r = answerable(retrieved=[chunk(text="hay")], answer_chunks=[9])
    assert r.reachable is True and r.rank is None


# Unanswerable subtypes
def test_score_separation_by_subtype():
    results = [
        answerable(retrieved=[chunk(text="needle", score=0.65)]),
        unanswerable(id="near", subtype="near_miss", retrieved=[chunk(score=0.77)]),
        unanswerable(id="off", subtype="off_topic", retrieved=[chunk(score=0.40)]),
        unanswerable(id="untagged", retrieved=[chunk(score=0.50)]),
    ]

    by_subtype = score_separation_by_subtype(results)

    assert list(by_subtype) == ["near_miss", "none", "off_topic"]
    # Every subtype is compared with the same relevant-chunk scores.
    assert all(s["relevant_chunk_score_min"] == 0.65 for s in by_subtype.values())
    assert by_subtype["near_miss"]["unanswerable_top1_max"] == 0.77
    assert by_subtype["near_miss"]["clean_threshold_possible"] is False
    assert by_subtype["off_topic"]["clean_threshold_possible"] is True
    assert by_subtype["none"]["unanswerable_n"] == 1


def test_score_separation_by_subtype_without_unanswerable():
    assert score_separation_by_subtype([answerable(retrieved=[chunk(text="needle")])]) == {}


def test_validate_question_subtype():
    unans = dict(type="unanswerable", location=None, doc=None, must_contain=None)
    assert validate_question(valid_row(**unans, subtype="near_miss")) == []
    assert validate_question(valid_row(**unans, subtype="off_topic")) == []
    assert validate_question(valid_row(**unans, subtype="trick"))
    assert validate_question(valid_row(subtype="near_miss"))   # answerable can't have one


def test_project_unanswerable_questions_are_all_tagged():
    unanswerable_rows = [q for q in load_questions(DEFAULT_QUESTIONS) if q["type"] == "unanswerable"]
    assert unanswerable_rows
    assert all(q.get("subtype") in {"near_miss", "off_topic"} for q in unanswerable_rows)


def test_question_record_includes_reachability():
    record = question_record(answerable(retrieved=[chunk(text="needle")], answer_chunks=[0, 5]))
    assert record["reachable"] is True
    assert record["answer_chunks"] == [0, 5]
