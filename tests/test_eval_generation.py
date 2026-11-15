"""
Unit tests for the generation evaluation tooling and the shared eval helpers.

Pure logic only: no embedding model, no ChromaDB, no Ollama.
Marked `eval`: evaluation tooling, not app code. Run with `pytest -m eval`.
"""

import json

import pytest

from evaluation.common import output_path, resolve_run_path, run_label
from evaluation.generation.metrics import (
    SCORING_VERSION,
    attribute,
    gold_rank,
    is_correct,
    is_exact_refusal,
    is_refusal,
    is_truncated,
    soft_refusal_phrases,
    summarize,
)
from evaluation.generation.run import build_row, format_summary, load_rows, main
from src.services.generation.generation_service import GenerationResult
from src.services.generation.ollama_client import OllamaGeneration
from src.services.generation.prompt_builder import REFUSAL_MESSAGE

pytestmark = pytest.mark.eval


# Refusals
def test_refusal_exact_sentence():
    assert is_refusal(REFUSAL_MESSAGE)


def test_refusal_ignores_case_spacing_and_extra_text():
    answer = "  THE provided documents do not contain enough\ninformation to answer this question. [1]"
    assert is_refusal(answer)
    assert is_refusal("Sorry. " + REFUSAL_MESSAGE)


@pytest.mark.parametrize("answer", [None, "", "The forwards were split into 5 clusters [1]."])
def test_not_a_refusal(answer):
    assert not is_refusal(answer)


# Reworded refusals seen in G0: refusal wording and no citation.
@pytest.mark.parametrize("answer", [
    "The provided context does not contain enough information to answer this question.",
    "The provided context does not contain information about the salaries of the players.",
    "I don't have enough information to answer this question.",
    "The answer cannot be determined from the provided context.",
    "The SHL CF/60 in the 19/20 season is not explicitly stated in the provided context.",
    "The text doesn’t mention hardware.",  # curly apostrophe
])
def test_reworded_refusal_without_citation_counts(answer):
    assert is_refusal(answer)
    assert not is_exact_refusal(answer)
    assert soft_refusal_phrases(answer) == []


def test_exact_refusal_is_flagged_separately():
    assert is_exact_refusal(REFUSAL_MESSAGE)
    assert is_exact_refusal("Sorry. " + REFUSAL_MESSAGE)


def test_cited_answer_with_refusal_wording_is_soft_not_refusal():
    answer = ("According to Table 5 in [1], D3.19-24 has 42 players. The specific defenders "
              "are not explicitly listed in the provided context.")
    assert not is_refusal(answer)
    assert soft_refusal_phrases(answer) == ["not explicitly"]
    assert soft_refusal_phrases("They used 15,360 configurations [1].") == []


# Correctness and context
def test_correct_uses_normalized_match():
    assert is_correct("They evaluated 15,360 configurations [1].", "15,360")
    assert is_correct("Vectors have length 13 for\ndefenders [2].", "length 13 for defenders")
    assert is_correct("Players near the centroid include Matt Caito.", "matt caito")
    assert is_correct("They evaluated 15360 configurations.", "15,360") is False
    assert is_correct(None, "15,360") is False
    assert is_correct("anything", None) is None


def ctx(doc, text, index=0):
    return {"document_name": doc, "chunk_index": index, "score": 0.5, "text": text}


def test_gold_rank_needs_doc_and_answer_text():
    context = [ctx("b.pdf", "the needle"), ctx("a.pdf", "no"), ctx("a.pdf", "a Needle here")]
    assert gold_rank(context, "a.pdf", "needle") == 3
    assert gold_rank(context, "c.pdf", "needle") is None
    assert gold_rank(context, None, None) is None


@pytest.mark.parametrize("count, expected", [(None, None), (8000, False), (8128, True), (8192, True)])
def test_truncation_margin(count, expected):
    assert is_truncated(count, 8192, margin=64) is expected


@pytest.mark.parametrize("answerable, blocked, correct, gold, expected", [
    (False, False, None, False, None),
    (True, True, False, False, "blocked"),
    (True, False, True, False, "correct"),
    (True, False, False, True, "generation_miss"),
    (True, False, False, False, "retrieval_miss"),
])
def test_attribution(answerable, blocked, correct, gold, expected):
    assert attribute(answerable, blocked, correct, gold) == expected


# Rows and summary
CONFIG = {
    "label": "t", "timestamp": "2026-11-13T00:00:00+00:00", "git": "abc", "model": "m",
    "temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 512, "top_k": 3,
    "max_context_chunks": 5, "prompt_version": "p", "reranker": "r", "neighbor_window": 1,
}

ANSWERABLE = {"id": "a1", "type": "direct", "location": "body", "question": "How many?",
              "doc": "a.pdf", "must_contain": "15,360"}
UNANSWERABLE = {"id": "u1", "type": "unanswerable", "subtype": "near_miss", "location": None,
                "question": "Salary?", "doc": None, "must_contain": None}


def result(answer, context, prompt_tokens=900, done_reason="stop"):
    return GenerationResult(
        query="q", answer=answer, context_chunks=context,
        prompt_eval_count=prompt_tokens, eval_count=20,
        ollama=OllamaGeneration(text=answer, prompt_eval_count=prompt_tokens, eval_count=20,
                                done_reason=done_reason),
    )


def test_build_row_correct_answer_with_gold_in_context():
    row = build_row(ANSWERABLE, CONFIG, 1234.567,
                    result=result("15,360 configurations [1].", [ctx("a.pdf", "grid of 15,360")]))
    assert row["correct"] is True
    assert row["gold_rank"] == 1
    assert row["attribution"] == "correct"
    assert row["refused"] is False
    assert row["truncated"] is False
    assert row["latency_ms"] == 1234.57
    assert row["context"][0]["text"] == "grid of 15,360"
    assert row["config"] is CONFIG


def test_build_row_false_refusal_is_generation_miss():
    row = build_row(ANSWERABLE, CONFIG, 10.0,
                    result=result(REFUSAL_MESSAGE, [ctx("a.pdf", "grid of 15,360")]))
    assert row["refused"] is True
    assert row["correct"] is False
    assert row["attribution"] == "generation_miss"


def test_build_row_blocked():
    row = build_row(ANSWERABLE, CONFIG, 1.0, result=None, blocked_reason="injection")
    assert row["status"] == "blocked"
    assert row["refused"] is False
    assert row["attribution"] == "blocked"
    assert row["prompt_eval_count"] is None
    assert row["truncated"] is None


def test_summary_counts_and_report():
    rows = [
        build_row(ANSWERABLE, CONFIG, 100.0,
                  result=result("15,360 [1].", [ctx("a.pdf", "15,360")])),
        build_row({**ANSWERABLE, "id": "a2"}, CONFIG, 200.0,
                  result=result("It is not mentioned.", [ctx("b.pdf", "other")], prompt_tokens=8150)),
        build_row({**ANSWERABLE, "id": "a3"}, CONFIG, 250.0,
                  result=result(REFUSAL_MESSAGE, [ctx("a.pdf", "grid of 15,360")])),
        build_row({**ANSWERABLE, "id": "a4"}, CONFIG, 260.0,
                  result=result("It is 7 [1], the rest is not mentioned.", [ctx("b.pdf", "other")])),
        build_row(UNANSWERABLE, CONFIG, 300.0, result=result(REFUSAL_MESSAGE, [ctx("a.pdf", "x")])),
        build_row({**UNANSWERABLE, "id": "u2", "subtype": "off_topic"}, CONFIG, 400.0,
                  result=result("Paris [1].", [ctx("a.pdf", "x")], done_reason="length")),
    ]
    s = summarize(rows)

    assert s["answerable"] == {
        "n": 4, "correct": 1, "false_refusals": 2,
        "false_refusals_gold_in_context": 1, "false_refusals_retrieval_miss": 1,
        "gold_in_context": 2,
    }
    assert s["unanswerable"]["refused"] == 1
    assert s["unanswerable"]["by_subtype"] == {"near_miss": {"n": 1, "refused": 1},
                                               "off_topic": {"n": 1, "refused": 0}}
    assert s["refusals"] == {"n": 3, "exact_sentence": 2}
    assert s["reworded_refusal_ids"] == ["a2"]
    assert s["attribution"] == {"correct": 1, "retrieval_miss": 2, "generation_miss": 1, "blocked": 0}
    assert s["soft_refusal_ids"] == ["a4"]
    assert s["truncated_ids"] == ["a2"]
    assert s["hit_num_predict_ids"] == ["u2"]
    assert s["latency_ms"]["max"] == 400.0

    report = format_summary(s, rows, CONFIG)
    assert "Unanswerable but answered (1)" in report
    assert "Reworded refusals to review" in report
    assert "Soft refusals to review" in report


# Rescoring
def test_rescore_recomputes_scores_in_place_and_keeps_answers(tmp_path, capsys):
    row = build_row(UNANSWERABLE, CONFIG, 300.0, result=result(
        "The provided context does not contain enough information to answer this question.",
        [ctx("a.pdf", "x")]))
    # What scoring v1 wrote for this answer: not a refusal, no refusal_exact field.
    v1_row = {k: v for k, v in row.items() if k != "refusal_exact"}
    v1_row.update(refused=False, soft_refusal_phrases=["does not contain"], scoring_version=1)
    path = tmp_path / "2026-11-13_gen_k3.jsonl"
    path.write_text(json.dumps(v1_row) + "\n")

    assert main(["--rescore", "gen_k3", "--out-dir", str(tmp_path)]) == 0

    [new] = load_rows(path)
    assert new["refused"] is True
    assert new["refusal_exact"] is False
    assert new["soft_refusal_phrases"] == []
    assert new["scoring_version"] == SCORING_VERSION
    assert new["answer"] == v1_row["answer"]
    assert new["context"] == v1_row["context"]
    out = capsys.readouterr().out
    assert "refused" in out and "u1" in out


def test_rescore_twice_changes_nothing(tmp_path, capsys):
    path = tmp_path / "2026-11-13_gen_k3.jsonl"
    row = build_row(ANSWERABLE, CONFIG, 1.0, result=result("15,360 [1].", [ctx("a.pdf", "15,360")]))
    path.write_text(json.dumps(row) + "\n")

    assert main(["--rescore", "gen_k3", "--out-dir", str(tmp_path)]) == 0
    assert load_rows(path) == [row]
    assert "Changed fields:\n  none" in capsys.readouterr().out


# Shared run naming
def test_run_label_by_suffix(tmp_path):
    assert run_label(tmp_path / "2026-11-13_baseline.json") == "baseline"
    assert run_label(tmp_path / "2026-11-13_gen_k3.jsonl", ".jsonl") == "gen_k3"
    assert run_label(tmp_path / "2026-11-13_gen_k3.judged.jsonl", ".jsonl") == "gen_k3.judged"
    assert run_label(tmp_path / "2026-11-13_gen_k3.judged.jsonl", ".judged.jsonl") == "gen_k3"


def test_resolve_generation_label_ignores_judged_file(tmp_path):
    (tmp_path / "2026-11-13_gen_k3.jsonl").write_text("{}\n")
    (tmp_path / "2026-11-13_gen_k3.judged.jsonl").write_text("{}\n")
    assert resolve_run_path("gen_k3", tmp_path, ".jsonl").name == "2026-11-13_gen_k3.jsonl"
    assert resolve_run_path("gen_k3", tmp_path, ".judged.jsonl").name == "2026-11-13_gen_k3.judged.jsonl"


def test_resolve_label_matching_several_files_is_an_error(tmp_path):
    (tmp_path / "2026-11-13_gen_k3.jsonl").write_text("{}\n")
    (tmp_path / "2026-11-14_gen_k3.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="several files"):
        resolve_run_path("gen_k3", tmp_path, ".jsonl")
    with pytest.raises(FileNotFoundError):
        resolve_run_path("gen_k5", tmp_path, ".jsonl")


def test_output_path_suffix(tmp_path):
    assert output_path(tmp_path, "gen_k3", overwrite=False, suffix=".jsonl").name.endswith("_gen_k3.jsonl")
