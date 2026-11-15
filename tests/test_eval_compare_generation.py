"""
Unit tests for evaluation/generation/compare_runs.py. Hand-written runs only.

Marked `eval`: evaluation tooling, not app code. Run with `pytest -m eval`.
"""

import json

import pytest

from evaluation.generation.compare_runs import (
    compare_questions,
    load_run,
    main,
    outcome,
    run_metrics,
)
from evaluation.generation.judge import judge_prompt_version, judged_row
from evaluation.generation.run import build_row
from src.services.generation.generation_service import GenerationResult
from src.services.generation.ollama_client import OllamaGeneration
from src.services.generation.prompt_builder import REFUSAL_MESSAGE

pytestmark = pytest.mark.eval


def config(top_k):
    return {"label": f"k{top_k}", "timestamp": "t", "git": "abc", "model": "m", "num_ctx": 8192, "top_k": top_k}


def row(qid, answer, top_k, answerable=True, prompt_tokens=1000):
    q = {"id": qid, "type": "direct" if answerable else "unanswerable", "question": "Q?",
         "doc": "a.pdf" if answerable else None, "must_contain": "15,360" if answerable else None,
         "subtype": None if answerable else "near_miss", "location": None}
    result = GenerationResult(query="q", answer=answer, context_chunks=[
        {"document_name": "a.pdf", "chunk_index": 0, "score": 0.5, "text": "grid of 15,360"}],
        prompt_eval_count=prompt_tokens, eval_count=20, ollama=OllamaGeneration(text=answer))
    return build_row(q, config(top_k), 1000.0, result=result)


def judgment(qid, correctness, supported=True):
    return {"id": qid, "correctness": correctness, "reason": "r",
            "claims": [{"text": "c", "supported": supported, "chunk": 1 if supported else None}]}


META = {"judge_prompt_version": judge_prompt_version(), "judge_model": "sonnet",
        "judge_model_resolved": "claude-sonnet-test"}


def write_run(dir, name, rows, judgments=None, meta=META):
    (dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if judgments is not None:
        by_id = {j["id"]: j for j in judgments}
        judged = []
        for r in rows:
            if r["refused"]:
                verdict = "refusal" if r["answerable"] else "correct_refusal"
                judged.append(judged_row(r, meta, verdict=verdict))
            else:
                judged.append(judged_row(r, meta, judgment=by_id[r["id"]]))
        (dir / f"{name}.judged.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judged))


@pytest.fixture
def runs(tmp_path):
    a = [row("a1", "15,360 [1].", 3), row("a2", REFUSAL_MESSAGE, 3), row("a3", REFUSAL_MESSAGE, 3),
         row("u1", REFUSAL_MESSAGE, 3, answerable=False)]
    b = [row("a1", "20 [1].", 5, prompt_tokens=2000), row("a2", "fifteen thousand [1].", 5, prompt_tokens=2000),
         row("a3", "Maybe 7 [1].", 5, prompt_tokens=2000), row("u1", "Paris [1].", 5, answerable=False)]
    write_run(tmp_path, "2026-11-13_gen_k3", a, [judgment("a1", "correct")])
    write_run(tmp_path, "2026-11-15_gen_k5", b,
              [judgment("a1", "incorrect", supported=False), judgment("a2", "correct"),
               judgment("a3", "incorrect"), judgment("u1", "incorrect")])
    return tmp_path


def test_outcomes():
    assert outcome(row("a", REFUSAL_MESSAGE, 3), None, use_judge=True) == "refusal"
    assert outcome(row("u", REFUSAL_MESSAGE, 3, answerable=False), None, True) == "correct_refusal"
    assert outcome(row("u", "Paris [1].", 3, answerable=False), None, True) == "answered"
    assert outcome(row("a", "15,360 [1].", 3), {"correctness": "partial"}, True) == "partial"
    assert outcome(row("a", "15,360 [1].", 3), None, use_judge=False) == "correct"


def test_compare_groups_by_score_change(runs):
    a, b = load_run("gen_k3", runs), load_run("gen_k5", runs)
    groups = compare_questions(a, b, use_judge=True)
    assert [c.id for c in groups["improved"]] == ["a2"]            # refusal -> correct
    assert [c.id for c in groups["regressed"]] == ["a1", "u1"]     # correct -> incorrect; refused -> answered
    assert [(c.id, c.before, c.after) for c in groups["changed"]] == [("a3", "refusal", "incorrect")]
    assert groups["regressed"][0].faith_after == 0.0


def test_judged_comparison_uses_judge_and_shows_cost(runs, capsys):
    assert main(["gen_k3", "gen_k5", "--results-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert "Outcomes from: the judge" in out
    assert "top_k: 3 -> 5" in out
    assert "1 improved, 2 regressed, 1 changed" in out
    assert "prompt tokens p50" in out and "+1,000 (+100%)" in out


def test_falls_back_to_strict_when_one_run_is_not_judged(runs, capsys):
    (runs / "2026-11-15_gen_k5.judged.jsonl").unlink()
    assert main(["gen_k3", "gen_k5", "--results-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert "STRICT must_contain match" in out
    assert "faithfulness mean" not in out


def test_different_judge_model_is_not_comparable(runs, capsys):
    b = [row("a1", "20 [1].", 5), row("a2", "x [1].", 5), row("a3", "y [1].", 5),
         row("u1", "Paris [1].", 5, answerable=False)]
    write_run(runs, "2026-11-15_gen_k5", b,
              [judgment(i, "incorrect") for i in ("a1", "a2", "a3", "u1")],
              meta={**META, "judge_model_resolved": "claude-sonnet-other"})
    assert main(["gen_k3", "gen_k5", "--results-dir", str(runs)]) == 1
    assert "different judge model" in capsys.readouterr().err


def test_correct_per_1k_prompt_tokens(runs):
    metrics = run_metrics(load_run("gen_k3", runs), use_judge=True)
    assert metrics["correct"] == 1
    assert metrics["prompt tokens total"] == 4000
    assert metrics["correct per 1k prompt tokens"] == 0.25
