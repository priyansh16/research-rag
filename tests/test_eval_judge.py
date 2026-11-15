"""
Unit tests for the generation judge (evaluation/generation/judge.py).

`claude -p` is never called: a fake returns canned JSON envelopes.
Marked `eval`: evaluation tooling, not app code. Run with `pytest -m eval`.
"""

import json

import pytest

from evaluation.generation import judge
from evaluation.generation.judge import (
    build_item,
    call_usage,
    count_agreement,
    fixed_verdict,
    judge_command,
    judged_row,
    parse_judgments,
    resolved_model,
    summarize_judged,
    validate_judgments,
)
from evaluation.generation.run import build_row, load_rows
from src.services.generation.generation_service import GenerationResult
from src.services.generation.ollama_client import OllamaGeneration
from src.services.generation.prompt_builder import REFUSAL_MESSAGE

pytestmark = pytest.mark.eval

MODEL_ID = "claude-sonnet-test-1"
CONFIG = {"label": "t", "timestamp": "x", "git": "abc", "model": "m", "num_ctx": 8192}


def ctx(doc, text):
    return {"document_name": doc, "chunk_index": 0, "score": 0.5, "text": text}


def gen_row(qid, answer, answerable=True, context=None, must_contain="15,360"):
    q = {"id": qid, "type": "direct" if answerable else "unanswerable", "question": f"Q {qid}?",
         "doc": "a.pdf" if answerable else None, "must_contain": must_contain if answerable else None,
         "subtype": None if answerable else "near_miss", "location": None}
    result = GenerationResult(query="q", answer=answer,
                              context_chunks=context or [ctx("a.pdf", "grid of 15,360")],
                              prompt_eval_count=900, eval_count=10,
                              ollama=OllamaGeneration(text=answer, done_reason="stop"))
    return build_row(q, CONFIG, 100.0, result=result)


def envelope(judgments, model=MODEL_ID, structured=True):
    env = {
        "type": "result", "is_error": False,
        "usage": {"input_tokens": 100, "cache_creation_input_tokens": 1000,
                  "cache_read_input_tokens": 4000, "output_tokens": 300},
        "total_cost_usd": 0.02, "duration_ms": 5000,
        "modelUsage": {"claude-haiku-x": {"outputTokens": 5}, model: {"outputTokens": 300}},
    }
    if structured:
        env["structured_output"] = {"judgments": judgments}
    else:
        env["result"] = "```json\n" + json.dumps({"judgments": judgments}) + "\n```"
    return env


def good_judgment(qid, correctness="correct"):
    return {"id": qid, "claims": [{"text": "15,360 configurations", "supported": True, "chunk": 1},
                                  {"text": "made up", "supported": False, "chunk": None}],
            "correctness": correctness, "reason": "matches"}


# Pieces
def test_fixed_verdicts():
    assert fixed_verdict(gen_row("a", "15,360 [1].")) is None
    assert fixed_verdict(gen_row("a", REFUSAL_MESSAGE)) == "refusal"
    assert fixed_verdict(gen_row("u", REFUSAL_MESSAGE, answerable=False)) == "correct_refusal"
    assert fixed_verdict(gen_row("u", "Paris [1].", answerable=False)) is None  # answered: judged


def test_build_item_numbers_context_like_the_prompt():
    row = gen_row("a", "x [2].", context=[ctx("a.pdf", "one"), ctx("b.pdf", "two")])
    item = build_item(row)
    assert [c["n"] for c in item["context"]] == [1, 2]
    assert item["context"][1] == {"n": 2, "source": "b.pdf", "text": "two"}
    assert item["reference"] == "15,360"


def test_judge_command_has_no_tools_and_no_bare():
    cmd = judge_command("sonnet")
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--bare" not in cmd  # --bare drops OAuth, so it can't use the Pro plan
    assert "--json-schema" in cmd and "--system-prompt" in cmd


def test_parse_structured_and_fenced_text():
    assert parse_judgments(envelope([good_judgment("a")]))[0]["id"] == "a"
    assert parse_judgments(envelope([good_judgment("a")], structured=False))[0]["id"] == "a"
    with pytest.raises(ValueError):
        parse_judgments({"is_error": True, "result": "usage limit"})
    with pytest.raises(ValueError):
        parse_judgments({"result": "not json"})


def test_model_and_usage_from_envelope():
    env = envelope([])
    assert resolved_model(env) == MODEL_ID
    assert call_usage(env) == {"input_tokens": 5100, "output_tokens": 300, "cost_usd": 0.02, "duration_ms": 5000}


def test_validate_judgments():
    items = [build_item(gen_row("a", "x [1].")), build_item(gen_row("b", "y [1]."))]
    assert validate_judgments([good_judgment("a"), good_judgment("b")], items) == []
    assert validate_judgments([good_judgment("a")], items)                             # missing id
    assert validate_judgments([good_judgment("a"), good_judgment("c")], items)         # wrong id
    assert validate_judgments([good_judgment("a"), good_judgment("b", "maybe")], items)
    bad_chunk = good_judgment("b")
    bad_chunk["claims"][0]["chunk"] = 5                                                # only 1 chunk
    assert validate_judgments([good_judgment("a"), bad_chunk], items)


def test_judged_row_faithfulness():
    meta = {"judge_prompt_version": "p", "judge_model": "sonnet", "judge_model_resolved": MODEL_ID}
    row = judged_row(gen_row("a", "x [1]."), meta, judgment=good_judgment("a"))
    assert (row["n_claims"], row["n_supported"], row["faithfulness"]) == (2, 1, 0.5)
    assert row["source"] == "judge"
    no_claims = judged_row(gen_row("a", "x [1]."), meta, judgment={**good_judgment("a"), "claims": []})
    assert no_claims["faithfulness"] is None


def test_summary_lists_strict_vs_judged_disagreements():
    gen = [gen_row("a", "15,360 [1]."), gen_row("b", "fifteen thousand [1].")]
    meta = {"judge_prompt_version": "p", "judge_model": "sonnet"}
    judged = {"a": judged_row(gen[0], meta, judgment=good_judgment("a")),
              "b": judged_row(gen[1], meta, judgment=good_judgment("b", "correct"))}
    s = summarize_judged(gen, judged)
    assert s["answerable"]["correct"] == 2
    assert s["answerable"]["strict_correct"] == 1
    assert [d["id"] for d in s["disagreements"]] == ["b"]
    assert s["faithfulness"]["mean"] == 0.5
    assert len(s["unsupported_claims"]) == 2


# Full runs with a fake `claude`
@pytest.fixture
def run_dir(tmp_path):
    rows = [
        gen_row("a1", "15,360 configurations [1]."),
        gen_row("a2", REFUSAL_MESSAGE),
        gen_row("a3", "Maybe 20 [1]."),
        gen_row("u1", REFUSAL_MESSAGE, answerable=False),
    ]
    path = tmp_path / "2026-11-13_gen_k3.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def fake_claude(calls, fail_first=0, model=MODEL_ID):
    def call(prompt, model_arg):
        calls.append(prompt)
        if len(calls) <= fail_first:
            raise RuntimeError("claude exited 1: boom")
        items = json.loads(prompt.split("\n", 2)[2])
        return envelope([good_judgment(i["id"]) for i in items], model=model)
    return call


def judged_file(run_dir):
    return load_rows(run_dir / "2026-11-13_gen_k3.judged.jsonl")


def test_judges_answers_fixes_refusals_and_resumes(run_dir, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(judge, "call_claude", fake_claude(calls))

    assert judge.main(["--label", "gen_k3", "--results-dir", str(run_dir)]) == 0
    rows = {r["id"]: r for r in judged_file(run_dir)}
    assert len(calls) == 1  # a1 + a3 in one batch
    assert rows["a2"]["correctness"] == "refusal" and rows["a2"]["source"] == "fixed"
    assert rows["u1"]["correctness"] == "correct_refusal"
    assert rows["a1"]["judge_model_resolved"] == MODEL_ID
    out = capsys.readouterr().out
    assert "judged correct 2" in out and "strict must_contain match: 1" in out

    assert judge.main(["--label", "gen_k3", "--results-dir", str(run_dir)]) == 0
    assert len(calls) == 1          # nothing left to judge
    assert len(judged_file(run_dir)) == 4


def test_retry_then_error_rows_are_redone_next_run(run_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(judge, "call_claude", fake_claude(calls, fail_first=2))

    judge.main(["--label", "gen_k3", "--results-dir", str(run_dir)])
    assert len(calls) == 2  # first try + one retry, both failed
    errors = [r for r in judged_file(run_dir) if r["source"] == "error"]
    assert sorted(r["id"] for r in errors) == ["a1", "a3"]

    judge.main(["--label", "gen_k3", "--results-dir", str(run_dir)])
    assert len(calls) == 3
    latest = {r["id"]: r for r in judged_file(run_dir)}  # later rows win
    assert latest["a1"]["source"] == "judge"


def test_max_batches_and_model_pin(run_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(judge, "call_claude", fake_claude(calls))
    judge.main(["--label", "gen_k3", "--results-dir", str(run_dir), "--batch-size", "1", "--max-batches", "1"])
    assert len(calls) == 1

    monkeypatch.setattr(judge, "call_claude", fake_claude(calls, model="claude-sonnet-newer"))
    with pytest.raises(SystemExit, match="Judge model changed"):
        judge.main(["--label", "gen_k3", "--results-dir", str(run_dir), "--batch-size", "1"])


def test_sample_sheet_and_agreement(run_dir, monkeypatch, capsys):
    monkeypatch.setattr(judge, "call_claude", fake_claude([]))
    judge.main(["--label", "gen_k3", "--results-dir", str(run_dir)])

    assert judge.main(["--label", "gen_k3", "--results-dir", str(run_dir), "--sample", "5"]) == 0
    sheet_path = run_dir / "2026-11-13_gen_k3.check.md"
    sheet = sheet_path.read_text()
    assert "## a1" in sheet and "## a3" in sheet and "## a2" not in sheet  # refusals aren't judged
    assert "- UNSUPPORTED: made up" in sheet

    sheet_path.write_text(sheet.replace("- [ ] a1 agree with claims", "- [x] a1 agree with claims"))
    capsys.readouterr()
    assert judge.main(["--label", "gen_k3", "--results-dir", str(run_dir), "--agreement"]) == 0
    out = capsys.readouterr().out
    assert "on claims: 1/2" in out and "on correctness: 0/2" in out


def test_count_agreement_ignores_other_lines():
    sheet = "- [x] a agree with claims\n- [ ] a agree with correctness\n- [x] something else\n"
    assert count_agreement(sheet) == {"claims": {"agree": 1, "total": 1}, "correctness": {"agree": 0, "total": 1}}
