# Generation Experiments

## Summary (96 questions: 79 answerable, 17 unanswerable)

| # | Run | Correct | Partial | Incorrect | Refused (answerable) | Unanswerable refused | Faithfulness | Prompt tokens p50 / p95 | Latency p50 / p95 | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 | **k=3** (`gen_k3`) | **51** | 1 | 10 | 17 (7 with gold in context) | **17/17** | **0.914** | 1,178 / 2,343 | 2.9 s / 5.3 s | **Default, kept** |
| G1 | k=5 (`gen_k5`) | 55 | 3 | 13 | 8 (3 with gold in context) | 17/17 | 0.881 | 1,938 / 3,532 | 4.8 s / 10.0 s | Not adopted: +4 correct for +63% tokens, ~2× p95 latency, lower faithfulness |

Correct / partial / incorrect are the judge's verdicts on the 79 answerable questions (strict `must_contain` match:
42 and 46). 1 question = 1.3 points. Judge by the Improved/Regressed lists in `comparisons/`, not aggregate deltas.
Latency is the full `GenerationService.generate()` call (guardrails, retrieval, rerank, Ollama) on a laptop, after warm-up.

---

## Setup (both runs)

- **Pipeline:** the app's own `GenerationService.generate()`, the call the API makes: guardrails → retrieval
  (vector top 15 → MiniLM rerank → drop ±1 neighbours → top k) → prompt → Ollama. No pipeline rebuilt in the eval.
- **Generator:** `llama3.2:3b` via Ollama, temperature 0, seed 42, `num_ctx` 8192, `num_predict` 512
  (made app defaults for this step, so answers are reproducible). Prompt version `7f77e7700d32`.
- **Questions:** `evaluation/retrieval/questions.jsonl` v2, unchanged during the step.
- **Judge:** `claude-sonnet-5-5` via `claude -p` (Pro plan), judge prompt `51afd5bcc9dd`, 6 answers per call,
  structured JSON output. Same model and prompt for both runs.
- **Runs:** G0 generated at `cfd9429`, G1 at `4a703c7` (only `TOP_K_RESULTS` differs). Both scored with scoring v2.

## Metrics

| Metric | How | Notes |
|---|---|---|
| Correctness (headline) | Judge: correct / partial / incorrect against the reference excerpt | Refusals and blocked queries get a fixed verdict, no judge call |
| Correctness (strict) | `must_contain` in the answer after `normalize_for_match` | Lower bound only; see "Strict matching" below |
| Faithfulness | Judge splits the answer into claims; each supported by the context or not | Context only, never the reference. Supported / all claims |
| Refusal | Answer contains `REFUSAL_MESSAGE`, or has refusal wording and no `[n]` citation | Exact-sentence use is reported separately as prompt compliance |
| Error attribution | Wrong answerable answer: gold chunk in context → generation miss, else retrieval miss | Gold chunk = retrieval relevance rule (doc + `must_contain`) |
| Cost | Ollama `prompt_eval_count` / `eval_count`, wall-clock latency | Truncation flagged near `num_ctx`: none in either run |

**Judge hand check** (G0, 15 answers stratified across supported / partly / unsupported and correct / partial /
incorrect, `results/2026-11-13_gen_k3.check.md`): agreed on claims **15/15**, on correctness **14/15**. The disagreement:
eri_q13. The judge accepted "Växjö had the best CF% with a CF% of 1… champions" as correct; the table gives Växjö's CF%
*rank* as 1, so I'd call it partial. That's the leniency the design expected (the judge sees the reference). The
judge's faithfulness check did flag the claim as unsupported. Small sample: this shows the judge is reliable here, not
that it never errs.

---

## G0: k=3 (baseline)

- **Hypothesis:** with the final retrieval setup (hit@3 64/79), the 3B model answers most questions it has the
  context for, refuses the unanswerable ones, and stays grounded.
- **Result:**
  - **79 answerable:** 64 had the gold chunk in context (= retrieval hit@3; same 15 misses as the retrieval eval).
    - Gold in context (64): **51 correct (80%)**, 6 wrong, 7 refused.
    - Gold missing (15): 10 refused (honest), 4 incorrect, 1 partial. **No correct answer without the gold chunk**:
      the model isn't answering from its own knowledge.
    - Losses split evenly: retrieval 15, generation 13.
  - **17 unanswerable:** 17/17 refused (13 near_miss, 4 off_topic).
  - **Faithfulness 0.914:** 54 of 62 answers fully supported. The unsupported ones are table misreads: gmm_q06
    (3 names from a misaligned OCR table, 0.0), oli_q11 (numbers assigned to the wrong leagues, 0.0), oli_q12 (the wrong
    paper's table, 0.0), gmm_q17 (0.33). Answers given after a retrieval miss stay faithful (5 answers, 0.90): wrong but
    true to their context, e.g. oli_q07 reports 65.1% for three event types instead of the passes-only figure.
  - **Refusals:** 7 with the gold chunk at rank 1 (eri_q06, eri_q09, ras_q06, gmm_q12, oli_q13, eri_q12, ras_q13).
    The exact refusal sentence was used in 20 of 34 refusals; the model often writes "The provided *context* does
    not contain…" (the prompt calls the passages "context").
  - **Cost:** prompt tokens p50 1,178 / p95 2,343 (no truncation risk at `num_ctx` 8192); latency p50 2.9 s, p95 5.3 s.
- **Decision:** reference run; k=3 stays the default.
- **Evidence:** `results/2026-11-13_gen_k3.jsonl`, `.judged.jsonl`, `.judge_calls.jsonl`, `.check.md`

## G1: k=5

- **Hypothesis:** 5 chunks instead of 3 bring in answers ranked 4–5 (gmm_q08, oli_q08, oli_q04 → retrieval hit@5 67;
  at most +3) without the extra chunks hurting answers that already work.
- **Change:** `TOP_K_RESULTS=5` (env var only; `GENERATION_MAX_CONTEXT_CHUNKS` is already 5).
- **Result vs G0:** correct 51 → 55 (**8 improved, 3 regressed**, 4 changed without a score change).
  - **Only 2 of 8 gains come from the extra chunks** (gmm_q08 at rank 4, oli_q04 at rank 5). The others already had the
    gold chunk at k=3: more context made the model refuse less (eri_q09, oli_q13, ras_q13) or answer differently
    (oli_q11 and gmm_q17 tables, bic_q12).
  - **Regressions are cross-paper mixing:** oli_q01 ("1820 games" is LINHAC_2025's number, faithfulness 1.0 → 0.0),
    ras_q11 ("5 clusters for forwards", also LINHAC_2025, 1.0 → 0.5), bic_q10. The hockey papers overlap in topic, and
    2 more chunks bring in more of the wrong paper.
  - **Honest refusals → wrong answers** (same score, listed as "changed"): gmm_q03, gmm_q19, ras_q12 had no gold chunk
    at either k. At k=3 the model refused; at k=5 it built answers from unrelated chunks (gmm_q03 "18-dimensional
    vectors", faithfulness 0.0). Also ras_q06 (gold chunk at rank 1 both times: refused at k=3, wrong at k=5).
  - **Faithfulness 0.914 → 0.881**; partly supported answers 5 → 13; 11 answers less grounded. After a retrieval
    miss, faithfulness fell **0.90 → 0.55**: with more context, the model has more material to build a plausible
    wrong answer from.
  - **Right when it answers:** 51/62 (82%) → 55/71 (77%).
  - **Unanswerable:** 17/17 at both k.
  - **Cost:** prompt tokens +63% (124k → 202k for the run), latency p50 2.9 → 4.8 s (+64%), p95 5.3 → 10.0 s (+90%),
    correct answers per 1k prompt tokens 0.411 → 0.273 (−34%). Two answers neared `num_predict` (gmm_q17 498,
    ras_q05 450). Judging also cost more: ~19.5k vs ~11.4k input tokens per call.
- **Decision rule (set before the run):** switch only if correctness improves on the Improved/Regressed lists,
  faithfulness and refusal accuracy don't drop, and the gain is worth the cost.
  ✅ correctness +4 (8 vs 3) · ❌ faithfulness dropped · ⚠️ unanswerable held, but 4 honest refusals became wrong answers
  · ❌ +63% tokens and ~2× p95 latency for +4 of 79.
- **Decision:** **not adopted; k=3 stays.** Main reason (with the user): latency and cost aren't worth +4 correct.
  k=5 makes the assistant more willing to answer but less trustworthy when it does.
- **Evidence:** `comparisons/2026-11-15_gen_k3_vs_gen_k5.txt`, `results/2026-11-15_gen_k5.jsonl`, `.judged.jsonl`

---

## Findings beyond k

- **Strict matching is unreliable in both directions.** `must_contain` was written to find the chunk, not to check
  answers. It missed 11 correct answers in G0 (bullets instead of "142, 113, 121", "263" vs "263 (out of 345)") and
  passed 1 wrong one (oli_q11: "4,014" present but assigned to the wrong league). Strict 42 vs judged 51.
- **Biggest generation problem: refusing with the answer in hand.** 7 refusals in G0 had the gold chunk at rank 1.
  k=5 recovered 3 of them only as a side effect of more context. Next lever: the refusal instruction in the prompt,
  not k.
- **Tables cause the made-up answers.** All four 0–0.33 faithfulness answers in G0 misread OCR'd tables, matching the
  retrieval weak spot.
- **Data issues found in the hand check:** MLSA23-RASMAS chunks #27 and #29 contain the same paragraph (two of three
  context slots in oli_q08); bic_q12's question is ambiguous (two modified BICs). Both on the backlog.
- **Judge cost:** ~134k tokens per run at k=3, ~245k at k=5, on the Pro plan; 1 of 24 calls failed and was retried
  automatically.
