# Retrieval Experiments

## Current question set (v2: 96 questions, 79 answerable)

| # | Experiment | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | Verdict |
|---|---|---|---|---|---|---|---|
| — | [Baseline v2](#baseline-v2) | 37 | **52** | 59 | 66 | 0.583 | Reference |
| E0 | [Answer reachability](#e0-answer-reachability) | — | — | — | — | — | Diagnostic: 78/79 answers reachable |
| E2 | [Neighbour dedup](#e2-neighbour-dedup) (window 1) | 37 | **54** | 62 | 68 | 0.597 | Held: +2 at k=3, re-test combined with E3 |
| E2 | [Neighbour dedup](#e2-neighbour-dedup) (window 2) | 37 | **53** | 61 | 66 | 0.589 | Not adopted: drops more unique text than w1 |
| E3 | [Cross-encoder reranker](#e3-cross-encoder-reranker) | | | | | | Planned |
| E4 | [Chunking overlap fix](#e4-chunking-overlap-fix) | | | | | | Planned |

Counts are out of 79 answerable questions; 1 question = 1.3 points, so ±1 question is within noise.
Judge by the Improved/Regressed lists in `comparisons/`, not by aggregate deltas.
hit@3 is the headline metric because the app sends 3 chunks to the LLM; this may move to hit@5 if `top_k` changes.

## Original question set (v1: 61 questions, 44 answerable)

| # | Experiment | hit@1 | hit@3 | hit@5 | MRR@10 | Verdict |
|---|---|---|---|---|---|---|
| — | Baseline (`baseline_reach`) | 19 | **30** | 35 | 0.576 | Reference |
| E0 | Answer reachability | — | — | — | — | Diagnostic: 43/44 answers reachable |
| E1 | [BGE query prefix](#e1-bge-query-prefix) | 18 | **31** | 33 | 0.561 | Not adopted, no effect |
| E2 | Neighbour dedup (window 1) | 19 | **30** | 37 | 0.592 | Tie at k=3, +2 at k=5 (superseded by v2 re-run) |

v1 and v2 numbers are not comparable with each other. E1 was not re-run on v2 (old branch layout, no effect on v1).

---

## Baseline v2

- **Setup:** `bge-small-en-v1.5`, cosine, 313 chunks (1200 chars, 2-element overlap), top_k 3.
- **Question set:** grown from 61 to 96 (79 answerable: 45 direct, 34 paraphrased; 65 body, 13 table, 1 figure;
  17 unanswerable unchanged). New questions written from the PDFs, not from chunks or retrieval output.
- **Matching fix:** `normalize_for_match` now folds accents (NFKD, combining marks dropped). Tables are stored as
  OCR text that loses diacritics ("Växjö" → "Vaxjo"), so eri_q13 was falsely unreachable. Only that question changed.
- **Result:** hit@3 52/79 (0.66), hit@10 66/79. Weakest: tables (hit@3 0.46), paraphrased (0.56), LINHAC_2025 (0.39).
  14 of 27 misses at k=3 rank 4–10 → reranking headroom.
- **Evidence:** `results/2026-11-03_baseline_v2.json`

## E0: Answer reachability

- **Hypothesis:** some misses happen because the answer never made it into any chunk.
- **Change:** diagnostic: check whether each answer appears in any indexed chunk of its paper.
- **Result:** v1 43/44, v2 78/79 reachable. Only oli_q07 is missing (the value is inside a figure image).
  Tables are extracted (as OCR text) but ranked low.
- **Finding:** answers often sit in 3 consecutive chunks: the 2-element overlap fills the top 3 with near-duplicates.
- **Decision:** kept in every run. Ranking, not ingestion, is the bottleneck.

## E1: BGE query prefix

- **Hypothesis:** BGE's recommended query instruction ranks answer chunks higher.
- **Change:** `"Represent this sentence for searching relevant passages: "` on queries only. No re-ingest. Branch `exp/eval_bge_prefix`.
- **Result (v1):** hit@3 +1 (ras_q06, rank 4 → 3), 0 regressed. hit@5 −2. Unanswerable separation unchanged.
- **Decision:** not adopted. Mixed ±1–2 question changes are noise; the same misses remain.
- **Evidence:** `comparisons/2026-11-02_baseline_reach_vs_bge_prefix.txt`

## E2: Neighbour dedup

- **Hypothesis:** dropping chunks adjacent to a higher-ranked chunk from the same paper frees top-3 slots taken by near-duplicates.
- **Change:** `RETRIEVAL_NEIGHBOR_WINDOW` drops chunks within ±1 or ±2 positions of a kept chunk from the same paper.
  Query-time only, no re-ingest. Branch `exp/eval_neighbor_dedup`.
- **Result (v1):** hit@3 unchanged for both windows (w1: 1 improved, 1 regressed; w2: 2 and 2); hit@5 35 → 37.
- **Result (v2):**
  - w1: hit@3 +2 (3 improved: bic_q01, gmm_q12, oli_q14; 1 regressed: bic_q05), hit@5 +3, hit@10 +2,
    doc_hit@3 +3. 14 more answers moved up without crossing k=3 (e.g. gmm_q03, gmm_q06, gmm_q17 enter the top 10).
  - w2: hit@3 +1 (4 improved, 3 regressed: bic_q05, oli_q13, gmm_q04), hit@10 ±0; bic_q10 and eri_q08 drop out of the top 10.
  - hit@1 and unanswerable top-1 scores unchanged by design: the top chunk is never dropped.
- **Failure mode:** bic_q05's answer is in chunks 44–45, next to a higher-ranked chunk 43 without it, so the answer
  chunks were dropped as "duplicates". Query-time dropping can't tell overlapping text from new text; a wider window drops more.
- **Decision:** w2 not adopted. w1 held (not merged, off by default): a small consistent gain, but it works around
  overlap that E4 fixes at ingest. Re-test w1 on top of the reranker in E3; merge only if the combination beats the reranker alone.
- **Evidence:** `comparisons/2026-11-03_baseline_v2_vs_neighbor_w1_v2.txt`, `..._vs_neighbor_w2_v2.txt`
  (v1: `comparisons/2026-11-02_baseline_reach_vs_neighbor_w1.txt`, `..._vs_neighbor_w2.txt`)

## E3: Cross-encoder reranker

- **Hypothesis:** most misses rank 4–10; re-scoring the top ~20 with a cross-encoder moves answer chunks into the top 3.
- **Change:** query-time only, no re-ingest. Branch `exp/eval_reranker`. Also run reranker + neighbour dedup w1.

## E4: Chunking overlap fix

- **Hypothesis:** chunks that don't repeat each other's text stop crowding the top 3, without dropping unique content as E2 did.
- **Change:** smaller overlap, split oversized elements, cap chunk size. Needs a re-ingest into a new collection.
