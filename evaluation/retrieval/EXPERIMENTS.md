# Retrieval Experiments

| # | Experiment | hit@1 | hit@3 | hit@5 | MRR@10 | Verdict |
|---|---|---|---|---|---|---|
| — | [Baseline](#baseline) | 19 | **30** | 35 | 0.576 | Reference |
| E0 | [Answer reachability](#e0-answer-reachability) | — | — | — | — | Diagnostic: 43/44 answers reachable |
| E1 | [BGE query prefix](#e1-bge-query-prefix) | 18 | **31** | 33 | 0.561 | Not adopted, no effect |
| E2 | [Neighbour dedup](#e2-neighbour-dedup) (window 1) | 19 | **30** | 37 | 0.592 | Not adopted: tie at k=3, +2 at k=5 |
| E3 | [Cross-encoder reranker](#e3-cross-encoder-reranker) | | | | | Planned |
| E4 | [Chunking overlap fix](#e4-chunking-overlap-fix) | | | | | Planned |

Counts are out of 44 answerable questions; ±1–2 questions is within noise.
hit@3 is the headline metric because the app sends 3 chunks to the LLM; this may move to hit@5 if `top_k` changes.

---

## Baseline

- **Setup:** `bge-small-en-v1.5`, cosine, 313 chunks (1200 chars, 2-element overlap), top_k 3.
- **Evidence:** `results/2026-11-02_baseline_reach.json`

## E0: Answer reachability

- **Hypothesis:** some misses happen because the answer never made it into any chunk.
- **Change:** diagnostic: check whether each answer appears in any indexed chunk of its paper.
- **Result:** 43/44 reachable. Only oli_q07 is missing (the value is inside a figure image). Tables are extracted but ranked low.
- **Finding:** answers often sit in 3 consecutive chunks: the 2-element overlap fills the top 3 with near-duplicates.
- **Decision:** kept in every run. Ranking, not ingestion, is the bottleneck.

## E1: BGE query prefix

- **Hypothesis:** BGE's recommended query instruction ranks answer chunks higher.
- **Change:** `"Represent this sentence for searching relevant passages: "` on queries only. No re-ingest. Branch `exp/eval_bge_prefix`.
- **Result:** hit@3 +1 (ras_q06, rank 4 → 3), 0 regressed. hit@5 −2. Unanswerable separation unchanged.
- **Decision:** not adopted. Mixed ±1–2 question changes are noise; the same misses remain.
- **Evidence:** `comparisons/2026-11-02_baseline_reach_vs_bge_prefix.txt`

## E2: Neighbour dedup

- **Hypothesis:** dropping chunks adjacent to a higher-ranked chunk from the same paper frees top-3 slots taken by near-duplicates.
- **Change:** `RETRIEVAL_NEIGHBOR_WINDOW` drops chunks within ±1 or ±2 positions of a kept chunk from the same paper. Query-time only, no re-ingest. Branch `exp/eval_neighbor_dedup`.
- **Result:** hit@3 unchanged for both windows (w1: 1 improved, 1 regressed; w2: 2 and 2). Below the cutoff answers moved up consistently (w1: all 10 rank changes upward), so hit@5 35 → 37.
- **Failure mode:** bic_q05's answer is in chunks 44–45, next to a higher-ranked chunk 43 without it, so both answer chunks were dropped as "duplicates". Query-time dropping can't tell overlapping text from new text.
- **Decision:** not adopted. Confirms overlap hurts ranking, but the fix belongs in chunking (E4). Worth revisiting (window 1) if `top_k` moves to 5.
- **Evidence:** `comparisons/2026-11-02_baseline_reach_vs_neighbor_w1.txt`, `..._vs_neighbor_w2.txt`

## E3: Cross-encoder reranker

- **Hypothesis:** most misses rank 4–10; re-scoring the top ~20 with a cross-encoder moves answer chunks into the top 3.
- **Change:** query-time only, no re-ingest. Branch `exp/eval_reranker`.

## E4: Chunking overlap fix

- **Hypothesis:** chunks that don't repeat each other's text stop crowding the top 3, without dropping unique content as E2 did.
- **Change:** smaller overlap, split oversized elements, cap chunk size. Needs a re-ingest into a new collection.
