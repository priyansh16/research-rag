# Retrieval Experiments

## Current question set (v2: 96 questions, 79 answerable)

| # | Experiment | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | p50 ms | Verdict |
|---|---|---|---|---|---|---|---|---|
| — | [Baseline v2](#baseline-v2) (no reranker) | 37 | **52** | 59 | 66 | 0.583 | 21 | Reference for E2, E3 |
| E0 | [Answer reachability](#e0-answer-reachability) | — | — | — | — | — | — | Diagnostic: 78/79 answers reachable |
| E2 | [Neighbour dedup](#e2-neighbour-dedup) w1 | 37 | **54** | 62 | 68 | 0.597 | — | +2 at k=3 (3 up, 1 down) |
| E2 | Neighbour dedup w2 | 37 | **53** | 61 | 66 | 0.589 | — | Not adopted: 3 regressions |
| E3 | [Reranker](#e3-cross-encoder-reranker) bge-reranker-base, 20 cand. | 50 | **60** | 65 | 68 | 0.715 | 941 | Not adopted: 6 regressions, 3× slower |
| E3 | Reranker MiniLM-L-6, 20 cand. | 55 | **61** | 65 | 71 | 0.755 | 301 | +9 at k=3, 0 regressions |
| E3 | Reranker MiniLM-L-6, 30 cand. | 56 | **62** | 65 | 71 | 0.763 | 434 | Same quality, too slow |
| E3 | **Reranker MiniLM-L-6, 15 cand.** (`rerank_minilm_c15`) | 55 | **62** | 66 | 70 | 0.756 | 231 | **Adopted, default** |
| E2+E3 | **Reranker c15 + neighbour dedup w1** (`rerank_c15_neighbor_w1`) | 55 | **64** | 67 | 69 | 0.760 | 228 | **Adopted, default. Final reference** |
| E4a | [Chunking: overlap cap + split long elements](#e4-chunking-overlap-fix) (`e4_rerank`) | 51 | **56** | 60 | 65 | 0.696 | 246 | Not adopted: −6 vs c15 (2 up, 8 down) |
| E4b | E4a + text-layer tables (`e4t_rerank`) | 50 | **56** | 59 | 65 | 0.686 | 254 | Not adopted: ±0 vs E4a (3 up, 3 down) |
| E4b+E2 | E4b + dedup w1 (`e4t_rerank_w1`) | 50 | **56** | 58 | 59 | 0.674 | — | No effect at k=3 without overlap |

Counts are out of 79 answerable questions; 1 question = 1.3 points, so ±1 question is within noise.
Latency is p50 of `search()` on a laptop CPU after warm-up (embedding + Chroma + reranking).
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
- **Result (v2, on top of the E3 reranker, 15 candidates):** w1 hit@3 62 → 64 (2 improved: bic_q06, oli_q14, both
  rank 4 → 3; 0 regressed), hit@5 +1, hit@10 −1, no latency cost. The reranker scores near-identical overlapping
  chunks near-identically, so it can't remove duplicates itself; dedup does. Dedup runs on the reranked pool
  (pool stays at 15), so it can return fewer than 10 results, which slightly understates hit@10.
- **Failure mode:** bic_q05's answer is in chunks 44–45, next to a higher-ranked chunk 43 without it, so the answer
  chunks were dropped as "duplicates". Query-time dropping can't tell overlapping text from new text; a wider window drops more.
  Still present with the reranker: bic_q14 (rank 5 → out).
- **Result (E4 chunks, no overlap):** no change at k=3; hit@10 −6 because neighbours are no longer duplicates, so
  dropping them only loses unique text (and the 15-candidate pool runs short).
- **Decision:** w2 not adopted. w1 cleared the bar (+2, 0 regressions, three consistent runs). It was first merged
  off, pending E4, which would remove overlap at ingest. E4 lost and the overlapping chunks stay, so **w1 is on by
  default** (`RETRIEVAL_NEIGHBOR_WINDOW = 1`). Revisit if chunking ever drops overlap.
- **Evidence:** `comparisons/2026-11-03_rerank_c15_vs_rerank_c15_neighbor_w1.txt`,
  `comparisons/2026-11-03_baseline_v2_vs_neighbor_w1_v2.txt`, `..._vs_neighbor_w2_v2.txt`
  (v1: `comparisons/2026-11-02_baseline_reach_vs_neighbor_w1.txt`, `..._vs_neighbor_w2.txt`)

## E3: Cross-encoder reranker

- **Hypothesis:** most misses rank 4–10; re-scoring a candidate pool with a cross-encoder moves answer chunks into the top 3.
- **Change:** `RETRIEVAL_RERANKER_MODEL` re-scores the top `RETRIEVAL_RERANK_CANDIDATES` vector-search chunks with a
  `sentence_transformers.CrossEncoder` and keeps top_k. Query-time only, no re-ingest. Branch `exp/eval_reranker`.
- **Result (models, 20 candidates, vs baseline_v2):**
  - MiniLM-L-6 (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22M): hit@3 52 → 61 (9 improved, **0 regressed**), hit@1 37 → 55,
    hit@10 66 → 71, MRR 0.583 → 0.755.
  - bge-reranker-base (278M): hit@3 52 → 60 (14 improved, **6 regressed**, incl. direct facts at rank 1 such as
    ras_q03, ras_q09 → rank 4). Scored off-topic questions near 1.0 against noisy number tables
    (unk_q10 "Zuckerberg's net worth" 0.995 on a GMM_BIC parameter table).
  - Biggest gains on LINHAC_2025 tables (gmm_q05/06/12/16/17 to rank 1–2): the cross-encoder reads query and
    OCR'd table together, where separate embeddings match poorly.
- **Result (pool size, MiniLM):** hit@3 15 / 20 / 30 candidates = 62 / 61 / 62; p50 231 / 301 / 434 ms. Flat: each
  size wins one question and loses another (30 rescues oli_q04 from vector rank 21–30; 15 avoids a distractor at 16–20
  that outranked bic_q10). 15 chosen on a tie, for latency.
- **Unanswerable:** reranker scores are unbounded logits (≈ −11 to +5), not cosine. Off-topic questions fall to ≈ −10,
  but near-miss questions (−10.6 … +3.1) still overlap relevant chunks (min −10.4): no clean threshold. Refusal stays a
  generation-side job.
- **Decision:** adopted. MiniLM-L-6 with 15 candidates is the default. `rerank_minilm_c15` is the reference run.
- **Caveats:** pool size depends on the corpus (answers can sit at vector rank 21–30 already); re-check after E4 and
  when papers are added. Pool size was chosen on the evaluation set, so it's recorded as "chosen on a tie", not "better".
  p95 is 340 ms; long chunks (up to 3,447 chars) cost the most, which E4's size cap should reduce.
- **Evidence:** `comparisons/2026-11-03_baseline_v2_vs_rerank_minilm.txt`, `..._vs_rerank_bge_base.txt`,
  `comparisons/2026-11-03_rerank_minilm_vs_c15.txt`, `..._vs_c30.txt`

## E4: Chunking overlap fix

- **Hypothesis:** chunks that don't repeat each other's text stop crowding the top 3, without dropping unique content
  as E2 did; text-layer tables fix OCR-garbled table text.
- **Change (branch `exp/eval_chunking`, re-ingested into new collections; the main collection is untouched):**
  - E4a `research_documents_e4`: `OVERLAP_ELEMENTS=1`, `CHUNK_OVERLAP_MAX_CHARS=300` (carry only short trailing
    elements, never past the size limit), `CHUNK_SPLIT_LONG_ELEMENTS=true` (split elements over 1,200 chars at
    sentence ends). 313 → 202 chunks; max chunk 3,447 → 1,279 chars.
  - E4b `research_documents_e4t`: E4a + `TABLE_TEXT_SOURCE=text_layer` (rebuild unstructured Table elements from the
    PDF text layer, one row per line, ` | ` at column gaps, instead of OCR'd flattened cells).
- **Result E4a:** hit@3 62 → 56 (2 improved, 8 regressed, most from rank 1–2), hit@1 55 → 51. Reachability unchanged
  (78/79), so these are ranking losses, not broken labels.
- **Why:** overlap gave each answer several copies. Chunks containing the answer per question: **2.65 → 1.29** on
  average; questions with the answer in a single chunk: 17 → 68. Each copy is another chance to enter the reranker's
  15 candidates and the top 3; removing overlap removed those chances. Overlap crowds the top 3 with duplicates (E2
  handles that at query time), but it also adds recall, and with a reranker the recall matters more.
  Latency did not drop (p95 340 → 355 ms): the reranker truncates at 512 tokens, so long chunks never cost extra.
- **Result E4b (tables):** ±0 at k=3 (3 up, 3 down). The 3 regressions are the LINHAC_2025 table questions
  (gmm_q06/12/17), from rank 1–2 to outside the top 10. In Table 5: the longer table text pushed its caption
  ("Defenders closest to the centroids…", nearly the question's wording) into another chunk; a junk first row
  (`gg g | / | / | /`) came from figure fragments in the bbox; narrow gaps merged cells. unstructured had already read
  this paper's tables correctly, and the OCR damage was specific to LINHAC_2022_Erik.
- **Caveat on the metric:** hit@k counts a question solved if *any* copy is in the top k and doesn't penalise near-
  duplicates filling the LLM context; that cost would show in generation eval, not here.
- **Decision:** not adopted, neither E4a nor E4b. Old chunking stays (no re-ingest of the main collection). Two
  always-on chunker bug fixes were ported to `feat/evaluation` (no empty chunk for an oversized first element;
  `OVERLAP_ELEMENTS=0` means no overlap), plus the `chunk chars` line in run reports. Untested: split-only (keep the
  2-element overlap, split only oversized elements so their tails get embedded).
- **Evidence:** `comparisons/2026-11-05_rerank_c15_vs_e4_rerank.txt`, `..._e4_rerank_vs_e4t_rerank.txt`,
  `..._e4t_rerank_vs_e4t_rerank_w1.txt`
