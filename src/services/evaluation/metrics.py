"""
Pure retrieval-evaluation metrics.

No I/O and no services here, so everything is unit-testable without
loading the embedding model or touching ChromaDB.

Relevance rule: a retrieved chunk is relevant to a question when it comes
from the expected document AND its text contains the question's
`must_contain` string (after normalization, see normalize_for_match).
"""

import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List, Optional, Sequence


# Whitespace plus every dash-like character PDF extraction produces:
# hyphen-minus, soft hyphen, U+2010..U+2015 (hyphens, en/em dash, bar),
# and the minus sign.
_IGNORED_CHARS = re.compile(r"[\s\-­‐-―−]+")


def normalize_for_match(text: str) -> str:
    """
    Normalize text for substring matching only (never for display).

    - NFKC turns ligatures into plain letters ("ﬁnite" -> "finite")
    - casefold makes matching case-insensitive
    - dropping whitespace and dashes makes line-break hyphenation
      ("right- skewed", "member- ship") and spacing differences match
    """
    return _IGNORED_CHARS.sub("", unicodedata.normalize("NFKC", text).casefold())


@dataclass(frozen=True)
class RetrievedChunk:
    document_name: str
    chunk_index: int
    score: float
    text: str


@dataclass(frozen=True)
class QuestionResult:
    """
    One question plus the ranked chunks retrieval returned for it.
    Unanswerable questions have doc and must_contain set to None.
    """

    id: str
    type: str
    location: Optional[str]
    doc: Optional[str]
    must_contain: Optional[str]
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def answerable(self) -> bool:
        return self.doc is not None

    @property
    def rank(self) -> Optional[int]:
        """1-based rank of the first relevant chunk, None if not retrieved."""
        if not self.answerable:
            return None
        return first_relevant_rank(self.retrieved, self.doc, self.must_contain)

    @property
    def doc_rank(self) -> Optional[int]:
        """1-based rank of the first chunk from the expected document."""
        if not self.answerable:
            return None
        return first_doc_rank(self.retrieved, self.doc)

    @property
    def top_score(self) -> Optional[float]:
        return self.retrieved[0].score if self.retrieved else None


# Per-question metrics
def is_relevant(chunk: RetrievedChunk, doc: str, must_contain: str) -> bool:
    return (
        chunk.document_name == doc
        and normalize_for_match(must_contain) in normalize_for_match(chunk.text)
    )


def first_relevant_rank(
    chunks: Sequence[RetrievedChunk],
    doc: str,
    must_contain: str,
) -> Optional[int]:
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk, doc, must_contain):
            return rank
    return None


def first_doc_rank(chunks: Sequence[RetrievedChunk], doc: str) -> Optional[int]:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.document_name == doc:
            return rank
    return None


def hit_at_k(rank: Optional[int], k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank is not None else 0.0


# Aggregates
def rate(flags: Sequence[bool]) -> Optional[float]:
    return round(sum(flags) / len(flags), 4) if flags else None


def summarize(
    results: Sequence[QuestionResult],
    ks: Sequence[int],
    doc_k: int,
) -> Dict:
    """
    hit@k for each k, MRR, and doc-level hit@doc_k over answerable results.

    MRR is truncated at max(ks): a relevant chunk ranked below the deepest
    retrieved position counts as 0, so the key is named mrr@<k_max>.
    """
    answerable = [r for r in results if r.answerable]
    k_max = max(ks)

    summary = {"n": len(answerable)}
    for k in ks:
        summary[f"hit@{k}"] = rate([hit_at_k(r.rank, k) for r in answerable])
    summary[f"mrr@{k_max}"] = (
        round(mean(reciprocal_rank(r.rank) for r in answerable), 4)
        if answerable else None
    )
    summary[f"doc_hit@{doc_k}"] = rate([hit_at_k(r.doc_rank, doc_k) for r in answerable])
    return summary


def group_by(results: Sequence[QuestionResult], attr: str) -> Dict[str, List[QuestionResult]]:
    """Group results by an attribute; None becomes the key 'none'."""
    groups: Dict[str, List[QuestionResult]] = defaultdict(list)
    for r in results:
        value = getattr(r, attr)
        groups["none" if value is None else str(value)].append(r)
    return dict(sorted(groups.items()))


def score_separation(results: Sequence[QuestionResult]) -> Dict:
    """
    Compare top-1 scores of unanswerable questions against the scores of
    relevant chunks for answerable ones. If every unanswerable top score is
    below every relevant-chunk score, a clean "no relevant context"
    threshold exists between them.
    """
    unanswerable_top = [
        r.top_score for r in results
        if not r.answerable and r.top_score is not None
    ]
    relevant_scores = [
        r.retrieved[r.rank - 1].score for r in results
        if r.answerable and r.rank is not None
    ]

    def _round(value: Optional[float]) -> Optional[float]:
        return round(value, 4) if value is not None else None

    max_unanswerable = max(unanswerable_top) if unanswerable_top else None
    min_relevant = min(relevant_scores) if relevant_scores else None

    return {
        "unanswerable_n": len(unanswerable_top),
        "unanswerable_top1_mean": _round(mean(unanswerable_top)) if unanswerable_top else None,
        "unanswerable_top1_max": _round(max_unanswerable),
        "relevant_chunk_n": len(relevant_scores),
        "relevant_chunk_score_min": _round(min_relevant),
        "relevant_chunk_score_median": _round(median(relevant_scores)) if relevant_scores else None,
        "clean_threshold_possible": (
            max_unanswerable < min_relevant
            if max_unanswerable is not None and min_relevant is not None
            else None
        ),
    }


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]
