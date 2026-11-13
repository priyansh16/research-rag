"""
Deterministic generation-evaluation metrics (no LLM judge).

Pure functions over strings and result rows: no services, no I/O, so all of
it is unit-testable without Ollama.

- refusal:     the answer contains REFUSAL_MESSAGE (after normalize_for_match)
- soft refusal: refusal-like wording ("not mentioned", ...) in an answer that
               is NOT a refusal. Listed for review, never counted as a refusal.
- correctness: an answerable question's `must_contain` is in the answer
               (after normalize_for_match)
- gold rank:   1-based position of the first context chunk that is relevant
               under the retrieval rule (expected doc AND must_contain in text)
- attribution: a wrong answerable answer is a retrieval miss when no relevant
               chunk was in the context, otherwise a generation miss
- truncation:  prompt_eval_count within a margin of num_ctx
"""

from typing import Dict, List, Optional, Sequence

from evaluation.common import normalize_for_match
from evaluation.retrieval.metrics import RetrievedChunk, is_relevant, percentile
from src.services.generation.prompt_builder import REFUSAL_MESSAGE


# Refusal-like phrases that don't use the fixed sentence. Checked after
# normalize_for_match, so case and spacing don't matter. Curly apostrophes
# are folded to straight ones first (see _normalize).
SOFT_REFUSAL_PHRASES = (
    "not mentioned",
    "does not mention",
    "do not mention",
    "doesn't mention",
    "does not contain",
    "do not contain",
    "doesn't contain",
    "does not provide",
    "do not provide",
    "doesn't provide",
    "does not specify",
    "doesn't specify",
    "not specified",
    "not provided",
    "not stated",
    "no information",
    "not enough information",
    "cannot be determined",
    "can't be determined",
    "unable to answer",
    "cannot answer",
    "can't answer",
)

# A prompt this close to num_ctx was probably cut: Ollama truncates long
# prompts to fit the window without an error.
TRUNCATION_MARGIN_TOKENS = 64

ATTRIBUTION_CORRECT = "correct"
ATTRIBUTION_RETRIEVAL_MISS = "retrieval_miss"
ATTRIBUTION_GENERATION_MISS = "generation_miss"
ATTRIBUTION_BLOCKED = "blocked"


def _normalize(text: str) -> str:
    return normalize_for_match(text.replace("’", "'"))


_REFUSAL_NEEDLE = _normalize(REFUSAL_MESSAGE)


# Per answer
def is_refusal(answer: Optional[str]) -> bool:
    """The answer starts with or contains the fixed refusal sentence."""
    return bool(answer) and _REFUSAL_NEEDLE in _normalize(answer)


def soft_refusal_phrases(answer: Optional[str]) -> List[str]:
    """Refusal-like phrases in an answer that is not a refusal (empty for refusals)."""
    if not answer or is_refusal(answer):
        return []
    text = _normalize(answer)
    return [phrase for phrase in SOFT_REFUSAL_PHRASES if _normalize(phrase) in text]


def is_correct(answer: Optional[str], must_contain: Optional[str]) -> Optional[bool]:
    """must_contain appears in the answer; None for unanswerable questions."""
    if must_contain is None:
        return None
    if not answer:
        return False
    return normalize_for_match(must_contain) in normalize_for_match(answer)


def gold_rank(context_chunks: Sequence[Dict], doc: Optional[str], must_contain: Optional[str]) -> Optional[int]:
    """1-based position of the first relevant chunk in the context, or None."""
    if doc is None or must_contain is None:
        return None
    for rank, chunk in enumerate(context_chunks, start=1):
        retrieved = RetrievedChunk(
            document_name=chunk.get("document_name"),
            chunk_index=chunk.get("chunk_index"),
            score=chunk.get("score") or 0.0,
            text=chunk.get("text") or "",
        )
        if is_relevant(retrieved, doc, must_contain):
            return rank
    return None


def is_truncated(
    prompt_eval_count: Optional[int],
    num_ctx: int,
    margin: int = TRUNCATION_MARGIN_TOKENS,
) -> Optional[bool]:
    """None when Ollama didn't report the prompt token count."""
    if prompt_eval_count is None:
        return None
    return prompt_eval_count >= num_ctx - margin


def attribute(
    answerable: bool,
    blocked: bool,
    correct: Optional[bool],
    gold_in_context: bool,
) -> Optional[str]:
    """Why an answerable question was answered wrong (None for unanswerable)."""
    if not answerable:
        return None
    if blocked:
        return ATTRIBUTION_BLOCKED
    if correct:
        return ATTRIBUTION_CORRECT
    return ATTRIBUTION_GENERATION_MISS if gold_in_context else ATTRIBUTION_RETRIEVAL_MISS


# Over a run
def _count(rows: Sequence[Dict], predicate) -> int:
    return sum(1 for r in rows if predicate(r))


def _stats(values: Sequence[float]) -> Dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"p50": None, "p95": None, "max": None}
    return {"p50": percentile(values, 50), "p95": percentile(values, 95), "max": max(values)}


def summarize(rows: Sequence[Dict]) -> Dict:
    """Aggregate metrics over the result rows written by run.py."""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    by_subtype: Dict[str, Dict] = {}
    for subtype in sorted({r.get("subtype") or "none" for r in unanswerable}):
        group = [r for r in unanswerable if (r.get("subtype") or "none") == subtype]
        by_subtype[subtype] = {"n": len(group), "refused": _count(group, lambda r: r["refused"])}

    attribution = {
        name: _count(answerable, lambda r, name=name: r["attribution"] == name)
        for name in (
            ATTRIBUTION_CORRECT,
            ATTRIBUTION_RETRIEVAL_MISS,
            ATTRIBUTION_GENERATION_MISS,
            ATTRIBUTION_BLOCKED,
        )
    }

    return {
        "n": len(rows),
        "answerable": {
            "n": len(answerable),
            "correct": _count(answerable, lambda r: r["correct"]),
            "false_refusals": _count(answerable, lambda r: r["refused"]),
            "gold_in_context": _count(answerable, lambda r: r["gold_in_context"]),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "refused": _count(unanswerable, lambda r: r["refused"]),
            "by_subtype": by_subtype,
        },
        "attribution": attribution,
        "soft_refusal_ids": [r["id"] for r in rows if r["soft_refusal_phrases"]],
        "blocked_ids": [r["id"] for r in rows if r["status"] == "blocked"],
        "truncated_ids": [r["id"] for r in rows if r["truncated"]],
        "hit_num_predict_ids": [r["id"] for r in rows if r.get("done_reason") == "length"],
        "latency_ms": _stats([r["latency_ms"] for r in rows]),
        "prompt_tokens": _stats([r["prompt_eval_count"] for r in rows]),
        "output_tokens": _stats([r["eval_count"] for r in rows]),
    }
