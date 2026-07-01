"""
Rule-based query guardrails.

Catches obvious prompt-injection / jailbreak attempts before a query
ever reaches retrieval or generation. This is intentionally cheap and
fast (regex/keyword based) — no model call, no added latency.

Designed to be swapped or extended later: `GuardrailsService.validate()`
is the single entrypoint. A future LLM-based classifier step can be
added inside this method (or as an additional check) without changing
how callers use it.
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from src.core.config import settings


class GuardrailViolationType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    SYSTEM_OVERRIDE = "system_override"
    EXCESSIVE_LENGTH = "excessive_length"
    EMPTY_QUERY = "empty_query"


@dataclass
class GuardrailResult:
    is_safe: bool
    violation_type: GuardrailViolationType | None = None
    reason: str | None = None
    matched_pattern: str | None = field(default=None, repr=False)


class GuardrailsService:
    """
    Validates user queries before they're used for retrieval/generation.

    Current implementation: rule-based pattern matching against known
    jailbreak/injection phrasing. This is a first line of defense, not
    a complete one — rule-based checks are bypassable by determined
    adversarial input. The TODO below documents the planned upgrade path.

    TODO (production hardening): add an LLM-based classifier step that
    asks a small/fast model to label the query as safe/unsafe before
    falling back to the rule-based result. Keep the rule-based check
    as a fast-path short-circuit (cheap, no model latency) and only
    invoke the classifier when the rule-based check passes, to keep
    average-case latency low.
    """

    # Patterns aimed at making the model ignore/override its instructions
    # or reveal/replace its system prompt. Kept lowercase; matched against
    # a lowercased, whitespace-normalized version of the query.
    _INJECTION_PATTERNS: list[str] = [
        r"ignore (all )?(previous|prior|above) instructions",
        r"disregard (all )?(previous|prior|above) instructions",
        r"forget (all )?(previous|prior|your) instructions",
        r"you are now (an?|the)",
        r"act as (an? )?(unrestricted|unfiltered|jailbroken|uncensored)",
        r"new instructions?:",
        r"system prompt",
        r"reveal (your|the) (system )?prompt",
        r"override (your|the) (rules|instructions|guidelines)",
        r"pretend (you are|to be)",
        r"jailbreak",
        r"dan mode",
        r"developer mode",
        r"\bsudo\b",
        r"<\s*system\s*>",
        r"\[\s*system\s*\]",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._INJECTION_PATTERNS
        ]
        logger.debug(
            "GuardrailsService initialized with {} patterns",
            len(self._compiled_patterns),
        )

    def validate(self, query: str) -> GuardrailResult:
        """
        Run all guardrail checks against a query.

        Returns a GuardrailResult. Callers should check `is_safe` and,
        if False, short-circuit before calling retrieval/generation.
        """
        if not query or not query.strip():
            return GuardrailResult(
                is_safe=False,
                violation_type=GuardrailViolationType.EMPTY_QUERY,
                reason="Query is empty.",
            )

        if len(query) > settings.GUARDRAILS_MAX_QUERY_LENGTH:
            logger.warning(
                "Query rejected: length {} exceeds max {}",
                len(query),
                settings.GUARDRAILS_MAX_QUERY_LENGTH,
            )
            return GuardrailResult(
                is_safe=False,
                violation_type=GuardrailViolationType.EXCESSIVE_LENGTH,
                reason=(
                    f"Query exceeds maximum length of "
                    f"{settings.GUARDRAILS_MAX_QUERY_LENGTH} characters."
                ),
            )

        normalized = " ".join(query.lower().split())

        for pattern in self._compiled_patterns:
            match = pattern.search(normalized)
            if match:
                logger.warning(
                    "Query rejected by guardrails: pattern matched.",
                )
                return GuardrailResult(
                    is_safe=False,
                    violation_type=GuardrailViolationType.PROMPT_INJECTION,
                    reason=(
                        "Query appears to contain an instruction-override "
                        "or prompt-injection attempt."
                    ),
                    matched_pattern=pattern.pattern,
                )

        return GuardrailResult(is_safe=True)