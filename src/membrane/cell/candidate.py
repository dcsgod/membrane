"""
Candidate generation φ — extracts structured memory candidates from raw input (A2.3, §45).

Phase 1: rule-based / deterministic extraction.
Phase 2+: embedding-assisted or LLM-assisted (proposals only; write gate decides).
"""

from __future__ import annotations

import re
from typing import Any

from membrane.core.models import (
    Candidate,
    MemoryType,
    Source,
    StructuredFields,
    TrustTier,
)


class RuleCandidateGenerator:
    """
    Deterministic candidate generator — no LLM, no embeddings.

    Extracts candidates from raw text using heuristic rules.
    """

    def __init__(self) -> None:
        self._preference_patterns = [
            (r"(user|he|she|they)\s+(prefer[s]?|like[s]?|want[s]?|love[s]?)\s+(.+)", 0),
            (r"(prefer[s]?|like[s]?|want[s]?|love[s]?)\s+(.+)", 1),
            (r"favorite\s+(.+)\s+is\s+(.+)", 2),
        ]

    def generate(
        self,
        content: str,
        context: dict[str, Any],
        source: Source | None = None,
        trust_tier: TrustTier = TrustTier.USER_ASSERTED,
    ) -> list[Candidate]:
        """Generate candidate memories from raw content."""
        if not content or not content.strip():
            return []

        candidates: list[Candidate] = []
        src = source or Source(type="user")

        # Always create one primary candidate from the raw content
        primary = Candidate(
            content=content.strip(),
            type=self._infer_type(content),
            confidence=self._estimate_confidence(content),
            importance=self._estimate_importance(content),
            source=src,
            trust_tier=trust_tier,
            structured=self._extract_structured(content),
        )
        candidates.append(primary)

        return candidates

    def _infer_type(self, content: str) -> MemoryType:
        lower = content.lower()

        # State memory: "X is Y", "price = N"
        if re.search(r"\b(is|=|equals?|costs?|price)\b", lower) and re.search(
            r"\b(currently|now|today|present)\b", lower
        ):
            return MemoryType.STATE

        # Preference
        if re.search(
            r"\b(prefer|like[s]?|want[s]?|love[s]?|favor[s]?|favorite|prefers?)\b", lower
        ):
            return MemoryType.PREFERENCE

        # Episodic: past events
        if re.search(r"\b(yesterday|last week|on monday|bought|purchased|did|happened|went)\b", lower):
            return MemoryType.EPISODIC

        # Working: task-related
        if re.search(r"\b(current task|working on|todo|in progress)\b", lower):
            return MemoryType.WORKING

        # Procedural: "when X, do Y"
        if re.search(r"\b(when|if|whenever|always|never|should|must)\b", lower) and re.search(
            r"\b(do|use|apply|follow|run)\b", lower
        ):
            return MemoryType.PROCEDURAL

        return MemoryType.SEMANTIC

    def _estimate_confidence(self, content: str) -> float:
        lower = content.lower()
        # Hedging language reduces confidence
        hedges = ["might", "maybe", "perhaps", "possibly", "could be", "i think", "not sure"]
        if any(h in lower for h in hedges):
            return 0.6
        # Definitive language increases confidence
        definitive = ["is", "are", "was", "will be", "always", "never", "definitely"]
        if any(d in lower.split() for d in definitive):
            return 0.9
        return 0.8

    def _estimate_importance(self, content: str) -> float:
        lower = content.lower()
        important = ["important", "critical", "essential", "must", "required", "always", "never"]
        if any(i in lower for i in important):
            return 0.8
        if len(content) < 20:
            return 0.3  # Very short, probably not important
        return 0.5

    def _extract_structured(self, content: str) -> StructuredFields | None:
        """Try to extract entity-attribute-value triples."""
        # Pattern: "SKU 123 price is 12.99"
        m = re.search(
            r"(?:sku|product|item|entity|user)[\s_]*([\w\d]+).*?(price|inventory|status|value|attribute)\s*(?:is|=|:)\s*([\w\d\.\$]+)",
            content.lower(),
        )
        if m:
            return StructuredFields(
                entity=m.group(1),
                attribute=m.group(2),
                value=m.group(3),
            )

        # Pattern: "The price of X is Y"
        m = re.search(
            r"(?:the\s+)?([\w]+)\s+of\s+([\w\s]+)\s+is\s+([\w\d\.\$]+)", content.lower()
        )
        if m:
            return StructuredFields(
                attribute=m.group(1),
                entity=m.group(2).strip(),
                value=m.group(3),
            )

        return None
