"""
Read gate (o_t) — determines which retrieved memories enter working memory (§8, A2.3).

Phase 1: deterministic multi-factor scoring.
"""

from __future__ import annotations

from datetime import datetime, timezone

from membrane.core.models import (
    GateOutput,
    Hit,
    MemoryRecord,
    MemoryScore,
    MemoryState,
    Query,
    TrustTier,
    TRUST_ORDER,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReadGate:
    """
    Rule-based read gate (o_t).

    Scores retrieved memories on multiple factors (§8, A2.8) and
    applies budget-aware selection (A19).
    """

    name = "read"
    version = "1.0"

    def __init__(
        self,
        threshold: float = 0.2,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.threshold = threshold
        self.weights = weights or {
            "lexical": 0.20,
            "semantic": 0.00,   # zero until embeddings enabled
            "entity_match": 0.15,
            "temporal_match": 0.10,
            "freshness": 0.15,
            "confidence": 0.10,
            "utility": 0.10,
            "access_history": 0.05,
            "strength": 0.10,
            "trust": 0.05,
            "redundancy": -0.10,  # penalty
        }

    def score_hits(
        self, hits: list[Hit], query: Query, context: dict
    ) -> list[tuple[Hit, MemoryScore]]:
        """Score a list of hits from the router."""
        now = context.get("now") or _now()
        scored: list[tuple[Hit, MemoryScore]] = []

        seen_content: set[str] = set()
        for hit in hits:
            mem = hit.memory
            ms = self._compute_score(mem, query, now, seen_content)
            scored.append((hit, ms))
            if ms.final_score >= self.threshold:
                seen_content.add(mem.content.lower()[:100])

        scored.sort(key=lambda x: x[1].final_score, reverse=True)
        return scored

    def _compute_score(
        self,
        memory: MemoryRecord,
        query: Query,
        now: datetime,
        seen: set[str],
    ) -> MemoryScore:
        w = self.weights

        lexical = self._lexical_score(memory, query.text)
        freshness = self._freshness(memory, now)
        confidence = memory.confidence
        utility = memory.utility
        importance = memory.importance
        strength = memory.strength
        trust = TRUST_ORDER.get(memory.trust_tier, 2) / 6.0
        entity_match = self._entity_match(memory, query)
        temporal_match = self._temporal_match(memory, query, now)
        redundancy = 1.0 if memory.content.lower()[:100] in seen else 0.0

        final = (
            w.get("lexical", 0) * lexical
            + w.get("entity_match", 0) * entity_match
            + w.get("temporal_match", 0) * temporal_match
            + w.get("freshness", 0) * freshness
            + w.get("confidence", 0) * confidence
            + w.get("utility", 0) * utility
            + w.get("access_history", 0) * utility  # use utility as proxy
            + w.get("strength", 0) * strength
            + w.get("trust", 0) * trust
            + w.get("redundancy", 0) * redundancy
        )
        final = max(0.0, min(1.0, final))

        return MemoryScore(
            memory_id=memory.id,
            relevance=lexical,
            freshness=freshness,
            confidence=confidence,
            utility=utility,
            importance=importance,
            temporal_match=temporal_match,
            strength=strength,
            trust=trust,
            entity_match=entity_match,
            lexical=lexical,
            semantic=0.0,
            redundancy=redundancy,
            final_score=final,
            breakdown={
                "lexical": w.get("lexical", 0) * lexical,
                "entity_match": w.get("entity_match", 0) * entity_match,
                "temporal_match": w.get("temporal_match", 0) * temporal_match,
                "freshness": w.get("freshness", 0) * freshness,
                "confidence": w.get("confidence", 0) * confidence,
                "utility": w.get("utility", 0) * utility,
                "strength": w.get("strength", 0) * strength,
                "trust": w.get("trust", 0) * trust,
                "redundancy_penalty": w.get("redundancy", 0) * redundancy,
            },
        )

    def _lexical_score(self, memory: MemoryRecord, query_text: str) -> float:
        """Simple token overlap score (no LLM, no embeddings)."""
        if not query_text:
            return 0.0
        query_tokens = set(query_text.lower().split())
        content_tokens = set(memory.content.lower().split())
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & content_tokens)
        return min(1.0, overlap / max(1, len(query_tokens) * 0.5))

    def _freshness(self, memory: MemoryRecord, now: datetime) -> float:
        """Recency score: higher for more recent memories."""
        import math

        age_days = (now - memory.created_at).total_seconds() / 86400.0
        return math.exp(-0.02 * age_days)  # half-life ~35 days

    def _entity_match(self, memory: MemoryRecord, query: Query) -> float:
        if query.entity and memory.structured:
            if memory.structured.entity == query.entity:
                if query.attribute and memory.structured.attribute == query.attribute:
                    return 1.0
                return 0.8
        return 0.0

    def _temporal_match(self, memory: MemoryRecord, query: Query, now: datetime) -> float:
        """Check if the memory is valid at the queried time."""
        target = query.at or now
        vf = memory.valid_from or memory.created_at
        vu = memory.valid_until

        if vf <= target:
            if vu is None or target <= vu:
                return 1.0
            return 0.0
        return 0.0

    def apply_budget(
        self,
        scored: list[tuple[Hit, MemoryScore]],
        limit: int = 10,
        token_budget: int | None = None,
    ) -> list[tuple[Hit, MemoryScore]]:
        """Select top-k memories within token budget (A19)."""
        result = []
        tokens_used = 0
        for hit, score in scored:
            if score.final_score < self.threshold:
                continue
            mem_tokens = len(hit.memory.content.split()) + 10  # rough estimate
            if token_budget and tokens_used + mem_tokens > token_budget:
                break
            result.append((hit, score))
            tokens_used += mem_tokens
            if len(result) >= limit:
                break
        return result
