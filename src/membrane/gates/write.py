"""
Write gate (i_t) — determines whether a candidate deserves durable memory (§9, A2.3).

Phase 1: deterministic rule-based scoring.
Phase 2+: drop-in replacement via the Gate protocol.
"""

from __future__ import annotations

from datetime import datetime, timezone

from membrane.core.models import Candidate, GateOutput, MemoryRecord, MemoryType, TrustTier


def _now() -> datetime:
    return datetime.now(timezone.utc)


class WriteGate:
    """
    Rule-based write gate (i_t).

    Computes a write score for each candidate in [0, 1].
    Score is driven by: novelty, confidence, importance, trust_tier,
    instruction_likeness (A17), and overlap with existing memories.
    """

    name = "write"
    version = "1.0"

    def __init__(
        self,
        threshold: float = 0.3,
        novelty_weight: float = 0.30,
        confidence_weight: float = 0.25,
        importance_weight: float = 0.25,
        trust_weight: float = 0.20,
        quarantine_below: TrustTier = TrustTier.EXTERNAL_CONTENT,
    ) -> None:
        self.threshold = threshold
        self.novelty_weight = novelty_weight
        self.confidence_weight = confidence_weight
        self.importance_weight = importance_weight
        self.trust_weight = trust_weight
        self.quarantine_below = quarantine_below

        self._trust_scores = {
            TrustTier.SYSTEM: 1.0,
            TrustTier.HUMAN_VERIFIED: 0.95,
            TrustTier.USER_ASSERTED: 0.85,
            TrustTier.TOOL_OUTPUT: 0.65,
            TrustTier.AGENT_INFERRED: 0.50,
            TrustTier.EXTERNAL_CONTENT: 0.30,
            TrustTier.UNTRUSTED: 0.05,
        }

    def evaluate(
        self,
        memories: list[MemoryRecord],
        candidates: list[Candidate],
        context: dict,
    ) -> GateOutput:
        output = GateOutput(gate_name=self.name, gate_version=self.version, threshold=self.threshold)

        existing_contents = {m.content.lower().strip() for m in memories if m.is_retrievable()}

        for i, candidate in enumerate(candidates):
            key = f"c_{i}"
            score, reason = self._score_candidate(candidate, existing_contents, context)
            output.scores[key] = score
            output.reasons[key] = reason
            output.features[key] = {
                "novelty": self._novelty(candidate, existing_contents),
                "confidence": candidate.confidence,
                "importance": candidate.importance,
                "trust": self._trust_scores.get(candidate.trust_tier, 0.5),
                "instruction_like": self._is_instruction_like(candidate.content),
                "type": candidate.type.value,
            }

        return output

    def _score_candidate(
        self,
        candidate: Candidate,
        existing_contents: set[str],
        context: dict,
    ) -> tuple[float, str]:
        # Hard rejection: instruction-like content from low-trust sources
        if self._is_instruction_like(candidate.content) and self._is_low_trust(
            candidate.trust_tier
        ):
            return 0.0, "instruction_like_from_low_trust_source → quarantine"

        novelty = self._novelty(candidate, existing_contents)
        trust = self._trust_scores.get(candidate.trust_tier, 0.5)

        score = (
            self.novelty_weight * novelty
            + self.confidence_weight * candidate.confidence
            + self.importance_weight * candidate.importance
            + self.trust_weight * trust
        )
        score = max(0.0, min(1.0, score))

        if score >= self.threshold:
            reason = (
                f"novelty={novelty:.2f} conf={candidate.confidence:.2f} "
                f"importance={candidate.importance:.2f} trust={trust:.2f} → write"
            )
        else:
            reason = (
                f"novelty={novelty:.2f} conf={candidate.confidence:.2f} "
                f"importance={candidate.importance:.2f} trust={trust:.2f} → reject (below threshold)"
            )
        return score, reason

    def _novelty(self, candidate: Candidate, existing: set[str]) -> float:
        content_lower = candidate.content.lower().strip()
        if content_lower in existing:
            return 0.0
        # Partial overlap check (rough substring)
        for e in existing:
            if len(e) > 10 and (e in content_lower or content_lower in e):
                return 0.3
        return 1.0

    def _is_instruction_like(self, content: str) -> bool:
        """
        Heuristic detector for instruction-like content (A17.2).
        A pluggable classifier may replace this.
        """
        lower = content.lower()
        triggers = [
            "ignore previous",
            "from now on",
            "always respond",
            "never respond",
            "disregard",
            "forget everything",
            "send to",
            "http://",
            "https://",
            "you must",
            "you should always",
            "system override",
        ]
        return any(t in lower for t in triggers)

    def _is_low_trust(self, tier: TrustTier) -> bool:
        from membrane.core.models import TRUST_ORDER

        return TRUST_ORDER.get(tier, 0) <= TRUST_ORDER.get(TrustTier.TOOL_OUTPUT, 3)
