"""
Update gate (u_t) — merge/revise existing memories with new candidates (§10, A2.3).

Detects when a candidate overlaps an existing memory and delegates
to the appropriate Delta strategy.
"""

from __future__ import annotations

from membrane.core.models import Candidate, GateOutput, MemoryRecord, MemoryType


class UpdateGate:
    """
    Rule-based update gate (u_t).

    Identifies (existing_memory, candidate) pairs that should be
    merged rather than written as separate memories.
    Scores are in [0, 1]; high score = should merge.
    """

    name = "update"
    version = "1.0"

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def evaluate(
        self,
        memories: list[MemoryRecord],
        candidates: list[Candidate],
        context: dict,
    ) -> GateOutput:
        output = GateOutput(gate_name=self.name, gate_version=self.version, threshold=self.threshold)

        for i, candidate in enumerate(candidates):
            for memory in memories:
                if not memory.is_retrievable():
                    continue
                score, reason = self._overlap_score(memory, candidate)
                key = f"{memory.id}::{i}"
                output.scores[key] = score
                output.reasons[key] = reason

        return output

    def _overlap_score(self, memory: MemoryRecord, candidate: Candidate) -> tuple[float, str]:
        """
        Score how much a candidate overlaps an existing memory.
        Exact structured match: 1.0 → supersede
        Content overlap: partial score → merge fields or append
        """
        # Exact structured state match: same entity+attribute → definite supersession
        if (
            memory.structured
            and candidate.structured
            and memory.structured.entity
            and memory.structured.entity == candidate.structured.entity
            and memory.structured.attribute
            and memory.structured.attribute == candidate.structured.attribute
        ):
            return 1.0, "exact_structured_match → supersede"

        # Same type + high lexical overlap
        if memory.type == candidate.type:
            overlap = self._token_overlap(memory.content, candidate.content)
            if overlap > 0.7:
                return overlap, f"high_token_overlap={overlap:.2f} → update"
            if overlap > 0.4:
                return overlap * 0.8, f"moderate_token_overlap={overlap:.2f} → maybe_update"

        return 0.0, "no_overlap → skip"

    def _token_overlap(self, a: str, b: str) -> float:
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta), len(tb))

    def find_supersession_pairs(
        self, memories: list[MemoryRecord], candidates: list[Candidate]
    ) -> list[tuple[MemoryRecord, Candidate, float]]:
        """Return (existing, candidate, score) pairs above threshold."""
        pairs = []
        gate_out = self.evaluate(memories, candidates, {})
        for i, candidate in enumerate(candidates):
            for memory in memories:
                key = f"{memory.id}::{i}"
                score = gate_out.scores.get(key, 0.0)
                if score >= self.threshold:
                    pairs.append((memory, candidate, score))
        return pairs
