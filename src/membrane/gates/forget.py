"""
Forget gate (f_t) — decides which memories to expire, archive, or supersede (§11, A2.3).

Phase 1: deterministic rule-based scoring.
Score of 1.0 = keep fully; score approaching 0 = candidate for forgetting.
"""

from __future__ import annotations

from datetime import datetime, timezone

from membrane.core.models import GateOutput, MemoryRecord, MemoryState, MemoryType


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Decay rates per memory type (A10)
_DEFAULT_DECAY_RATES: dict[MemoryType, float] = {
    MemoryType.WORKING: 1.0,        # Fastest — decays quickly
    MemoryType.EPISODIC: 0.3,
    MemoryType.SEMANTIC: 0.1,
    MemoryType.PREFERENCE: 0.05,
    MemoryType.PROCEDURAL: 0.05,
    MemoryType.STATE: 0.0,          # Not decayed by age; invalidated by supersession
    MemoryType.OUTCOME: 0.05,
    MemoryType.DECISION: 0.02,
    MemoryType.CAUSAL: 0.02,
    MemoryType.CONSOLIDATED: 0.05,
}


class ForgetGate:
    """
    Rule-based forget gate (f_t).

    Returns scores in [0, 1] for each active memory.
    Low score → candidate for archive/expire/supersede.
    """

    name = "forget"
    version = "1.0"

    def __init__(
        self,
        threshold: float = 0.2,
        age_weight: float = 0.30,
        utility_weight: float = 0.25,
        access_weight: float = 0.20,
        importance_weight: float = 0.25,
        decay_rates: dict[MemoryType, float] | None = None,
    ) -> None:
        self.threshold = threshold
        self.age_weight = age_weight
        self.utility_weight = utility_weight
        self.access_weight = access_weight
        self.importance_weight = importance_weight
        self.decay_rates = decay_rates or dict(_DEFAULT_DECAY_RATES)

    def evaluate(
        self,
        memories: list[MemoryRecord],
        candidates: list | None,
        context: dict,
    ) -> GateOutput:
        now = context.get("now") or _now()
        output = GateOutput(gate_name=self.name, gate_version=self.version, threshold=self.threshold)

        for memory in memories:
            if memory.state not in (MemoryState.ACTIVE, MemoryState.WRITTEN):
                continue
            score, reason, features = self._score_memory(memory, now)
            output.scores[memory.id] = score
            output.reasons[memory.id] = reason
            output.features[memory.id] = features

        return output

    def _score_memory(
        self, memory: MemoryRecord, now: datetime
    ) -> tuple[float, str, dict]:
        # Hard keep: if expired, force forget
        if memory.expires_at and now >= memory.expires_at:
            return 0.0, "expired", {"expired": True}

        # Hard keep: high-importance system memories
        if memory.trust_tier.value == "system" and memory.importance >= 0.9:
            return 1.0, "system_high_importance_keep", {}

        # Age decay
        age_days = (now - memory.created_at).total_seconds() / 86400.0
        decay_rate = self.decay_rates.get(memory.type, 0.1)
        import math

        age_factor = math.exp(-decay_rate * age_days)

        # Access recency
        if memory.last_accessed_at:
            days_since_access = (now - memory.last_accessed_at).total_seconds() / 86400.0
            access_factor = math.exp(-0.05 * days_since_access)
        else:
            access_factor = 0.3  # never accessed

        # Utility and importance
        util_factor = memory.utility
        imp_factor = memory.importance

        keep_score = (
            self.age_weight * age_factor
            + self.access_weight * access_factor
            + self.utility_weight * util_factor
            + self.importance_weight * imp_factor
        )
        keep_score = max(0.0, min(1.0, keep_score))

        features = {
            "age_days": round(age_days, 2),
            "age_factor": round(age_factor, 3),
            "access_factor": round(access_factor, 3),
            "utility": memory.utility,
            "importance": memory.importance,
            "decay_rate": decay_rate,
        }
        action = "keep" if keep_score >= self.threshold else "forget_candidate"
        reason = (
            f"age_factor={age_factor:.3f} access_factor={access_factor:.3f} "
            f"utility={memory.utility:.2f} importance={memory.importance:.2f} → {action}"
        )
        return keep_score, reason, features
