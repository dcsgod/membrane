"""
Policy Engine (A13) — converts gate scores to lifecycle actions and enforces hard rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from membrane.core.models import (
    Candidate,
    GateOutput,
    MemoryRecord,
    MemoryState,
    MemoryType,
    TrustTier,
    TRUST_ORDER,
)


@dataclass
class WriteDecision:
    action: str          # write|quarantine|reject
    reason: str
    candidate_idx: int


@dataclass
class ForgetDecision:
    action: str          # keep|archive|expire|supersede|delete
    memory_id: str
    reason: str


@dataclass
class UpdateDecision:
    action: str          # supersede|merge_fields|append_evidence|conflict|skip
    existing_id: str
    candidate_idx: int
    reason: str


class PolicyEngine:
    """
    Converts continuous gate scores to discrete lifecycle actions (A13, A2.6).
    Enforces hard safety rules AFTER gates — these cannot be overridden.
    """

    def __init__(
        self,
        preset: str = "conservative",
        write_threshold: float = 0.3,
        forget_threshold: float = 0.2,
        update_threshold: float = 0.6,
        read_threshold: float = 0.2,
        quarantine_below: TrustTier = TrustTier.EXTERNAL_CONTENT,
        max_memories_per_recall: int = 10,
    ) -> None:
        self.preset = preset
        self.write_threshold = write_threshold
        self.forget_threshold = forget_threshold
        self.update_threshold = update_threshold
        self.read_threshold = read_threshold
        self.quarantine_below = quarantine_below
        self.max_memories_per_recall = max_memories_per_recall

        # Adjust thresholds based on preset
        if preset == "aggressive":
            self.write_threshold = 0.15
            self.forget_threshold = 0.35
        elif preset == "conservative":
            self.write_threshold = 0.45
            self.forget_threshold = 0.1
        elif preset == "strict_audit":
            self.write_threshold = 0.5
            self.forget_threshold = 0.05

    def decide_writes(
        self,
        candidates: list[Candidate],
        write_gate: GateOutput,
    ) -> list[WriteDecision]:
        """Convert write gate scores to write/quarantine/reject decisions."""
        decisions = []
        for i, candidate in enumerate(candidates):
            key = f"c_{i}"
            score = write_gate.scores.get(key, 0.0)
            reason = write_gate.reasons.get(key, "")

            # Hard rule: instruction-like content from low-trust → quarantine (A17.2)
            if "instruction_like_from_low_trust" in reason:
                decisions.append(WriteDecision("quarantine", reason, i))
                continue

            # Hard rule: quarantine_below tier → quarantine
            trust_score = TRUST_ORDER.get(candidate.trust_tier, 0)
            quarantine_score = TRUST_ORDER.get(self.quarantine_below, 1)
            if trust_score < quarantine_score:
                decisions.append(
                    WriteDecision(
                        "quarantine",
                        f"trust_tier={candidate.trust_tier.value} below quarantine threshold",
                        i,
                    )
                )
                continue

            if score >= self.write_threshold:
                decisions.append(WriteDecision("write", reason, i))
            else:
                decisions.append(WriteDecision("reject", reason, i))

        return decisions

    def decide_forgets(
        self,
        memories: list[MemoryRecord],
        forget_gate: GateOutput,
    ) -> list[ForgetDecision]:
        """Convert forget gate scores to keep/archive/expire decisions."""
        decisions = []
        for memory in memories:
            score = forget_gate.scores.get(memory.id, 1.0)  # default: keep
            reason = forget_gate.reasons.get(memory.id, "")

            # Hard rule: never silently delete (A2.8) — always archive first
            if score < self.forget_threshold:
                if "expired" in reason:
                    action = "expire"
                else:
                    action = "archive"
                decisions.append(ForgetDecision(action, memory.id, reason))
            else:
                decisions.append(ForgetDecision("keep", memory.id, reason))

        return decisions

    def decide_updates(
        self,
        candidates: list[Candidate],
        existing_memories: list[MemoryRecord],
        update_gate: GateOutput,
    ) -> list[UpdateDecision]:
        """Convert update gate scores to supersede/merge/skip decisions."""
        decisions = []
        for i, candidate in enumerate(candidates):
            for memory in existing_memories:
                key = f"{memory.id}::{i}"
                score = update_gate.scores.get(key, 0.0)
                reason = update_gate.reasons.get(key, "")

                if score >= self.update_threshold:
                    if "exact_structured_match" in reason:
                        action = "supersede"
                    elif "high_token_overlap" in reason:
                        action = "supersede"
                    else:
                        action = "merge_fields"
                    decisions.append(UpdateDecision(action, memory.id, i, reason))

        return decisions
