"""Rule-based Memory Query Planner (A4)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from membrane.core.models import Context, Namespace, Query


class Intent(str, Enum):
    """Query intent classification (A4.3)."""

    STATE = "state"               # Exact entity/attribute state
    SEMANTIC = "semantic"         # Similarity / preference memory
    TEMPORAL = "temporal"         # Timeline / history scan
    CAUSAL = "causal"             # Causal graph traversal
    PROVENANCE = "provenance"     # Why does the system believe X?
    HYBRID = "hybrid"             # Low-confidence → fan-out
    NONE = "none"                 # No memory needed


class SubstrateKind(str, Enum):
    SQL = "sql"
    KV = "kv"
    VECTOR = "vector"
    GRAPH = "graph"
    TEMPORAL = "temporal"
    LEXICAL = "lexical"


@dataclass
class PlanStep:
    substrate: SubstrateKind
    operation: str            # lookup|knn|traverse|range_scan|fts|timeline
    params: dict[str, Any] = field(default_factory=dict)
    merge: str = "union"      # union|intersect|rrf|first_hit


@dataclass
class QueryPlan:
    plan_id: str
    intent: Intent
    steps: list[PlanStep] = field(default_factory=list)
    fallbacks: list[PlanStep] = field(default_factory=list)
    needs_memory: bool = True
    rationale: dict[str, Any] = field(default_factory=dict)


class RulePlanner:
    """
    Deterministic rule-based query planner (A4.4, Phase 1).

    Classifies query intent from text patterns, explicit parameters,
    and entity registry hits. No LLM, no embeddings required.

    Normative examples from A4.2 are covered by this planner.
    """

    # Patterns that signal structured/state intent
    _STATE_PATTERNS = [
        r"\b(what(?:'s| is| was)(?: the)?)\s+(?:current|today'?s?|latest|present)\b",
        r"\b(?:current|latest|present|today'?s?)\s+\w+",
        r"\bwhat(?:'s| is| was)\s+(?:the\s+)?\w+\s+(?:of|for)\b",
        r"\bprice\b",
        r"\binventory\b",
        r"\bstatus\b",
        r"\bvalue of\b",
    ]
    _STATE_RE = [re.compile(p, re.IGNORECASE) for p in _STATE_PATTERNS]

    # Temporal patterns
    _TEMPORAL_PATTERNS = [
        r"\b(?:yesterday|last week|last month|last year|ago|since|between|from|during)\b",
        r"\bwhat changed\b",
        r"\bhistory\b",
        r"\btimeline\b",
        r"\bevolution\b",
        r"\bwhen did\b",
        r"\bwhat was .+ (?:on|at|in)\b",
    ]
    _TEMPORAL_RE = [re.compile(p, re.IGNORECASE) for p in _TEMPORAL_PATTERNS]

    # Provenance patterns
    _PROV_PATTERNS = [
        r"\bwhy (?:do|does|did|should|is|are)\b",
        r"\bwhy (?:you |the system |do you )?believe\b",
        r"\bwhere did (?:you|this|that) (?:come from|learn)\b",
        r"\bsource of\b",
        r"\bprovenance\b",
    ]
    _PROV_RE = [re.compile(p, re.IGNORECASE) for p in _PROV_PATTERNS]

    # Causal patterns
    _CAUSAL_PATTERNS = [
        r"\bwhy (?:did|does)\b.*\b(?:decline|increase|change|happen|occur)\b",
        r"\bcaused?\b",
        r"\bbecause of\b",
        r"\bimpact of\b",
        r"\beffect of\b",
    ]
    _CAUSAL_RE = [re.compile(p, re.IGNORECASE) for p in _CAUSAL_PATTERNS]

    # Semantic / preference patterns
    _SEMANTIC_PATTERNS = [
        r"\b(?:prefer|prefer[s]?|like[s]?|want[s]?|usually|typically|tend[s]? to)\b",
        r"\bwhat (?:does|do) .+ (?:prefer|like|want|usually)\b",
        r"\bsemantic\b",
        r"\bsimilar\b",
    ]
    _SEMANTIC_RE = [re.compile(p, re.IGNORECASE) for p in _SEMANTIC_PATTERNS]

    def plan(self, query: Query, context: Context | None = None) -> QueryPlan:
        """
        Classify intent and produce a QueryPlan.
        Deterministic — no network, no LLM, no embeddings.
        """
        import uuid

        plan_id = f"plan_{uuid.uuid4().hex[:10]}"
        text = query.text.strip()

        # Explicit parameters override intent detection
        if query.entity and query.attribute:
            return QueryPlan(
                plan_id=plan_id,
                intent=Intent.STATE,
                needs_memory=True,
                steps=[
                    PlanStep(
                        substrate=SubstrateKind.SQL,
                        operation="lookup",
                        params={
                            "entity": query.entity,
                            "attribute": query.attribute,
                            "at": query.at,
                        },
                    )
                ],
                fallbacks=[PlanStep(SubstrateKind.LEXICAL, "fts", {"q": text})],
                rationale={"reason": "explicit_entity_attribute", "entity": query.entity},
            )

        if query.entity and not query.attribute:
            # Timeline scan
            return QueryPlan(
                plan_id=plan_id,
                intent=Intent.TEMPORAL,
                needs_memory=True,
                steps=[
                    PlanStep(
                        substrate=SubstrateKind.TEMPORAL,
                        operation="timeline",
                        params={"entity": query.entity, "at": query.at},
                    )
                ],
                fallbacks=[PlanStep(SubstrateKind.LEXICAL, "fts", {"q": text})],
                rationale={"reason": "explicit_entity_no_attribute"},
            )

        # Pattern matching
        intent, confidence = self._classify(text)
        rationale: dict[str, Any] = {"intent": intent.value, "confidence": confidence, "text": text}

        if intent == Intent.NONE:
            return QueryPlan(
                plan_id=plan_id,
                intent=Intent.NONE,
                needs_memory=False,
                rationale=rationale,
            )

        if intent == Intent.STATE:
            steps = [PlanStep(SubstrateKind.SQL, "lookup", {"q": text})]
            fallbacks = [PlanStep(SubstrateKind.LEXICAL, "fts", {"q": text})]
        elif intent == Intent.TEMPORAL:
            steps = [
                PlanStep(SubstrateKind.TEMPORAL, "range_scan", {"q": text, "at": query.at})
            ]
            fallbacks = [PlanStep(SubstrateKind.LEXICAL, "fts", {"q": text})]
        elif intent == Intent.PROVENANCE:
            steps = [PlanStep(SubstrateKind.GRAPH, "traverse", {"q": text})]
            fallbacks = [PlanStep(SubstrateKind.LEXICAL, "fts", {"q": text})]
        elif intent == Intent.CAUSAL:
            steps = [
                PlanStep(SubstrateKind.GRAPH, "traverse", {"q": text}),
                PlanStep(SubstrateKind.SQL, "lookup", {"q": text}, merge="union"),
            ]
            fallbacks = [PlanStep(SubstrateKind.LEXICAL, "fts", {"q": text})]
        elif intent == Intent.SEMANTIC:
            steps = [PlanStep(SubstrateKind.LEXICAL, "fts", {"q": text})]
            fallbacks = [PlanStep(SubstrateKind.SQL, "lookup", {"q": text})]
        else:  # HYBRID
            steps = [
                PlanStep(SubstrateKind.LEXICAL, "fts", {"q": text}, merge="rrf"),
                PlanStep(SubstrateKind.SQL, "lookup", {"q": text}, merge="rrf"),
            ]
            fallbacks = []

        return QueryPlan(
            plan_id=plan_id,
            intent=intent,
            needs_memory=True,
            steps=steps,
            fallbacks=fallbacks,
            rationale=rationale,
        )

    def _classify(self, text: str) -> tuple[Intent, float]:
        """Rule-based intent classifier. Returns (intent, confidence)."""
        scores: dict[Intent, float] = {}

        for pattern in self._STATE_RE:
            if pattern.search(text):
                scores[Intent.STATE] = scores.get(Intent.STATE, 0) + 0.3

        for pattern in self._TEMPORAL_RE:
            if pattern.search(text):
                scores[Intent.TEMPORAL] = scores.get(Intent.TEMPORAL, 0) + 0.3

        for pattern in self._PROV_RE:
            if pattern.search(text):
                scores[Intent.PROVENANCE] = scores.get(Intent.PROVENANCE, 0) + 0.4

        for pattern in self._CAUSAL_RE:
            if pattern.search(text):
                scores[Intent.CAUSAL] = scores.get(Intent.CAUSAL, 0) + 0.4

        for pattern in self._SEMANTIC_RE:
            if pattern.search(text):
                scores[Intent.SEMANTIC] = scores.get(Intent.SEMANTIC, 0) + 0.3

        if not scores:
            # Default: lexical search (covers most generic queries)
            return Intent.SEMANTIC, 0.5

        best_intent = max(scores, key=lambda k: scores[k])
        best_score = min(1.0, scores[best_intent])

        # Low confidence → HYBRID
        if best_score < 0.3:
            return Intent.HYBRID, best_score

        return best_intent, best_score

    def explain_plan(self, query: Query, context: Context | None = None) -> dict[str, Any]:
        """
        Return the plan and rationale without executing it (A4.5 dry run).
        """
        plan = self.plan(query, context)
        return {
            "plan_id": plan.plan_id,
            "intent": plan.intent.value,
            "needs_memory": plan.needs_memory,
            "steps": [
                {
                    "substrate": s.substrate.value,
                    "operation": s.operation,
                    "params": s.params,
                    "merge": s.merge,
                }
                for s in plan.steps
            ],
            "fallbacks": [
                {
                    "substrate": s.substrate.value,
                    "operation": s.operation,
                }
                for s in plan.fallbacks
            ],
            "rationale": plan.rationale,
        }
