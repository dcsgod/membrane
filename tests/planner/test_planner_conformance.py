"""
Planner conformance suite (A4.5).

Tests the normative examples from A4.2.
"""

from __future__ import annotations

import pytest
from membrane.core.models import Query
from membrane.planner.rule import Intent, RulePlanner


PLANNER_CASES = [
    # (query, expected_intents)
    ("What's yesterday's sales?", {Intent.TEMPORAL, Intent.HYBRID}),
    ("What does this customer usually prefer?", {Intent.SEMANTIC, Intent.HYBRID}),
    ("What changed about SKU 123?", {Intent.TEMPORAL, Intent.HYBRID}),
    ("What caused sales to decline?", {Intent.CAUSAL, Intent.HYBRID}),
    ("What was the price last month?", {Intent.TEMPORAL, Intent.HYBRID, Intent.STATE}),
    ("Why do you believe this?", {Intent.PROVENANCE, Intent.HYBRID}),
    ("What is the current status of project X?", {Intent.STATE, Intent.HYBRID}),
    ("User preferences for dark mode", {Intent.SEMANTIC, Intent.HYBRID}),
]


class TestPlannerConformance:
    """Planner conformance tests (A4.5)."""

    def setup_method(self):
        self.planner = RulePlanner()

    @pytest.mark.parametrize("query,expected_intents", PLANNER_CASES)
    def test_intent_classification(self, query: str, expected_intents: set):
        q = Query(text=query)
        plan = self.planner.plan(q)
        assert plan.intent in expected_intents, (
            f"Query: {query!r}\n"
            f"Expected one of: {[i.value for i in expected_intents]}\n"
            f"Got: {plan.intent.value}"
        )

    def test_explicit_entity_attribute_forces_state_intent(self):
        q = Query(text="price", entity="sku_123", attribute="price")
        plan = self.planner.plan(q)
        assert plan.intent == Intent.STATE
        assert plan.steps[0].operation == "lookup"

    def test_explicit_entity_only_forces_temporal_intent(self):
        q = Query(text="history", entity="sku_123")
        plan = self.planner.plan(q)
        assert plan.intent == Intent.TEMPORAL

    def test_plan_has_steps_when_needs_memory(self):
        q = Query(text="user preferences")
        plan = self.planner.plan(q)
        if plan.needs_memory:
            assert len(plan.steps) > 0

    def test_plan_has_rationale(self):
        q = Query(text="why did sales decline?")
        plan = self.planner.plan(q)
        assert plan.rationale is not None
        assert "intent" in plan.rationale or "reason" in plan.rationale

    def test_explain_plan_returns_dict(self):
        q = Query(text="what does the user like?")
        plan_dict = self.planner.explain_plan(q)
        assert isinstance(plan_dict, dict)
        assert "intent" in plan_dict
        assert "steps" in plan_dict
        assert "needs_memory" in plan_dict
        assert "rationale" in plan_dict
