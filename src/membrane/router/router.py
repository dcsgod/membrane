"""
Memory Router — executes QueryPlans against substrates (A5).
"""

from __future__ import annotations

from datetime import datetime

from membrane.core.models import Hit, MemoryRecord, MemoryScore, MemoryState, Namespace, Query
from membrane.planner.rule import Intent, PlanStep, QueryPlan, SubstrateKind
from membrane.substrates.sqlite.substrate import SQLiteSubstrate


class MemoryRouter:
    """
    Executes a QueryPlan against the available substrates (A5).

    v0.1: single SQLite substrate for all kinds.
    v0.5+: parallel fan-out with per-substrate timeouts.
    """

    def __init__(
        self,
        substrate: SQLiteSubstrate,
        failure_mode: str = "best_effort",  # strict|best_effort|disabled
    ) -> None:
        self.substrate = substrate
        self.failure_mode = failure_mode

    def execute(
        self, plan: QueryPlan, query: Query, namespace: Namespace
    ) -> tuple[list[Hit], bool, str | None]:
        """
        Execute the plan and return (hits, degraded, reason).

        Returns:
            hits: scored memory candidates
            degraded: True if we fell back to a fallback plan
            reason: degradation reason if degraded
        """
        if not plan.needs_memory or plan.intent == Intent.NONE:
            return [], False, None

        hits: list[Hit] = []
        degraded = False
        reason: str | None = None

        try:
            for step in plan.steps:
                step_hits = self._execute_step(step, query, namespace)
                hits.extend(step_hits)
        except Exception as exc:
            if self.failure_mode == "strict":
                raise
            degraded = True
            reason = str(exc)
            # Try fallbacks
            for step in plan.fallbacks:
                try:
                    hits.extend(self._execute_step(step, query, namespace))
                except Exception:
                    pass

        # Deduplicate by memory id
        seen: set[str] = set()
        unique_hits: list[Hit] = []
        for h in hits:
            if h.memory.id not in seen:
                unique_hits.append(h)
                seen.add(h.memory.id)

        return unique_hits, degraded, reason

    def _execute_step(
        self, step: PlanStep, query: Query, namespace: Namespace
    ) -> list[Hit]:
        if step.substrate == SubstrateKind.SQL:
            return self._sql_lookup(step, query, namespace)
        elif step.substrate == SubstrateKind.LEXICAL:
            return self._fts_search(step, query, namespace)
        elif step.substrate == SubstrateKind.TEMPORAL:
            return self._temporal_scan(step, query, namespace)
        elif step.substrate == SubstrateKind.GRAPH:
            return self._graph_traverse(step, query, namespace)
        elif step.substrate == SubstrateKind.VECTOR:
            # Vector not available in v0.1 without embeddings extra
            return []
        else:
            return self._fts_search(step, query, namespace)

    def _sql_lookup(self, step: PlanStep, query: Query, namespace: Namespace) -> list[Hit]:
        """Exact entity+attribute lookup or general list."""
        params = step.params
        entity = params.get("entity") or query.entity
        attribute = params.get("attribute") or query.attribute
        at = params.get("at") or query.at

        if entity and attribute:
            records = self.substrate.get_by_entity_attribute(
                entity, attribute, namespace, at=at
            )
        else:
            records = self.substrate.list_by_namespace(
                namespace,
                state_filter=query.state_filter,
                type_filter=query.type_filter,
                limit=query.limit * 2,
            )

        return [self._make_hit(r, "sql") for r in records]

    def _fts_search(self, step: PlanStep, query: Query, namespace: Namespace) -> list[Hit]:
        """Full-text search via FTS5."""
        q = step.params.get("q", query.text)
        records = self.substrate.search_fts(
            q, namespace, state_filter=query.state_filter, limit=query.limit * 2
        )
        return [self._make_hit(r, "lexical") for r in records]

    def _temporal_scan(self, step: PlanStep, query: Query, namespace: Namespace) -> list[Hit]:
        """Timeline scan for an entity or keyword."""
        entity = step.params.get("entity") or query.entity
        if entity:
            records = self.substrate.timeline(entity, namespace)
        else:
            # Fall back to FTS with state filter including superseded
            records = self.substrate.search_fts(
                query.text,
                namespace,
                state_filter=[MemoryState.ACTIVE, MemoryState.SUPERSEDED],
                limit=query.limit * 2,
            )
        return [self._make_hit(r, "temporal") for r in records]

    def _graph_traverse(self, step: PlanStep, query: Query, namespace: Namespace) -> list[Hit]:
        """Graph traversal — for v0.1, use FTS as a fallback."""
        return self._fts_search(step, query, namespace)

    def _make_hit(self, record: MemoryRecord, substrate: str) -> Hit:
        return Hit(
            memory=record,
            score=MemoryScore(memory_id=record.id),
            substrate=substrate,
        )

    def write(
        self,
        record: MemoryRecord,
        *,
        provenance_from: str | None = None,
    ) -> None:
        """Write a memory to the primary substrate and queue secondary indexes."""
        self.substrate.put(record)
        if provenance_from:
            self.substrate.add_provenance_edge(
                from_id=record.id,
                to_id=provenance_from,
                relation="derived_from",
            )
