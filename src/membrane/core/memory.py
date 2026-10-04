"""
Memory — the public façade for Membrane (§24, A3.3).

This is the primary API that developers interact with.
All operations route through the MembraneCell step function.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from membrane.cell.candidate import RuleCandidateGenerator
from membrane.core.clock import Clock, SystemClock
from membrane.core.exceptions import (
    IdempotencyConflictError,
    MissingExtraError,
    SnapshotNotFoundError,
)
from membrane.core.models import (
    Candidate,
    Context,
    ExplainResult,
    MemoryEvent,
    MemoryRecord,
    MemoryScore,
    MemoryState,
    MemoryType,
    Namespace,
    Query,
    RecallResult,
    Snapshot,
    Source,
    StructuredFields,
    TrustTier,
)
from membrane.gates.forget import ForgetGate
from membrane.gates.read import ReadGate
from membrane.gates.update import UpdateGate
from membrane.gates.write import WriteGate
from membrane.planner.rule import RulePlanner
from membrane.policy.engine import PolicyEngine
from membrane.router.router import MemoryRouter
from membrane.state.machine import get_state_machine
from membrane.substrates.sqlite.substrate import SQLiteSubstrate


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Memory:
    """
    The programmable memory layer for AI.

    Quickstart::

        from membrane import Memory

        memory = Memory()
        memory.remember("The user prefers Python over JavaScript.", user_id="u1")
        result = memory.recall("What does this user prefer?", user_id="u1")
        print(result)

    No API key. No Docker. No external database. No LLM required.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        policy: str = "conservative",
        failure_mode: str = "best_effort",
        clock: Clock | None = None,
        default_namespace: str = "default",
    ) -> None:
        self._clock = clock or SystemClock()
        self._default_namespace = default_namespace
        self._failure_mode = failure_mode

        # Storage
        self._substrate = SQLiteSubstrate(db_path)

        # Gates (Phase 1: rule-based)
        self._candidate_fn = RuleCandidateGenerator()
        self._write_gate = WriteGate()
        self._forget_gate = ForgetGate()
        self._update_gate = UpdateGate()
        self._read_gate = ReadGate()

        # Planner & Router
        self._planner = RulePlanner()
        self._router = MemoryRouter(self._substrate, failure_mode=failure_mode)

        # Policy Engine
        self._policy = PolicyEngine(preset=policy)

        # State machine
        self._sm = get_state_machine()

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def remember(
        self,
        content: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str = "default",
        type: MemoryType | str | None = None,
        confidence: float = 0.9,
        importance: float = 0.5,
        trust_tier: TrustTier | str = TrustTier.USER_ASSERTED,
        source: Source | None = None,
        structured: StructuredFields | None = None,
        idempotency_key: str | None = None,
        persist: bool = True,
        metadata: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryRecord | None:
        """
        Remember new information (write path: candidate → write gate → policy → store).

        Args:
            content: The information to remember.
            user_id: Namespace user scope.
            persist: If False, content is treated as session-only working memory.

        Returns:
            The created MemoryRecord, or None if rejected.
        """
        now = self._clock.now()
        ns = Namespace(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

        # Idempotency check (§54)
        if idempotency_key:
            existing = self._substrate.get_by_idempotency_key(idempotency_key)
            if existing:
                return existing

        # Build candidate
        if isinstance(trust_tier, str):
            trust_tier = TrustTier(trust_tier)
        if isinstance(type, str):
            type = MemoryType(type)

        candidate = Candidate(
            content=content,
            type=type or self._candidate_fn._infer_type(content),
            confidence=confidence,
            importance=importance,
            source=source or Source(type="user"),
            trust_tier=trust_tier,
            structured=structured,
            metadata=metadata or {},
            idempotency_key=idempotency_key,
            expires_at=expires_at,
        )

        if not persist:
            candidate.type = MemoryType.WORKING

        # Get active memories for overlap/update detection
        active = self._substrate.list_by_namespace(ns, limit=50)

        # Gate evaluation
        context = {"now": now, "namespace": ns}
        write_out = self._write_gate.evaluate(active, [candidate], context)
        update_out = self._update_gate.evaluate(active, [candidate], context)

        # Policy decisions
        write_decisions = self._policy.decide_writes([candidate], write_out)
        update_decisions = self._policy.decide_updates([candidate], active, update_out)

        decision = write_decisions[0] if write_decisions else None
        if not decision:
            return None

        if decision.action == "reject":
            # Emit rejection event
            self._emit_event(
                memory_id=f"rejected_{uuid.uuid4().hex[:8]}",
                event_type="memory.rejected",
                namespace_key=ns.to_key(),
                payload={"content": content[:100], "reason": decision.reason},
            )
            return None

        # Check for supersession first
        for upd in update_decisions:
            if upd.action in ("supersede", "merge_fields"):
                old = self._substrate.get(upd.existing_id)
                if old:
                    return self._supersede(old, candidate, ns, now, upd.action)

        # Create new memory record
        record = self._candidate_to_record(candidate, ns, now)

        if decision.action == "quarantine":
            record.state = MemoryState.QUARANTINED
        else:
            record.state = MemoryState.ACTIVE

        # Persist
        self._substrate.put(record)

        # Emit event
        event_type = "memory.created" if record.state == MemoryState.ACTIVE else "memory.quarantined"
        self._emit_event(
            memory_id=record.id,
            event_type=event_type,
            namespace_key=ns.to_key(),
            payload={"content": content[:200], "type": record.type.value, "reason": decision.reason},
            gate_scores={"write_score": write_out.scores.get("c_0", 0.0)},
        )

        return record

    def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str = "default",
        limit: int = 10,
        type_filter: list[MemoryType | str] | None = None,
        entity: str | None = None,
        attribute: str | None = None,
        at: datetime | None = None,
        as_of: datetime | None = None,
        include_superseded: bool = False,
        token_budget: int | None = None,
        return_trace: bool = False,
        format: str = "raw",
    ) -> RecallResult:
        """
        Retrieve relevant memories (read path: query → planner → router → read gate).

        Args:
            query: Natural language query.
            limit: Maximum number of memories to return.
            token_budget: Maximum tokens for context assembly (A19).
            return_trace: If True, include the query plan in the result.

        Returns:
            RecallResult with scored memories, token usage, and near-misses.
        """
        now = self._clock.now()
        ns = Namespace(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

        tf = None
        if type_filter:
            tf = [MemoryType(t) if isinstance(t, str) else t for t in type_filter]

        state_filter = [MemoryState.ACTIVE]
        if include_superseded:
            state_filter.append(MemoryState.SUPERSEDED)

        q = Query(
            text=query,
            namespace=ns,
            type_filter=tf,
            state_filter=state_filter,
            entity=entity,
            attribute=attribute,
            at=at,
            as_of=as_of,
            limit=limit,
        )

        # Plan
        plan = self._planner.plan(q)

        if not plan.needs_memory:
            return RecallResult(query=query)

        # Route → retrieve hits
        context = {"now": now, "namespace": ns}
        hits, degraded, deg_reason = self._router.execute(plan, q, ns)

        # Read gate scoring
        scored = self._read_gate.score_hits(hits, q, context)

        # Budget-aware selection (A19)
        selected = self._read_gate.apply_budget(
            scored, limit=limit, token_budget=token_budget
        )

        memories = [h.memory for h, _ in selected]
        scores = [s for _, s in selected]

        # Near-misses (first excluded candidates)
        near_misses: list[dict] = []
        excluded = [
            (h, s)
            for h, s in scored
            if h.memory.id not in {m.id for m in memories}
        ][:3]
        for h, s in excluded:
            near_misses.append(
                {
                    "memory_id": h.memory.id,
                    "content": h.memory.content[:100],
                    "final_score": s.final_score,
                    "reason": "below_threshold_or_budget",
                }
            )

        # Touch accessed memories
        for m in memories:
            self._substrate.touch(m.id)

        # Token count (approximate)
        tokens_used = sum(len(m.content.split()) + 10 for m in memories)

        result = RecallResult(
            query=query,
            memories=memories,
            scores=scores,
            tokens_used=tokens_used,
            dropped=near_misses,
            plan_id=plan.plan_id if return_trace else None,
            degraded=degraded,
            degraded_reason=deg_reason,
        )

        # Emit event
        self._emit_event(
            memory_id="*",
            event_type="memory.recalled",
            namespace_key=ns.to_key(),
            payload={
                "query": query,
                "recall_id": result.recall_id,
                "returned": len(memories),
                "plan_id": plan.plan_id,
                "degraded": degraded,
            },
        )

        return result

    def forget(
        self,
        memory_id: str,
        *,
        mode: str = "archive",  # archive|delete|expire
        reason: str = "manual",
    ) -> bool:
        """
        Forget a memory (policy-driven, auditable, not destructive by default).

        Default mode is 'archive', not 'delete'. History is preserved.
        """
        record = self._substrate.get(memory_id)
        if not record:
            return False

        new_state: MemoryState
        if mode == "expire":
            new_state = MemoryState.EXPIRED
        elif mode == "delete":
            new_state = MemoryState.FORGOTTEN
        else:
            new_state = MemoryState.ARCHIVED

        # Validate transition
        self._sm.validate(record.state, new_state, memory_id)

        self._substrate.update_state(memory_id, new_state, now=self._clock.now())

        self._emit_event(
            memory_id=memory_id,
            event_type=f"memory.{mode}d" if mode != "archive" else "memory.archived",
            namespace_key=record.namespace.to_key(),
            payload={"reason": reason, "mode": mode},
        )
        return True

    def forget_user(self, user_id: str, tenant_id: str = "default") -> int:
        """Archive all memories for a user (GDPR-style right to erasure)."""
        ns = Namespace(tenant_id=tenant_id, user_id=user_id)
        memories = self._substrate.list_by_namespace(
            ns, state_filter=[MemoryState.ACTIVE, MemoryState.WRITTEN]
        )
        count = 0
        for m in memories:
            try:
                self._sm.validate(m.state, MemoryState.ARCHIVED, m.id)
                self._substrate.update_state(m.id, MemoryState.ARCHIVED, now=self._clock.now())
                count += 1
            except Exception:
                pass
        if count:
            self._emit_event(
                memory_id="*",
                event_type="memory.user_erased",
                namespace_key=ns.to_key(),
                payload={"user_id": user_id, "count": count},
            )
        return count

    def update(
        self,
        memory_id: str,
        content: str,
        *,
        reason: str = "manual_update",
    ) -> MemoryRecord | None:
        """
        Update a memory's content, creating a new version and superseding the old.
        """
        old = self._substrate.get(memory_id)
        if not old:
            return None

        now = self._clock.now()

        # Create new version
        new_record = old.model_copy(
            update={
                "id": f"m_{uuid.uuid4().hex[:16]}",
                "content": content,
                "version": old.version + 1,
                "supersedes": old.id,
                "state": MemoryState.ACTIVE,
                "created_at": now,
                "recorded_at": now,
                "updated_at": now,
            }
        )

        # Supersede old
        self._sm.validate(old.state, MemoryState.SUPERSEDED, old.id)
        self._substrate.update_state(
            old.id, MemoryState.SUPERSEDED, superseded_by=new_record.id, now=now
        )

        # Write new
        self._substrate.put(new_record)

        # Provenance edge
        self._substrate.add_provenance_edge(new_record.id, old.id, "supersedes")

        self._emit_event(
            memory_id=new_record.id,
            event_type="memory.updated",
            namespace_key=new_record.namespace.to_key(),
            payload={
                "superseded_id": old.id,
                "reason": reason,
                "new_content": content[:200],
            },
        )
        return new_record

    def explain(self, ref_id: str) -> ExplainResult:
        """
        Explain a memory or recall decision (A18).

        Args:
            ref_id: A memory_id or recall_id.

        Returns:
            ExplainResult with score breakdown, gate scores, plan, near-misses, and provenance.
        """
        # Try as memory_id first
        record = self._substrate.get(ref_id)
        if record:
            events = self._substrate.list_events(memory_id=ref_id, limit=20)
            prov = self._substrate.get_provenance(ref_id)
            gate_scores = {}
            for e in events:
                if e.gate_scores:
                    gate_scores.update(e.gate_scores)

            return ExplainResult(
                ref_id=ref_id,
                ref_type="memory",
                summary=(
                    f"Memory '{record.content[:80]}' — "
                    f"state={record.state.value}, "
                    f"confidence={record.confidence:.2f}, "
                    f"utility={record.utility:.2f}, "
                    f"strength={record.strength:.2f}, "
                    f"trust={record.trust_tier.value}"
                ),
                gate_scores=gate_scores,
                events=events,
                provenance=[{"relation": p["relation"], "from": p["from_id"], "to": p["to_id"]} for p in prov],
            )

        return ExplainResult(
            ref_id=ref_id,
            ref_type="unknown",
            summary=f"No memory or recall found for ref_id={ref_id!r}",
        )

    def explain_plan(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Return the query plan without executing it (A4.5 dry run)."""
        q = Query(text=query)
        return self._planner.explain_plan(q)

    def provenance(self, memory_id: str) -> list[dict]:
        """Return the full provenance lineage for a memory (A8.2)."""
        return self._substrate.get_provenance(memory_id)

    def why(self, memory_id: str) -> str:
        """Human-readable provenance chain: 'Why does the system believe this?' (A8.2)."""
        record = self._substrate.get(memory_id)
        if not record:
            return f"Memory {memory_id!r} not found."
        edges = self._substrate.get_provenance(memory_id)
        lines = [f"Memory: {record.content[:120]}"]
        lines.append(f"  Source: {record.source.type} (trust: {record.trust_tier.value})")
        lines.append(f"  Written: {record.recorded_at.strftime('%Y-%m-%d %H:%M UTC')}")
        if record.supersedes:
            lines.append(f"  Supersedes: {record.supersedes}")
        for e in edges:
            lines.append(f"  Provenance: {e.get('relation', '?')} → {e.get('to_id', '?')}")
        return "\n".join(lines)

    def timeline(
        self,
        entity: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        user_id: str | None = None,
        tenant_id: str = "default",
    ) -> list[MemoryRecord]:
        """Return entity state history (§57, A16.2)."""
        ns = Namespace(tenant_id=tenant_id, user_id=user_id)
        return self._substrate.timeline(entity, ns, start=start, end=end)

    def snapshot(self, label: str | None = None, *, user_id: str | None = None, tenant_id: str = "default") -> Snapshot:
        """Create a lightweight snapshot (pointer into event log) (A16.2)."""
        ns = Namespace(tenant_id=tenant_id, user_id=user_id)
        snap = Snapshot(label=label, namespace_key=ns.to_key())
        self._substrate.create_snapshot(snap)
        self._emit_event(
            memory_id="*",
            event_type="snapshot.created",
            namespace_key=ns.to_key(),
            payload={"snapshot_id": snap.id, "label": label},
        )
        return snap

    def diff(self, from_snap: Snapshot | str, to_snap: Snapshot | str | None = None, *, tenant_id: str = "default") -> list[MemoryEvent]:
        """Return events between two snapshots (A16.2)."""
        ns_key = f"{tenant_id}"

        if isinstance(from_snap, str):
            s = self._substrate.get_snapshot(from_snap)
            if not s:
                raise SnapshotNotFoundError(from_snap)
            from_cursor = s.event_cursor
            ns_key = s.namespace_key
        else:
            from_cursor = from_snap.event_cursor
            ns_key = from_snap.namespace_key

        to_cursor: str | None = None
        if to_snap is not None:
            if isinstance(to_snap, str):
                s2 = self._substrate.get_snapshot(to_snap)
                if s2:
                    to_cursor = s2.event_cursor
            else:
                to_cursor = to_snap.event_cursor

        return self._substrate.diff(from_cursor, to_cursor, ns_key)

    def rollback(self, to: Snapshot | str, *, tenant_id: str = "default") -> int:
        """
        Rollback to a snapshot by archiving memories created after it.
        Appends compensating events — history is never rewritten (A16.2).
        """
        if isinstance(to, str):
            snap = self._substrate.get_snapshot(to)
            if not snap:
                raise SnapshotNotFoundError(to)
        else:
            snap = to

        events = self.diff(snap)
        rolled_back = 0
        now = self._clock.now()

        for event in events:
            if event.event_type == "memory.created":
                record = self._substrate.get(event.memory_id)
                if record and record.state == MemoryState.ACTIVE:
                    try:
                        self._sm.validate(record.state, MemoryState.ARCHIVED, record.id)
                        self._substrate.update_state(record.id, MemoryState.ARCHIVED, now=now)
                        self._emit_event(
                            memory_id=record.id,
                            event_type="memory.rollback_archived",
                            namespace_key=record.namespace.to_key(),
                            payload={"snapshot_id": snap.id},
                        )
                        rolled_back += 1
                    except Exception:
                        pass

        return rolled_back

    def health(self) -> dict[str, Any]:
        """Return memory health information (A22)."""
        h = self._substrate.health()
        return h

    def list_memories(
        self,
        *,
        user_id: str | None = None,
        tenant_id: str = "default",
        state: MemoryState | None = MemoryState.ACTIVE,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """List memories for inspection."""
        ns = Namespace(tenant_id=tenant_id, user_id=user_id)
        return self._substrate.list_by_namespace(
            ns,
            state_filter=[state] if state else None,
            limit=limit,
        )

    def review_queue(self, *, tenant_id: str = "default") -> list[MemoryRecord]:
        """Return quarantined memories awaiting review (A17.2)."""
        ns = Namespace(tenant_id=tenant_id)
        return self._substrate.list_by_namespace(
            ns, state_filter=[MemoryState.QUARANTINED]
        )

    def approve(self, memory_id: str) -> bool:
        """Approve a quarantined memory (A17.2)."""
        record = self._substrate.get(memory_id)
        if not record or record.state != MemoryState.QUARANTINED:
            return False
        self._sm.validate(record.state, MemoryState.WRITTEN, memory_id)
        self._substrate.update_state(memory_id, MemoryState.ACTIVE, now=self._clock.now())
        self._emit_event(
            memory_id=memory_id,
            event_type="memory.approved",
            namespace_key=record.namespace.to_key(),
            payload={},
        )
        return True

    def reject(self, memory_id: str, reason: str = "") -> bool:
        """Reject a quarantined memory (A17.2)."""
        record = self._substrate.get(memory_id)
        if not record or record.state != MemoryState.QUARANTINED:
            return False
        self._sm.validate(record.state, MemoryState.REJECTED, memory_id)
        self._substrate.update_state(memory_id, MemoryState.REJECTED, now=self._clock.now())
        self._emit_event(
            memory_id=memory_id,
            event_type="memory.rejected",
            namespace_key=record.namespace.to_key(),
            payload={"reason": reason},
        )
        return True

    def remember_conversation(
        self,
        messages: list[dict],
        *,
        user_id: str | None = None,
        tenant_id: str = "default",
    ) -> list[MemoryRecord]:
        """
        Extract and remember memories from a conversation (§45).

        Messages format: [{"role": "user"|"assistant", "content": "..."}]
        Uses rule-based extraction by default; LLM is optional.
        """
        created = []
        for msg in messages:
            content = msg.get("content", "")
            if not content or msg.get("role") == "system":
                continue
            record = self.remember(
                content=content,
                user_id=user_id,
                tenant_id=tenant_id,
                source=Source(type="conversation"),
                confidence=0.7,
            )
            if record:
                created.append(record)
        return created

    def consolidate(
        self,
        *,
        user_id: str | None = None,
        tenant_id: str = "default",
        min_cluster_size: int = 3,
    ) -> int:
        """
        Consolidate repeated episodic memories into semantic summaries (A12).

        Phase 1: simple frequency/cluster detection (no LLM).
        """
        ns = Namespace(tenant_id=tenant_id, user_id=user_id)
        memories = self._substrate.list_by_namespace(
            ns, state_filter=[MemoryState.ACTIVE], limit=200
        )

        # Group by type
        episodic = [m for m in memories if m.type == MemoryType.EPISODIC]
        consolidated_count = 0

        if len(episodic) < min_cluster_size:
            return 0

        # Simple entity-based clustering
        entity_groups: dict[str, list[MemoryRecord]] = {}
        for mem in episodic:
            key = (mem.structured.entity if mem.structured and mem.structured.entity else "general")
            entity_groups.setdefault(key, []).append(mem)

        for entity_key, group in entity_groups.items():
            if len(group) < min_cluster_size:
                continue

            # Create consolidated summary
            summary_content = f"Consolidated from {len(group)} episodic memories"
            if entity_key != "general":
                summary_content = f"Pattern for {entity_key}: " + "; ".join(
                    m.content[:50] for m in group[:3]
                )

            consolidated = MemoryRecord(
                content=summary_content,
                type=MemoryType.CONSOLIDATED,
                state=MemoryState.ACTIVE,
                confidence=0.8,
                importance=0.7,
                namespace=ns,
                consolidated_from=[m.id for m in group],
                metadata={
                    "consolidation_method": "frequency_cluster",
                    "source_count": len(group),
                    "compression_ratio": len(group),
                },
            )
            self._substrate.put(consolidated)

            # Archive source memories
            for m in group:
                try:
                    self._sm.validate(m.state, MemoryState.CONSOLIDATED_INTO, m.id)
                    self._substrate.update_state(
                        m.id, MemoryState.CONSOLIDATED_INTO, now=self._clock.now()
                    )
                    # Transition to archived
                    self._substrate.update_state(
                        m.id, MemoryState.ARCHIVED, now=self._clock.now()
                    )
                except Exception:
                    pass

            # Provenance edges
            for m in group:
                self._substrate.add_provenance_edge(
                    from_id=consolidated.id,
                    to_id=m.id,
                    relation="consolidates",
                )

            self._emit_event(
                memory_id=consolidated.id,
                event_type="memory.consolidated",
                namespace_key=ns.to_key(),
                payload={
                    "source_ids": [m.id for m in group],
                    "method": "frequency_cluster",
                    "compression_ratio": len(group),
                },
            )
            consolidated_count += 1

        return consolidated_count

    def reflect(
        self,
        *,
        user_id: str | None = None,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """
        Run reflection: detect patterns, stale memories, contradictions (§13, A22).
        """
        ns = Namespace(tenant_id=tenant_id, user_id=user_id)
        memories = self._substrate.list_by_namespace(
            ns, state_filter=[MemoryState.ACTIVE], limit=200
        )
        now = self._clock.now()

        stale = []
        contradictions = []
        seen_entities: dict[str, list[MemoryRecord]] = {}

        for mem in memories:
            # Stale detection
            if mem.utility < 0.2 and mem.strength < 0.2:
                stale.append(mem.id)

            # Contradiction: same entity+attribute with different values
            if mem.structured and mem.structured.entity and mem.structured.attribute:
                key = f"{mem.structured.entity}:{mem.structured.attribute}"
                seen_entities.setdefault(key, []).append(mem)

        for key, group in seen_entities.items():
            if len(group) > 1:
                values = [str(m.structured.value) for m in group if m.structured]  # type: ignore
                if len(set(values)) > 1:
                    contradictions.append({
                        "entity_attribute": key,
                        "memory_ids": [m.id for m in group],
                        "values": values,
                    })

        return {
            "total_memories": len(memories),
            "stale_candidates": stale[:10],
            "contradictions": contradictions[:5],
            "health_score": max(0, 100 - len(stale) * 2 - len(contradictions) * 5),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _supersede(
        self,
        old: MemoryRecord,
        candidate: Candidate,
        ns: Namespace,
        now: datetime,
        action: str,
    ) -> MemoryRecord:
        """Supersede an existing memory with a new candidate."""
        new_record = MemoryRecord(
            content=candidate.content,
            type=candidate.type,
            state=MemoryState.ACTIVE,
            confidence=candidate.confidence,
            importance=candidate.importance,
            source=candidate.source,
            trust_tier=candidate.trust_tier,
            structured=candidate.structured,
            namespace=ns,
            version=old.version + 1,
            supersedes=old.id,
            created_at=now,
            recorded_at=now,
            metadata=candidate.metadata,
            idempotency_key=candidate.idempotency_key,
            expires_at=candidate.expires_at,
        )

        # Supersede old
        self._sm.validate(old.state, MemoryState.SUPERSEDED, old.id)
        self._substrate.update_state(
            old.id, MemoryState.SUPERSEDED, superseded_by=new_record.id, now=now
        )

        # Write new
        self._substrate.put(new_record)
        self._substrate.add_provenance_edge(new_record.id, old.id, "supersedes")

        self._emit_event(
            memory_id=new_record.id,
            event_type="memory.superseded",
            namespace_key=ns.to_key(),
            payload={"superseded_id": old.id, "action": action},
        )
        return new_record

    def _candidate_to_record(
        self, candidate: Candidate, ns: Namespace, now: datetime
    ) -> MemoryRecord:
        return MemoryRecord(
            content=candidate.content,
            type=candidate.type,
            state=MemoryState.CANDIDATE,
            confidence=candidate.confidence,
            importance=candidate.importance,
            source=candidate.source,
            trust_tier=candidate.trust_tier,
            structured=candidate.structured,
            namespace=ns,
            created_at=now,
            recorded_at=now,
            metadata=candidate.metadata,
            idempotency_key=candidate.idempotency_key,
            expires_at=candidate.expires_at,
        )

    def _emit_event(
        self,
        memory_id: str,
        event_type: str,
        namespace_key: str,
        payload: dict | None = None,
        gate_scores: dict | None = None,
    ) -> MemoryEvent:
        event = MemoryEvent(
            memory_id=memory_id,
            event_type=event_type,
            timestamp=self._clock.now(),
            namespace_key=namespace_key,
            payload=payload or {},
            gate_scores=gate_scores or {},
        )
        self._substrate.append_event(event)
        return event
