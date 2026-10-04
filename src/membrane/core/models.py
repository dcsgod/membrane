"""
Core data models for Membrane.

These are the authoritative data types for the memory system.
All storage backends, gates, and policies operate on these models.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    """Return current UTC datetime. Always use Clock in core code."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"m_{uuid.uuid4().hex[:16]}"


def _new_event_id() -> str:
    return f"e_{uuid.uuid4().hex[:16]}"


class MemoryType(str, Enum):
    """Memory types that determine storage, decay, and policy defaults."""

    SEMANTIC = "semantic"          # Stable facts
    EPISODIC = "episodic"          # Events/experiences
    WORKING = "working"            # Short-lived task state (TTL)
    PROCEDURAL = "procedural"      # Reusable strategies/workflows
    PREFERENCE = "preference"      # User/application preferences
    STATE = "state"                # Current entity state (temporal)
    OUTCOME = "outcome"            # Decision outcomes (A9)
    DECISION = "decision"          # Agent decisions with rationale (A9)
    CAUSAL = "causal"              # Cause-effect edges with confidence (R6)
    CONSOLIDATED = "consolidated"  # Derived from consolidation (A12)


class MemoryState(str, Enum):
    """
    The full memory lifecycle state machine (A7.2).

    Transitions are validated by the StateMachine and every
    transition emits a MemoryEvent.
    """

    CANDIDATE = "candidate"
    WRITTEN = "written"
    ACTIVE = "active"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    CONSOLIDATED_INTO = "consolidated_into"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


class TrustTier(str, Enum):
    """
    Trust hierarchy for memories (A17).

    system > human_verified > user_asserted > tool_output >
    agent_inferred > external_content > untrusted
    """

    SYSTEM = "system"
    HUMAN_VERIFIED = "human_verified"
    USER_ASSERTED = "user_asserted"
    TOOL_OUTPUT = "tool_output"
    AGENT_INFERRED = "agent_inferred"
    EXTERNAL_CONTENT = "external_content"
    UNTRUSTED = "untrusted"

    def __lt__(self, other: "TrustTier") -> bool:
        order = list(TrustTier)
        return order.index(self) > order.index(other)  # system=0 is highest


TRUST_ORDER = {
    TrustTier.SYSTEM: 6,
    TrustTier.HUMAN_VERIFIED: 5,
    TrustTier.USER_ASSERTED: 4,
    TrustTier.TOOL_OUTPUT: 3,
    TrustTier.AGENT_INFERRED: 2,
    TrustTier.EXTERNAL_CONTENT: 1,
    TrustTier.UNTRUSTED: 0,
}


class Namespace(BaseModel):
    """Composite namespace for multi-tenancy (§27)."""

    tenant_id: str = "default"
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_key(self) -> str:
        """Canonical string key for namespace-scoped lookups."""
        parts = [self.tenant_id]
        if self.user_id:
            parts.append(f"u:{self.user_id}")
        if self.agent_id:
            parts.append(f"a:{self.agent_id}")
        if self.session_id:
            parts.append(f"s:{self.session_id}")
        return "/".join(parts)

    def matches(self, other: "Namespace") -> bool:
        """True if this namespace is at-or-within other (scope check)."""
        if self.tenant_id != other.tenant_id:
            return False
        if other.user_id and self.user_id != other.user_id:
            return False
        if other.agent_id and self.agent_id != other.agent_id:
            return False
        if other.session_id and self.session_id != other.session_id:
            return False
        return True


class Source(BaseModel):
    """Provenance source for a memory (§28, A8)."""

    type: str = "user"  # conversation|document|database|api|tool|agent|human|system|import|replay|policy|outcome|inference|consolidation
    provider: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    url: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredFields(BaseModel):
    """
    Structured knowledge block enabling exact lookups and
    contradiction detection without embeddings (§14, A14).
    """

    entity: Optional[str] = None       # e.g. "product_123"
    attribute: Optional[str] = None    # e.g. "price"
    value: Optional[Any] = None        # e.g. 12.99
    unit: Optional[str] = None         # e.g. "USD"
    qualifiers: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    """
    The core memory record (A7.1).

    This is the authoritative, versioned, temporal, provenance-carrying
    state object — not a text chunk.
    """

    # Identity
    id: str = Field(default_factory=_new_id)
    content: str
    type: MemoryType = MemoryType.SEMANTIC
    idempotency_key: Optional[str] = None

    # Lifecycle state (A7.2)
    state: MemoryState = MemoryState.CANDIDATE

    # Scoring signals [0, 1] (§31, A10)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    utility: float = Field(default=0.5, ge=0.0, le=1.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)

    # Bitemporal timestamps (§29, A16)
    recorded_at: datetime = Field(default_factory=_now)   # transaction time
    created_at: datetime = Field(default_factory=_now)
    observed_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None                  # valid time
    valid_until: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Provenance & trust (A8, A17)
    source: Source = Field(default_factory=Source)
    trust_tier: TrustTier = TrustTier.USER_ASSERTED

    # Versioning & supersession (§10, A16)
    version: int = 1
    supersedes: Optional[str] = None        # memory_id this replaces
    superseded_by: Optional[str] = None     # memory_id that replaced this

    # Namespace (§27)
    namespace: Namespace = Field(default_factory=Namespace)

    # Structured knowledge (§14, A14)
    structured: Optional[StructuredFields] = None

    # PII & privacy (A17.4)
    pii: list[str] = Field(default_factory=list)

    # Flexible metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Embedding reference (lazy, §14, A25)
    embedding_reference: Optional[str] = None

    # Consolidation links (A12)
    consolidated_from: list[str] = Field(default_factory=list)

    @field_validator("confidence", "importance", "utility", "strength", mode="before")
    @classmethod
    def clamp_unit_interval(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def is_active(self) -> bool:
        return self.state == MemoryState.ACTIVE

    def is_retrievable(self) -> bool:
        return self.state in (MemoryState.ACTIVE, MemoryState.WRITTEN)

    def namespace_key(self) -> str:
        return self.namespace.to_key()


class GateOutput(BaseModel):
    """
    Output of a gate evaluation (A2.8).

    Scores are in [0, 1].
    All metadata is preserved for explain() (A18).
    """

    scores: dict[str, float] = Field(default_factory=dict)  # memory_id -> score
    features: dict[str, Any] = Field(default_factory=dict)  # gate input features
    reasons: dict[str, str] = Field(default_factory=dict)   # memory_id -> reason
    policy_id: str = "default"
    gate_version: str = "1.0"
    gate_name: str = ""
    threshold: float = 0.5


class Candidate(BaseModel):
    """
    A proposed memory from candidate generation φ (A2.3).
    """

    content: str
    type: MemoryType = MemoryType.SEMANTIC
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Source = Field(default_factory=Source)
    trust_tier: TrustTier = TrustTier.USER_ASSERTED
    structured: Optional[StructuredFields] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = None
    expires_at: Optional[datetime] = None


class MemoryScore(BaseModel):
    """
    Detailed score breakdown for a memory retrieval (§31, A18).
    All values in [0, 1].
    """

    memory_id: str
    relevance: float = 0.0
    freshness: float = 0.0
    confidence: float = 0.0
    utility: float = 0.0
    importance: float = 0.0
    temporal_match: float = 0.0
    strength: float = 0.0       # A10
    trust: float = 0.0          # A17
    entity_match: float = 0.0
    lexical: float = 0.0
    semantic: float = 0.0
    redundancy: float = 0.0     # penalty
    final_score: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)


class Hit(BaseModel):
    """A scored retrieval candidate from a substrate."""

    memory: MemoryRecord
    score: MemoryScore
    substrate: str = "sqlite"


class RecallResult(BaseModel):
    """
    The output of a recall() call (A2.5, A19).
    """

    recall_id: str = Field(default_factory=lambda: f"r_{uuid.uuid4().hex[:12]}")
    query: str
    memories: list[MemoryRecord] = Field(default_factory=list)
    scores: list[MemoryScore] = Field(default_factory=list)
    tokens_used: int = 0
    dropped: list[dict[str, Any]] = Field(default_factory=list)  # near-misses
    plan_id: Optional[str] = None
    degraded: bool = False
    degraded_reason: Optional[str] = None

    def to_text(self, format: str = "prompt") -> str:
        """Render memories as an injectable text block (A19)."""
        if not self.memories:
            return ""
        lines = ["<memories>"]
        for mem in self.memories:
            tier = mem.trust_tier.value
            mtype = mem.type.value
            lines.append(
                f'  <memory id="{mem.id}" type="{mtype}" trust="{tier}" '
                f'confidence="{mem.confidence:.2f}">'
            )
            lines.append(f"    {mem.content}")
            lines.append("  </memory>")
        lines.append("</memories>")
        return "\n".join(lines)


class MemoryEvent(BaseModel):
    """
    Append-only event record for every memory lifecycle transition (§56, A7.2).
    Forms a hash-chained tamper-evident log (A17.4).
    """

    id: str = Field(default_factory=_new_event_id)
    memory_id: str
    event_type: str   # memory.created|read|updated|superseded|archived|forgotten|...
    timestamp: datetime = Field(default_factory=_now)
    actor: str = "system"
    namespace_key: str = "default"
    policy_id: str = "default"
    policy_version: str = "1.0"
    gate_scores: dict[str, float] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None
    prev_hash: Optional[str] = None  # for tamper-evident chaining
    hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of canonical event representation."""
        import hashlib
        import json

        canonical = json.dumps(
            {
                "id": self.id,
                "memory_id": self.memory_id,
                "event_type": self.event_type,
                "timestamp": self.timestamp.isoformat(),
                "actor": self.actor,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


class Context(BaseModel):
    """
    Runtime context for a memory operation (A2.5).
    """

    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    tenant_id: str = "default"
    task: Optional[str] = None
    time: Optional[datetime] = None
    environment: dict[str, Any] = Field(default_factory=dict)
    token_budget: Optional[int] = None

    def to_namespace(self) -> Namespace:
        return Namespace(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )


class Query(BaseModel):
    """A structured memory query."""

    text: str
    namespace: Optional[Namespace] = None
    type_filter: Optional[list[MemoryType]] = None
    state_filter: list[MemoryState] = Field(default_factory=lambda: [MemoryState.ACTIVE])
    entity: Optional[str] = None
    attribute: Optional[str] = None
    at: Optional[datetime] = None       # valid time
    as_of: Optional[datetime] = None    # record time
    limit: int = 10
    min_score: float = 0.0
    include_superseded: bool = False


class Snapshot(BaseModel):
    """A lightweight pointer into the event log (A16.2)."""

    id: str = Field(default_factory=lambda: f"snap_{uuid.uuid4().hex[:12]}")
    label: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    event_cursor: Optional[str] = None   # last event_id at snapshot time
    namespace_key: str = "default"


class ExplainResult(BaseModel):
    """
    Explanation of a memory decision (A18).
    """

    ref_id: str          # recall_id or memory_id
    ref_type: str        # "recall" | "memory" | "plan"
    summary: str
    scores: list[MemoryScore] = Field(default_factory=list)
    gate_scores: dict[str, Any] = Field(default_factory=dict)
    plan: Optional[dict[str, Any]] = None
    near_misses: list[dict[str, Any]] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    events: list[MemoryEvent] = Field(default_factory=list)
