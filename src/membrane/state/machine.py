"""
Memory lifecycle state machine (A7.2).

Defines valid transitions and enforces them at runtime.
Every transition emits a MemoryEvent.
"""

from __future__ import annotations



from membrane.core.models import MemoryState
from membrane.core.exceptions import InvalidTransitionError

# Adjacency list of valid transitions (A7.2)
# Format: from_state -> set of allowed to_states
_VALID_TRANSITIONS: dict[MemoryState, set[MemoryState]] = {
    MemoryState.CANDIDATE: {
        MemoryState.WRITTEN,
        MemoryState.REJECTED,
        MemoryState.QUARANTINED,
    },
    MemoryState.WRITTEN: {
        MemoryState.ACTIVE,
        MemoryState.REJECTED,
    },
    MemoryState.ACTIVE: {
        MemoryState.SUPERSEDED,
        MemoryState.EXPIRED,
        MemoryState.CONSOLIDATED_INTO,
        MemoryState.ARCHIVED,
        MemoryState.QUARANTINED,   # can be retroactively quarantined
    },
    MemoryState.QUARANTINED: {
        MemoryState.WRITTEN,   # approved → proceed with write
        MemoryState.REJECTED,
    },
    MemoryState.SUPERSEDED: {
        MemoryState.ARCHIVED,
    },
    MemoryState.EXPIRED: {
        MemoryState.ARCHIVED,
    },
    MemoryState.CONSOLIDATED_INTO: {
        MemoryState.ARCHIVED,
    },
    MemoryState.ARCHIVED: {
        MemoryState.FORGOTTEN,
    },
    # Terminal states — no further transitions
    MemoryState.REJECTED: set(),
    MemoryState.FORGOTTEN: set(),
}


class StateMachine:
    """Validates and applies memory lifecycle transitions (A7.2)."""

    def validate(self, from_state: MemoryState, to_state: MemoryState, memory_id: str = "") -> None:
        """Raise InvalidTransitionError if the transition is illegal."""
        allowed = _VALID_TRANSITIONS.get(from_state, set())
        if to_state not in allowed:
            raise InvalidTransitionError(from_state.value, to_state.value, memory_id)

    def can_transition(self, from_state: MemoryState, to_state: MemoryState) -> bool:
        """Return True if the transition is valid."""
        return to_state in _VALID_TRANSITIONS.get(from_state, set())

    def allowed_transitions(self, from_state: MemoryState) -> frozenset[MemoryState]:
        """Return all valid next states from a given state."""
        return frozenset(_VALID_TRANSITIONS.get(from_state, set()))

    def is_terminal(self, state: MemoryState) -> bool:
        """Return True if no further transitions are allowed."""
        return not bool(_VALID_TRANSITIONS.get(state))

    def is_retrievable(self, state: MemoryState) -> bool:
        """
        Return True if a memory in this state may be returned in recall.
        Only ACTIVE and WRITTEN memories are retrievable by default.
        SUPERSEDED can be returned via include_superseded=True queries.
        """
        return state in (MemoryState.ACTIVE, MemoryState.WRITTEN)


# Module-level singleton
_state_machine = StateMachine()


def get_state_machine() -> StateMachine:
    return _state_machine
