"""
Exceptions for Membrane (§69).
"""

from __future__ import annotations


class MembraneError(Exception):
    """Base class for all Membrane errors."""


class InvalidTransitionError(MembraneError):
    """Raised when an illegal state transition is attempted (A7.2)."""

    def __init__(self, from_state: str, to_state: str, memory_id: str = "") -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.memory_id = memory_id
        super().__init__(
            f"Invalid transition {from_state!r} → {to_state!r} for memory {memory_id!r}"
        )


class NamespaceViolationError(MembraneError):
    """Raised when a cross-namespace access is detected (§27, A5.1)."""

    def __init__(self, message: str = "Cross-namespace access denied") -> None:
        super().__init__(message)


class MissingExtraError(MembraneError):
    """
    Raised when an optional feature is used without the required extra installed.

    Example: using embeddings without `pip install membrane-memory[embeddings-local]`
    """

    def __init__(self, feature: str, extra: str) -> None:
        self.feature = feature
        self.extra = extra
        super().__init__(
            f"Feature {feature!r} requires the [{extra}] extra. "
            f"Install it with: pip install membrane-memory[{extra}]"
        )


class IdempotencyConflictError(MembraneError):
    """Raised when an idempotency key conflicts with a different payload (§54)."""

    def __init__(self, key: str, existing_id: str) -> None:
        self.key = key
        self.existing_id = existing_id
        super().__init__(
            f"Idempotency key {key!r} already exists as memory {existing_id!r}"
        )


class SubstrateError(MembraneError):
    """Raised by a substrate when an operation fails."""

    def __init__(self, substrate: str, operation: str, reason: str) -> None:
        self.substrate = substrate
        self.operation = operation
        self.reason = reason
        super().__init__(f"Substrate {substrate!r} failed on {operation!r}: {reason}")


class PolicyViolationError(MembraneError):
    """Raised when a hard safety rule is violated (A2.8)."""

    def __init__(self, rule: str, details: str = "") -> None:
        self.rule = rule
        super().__init__(f"Policy violation [{rule}]: {details}")


class QuarantinedError(MembraneError):
    """Raised when a quarantined memory is accessed through a read path (A17.2)."""

    def __init__(self, memory_id: str) -> None:
        super().__init__(
            f"Memory {memory_id!r} is quarantined and cannot be retrieved until approved."
        )


class SnapshotNotFoundError(MembraneError):
    """Raised when a snapshot is not found (A16.2)."""

    def __init__(self, snapshot_id: str) -> None:
        super().__init__(f"Snapshot {snapshot_id!r} not found.")
