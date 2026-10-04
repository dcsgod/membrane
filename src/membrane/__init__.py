"""
Membrane — The programmable memory layer for AI.

A programmable memory runtime that dynamically manages information
across heterogeneous storage systems for AI applications.

Usage::

    from membrane import Memory

    memory = Memory()
    memory.remember("The user prefers Python over JavaScript.", user_id="u1")
    result = memory.recall("What does this user prefer?", user_id="u1")
    print(result)

No API key. No Docker. No external database.
"""

from membrane.core.memory import Memory
from membrane.core.models import (
    MemoryRecord,
    MemoryType,
    MemoryState,
    RecallResult,
    MemoryScore,
    Namespace,
    Source,
    TrustTier,
)
from membrane.core.exceptions import (
    MembraneError,
    InvalidTransitionError,
    NamespaceViolationError,
    MissingExtraError,
    IdempotencyConflictError,
)

__version__ = "0.1.0"
__all__ = [
    "Memory",
    "MemoryRecord",
    "MemoryType",
    "MemoryState",
    "RecallResult",
    "MemoryScore",
    "Namespace",
    "Source",
    "TrustTier",
    "MembraneError",
    "InvalidTransitionError",
    "NamespaceViolationError",
    "MissingExtraError",
    "IdempotencyConflictError",
    "__version__",
]
