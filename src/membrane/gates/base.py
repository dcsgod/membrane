"""Gate protocol and base classes (A2.8)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from membrane.core.models import GateOutput, MemoryRecord, Candidate


@runtime_checkable
class Gate(Protocol):
    """
    All gates implement this protocol (A2.8).

    A gate is a pure function of GateInput → GateOutput.
    Gates MUST NOT perform storage I/O.
    Storage happens in apply() and in ψ.
    """

    name: str
    version: str

    def evaluate(
        self,
        memories: list[MemoryRecord],
        candidates: list[Candidate] | None,
        context: dict,
    ) -> GateOutput:
        """Return gate output with scores in [0, 1]."""
        ...
