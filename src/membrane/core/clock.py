"""
Injectable Clock abstraction (A26).

No code in the core may call datetime.now() directly.
Always use the injected Clock so that tests can freeze or step time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone


class Clock(ABC):
    """Protocol for time injection (A26)."""

    @abstractmethod
    def now(self) -> datetime:
        """Return current time in UTC."""
        ...

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Sleep (may be a no-op in tests)."""
        ...


class SystemClock(Clock):
    """Default: uses the real wall clock."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        import time

        time.sleep(seconds)


class FrozenClock(Clock):
    """Frozen at a specific instant — time never advances (for unit tests)."""

    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at

    def sleep(self, seconds: float) -> None:
        pass  # no-op


class SteppedClock(Clock):
    """
    Advances by a fixed step each call to now() (for property-based tests).
    """

    def __init__(self, start: datetime, step: timedelta = timedelta(seconds=1)) -> None:
        self._current = start
        self._step = step

    def now(self) -> datetime:
        t = self._current
        self._current += self._step
        return t

    def sleep(self, seconds: float) -> None:
        self._current += timedelta(seconds=seconds)


class SimulatedClock(Clock):
    """
    Fully controllable clock for simulation (A26).
    Call .advance(td) to move time forward explicitly.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._current = start or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._current

    def advance(self, by: timedelta | float) -> None:
        if isinstance(by, (int, float)):
            by = timedelta(seconds=float(by))
        self._current += by

    def set(self, t: datetime) -> None:
        self._current = t

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


# Module-level default (may be overridden in tests)
_default_clock: Clock = SystemClock()


def get_default_clock() -> Clock:
    return _default_clock


def set_default_clock(clock: Clock) -> None:
    global _default_clock
    _default_clock = clock
