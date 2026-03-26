"""Pipeline protocols and shared range types.

Protocols define the interface shape for the three solver stages.
Concrete parameter types (ArchetypeSignature, AbstractPlan, etc.) are
Phase 2-5 deliverables — protocols use Any for now to avoid circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..primitives.symbols import SymbolTier


@dataclass(frozen=True, slots=True)
class Range:
    """Integer range [min_val, max_val] inclusive, with validation."""

    min_val: int
    max_val: int

    def __post_init__(self) -> None:
        if self.min_val > self.max_val:
            raise ValueError(
                f"Range min_val ({self.min_val}) > max_val ({self.max_val})"
            )

    def contains(self, value: int) -> bool:
        """True if value is within [min_val, max_val]."""
        return self.min_val <= value <= self.max_val


@dataclass(frozen=True, slots=True)
class RangeFloat:
    """Float range [min_val, max_val] inclusive, with validation."""

    min_val: float
    max_val: float

    def __post_init__(self) -> None:
        if self.min_val > self.max_val:
            raise ValueError(
                f"RangeFloat min_val ({self.min_val}) > max_val ({self.max_val})"
            )

    def contains(self, value: float) -> bool:
        """True if value is within [min_val, max_val]."""
        return self.min_val <= value <= self.max_val


@dataclass(frozen=True, slots=True)
class TerminalNearMissSpec:
    """Near-miss constraint for the final dead board after all cascades.

    Shared between ArchetypeSignature and NarrativeArc — lives here to
    avoid circular imports between archetypes.registry and narrative.arc.
    """

    count: Range
    symbol_tier: SymbolTier | None


@runtime_checkable
class SpatialSolver(Protocol):
    """Solves spatial placement — where clusters, boosters, and scatters go."""

    def solve(self, plan: Any, gravity_dag: Any, variance: Any) -> Any: ...


@runtime_checkable
class BoardFiller(Protocol):
    """Fills unconstrained cells via WFC while preventing unintended clusters."""

    def fill(self, board: Any, pinned: Any, constraints: Any) -> Any: ...
