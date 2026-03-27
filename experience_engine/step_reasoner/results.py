"""Step results — outcome records from a completed cascade step.

SpawnRecord and FireRecord capture booster lifecycle events.
StepResult aggregates all outcomes and is consumed by
ProgressTracker.update() to advance cumulative state.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..primitives.board import Position
from ..primitives.symbols import SymbolTier
from .progress import ClusterRecord


@dataclass(frozen=True, slots=True)
class SpawnRecord:
    """A booster that was spawned during a completed step."""

    booster_type: str
    position: Position
    # Index into StepResult.clusters identifying which cluster spawned this booster
    source_cluster_index: int
    step_index: int
    # Directional orientation for boosters that fire along an axis (e.g. "H"/"V" for rockets)
    orientation: str | None = None


@dataclass(frozen=True, slots=True)
class FireRecord:
    """A booster that fired during a completed step."""

    booster_type: str
    position: Position
    # Number of cells cleared by this booster's effect
    affected_count: int
    # Whether this fire was triggered by another booster's chain reaction
    chain_triggered: bool
    step_index: int


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of executing one cascade step.

    Frozen after construction. Produced by the execution layer (Step 7),
    consumed by ProgressTracker.update() and validation.
    """

    step_index: int
    clusters: tuple[ClusterRecord, ...]
    spawns: tuple[SpawnRecord, ...]
    fires: tuple[FireRecord, ...]
    # Symbol tier observed at this step (for narrative arc tracking)
    symbol_tier: SymbolTier | None
    # Total centipayout contributed by this step
    step_payout: int
