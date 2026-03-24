"""Data structures for the CSP spatial solver.

Frozen dataclasses represent solver outputs (immutable after construction).
SolverContext is mutable — used during the solve process to track placements
and support backtracking via snapshot/restore.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config.schema import BoardConfig
from ..primitives.board import Position
from ..primitives.symbols import Symbol


# ---------------------------------------------------------------------------
# Solver outputs — frozen, immutable after construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """A planned cluster: symbol + connected positions on the board.

    wild_positions tracks which positions in the cluster are wild symbols
    that participated via bridging (C-COMPAT-2, R-WILD-5).
    """

    symbol: Symbol
    positions: frozenset[Position]
    size: int
    # Wild positions that contributed to this cluster via bridging
    wild_positions: frozenset[Position] = frozenset()


@dataclass(frozen=True, slots=True)
class NearMissAssignment:
    """A planned near-miss group: symbol + connected positions.

    Size must equal min_cluster_size - 1 (from config). The group is isolated:
    no orthogonal neighbor outside the group holds the same symbol.
    """

    symbol: Symbol
    positions: frozenset[Position]
    size: int


@dataclass(frozen=True, slots=True)
class BoosterPlacement:
    """A planned booster at a specific board position."""

    booster_type: Symbol
    position: Position


@dataclass(frozen=True, slots=True)
class WildPlacement:
    """A planned wild symbol at a specific position with its behavior.

    CSP places wilds explicitly — WFC never generates them. Behavior drives
    constraint validation: spawn requires centroid placement, bridge requires
    adjacency to 2+ clusters, idle requires no multi-cluster adjacency.
    """

    position: Position
    # "spawn" = placed at centroid of 7-8 cluster, "bridge" = connecting groups, "idle" = not bridging
    behavior: str


@dataclass(frozen=True, slots=True)
class SpatialStep:
    """One cascade step's spatial assignments — clusters, near-misses, scatters, boosters, wilds."""

    clusters: tuple[ClusterAssignment, ...]
    near_misses: tuple[NearMissAssignment, ...]
    scatter_positions: frozenset[Position]
    boosters: tuple[BoosterPlacement, ...]
    # Wild symbols placed at this step — spawn, bridge, or idle behavior
    wild_placements: tuple[WildPlacement, ...] = ()


@dataclass(frozen=True, slots=True)
class SpatialPlan:
    """Complete spatial plan — one SpatialStep per cascade step."""

    steps: tuple[SpatialStep, ...]


# ---------------------------------------------------------------------------
# Mutable solver state — used during construction, not part of output
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _ContextSnapshot:
    """Frozen snapshot of SolverContext for backtracking."""

    occupied: frozenset[Position]
    clusters: tuple[ClusterAssignment, ...]
    near_misses: tuple[NearMissAssignment, ...]
    scatter_positions: frozenset[Position]
    booster_placements: tuple[BoosterPlacement, ...]
    wild_placements: tuple[WildPlacement, ...]


class SolverContext:
    """Mutable state passed to constraints during the solve process.

    Tracks all placements made so far. Supports snapshot/restore for
    backtracking when a constraint is violated.
    """

    __slots__ = (
        "board_config",
        "occupied",
        "clusters",
        "near_misses",
        "scatter_positions",
        "booster_placements",
        "wild_placements",
    )

    def __init__(self, board_config: BoardConfig) -> None:
        self.board_config: BoardConfig = board_config
        self.occupied: set[Position] = set()
        self.clusters: list[ClusterAssignment] = []
        self.near_misses: list[NearMissAssignment] = []
        self.scatter_positions: set[Position] = set()
        self.booster_placements: list[BoosterPlacement] = []
        self.wild_placements: list[WildPlacement] = []

    def snapshot(self) -> _ContextSnapshot:
        """Capture current state for later restore."""
        return _ContextSnapshot(
            occupied=frozenset(self.occupied),
            clusters=tuple(self.clusters),
            near_misses=tuple(self.near_misses),
            scatter_positions=frozenset(self.scatter_positions),
            booster_placements=tuple(self.booster_placements),
            wild_placements=tuple(self.wild_placements),
        )

    def restore(self, snap: _ContextSnapshot) -> None:
        """Restore state from a previous snapshot."""
        self.occupied = set(snap.occupied)
        self.clusters = list(snap.clusters)
        self.near_misses = list(snap.near_misses)
        self.scatter_positions = set(snap.scatter_positions)
        self.booster_placements = list(snap.booster_placements)
        self.wild_placements = list(snap.wild_placements)

    def reset(self) -> None:
        """Clear all placements — start fresh for a new retry."""
        self.occupied.clear()
        self.clusters.clear()
        self.near_misses.clear()
        self.scatter_positions.clear()
        self.booster_placements.clear()
        self.wild_placements.clear()
