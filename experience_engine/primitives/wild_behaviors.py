"""Wild placement behaviors — protocol + per-behavior implementations.

Replaces the `if behavior == "spawn" / "bridge" / "idle"` chain in
`spatial_solver.solver._place_wild` with a single registry. Adding a new
wild behavior (e.g. "absorb") requires only one new class + one registry
entry — no edits to the call site.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable, Protocol

from .board import Position

if TYPE_CHECKING:
    from ..spatial_solver.data_types import SolverContext


# Type alias for the adjacency-filter callback the solver provides to
# behaviors so they can look up positions by cluster adjacency without
# knowing about SolverContext internals.
AdjacencyFilter = Callable[[list[Position], int, int | None], list[Position]]
WeightedChoice = Callable[
    [list[Position], dict[Position, float] | None, random.Random],
    Position | None,
]


class WildPlacementBehavior(Protocol):
    """Encapsulates a single wild-placement behavior (spawn / bridge / idle)
    consumed by the spatial solver.
    """

    def select_position(
        self,
        available: list[Position],
        context: "SolverContext",
        adjacent_to_clusters: AdjacencyFilter,
        weighted_choice: WeightedChoice,
        spatial_bias: dict[Position, float] | None,
        rng: random.Random,
    ) -> Position | None:
        """Pick a wild position from available candidates."""
        ...


class _SpawnBehavior:
    """Wild placed near a cluster — adjacent to maximize centroid-like
    placement.
    """

    def select_position(
        self, available, context, adjacent_to_clusters, weighted_choice,
        spatial_bias, rng,
    ) -> Position | None:
        candidates = adjacent_to_clusters(available, 1, None)
        if not candidates:
            # Fallback: any available position
            candidates = available
        return weighted_choice(candidates, spatial_bias, rng)


class _BridgeBehavior:
    """Wild must be adjacent to 2+ distinct clusters. No fallback — bridge
    is a strict constraint.
    """

    def select_position(
        self, available, context, adjacent_to_clusters, weighted_choice,
        spatial_bias, rng,
    ) -> Position | None:
        candidates = adjacent_to_clusters(available, 2, None)
        if not candidates:
            return None
        return weighted_choice(candidates, spatial_bias, rng)


class _IdleBehavior:
    """Wild must NOT be adjacent to 2+ clusters — keeps the wild from
    accidentally bridging.
    """

    def select_position(
        self, available, context, adjacent_to_clusters, weighted_choice,
        spatial_bias, rng,
    ) -> Position | None:
        candidates = adjacent_to_clusters(available, 0, 1)
        if not candidates:
            candidates = available
        return weighted_choice(candidates, spatial_bias, rng)


# Registry — single source of truth for the solver call site.
WILD_BEHAVIORS: dict[str, WildPlacementBehavior] = {
    "spawn": _SpawnBehavior(),
    "bridge": _BridgeBehavior(),
    "idle": _IdleBehavior(),
}
