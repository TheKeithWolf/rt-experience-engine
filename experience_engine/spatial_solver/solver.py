"""CSP Spatial Solver — constructive placement with pluggable constraints.

Grows connected cluster shapes via random frontier expansion, places near-miss
groups, scatters, and boosters, then validates all registered constraints.
Retries on failure up to config limits (max_construction_retries, csp_max_solve_time_ms).

All tweakable constants come from MasterConfig — zero hardcoded values.
"""

from __future__ import annotations

import random
import time
from typing import Any

from ..config.schema import MasterConfig
from ..primitives.board import Position, orthogonal_neighbors
from ..primitives.symbols import Symbol
from ..primitives.wild_behaviors import WILD_BEHAVIORS
from .constraints import (
    ClusterConnectivity,
    ClusterNonOverlap,
    NearMissIsolation,
    SameSymbolNonAdjacency,
    ScatterNonOverlap,
    SpatialConstraint,
)
from .data_types import (
    BoosterPlacement,
    ClusterAssignment,
    NearMissAssignment,
    SolverContext,

    SpatialStep,
    WildPlacement,
)



class SolveFailed(Exception):
    """Raised when the CSP solver exhausts retries or time budget."""


class CSPSpatialSolver:
    """Constructive CSP solver for spatial placement.

    Implements the SpatialSolver protocol. Constraints are pluggable:
    add_constraint() registers new constraints without modifying solver logic.

    Default constraints (registered at __init__):
    - ClusterConnectivity — orthogonal BFS connectivity
    - ClusterNonOverlap — no shared positions between entities
    - NearMissIsolation — isolated groups of min_cluster_size - 1
    - ScatterNonOverlap — scatters don't overlap clusters/near-misses
    """

    def __init__(self, config: MasterConfig) -> None:
        self._board_config = config.board
        self._symbol_config = config.symbols
        # Standard symbol names list — used by solve() to map tier → concrete symbols
        self._config_symbols_standard: list[str] = list(config.symbols.standard)
        self._max_retries: int = config.solvers.max_construction_retries
        # Convert ms to seconds for time.monotonic() comparison
        self._max_time_s: float = config.solvers.csp_max_solve_time_ms / 1000.0

        # All valid board positions — precomputed once for seed/scatter selection
        self._all_positions: list[Position] = [
            Position(reel, row)
            for reel in range(config.board.num_reels)
            for row in range(config.board.num_rows)
        ]

        # Default constraints with thresholds from config
        self._constraints: list[SpatialConstraint] = [
            ClusterConnectivity(),
            ClusterNonOverlap(),
            NearMissIsolation(config.board.min_cluster_size),
            ScatterNonOverlap(),
            SameSymbolNonAdjacency(),
        ]

    def add_constraint(self, constraint: SpatialConstraint) -> None:
        """Register an additional constraint — checked after each placement attempt.

        New constraint types (booster centroid, gravity mapping, chain relations)
        are added in later phases via this method, requiring zero solver changes.
        """
        self._constraints.append(constraint)

    def remove_constraint_type(self, constraint_type: type) -> None:
        """Remove all constraints of a specific type.

        Used by the cascade pipeline to swap out PostGravityMapping between
        steps — each step has a different set of allowed positions.
        """
        self._constraints = [
            c for c in self._constraints if not isinstance(c, constraint_type)
        ]

    def solve_step(
        self,
        cluster_specs: list[tuple[Symbol, int]],
        near_miss_specs: list[tuple[Symbol, int]],
        scatter_count: int,
        booster_specs: list[tuple[Symbol, Position | None]],
        spatial_bias: dict[Position, float] | None = None,
        rng: random.Random | None = None,
        wild_specs: list[str] | None = None,
    ) -> SpatialStep:
        """Solve a single step — place clusters, wilds, near-misses, scatters, boosters.

        Args:
            cluster_specs: (symbol, target_size) for each cluster to place.
            near_miss_specs: (symbol, target_size) for each near-miss group.
            scatter_count: number of scatter positions to place.
            booster_specs: (booster_type, hint_position_or_None) for each booster.
            spatial_bias: optional position → weight map for biased placement.
                Positions with higher weight are preferred during seed selection
                and frontier expansion (CONSTRAINT-CSP-4: soft constraints).
            rng: random number generator for reproducibility.
            wild_specs: list of wild behavior strings ("spawn", "bridge", "idle")
                for each wild to place at this step. None or empty = no wilds.

        Returns:
            SpatialStep with all placements.

        Raises:
            SolveFailed: if max_construction_retries or time budget exhausted.
        """
        if rng is None:
            rng = random.Random()
        if wild_specs is None:
            wild_specs = []

        context = SolverContext(self._board_config)
        start_time = time.monotonic()

        for attempt in range(self._max_retries):
            # Check time budget at start of each retry
            elapsed_s = time.monotonic() - start_time
            if elapsed_s > self._max_time_s:
                raise SolveFailed(
                    f"Time budget exceeded ({self._max_time_s * 1000:.0f}ms) "
                    f"after {attempt} attempts"
                )

            context.reset()
            success = True

            # 1. Place clusters — grow connected shapes
            for symbol, target_size in cluster_specs:
                available = self._available_positions(context)
                shape = self._grow_connected_shape(
                    target_size, available, rng, spatial_bias
                )
                if shape is None:
                    success = False
                    break
                assignment = ClusterAssignment(
                    symbol=symbol,
                    positions=shape,
                    size=len(shape),
                )
                context.clusters.append(assignment)
                context.occupied.update(shape)

            if not success:
                continue

            # 2. Place wilds — behavior determines placement strategy
            for behavior in wild_specs:
                wild_pos = self._place_wild(context, behavior, rng, spatial_bias)
                if wild_pos is None:
                    success = False
                    break
                placement = WildPlacement(position=wild_pos, behavior=behavior)
                context.wild_placements.append(placement)
                context.occupied.add(wild_pos)

            if not success:
                continue

            # 3. Place near-misses — same growth algorithm
            for symbol, target_size in near_miss_specs:
                available = self._available_positions(context)
                shape = self._grow_connected_shape(
                    target_size, available, rng, spatial_bias
                )
                if shape is None:
                    success = False
                    break
                assignment = NearMissAssignment(
                    symbol=symbol,
                    positions=shape,
                    size=len(shape),
                )
                context.near_misses.append(assignment)
                context.occupied.update(shape)

            if not success:
                continue

            # 4. Place scatters — random unoccupied positions
            if scatter_count > 0:
                available = self._available_positions(context)
                if len(available) < scatter_count:
                    continue
                scatter_positions = self._pick_positions(
                    scatter_count, available, rng, spatial_bias
                )
                context.scatter_positions.update(scatter_positions)
                context.occupied.update(scatter_positions)

            # 5. Place boosters — at hint position or random unoccupied
            for booster_type, hint_pos in booster_specs:
                if hint_pos is not None and hint_pos not in context.occupied:
                    pos = hint_pos
                else:
                    available = self._available_positions(context)
                    if not available:
                        success = False
                        break
                    pos = _weighted_choice(available, spatial_bias, rng)
                placement_b = BoosterPlacement(
                    booster_type=booster_type,
                    position=pos,
                )
                context.booster_placements.append(placement_b)
                context.occupied.add(pos)

            if not success:
                continue

            # 6. Validate all constraints
            if self._check_all_constraints(context):
                return self._freeze_step(context)

        raise SolveFailed(
            f"Exhausted {self._max_retries} retries without finding a valid placement"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _place_wild(
        self,
        context: SolverContext,
        behavior: str,
        rng: random.Random,
        spatial_bias: dict[Position, float] | None,
    ) -> Position | None:
        """Find a position for a wild symbol based on its behavior.

        B2: dispatch is via the WILD_BEHAVIORS registry — one entry per
        behavior. Adding a new behavior requires only registering a new
        WildPlacementBehavior; no edits here.
        """
        available = self._available_positions(context)
        if not available:
            return None
        try:
            wild_behavior = WILD_BEHAVIORS[behavior]
        except KeyError as exc:
            raise KeyError(
                f"Unknown wild placement behavior: {behavior!r}. "
                f"Register a WildPlacementBehavior in primitives.wild_behaviors."
            ) from exc

        # Adjacency filter: solver-private helper exposed as a callback so
        # behaviors don't need access to SolverContext internals.
        def _filter(
            positions: list[Position],
            min_clusters: int,
            max_clusters: int | None,
        ) -> list[Position]:
            return self._positions_adjacent_to_clusters(
                positions, context,
                min_clusters=min_clusters, max_clusters=max_clusters,
            )

        return wild_behavior.select_position(
            available, context, _filter, _weighted_choice, spatial_bias, rng,
        )

    def _positions_adjacent_to_clusters(
        self,
        available: list[Position],
        context: SolverContext,
        min_clusters: int = 0,
        max_clusters: int | None = None,
    ) -> list[Position]:
        """Filter available positions by how many distinct clusters they neighbor.

        Returns positions adjacent to at least min_clusters and at most max_clusters
        distinct cluster assignments. Used for wild placement strategies.
        """
        result: list[Position] = []
        for pos in available:
            adjacent_count = 0
            for cluster in context.clusters:
                for neighbor in orthogonal_neighbors(pos, self._board_config):
                    if neighbor in cluster.positions:
                        adjacent_count += 1
                        break
            if adjacent_count >= min_clusters:
                if max_clusters is None or adjacent_count <= max_clusters:
                    result.append(pos)
        return result

    def _available_positions(self, context: SolverContext) -> list[Position]:
        """Board positions not yet occupied by any assignment."""
        return [p for p in self._all_positions if p not in context.occupied]

    def _grow_connected_shape(
        self,
        size: int,
        available: list[Position],
        rng: random.Random,
        spatial_bias: dict[Position, float] | None,
    ) -> frozenset[Position] | None:
        """Grow a connected shape of exactly `size` via random frontier expansion.

        Picks a seed position, then expands by randomly selecting from the
        frontier (orthogonal neighbors that are available and not yet in shape).
        Returns None if the frontier is exhausted before reaching target size.
        """
        if len(available) < size:
            return None

        available_set = set(available)
        seed = _weighted_choice(available, spatial_bias, rng)
        shape: set[Position] = {seed}
        available_set.discard(seed)

        # Frontier: available orthogonal neighbors of current shape
        frontier: set[Position] = set()
        for neighbor in orthogonal_neighbors(seed, self._board_config):
            if neighbor in available_set:
                frontier.add(neighbor)

        while len(shape) < size:
            if not frontier:
                return None

            # Pick from frontier, weighted by spatial bias
            chosen = _weighted_choice(list(frontier), spatial_bias, rng)
            shape.add(chosen)
            frontier.discard(chosen)
            available_set.discard(chosen)

            # Add new neighbors of the chosen position to the frontier
            for neighbor in orthogonal_neighbors(chosen, self._board_config):
                if neighbor in available_set and neighbor not in shape:
                    frontier.add(neighbor)

        return frozenset(shape)

    def _pick_positions(
        self,
        count: int,
        available: list[Position],
        rng: random.Random,
        spatial_bias: dict[Position, float] | None,
    ) -> frozenset[Position]:
        """Pick `count` distinct positions from available, weighted by bias."""
        picked: set[Position] = set()
        remaining = list(available)
        for _ in range(count):
            chosen = _weighted_choice(remaining, spatial_bias, rng)
            picked.add(chosen)
            remaining.remove(chosen)
        return frozenset(picked)

    def _check_all_constraints(self, context: SolverContext) -> bool:
        """Validate all registered constraints against current state."""
        return all(c.is_satisfied(context) for c in self._constraints)

    @staticmethod
    def _freeze_step(context: SolverContext) -> SpatialStep:
        """Convert mutable context into a frozen SpatialStep output."""
        return SpatialStep(
            clusters=tuple(context.clusters),
            near_misses=tuple(context.near_misses),
            scatter_positions=frozenset(context.scatter_positions),
            boosters=tuple(context.booster_placements),
            wild_placements=tuple(context.wild_placements),
        )


def _weighted_choice(
    candidates: list[Position],
    bias: dict[Position, float] | None,
    rng: random.Random,
) -> Position:
    """Pick a position from candidates, weighted by bias map.

    When bias is None, selection is uniform. Positions not in the bias map
    receive weight 1.0 (neutral — no preference).
    """
    if bias is None or not bias:
        return rng.choice(candidates)

    weights = [bias.get(pos, 1.0) for pos in candidates]
    # random.choices returns a list; we want a single element
    return rng.choices(candidates, weights=weights, k=1)[0]
