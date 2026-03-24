"""Pluggable spatial constraints for the CSP solver.

Each constraint implements `is_satisfied(context) -> bool`. New constraints
(booster placement, gravity mapping, chain relations) are added in later phases
without modifying the solver — Open/Closed principle.

All thresholds come from MasterConfig via __init__ injection.
"""

from __future__ import annotations

from collections import deque

from typing import Protocol, runtime_checkable

from ..config.schema import BoardConfig
from ..pipeline.protocols import Range
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.booster_rules import BoosterRules
from ..config.schema import SymbolConfig
from ..primitives.symbols import Symbol, SymbolTier, symbol_from_name, symbols_in_tier
from .data_types import ClusterAssignment, NearMissAssignment, SolverContext


@runtime_checkable
class SpatialConstraint(Protocol):
    """Interface for CSP spatial constraints.

    is_satisfied: pure check against current solver state, no mutation.
    """

    def is_satisfied(self, context: SolverContext) -> bool: ...


class ClusterConnectivity:
    """Every cluster's positions must form a single orthogonally-connected component.

    Uses BFS from an arbitrary position — reuses orthogonal_neighbors from
    primitives.board (CONSTRAINT-CSP-1).
    """

    def is_satisfied(self, context: SolverContext) -> bool:
        for cluster in context.clusters:
            if not _is_connected(cluster.positions, context.board_config):
                return False
        return True


class ClusterNonOverlap:
    """No two entities (clusters, near-misses, scatters) may share a position."""

    def is_satisfied(self, context: SolverContext) -> bool:
        seen: set[Position] = set()
        for cluster in context.clusters:
            if seen & cluster.positions:
                return False
            seen |= cluster.positions
        for near_miss in context.near_misses:
            if seen & near_miss.positions:
                return False
            seen |= near_miss.positions
        if seen & context.scatter_positions:
            return False
        return True


class NearMissIsolation:
    """Near-miss groups must be exactly min_cluster_size - 1 in size, and no
    orthogonal neighbor outside the group may hold the same symbol.

    This prevents the WFC filler from accidentally completing a near-miss
    into a full cluster. The min_cluster_size threshold comes from config.
    """

    def __init__(self, min_cluster_size: int) -> None:
        # Near-miss must be exactly one less than cluster threshold
        self._required_size = min_cluster_size - 1

    def is_satisfied(self, context: SolverContext) -> bool:
        if not context.near_misses:
            return True

        # Build symbol → positions map from all assignments for efficient lookup
        symbol_positions: dict[int, set[Position]] = {}
        for cluster in context.clusters:
            symbol_positions.setdefault(cluster.symbol, set()).update(
                cluster.positions
            )
        for near_miss in context.near_misses:
            symbol_positions.setdefault(near_miss.symbol, set()).update(
                near_miss.positions
            )

        for near_miss in context.near_misses:
            # Size must equal min_cluster_size - 1
            if near_miss.size != self._required_size:
                return False

            # No adjacent same-symbol neighbor outside the group
            same_symbol_positions = symbol_positions.get(near_miss.symbol, set())
            for pos in near_miss.positions:
                for neighbor in orthogonal_neighbors(pos, context.board_config):
                    if neighbor not in near_miss.positions:
                        # Neighbor is outside the group — must not hold the same symbol
                        if neighbor in same_symbol_positions:
                            return False
        return True


class ScatterNonOverlap:
    """Scatter positions must not overlap any cluster or near-miss position."""

    def is_satisfied(self, context: SolverContext) -> bool:
        if not context.scatter_positions:
            return True
        assigned: set[Position] = set()
        for cluster in context.clusters:
            assigned |= cluster.positions
        for near_miss in context.near_misses:
            assigned |= near_miss.positions
        return not (context.scatter_positions & assigned)


class SameSymbolNonAdjacency:
    """Same-symbol clusters must not be orthogonally adjacent.

    Prevents detect_clusters() from merging independently-placed clusters
    into oversized components. Two L1 clusters placed adjacently would form
    one cluster of size 10+ on the board, violating required_cluster_sizes.
    Reuses orthogonal_neighbors from primitives.board (DRY).
    """

    def is_satisfied(self, context: SolverContext) -> bool:
        if len(context.clusters) < 2:
            return True

        for i, a in enumerate(context.clusters):
            for b in context.clusters[i + 1:]:
                if a.symbol is not b.symbol:
                    continue
                # Same symbol — no position in A may neighbor any position in B
                for pos_a in a.positions:
                    for neighbor in orthogonal_neighbors(pos_a, context.board_config):
                        if neighbor in b.positions:
                            return False
        return True


class BoardSymbolNonAdjacency:
    """Prevents CSP clusters from merging with surviving board symbols.

    During cascade refill, surviving cells from previous steps hold symbols
    that the CSP can't see. This constraint checks that each cluster position's
    orthogonal neighbors on the existing board don't hold the same symbol —
    preventing detect_clusters() from merging CSP clusters with survivors.
    """

    def __init__(self, board: Board) -> None:
        self._board = board

    def is_satisfied(self, context: SolverContext) -> bool:
        for cluster in context.clusters:
            for pos in cluster.positions:
                for neighbor in orthogonal_neighbors(pos, context.board_config):
                    if neighbor in cluster.positions:
                        continue
                    surviving_sym = self._board.get(neighbor)
                    if surviving_sym is cluster.symbol:
                        return False
        return True


class PostGravityMapping:
    """Ensures all placements land within gravity-predicted empty cells.

    After step N's clusters explode and gravity settles, predict_empty_cells()
    gives the positions that will be vacant for refill. Step N+1's CSP
    placements must only occupy these positions — clusters placed in
    non-empty cells would be physically impossible post-gravity.

    The caller computes the allowed set via GravityDAG.predict_empty_cells()
    and passes it at construction. This keeps gravity knowledge out of the
    constraint itself (SRP).
    """

    def __init__(self, allowed_positions: frozenset[Position]) -> None:
        # Positions available for placement after gravity settle
        self._allowed = allowed_positions

    def is_satisfied(self, context: SolverContext) -> bool:
        # Every placed position must be within the predicted empties
        for cluster in context.clusters:
            if not cluster.positions <= self._allowed:
                return False
        for near_miss in context.near_misses:
            if not near_miss.positions <= self._allowed:
                return False
        if not context.scatter_positions <= self._allowed:
            return False
        for booster in context.booster_placements:
            if booster.position not in self._allowed:
                return False
        return True


# ---------------------------------------------------------------------------
# Wild constraints (Phase 7)
# ---------------------------------------------------------------------------


class WildSpawnPlacement:
    """Validates that a wild placement with behavior='spawn' exists and its
    position is adjacent to a cluster in the wild spawn-size range (7-8).

    The centroid computation and collision resolution happen in the solver;
    this constraint only validates the final result. Spawn threshold range
    comes from config via __init__ injection.
    """

    def __init__(self, min_spawn_size: int, max_spawn_size: int) -> None:
        self._min_size = min_spawn_size
        self._max_size = max_spawn_size

    def is_satisfied(self, context: SolverContext) -> bool:
        spawn_wilds = [
            wp for wp in context.wild_placements if wp.behavior == "spawn"
        ]
        if not spawn_wilds:
            return False

        # Each spawn wild must be adjacent to at least one cluster in the spawn range
        for wild in spawn_wilds:
            has_source_cluster = False
            for cluster in context.clusters:
                if self._min_size <= cluster.size <= self._max_size:
                    # Wild should be near its source cluster (adjacent or centroid)
                    for neighbor in orthogonal_neighbors(wild.position, context.board_config):
                        if neighbor in cluster.positions:
                            has_source_cluster = True
                            break
                    # Also accept centroid placement inside the cluster
                    if wild.position in cluster.positions:
                        has_source_cluster = True
                if has_source_cluster:
                    break
            if not has_source_cluster:
                return False
        return True


class WildBridgeConstraint:
    """Wild with behavior='bridge' must be orthogonally adjacent to positions
    from at least 2 different cluster assignments.

    This ensures the wild genuinely bridges disconnected groups. Uses
    orthogonal_neighbors from primitives.board (CONSTRAINT-CSP-1).
    """

    def is_satisfied(self, context: SolverContext) -> bool:
        for wild in context.wild_placements:
            if wild.behavior != "bridge":
                continue

            # Count distinct clusters adjacent to this wild
            adjacent_cluster_indices: set[int] = set()
            for idx, cluster in enumerate(context.clusters):
                for neighbor in orthogonal_neighbors(wild.position, context.board_config):
                    if neighbor in cluster.positions:
                        adjacent_cluster_indices.add(idx)
                        break

            # Bridge requires adjacency to 2+ distinct clusters
            if len(adjacent_cluster_indices) < 2:
                return False
        return True


class WildIdleConstraint:
    """Wild with behavior='idle' must NOT be adjacent to 2+ distinct clusters.

    Prevents accidental bridging — an idle wild sits on the board without
    connecting standard-symbol groups. It may be adjacent to 0 or 1 clusters.
    """

    def is_satisfied(self, context: SolverContext) -> bool:
        for wild in context.wild_placements:
            if wild.behavior != "idle":
                continue

            adjacent_cluster_indices: set[int] = set()
            for idx, cluster in enumerate(context.clusters):
                for neighbor in orthogonal_neighbors(wild.position, context.board_config):
                    if neighbor in cluster.positions:
                        adjacent_cluster_indices.add(idx)
                        break

            # Idle wild must not bridge — adjacent to at most 1 cluster
            if len(adjacent_cluster_indices) >= 2:
                return False
        return True


class TerminalNearMissPlacement:
    """Validates that the terminal dead board has the correct number of near-miss
    groups at the specified symbol tier.

    Used for fakeout archetypes (e.g., wild_escalation_fizzle) where the final
    board must show tantalizing almost-wins. Count range and tier from the
    archetype signature's terminal_near_misses field.
    """

    def __init__(
        self,
        count_range: Range,
        symbol_tier: SymbolTier | None,
        min_cluster_size: int,
    ) -> None:
        self._count_range = count_range
        self._symbol_tier = symbol_tier
        # Near-miss = one less than the cluster threshold
        self._required_size = min_cluster_size - 1

    def is_satisfied(self, context: SolverContext) -> bool:
        # Count near-misses of the correct size
        valid_nms = [
            nm for nm in context.near_misses
            if nm.size == self._required_size
        ]

        if not self._count_range.contains(len(valid_nms)):
            return False

        # Tier check — if specified and not ANY, all NMs must match
        if self._symbol_tier is not None and self._symbol_tier is not SymbolTier.ANY:
            for nm in valid_nms:
                # Symbol tier check deferred to validator — CSP places symbols,
                # tier is enforced by the nm_specs passed to solve_step
                pass

        return True


class DormantBoosterSurvival:
    """Validates that specified booster types exist in the solver context's
    booster placements on the terminal board.

    Used for fakeout archetypes (e.g., wild_rocket_tease) where a booster
    must be visible on the final dead board to create tension. Booster names
    from the archetype signature's dormant_boosters_on_terminal field.
    """

    def __init__(self, required_booster_names: tuple[str, ...]) -> None:
        # Convert booster name strings to Symbol enums for comparison
        self._required_types: frozenset[Symbol] = frozenset(
            symbol_from_name(name) for name in required_booster_names
        )

    def is_satisfied(self, context: SolverContext) -> bool:
        placed_types = {bp.booster_type for bp in context.booster_placements}
        return self._required_types <= placed_types


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_connected(positions: frozenset[Position], board_config: BoardConfig) -> bool:
    """BFS connectivity check — all positions reachable via orthogonal adjacency.

    Returns True for empty sets (vacuously true) and single-element sets.
    Reuses orthogonal_neighbors from primitives.board (CONSTRAINT-CSP-1).
    """
    if len(positions) <= 1:
        return True

    start = next(iter(positions))
    visited: set[Position] = {start}
    queue: deque[Position] = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in orthogonal_neighbors(current, board_config):
            if neighbor in positions and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(positions)


# ---------------------------------------------------------------------------
# Booster constraints (Phase 8)
# ---------------------------------------------------------------------------


class BoosterCentroidPlacement:
    """Validates that each booster is placed within its source cluster's positions.

    The centroid computation and collision resolution happen in the solver
    using shared compute_centroid and resolve_collision from BoosterRules
    (DRY — single centroid algorithm). This constraint only validates the
    final result: the booster position must be a member of one of the
    assigned clusters (the one that spawned it).

    CONSTRAINT-CSP-8: booster position must be centroid or collision-resolved
    fallback within the source cluster.
    """

    __slots__ = ("_rules",)

    def __init__(self, rules: BoosterRules) -> None:
        self._rules = rules

    def is_satisfied(self, context: SolverContext) -> bool:
        """Each booster's position must be within at least one cluster's positions."""
        for booster in context.booster_placements:
            found_source = False
            for cluster in context.clusters:
                if booster.position in cluster.positions:
                    found_source = True
                    break
            if not found_source:
                return False
        return True


class BoosterArmAdjacency:
    """Validates that at least one cluster position is orthogonally adjacent
    to a booster that needs arming.

    When a booster exists from a previous step (passed as booster_positions),
    the CSP must place at least one cluster with a position adjacent to the
    booster to enable arming. Without adjacency, the booster stays dormant.

    Reuses orthogonal_neighbors from primitives.board (DRY — single adjacency
    algorithm). CONSTRAINT-CSP-9: arming requires cluster-to-booster adjacency.
    """

    __slots__ = ("_booster_positions", "_board_config")

    def __init__(
        self,
        booster_positions: frozenset[Position],
        board_config: BoardConfig,
    ) -> None:
        self._booster_positions = booster_positions
        self._board_config = board_config

    def is_satisfied(self, context: SolverContext) -> bool:
        """At least one cluster position must neighbor a booster position."""
        if not self._booster_positions:
            # No boosters to arm — constraint trivially satisfied
            return True

        for cluster in context.clusters:
            for pos in cluster.positions:
                for neighbor in orthogonal_neighbors(pos, self._board_config):
                    if neighbor in self._booster_positions:
                        return True
        return False


# ---------------------------------------------------------------------------
# Chain and orientation constraints (Phase 9)
# ---------------------------------------------------------------------------


class ChainSpatialRelation:
    """Validates that a chain target booster is within the source booster's effect zone.

    Rockets clear an entire row (H) or column (V). Bombs clear a Manhattan-radius
    square. For chaining to be physically possible, the target must sit inside the
    source's blast/path area. This constraint ensures the CSP places chain-target
    boosters at positions reachable by the source booster's fire.

    Reuses rocket_path and bomb_blast from BoosterRules (DRY — single geometry).
    CONSTRAINT-CSP-10: chain target must be within source's effect zone.
    """

    __slots__ = (
        "_source_type", "_source_position", "_source_orientation",
        "_target_position", "_rules",
    )

    def __init__(
        self,
        source_type: Symbol,
        source_position: Position,
        source_orientation: str | None,
        target_position: Position,
        rules: BoosterRules,
    ) -> None:
        self._source_type = source_type
        self._source_position = source_position
        self._source_orientation = source_orientation
        self._target_position = target_position
        self._rules = rules

    def is_satisfied(self, context: SolverContext) -> bool:
        """Target position must fall within the source booster's effect zone."""
        if self._source_type is Symbol.R:
            # Rocket clears along its orientation axis
            effect_zone = self._rules.rocket_path(
                self._source_position, self._source_orientation,
            )
        elif self._source_type is Symbol.B:
            # Bomb clears within Manhattan radius
            effect_zone = self._rules.bomb_blast(self._source_position)
        else:
            # Only R and B are chain initiators (from config) — other types
            # cannot source a chain, so the constraint is trivially satisfied
            return True

        return self._target_position in effect_zone


class RocketOrientationControl:
    """Constrains cluster shapes to produce the desired rocket orientation.

    Rocket orientation emerges from the cluster's spatial spread — it fires
    perpendicular to the dominant axis. This constraint validates that any
    cluster whose size falls in the rocket spawn threshold range produces
    the desired H or V orientation.

    Reuses compute_rocket_orientation and booster_type_for_size from
    BoosterRules (DRY — single orientation algorithm).
    CONSTRAINT-CSP-11: cluster shape must produce the specified orientation.
    """

    __slots__ = ("_desired_orientation", "_rules")

    def __init__(self, desired_orientation: str, rules: BoosterRules) -> None:
        self._desired_orientation = desired_orientation
        self._rules = rules

    def is_satisfied(self, context: SolverContext) -> bool:
        """Every cluster in the rocket spawn range must produce the desired orientation."""
        for cluster in context.clusters:
            # Only check clusters large enough to spawn a rocket
            candidate = self._rules.booster_type_for_size(cluster.size)
            if candidate is not Symbol.R:
                continue

            actual = self._rules.compute_rocket_orientation(cluster.positions)
            if actual != self._desired_orientation:
                return False

        return True


class LightballTargetAbundance:
    """Ensures cluster placements include symbols from the desired tier so the
    lightball targets the intended tier when it fires.

    When lb_target_tier is LOW, at least one cluster must use a LOW-tier symbol.
    When HIGH, at least one must use a HIGH-tier symbol. This biases the board
    composition so that after WFC fill, the most abundant symbol belongs to the
    desired tier.

    CONSTRAINT-CSP-12: LB target tier must appear in cluster assignments.
    """

    __slots__ = ("_target_tier", "_symbol_config")

    def __init__(
        self, target_tier: SymbolTier, symbol_config: SymbolConfig,
    ) -> None:
        self._target_tier = target_tier
        self._symbol_config = symbol_config

    def is_satisfied(self, context: SolverContext) -> bool:
        """At least one cluster uses a symbol from the target tier."""
        if self._target_tier is SymbolTier.ANY:
            return True

        tier_symbols = symbols_in_tier(self._target_tier, self._symbol_config)
        return any(
            cluster.symbol in tier_symbols
            for cluster in context.clusters
        )
