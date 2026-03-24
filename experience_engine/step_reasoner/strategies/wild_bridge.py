"""Wild bridge strategy — forms a cluster through a wild symbol.

The wild is at a known position on the board. This strategy fills empty
cells so that same-symbol cells end up on BOTH sides of the wild,
bridging them into a single cluster via detect_clusters wild-aware BFS.
"""

from __future__ import annotations

import random

from ...board_filler.propagators import (
    NoClusterPropagator,
    NoSpecialSymbolPropagator,
    WildBridgePropagator,
)
from ...config.schema import MasterConfig
from ...primitives.board import Position, orthogonal_neighbors
from ...primitives.cluster_detection import detect_clusters
from ...primitives.symbols import Symbol, SymbolTier
from ..context import BoardContext
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.cluster_builder import ClusterBuilder
from ..services.forward_simulator import ForwardSimulator
from ..services.seed_planner import SeedPlanner, build_cluster_exclusions
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


class WildBridgeStrategy:
    """Plans a cluster that uses a wild as a bridge.

    Finds empty cells that, when filled with the bridge symbol, form a
    connected cluster through the wild. The wild counts as one member
    of the cluster. Bridge positions must be on both sides of the wild
    for a genuine bridge effect.
    """

    __slots__ = (
        "_config", "_forward_sim", "_cluster_builder",
        "_seed_planner", "_rng",
    )

    def __init__(
        self,
        config: MasterConfig,
        forward_sim: ForwardSimulator,
        cluster_builder: ClusterBuilder,
        seed_planner: SeedPlanner,
        rng: random.Random,
    ) -> None:
        self._config = config
        self._forward_sim = forward_sim
        self._cluster_builder = cluster_builder
        self._seed_planner = seed_planner
        self._rng = rng

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        wild_pos = context.active_wilds[0]

        # Find surviving symbols adjacent to the wild
        wild_neighbors = orthogonal_neighbors(wild_pos, self._config.board)
        occupied_neighbors = {
            pos: context.surviving_symbols[pos]
            for pos in wild_neighbors
            if pos in context.surviving_symbols
        }

        # Select bridge symbol — prefer a symbol already adjacent to the wild
        bridge_symbol = self._select_bridge_symbol(
            occupied_neighbors, context, progress, signature, variance,
        )
        bridge_tier = SymbolTier.LOW if bridge_symbol.value <= 4 else SymbolTier.HIGH

        # Count existing same-symbol cells adjacent to the wild
        existing_same = [
            pos for pos, sym in occupied_neighbors.items()
            if sym is bridge_symbol
        ]

        # Target size: cluster must be >= min_cluster_size, wild counts as 1
        target_size = self._select_target_size(signature, progress)
        needed = target_size - len(existing_same) - 1  # -1 for the wild itself
        needed = max(needed, 1)  # at minimum place 1 cell

        # Boundary analysis for merge-aware placement — bridge clusters ACCEPT merge
        # since the goal is connecting through the wild
        boundary = self._cluster_builder.analyze_boundary(context)
        from ..services.merge_policy import MergePolicy

        result = self._cluster_builder.find_positions(
            context, needed, self._rng, variance,
            symbol=bridge_symbol, boundary=boundary, merge_policy=MergePolicy.ACCEPT,
            must_be_adjacent_to=frozenset({wild_pos}),
        )
        bridge_positions = result.planned_positions

        # Forward simulate: verify the bridge forms
        placements = {pos: bridge_symbol for pos in bridge_positions}
        hypothetical = self._forward_sim.build_hypothetical(context.board, placements)
        detected = detect_clusters(hypothetical, self._config)

        # Verify a cluster of the right size exists through the wild
        bridge_ok = any(
            c.symbol is bridge_symbol
            and wild_pos in c.wild_positions
            and c.size >= target_size
            for c in detected
        )
        if not bridge_ok:
            # Retry with AVOID policy and different positions
            result = self._cluster_builder.find_positions(
                context, needed, self._rng, variance,
                symbol=bridge_symbol, boundary=boundary, merge_policy=MergePolicy.AVOID,
                must_be_adjacent_to=frozenset({wild_pos}),
                avoid_positions=bridge_positions,
            )
            bridge_positions = result.planned_positions

        # Backward reasoning: seed the next step if not terminal
        strategic_cells: dict[Position, Symbol] = {}
        if not progress.must_terminate_soon():
            settle_result = self._forward_sim.simulate_explosion(
                hypothetical, bridge_positions,
            )
            # Prevent strategic seeds from merging into the bridge cluster
            exclusions = build_cluster_exclusions(
                [(frozenset(bridge_positions), bridge_symbol)],
                self._config.board,
            )
            strategic_cells = self._seed_planner.plan_generic_seeds(
                settle_result, progress, signature, variance, self._rng,
                exclusions=exclusions,
            )

        constrained = {pos: bridge_symbol for pos in bridge_positions}

        return StepIntent(
            step_type=StepType.CASCADE_CLUSTER,
            constrained_cells=constrained,
            strategic_cells=strategic_cells,
            expected_cluster_count=Range(1, 1),
            expected_cluster_sizes=[Range(target_size, target_size)],
            expected_cluster_tier=bridge_tier,
            expected_spawns=[],
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=[
                NoSpecialSymbolPropagator(self._config.symbols),
                NoClusterPropagator(self._config.board.min_cluster_size),
                WildBridgePropagator(frozenset(context.active_wilds)),
            ],
            wfc_symbol_weights=variance.symbol_weights,
            predicted_post_gravity=None,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            # Both bridge cluster and wilds explode — union feeds gravity-aware WFC
            planned_explosion=frozenset(bridge_positions) | frozenset(context.active_wilds),
            is_terminal=False,
        )

    def _select_bridge_symbol(
        self,
        occupied_neighbors: dict[Position, Symbol],
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> Symbol:
        """Select the symbol to bridge through the wild.

        Prefers symbols already adjacent to the wild (less new cells needed).
        Falls back to ClusterBuilder symbol selection if no adjacent match.
        """
        # Count how many of each symbol are already adjacent
        from collections import Counter
        neighbor_counts = Counter(occupied_neighbors.values())

        # Filter to standard symbols only
        from ...primitives.symbols import is_standard
        standard_neighbors = {
            sym: count for sym, count in neighbor_counts.items()
            if is_standard(sym, self._config.symbols)
        }

        if standard_neighbors:
            # Prefer the symbol with the most adjacent cells — needs fewer bridge cells
            best = max(standard_neighbors, key=standard_neighbors.get)
            return best

        # No standard symbols adjacent — use ClusterBuilder selection
        return self._cluster_builder.select_symbol(
            progress, signature, variance, self._rng,
        )

    def _select_target_size(
        self,
        signature: ArchetypeSignature,
        progress: ProgressTracker,
    ) -> int:
        """Determine the target cluster size for the bridge."""
        if signature.required_cluster_sizes:
            # Use the first size spec's minimum
            return signature.required_cluster_sizes[0].min_val
        return self._config.board.min_cluster_size
