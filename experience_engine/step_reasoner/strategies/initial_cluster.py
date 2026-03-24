"""Initial cluster strategy — step 0 for archetypes that need clusters.

Places cluster(s) on an empty board, forward-simulates gravity to predict
post-explosion board state, then backward-reasons about strategic seed
placements for future cascade steps. Supports multi-cluster archetypes
(e.g., t1_multi_cascade) via ClusterBuilder.build_multi_cluster().
"""

from __future__ import annotations

import random

from ...board_filler.propagators import ClusterBoundaryPropagator, NoClusterPropagator, NoSpecialSymbolPropagator
from ...config.schema import MasterConfig
from ...primitives.board import Position
from ...primitives.booster_rules import BoosterRules
from ...primitives.symbols import Symbol, SymbolTier
from ..context import BoardContext
from ..evaluators import SpawnEvaluator
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.cluster_builder import ClusterBuilder
from ..services.forward_simulator import ForwardSimulator
from ..services.near_miss_planner import NearMissPlanner
from ..services.seed_planner import SeedPlanner
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


class InitialClusterStrategy:
    """Builds the initial board with cluster(s), scatters, near-misses, and strategic seeds.

    This strategy does the most forward simulation because the initial board
    determines the entire cascade trajectory. It reasons about cluster
    placement, booster spawn prediction, and backward seeding for future steps.
    Near-misses are placed via NearMissPlanner when the archetype requires them.
    """

    __slots__ = (
        "_config", "_forward_sim", "_cluster_builder", "_seed_planner",
        "_spawn_eval", "_near_miss_planner", "_booster_rules", "_rng",
    )

    def __init__(
        self,
        config: MasterConfig,
        forward_sim: ForwardSimulator,
        cluster_builder: ClusterBuilder,
        seed_planner: SeedPlanner,
        spawn_eval: SpawnEvaluator,
        near_miss_planner: NearMissPlanner,
        rng: random.Random,
    ) -> None:
        self._config = config
        self._forward_sim = forward_sim
        self._cluster_builder = cluster_builder
        self._seed_planner = seed_planner
        self._spawn_eval = spawn_eval
        self._near_miss_planner = near_miss_planner
        self._booster_rules = BoosterRules(config.boosters, config.board, config.symbols)
        self._rng = rng

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        # Place N clusters via shared multi-cluster loop (DRY — ClusterBuilder)
        cluster_count = self._cluster_builder.select_cluster_count(
            signature.required_cluster_count, self._rng,
        )
        multi_result = self._cluster_builder.build_multi_cluster(
            context, cluster_count, list(signature.required_cluster_sizes),
            progress, signature, variance, self._rng,
        )

        # Derive tier from the first cluster symbol placed
        first_sym = multi_result.cluster_symbols[0]
        cluster_tier = SymbolTier.LOW if first_sym.value <= 4 else SymbolTier.HIGH

        # Detect booster spawns from each cluster's size
        expected_spawns: list[str] = []
        for size in multi_result.cluster_sizes:
            booster_type = self._spawn_eval.booster_for_size(size)
            if booster_type:
                expected_spawns.append(booster_type)

        # Forward simulate: all clusters explode simultaneously
        hypothetical = self._forward_sim.build_hypothetical(
            context.board, multi_result.all_constrained,
        )
        settle_result = self._forward_sim.simulate_explosion(
            hypothetical, multi_result.all_occupied,
        )
        predicted_post_gravity = settle_result

        # Backward reasoning: plant strategic seeds for the next step
        # Use first cluster's booster type for seed planning (if any)
        first_booster = expected_spawns[0] if expected_spawns else None
        strategic_cells = self._plan_strategic_seeds(
            context, multi_result.all_occupied, first_sym, first_booster,
            settle_result, progress, signature, variance,
        )

        # Scatter placement if required — avoid all cluster positions
        constrained: dict[Position, Symbol] = dict(multi_result.all_constrained)
        scatter_count = self._resolve_scatter_count(signature)
        if scatter_count > 0:
            avoid = frozenset(constrained) | frozenset(strategic_cells)
            scatter_positions = self._place_scatters(context, scatter_count, variance, avoid)
            for pos in scatter_positions:
                constrained[pos] = Symbol.S

        # Near-miss placement — pass all cluster symbols so the planner avoids merging
        nm_result = self._near_miss_planner.place(
            context, signature, variance,
            avoid=frozenset(constrained) | frozenset(strategic_cells),
            cluster_symbols=frozenset(multi_result.cluster_symbols),
            cluster_positions=multi_result.all_occupied,
        )
        constrained.update(nm_result.constrained_cells)

        # Propagators for WFC noise fill — one ClusterBoundaryPropagator per cluster
        # group prevents same-symbol survivors at each cluster's edge
        cluster_groups = [
            (r.planned_positions, sym)
            for r, sym in zip(multi_result.clusters, multi_result.cluster_symbols)
        ]
        propagators = self._select_propagators(
            has_near_misses=bool(nm_result.groups),
            cluster_groups=cluster_groups,
        )

        return StepIntent(
            step_type=StepType.INITIAL,
            constrained_cells=constrained,
            strategic_cells=strategic_cells,
            expected_cluster_count=Range(cluster_count, cluster_count),
            expected_cluster_sizes=[
                Range(s, s) for s in multi_result.cluster_sizes
            ],
            expected_cluster_tier=cluster_tier,
            expected_spawns=expected_spawns,
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=propagators,
            wfc_symbol_weights=variance.symbol_weights,
            predicted_post_gravity=predicted_post_gravity,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            # All cluster positions will explode — gates gravity-aware WFC mechanisms
            planned_explosion=frozenset(multi_result.all_occupied),
            is_terminal=False,
        )

    def _plan_strategic_seeds(
        self,
        context: BoardContext,
        cluster_positions: frozenset[Position],
        cluster_symbol: Symbol,
        booster_type: str | None,
        settle_result,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> dict[Position, Symbol]:
        """Backward reasoning — determine what future steps need and seed accordingly."""
        if progress.must_terminate_soon():
            return {}

        # Predict where the booster will land after this explosion + gravity
        if booster_type:
            centroid = self._booster_rules.compute_centroid(cluster_positions)
            booster_landing = self._forward_sim.predict_booster_landing(
                centroid, context.board, cluster_positions,
            )

            if self._next_step_needs_wild_bridge(signature, progress):
                return self._seed_planner.plan_bridge_seeds(
                    booster_landing, settle_result, cluster_symbol,
                    self._config.board.min_cluster_size - 1,
                    variance, self._rng,
                )

            if self._next_step_needs_arming_cluster(signature, progress):
                return self._seed_planner.plan_arm_seeds(
                    booster_landing, settle_result, variance, self._rng,
                )

        return self._seed_planner.plan_generic_seeds(
            settle_result, progress, signature, variance, self._rng,
        )

    def _next_step_needs_wild_bridge(
        self,
        signature: ArchetypeSignature,
        progress: ProgressTracker,
    ) -> bool:
        """Check if the next cascade step requires a wild bridge."""
        next_step = progress.steps_completed + 1
        if signature.cascade_steps and next_step < len(signature.cascade_steps):
            return signature.cascade_steps[next_step].wild_behavior == "bridge"
        return False

    def _next_step_needs_arming_cluster(
        self,
        signature: ArchetypeSignature,
        progress: ProgressTracker,
    ) -> bool:
        """Check if the next step needs to arm a dormant booster."""
        next_step = progress.steps_completed + 1
        if signature.cascade_steps and next_step < len(signature.cascade_steps):
            step_spec = signature.cascade_steps[next_step]
            return step_spec.must_arm_booster is not None
        return False

    def _resolve_scatter_count(self, signature: ArchetypeSignature) -> int:
        """Pick a scatter count within the signature's range."""
        rng = signature.required_scatter_count
        if rng.max_val <= 0:
            return 0
        return self._rng.randint(rng.min_val, rng.max_val)

    def _place_scatters(
        self,
        context: BoardContext,
        count: int,
        variance: VarianceHints,
        avoid: frozenset[Position],
    ) -> list[Position]:
        """Select scatter positions weighted by spatial bias, avoiding occupied cells."""
        candidates = [p for p in context.board.all_positions() if p not in avoid]
        if not candidates:
            return []
        selected: list[Position] = []
        remaining = list(candidates)
        for _ in range(min(count, len(remaining))):
            weights = [variance.spatial_bias.get(p, 1.0) for p in remaining]
            chosen = self._rng.choices(remaining, weights=weights, k=1)[0]
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def _select_propagators(
        self,
        has_near_misses: bool = False,
        cluster_groups: list[tuple[frozenset[Position], Symbol]] | None = None,
    ) -> list:
        """Select WFC propagators for noise fill around the cluster(s).

        NoClusterPropagator prevents WFC from extending placed clusters
        or forming new accidental clusters. One ClusterBoundaryPropagator
        per cluster group forbids each cluster's symbol at adjacent cells —
        prevents same-symbol survivors from bordering the refill zone
        after explosion + gravity.
        When no near-misses are present, MaxComponentPropagator caps fill
        components below near-miss size to prevent accidental groups.
        """
        from ...board_filler.propagators import MaxComponentPropagator

        propagators = [
            NoSpecialSymbolPropagator(self._config.symbols),
            NoClusterPropagator(self._config.board.min_cluster_size),
        ]
        # Isolate each cluster's boundary — WFC cannot place the cluster symbol
        # adjacent to cluster positions, preventing merge on the next cascade step
        if cluster_groups:
            for positions, symbol in cluster_groups:
                propagators.append(ClusterBoundaryPropagator(
                    positions, symbol, self._config.board,
                ))
        # When near-misses are pinned on the board, skip the component cap —
        # the cap (size 3) would overconstrain the fill around size-4 groups.
        # When no near-misses exist, enforce the cap to prevent accidental
        # near-miss-sized components that confuse the validator.
        if not has_near_misses:
            nm_fill_cap = self._config.board.min_cluster_size - 2
            propagators.append(MaxComponentPropagator(nm_fill_cap))
        return propagators
