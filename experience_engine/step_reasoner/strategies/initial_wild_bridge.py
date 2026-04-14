"""Initial wild bridge strategy — step 0 for arcs whose next phase is a wild bridge.

Places a wild-spawning cluster, forward-simulates to predict the Wild's
post-gravity landing, then uses BridgePathTracer to deterministically trace
the path from the Wild to the refill zone. Bridge symbols are assigned to
the traced path positions (pre-gravity coordinates) so that WildBridgeStrategy
at step 1 can complete the bridge with minimal shortfall.

Replaces the probabilistic bridge branch that was in InitialClusterStrategy.
"""

from __future__ import annotations

import random

from ...board_filler.propagators import (
    ClusterBoundaryPropagator, NearMissAwareDeadPropagator, NearMissGroup,
    NoClusterPropagator, NoSpecialSymbolPropagator,
)
from ...config.schema import MasterConfig
from ...primitives.board import Position
from ...primitives.symbols import Symbol, SymbolTier
from ..context import BoardContext
from ..evaluators import SpawnEvaluator
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.bridge_path_tracer import BridgePathTracer
from ..services.cluster_builder import ClusterBuilder
from ..services.forward_simulator import ForwardSimulator
from ..services.influence_map import DemandSpec
from ..services.landing_evaluator import BoosterLandingEvaluator
from ..services.near_miss_planner import NearMissPlanner
from ..services.seed_planner import SeedPlanner, build_cluster_exclusions
from ..services.spatial_context import StepSpatialContext
from ..services.utility_scorer import ScoringContext
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


class InitialWildBridgeStrategy:
    """Step 0 for wild bridge arcs: cluster + deterministic bridge path setup.

    Single responsibility: place a wild-spawning cluster, predict where the
    Wild lands, trace the bridge path from landing to refill zone, and assign
    bridge symbols to the traced path. WildBridgeStrategy at step 1 observes
    the planted path via _scan_bridge_candidates() and places only the shortfall.

    All dependencies are injected — no self-construction.
    """

    __slots__ = (
        "_config", "_forward_sim", "_cluster_builder", "_seed_planner",
        "_spawn_eval", "_near_miss_planner", "_landing_eval",
        "_bridge_tracer", "_rng", "_spatial",
    )

    def __init__(
        self,
        config: MasterConfig,
        forward_sim: ForwardSimulator,
        cluster_builder: ClusterBuilder,
        seed_planner: SeedPlanner,
        spawn_eval: SpawnEvaluator,
        near_miss_planner: NearMissPlanner,
        landing_eval: BoosterLandingEvaluator,
        bridge_tracer: BridgePathTracer,
        rng: random.Random,
        spatial: StepSpatialContext | None = None,
    ) -> None:
        self._config = config
        self._forward_sim = forward_sim
        self._cluster_builder = cluster_builder
        self._seed_planner = seed_planner
        self._spawn_eval = spawn_eval
        self._near_miss_planner = near_miss_planner
        self._landing_eval = landing_eval
        self._bridge_tracer = bridge_tracer
        self._rng = rng
        self._spatial = spatial

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        # Wild bridge arcs use a single cluster — cap count to 1
        step_sizes = progress.current_step_size_ranges()

        # Cap to wild spawn budget when every size range spawns a Wild
        cluster_count = 1
        if self._all_sizes_spawn_wilds(step_sizes):
            wild_budget = progress.remaining_booster_spawns().get("W")
            if wild_budget is not None:
                cluster_count = min(cluster_count, wild_budget.max_val)

        multi_result = self._cluster_builder.build_multi_cluster(
            context, cluster_count, list(step_sizes),
            progress, signature, variance, self._rng,
        )

        # Derive tier from the cluster symbol placed
        if not multi_result.cluster_symbols:
            cluster_tier = SymbolTier.ANY
        else:
            first_sym = multi_result.cluster_symbols[0]
            cluster_tier = SymbolTier.LOW if first_sym.value <= 4 else SymbolTier.HIGH

        # Detect booster spawn — should be "W" for wild bridge arcs
        expected_spawns: list[str] = []
        for size in multi_result.cluster_sizes:
            booster_type = self._spawn_eval.booster_for_size(size)
            if booster_type:
                expected_spawns.append(booster_type)

        # Forward simulate the cluster explosion + gravity
        hypothetical = self._forward_sim.build_hypothetical(
            context.board, multi_result.all_constrained,
        )
        settle_result = self._forward_sim.simulate_explosion(
            hypothetical, multi_result.all_occupied,
        )

        # Predict Wild landing — score validates the landing has viable refill adjacency
        first_booster = expected_spawns[0] if expected_spawns else None
        cluster_positions = frozenset(multi_result.all_occupied)
        if not first_booster:
            raise ValueError("Wild bridge arc requires a booster-spawning cluster size")

        ctx, score = self._landing_eval.evaluate_and_score(
            cluster_positions, context.board, first_booster,
        )
        booster_landing = ctx.landing_position

        if score < self._config.reasoner.min_booster_landing_score:
            raise ValueError(
                f"{first_booster} at {booster_landing} scored {score:.2f} — "
                f"no viable refill adjacency, retry cluster placement"
            )

        # Resolve bridge target size from the next phase's cluster sizes
        next_phase = progress.peek_next_phase()
        if next_phase is not None and next_phase.cluster_sizes:
            target_bridge_size = next_phase.cluster_sizes[0].min_val
        else:
            target_bridge_size = self._config.board.min_cluster_size

        # Trace the bridge path from wild landing to refill zone
        bridge_plan = self._bridge_tracer.plan(
            booster_landing, settle_result, target_bridge_size,
        )

        # Build strategic cells: bridge_symbol at each pre-gravity path position
        bridge_symbol = multi_result.cluster_symbols[0]
        cluster_groups = [
            (r.planned_positions, sym)
            for r, sym in zip(multi_result.clusters, multi_result.cluster_symbols)
        ]
        exclusions = build_cluster_exclusions(cluster_groups, self._config.board)
        strategic_cells = self._seed_planner._filter_excluded(
            {pre_pos: bridge_symbol for pre_pos in bridge_plan.path_pre_to_post},
            exclusions,
        )

        # Spatial intelligence — compute reserve zone around refill centroid
        # so next step's WFC doesn't fill bridge path positions with wrong symbols
        reserve_zone: frozenset[Position] | None = None
        if self._spatial is not None:
            demand = DemandSpec(
                centroid=bridge_plan.refill_centroid,
                cluster_size=target_bridge_size,
                booster_type=first_booster,
            )
            influence = self._spatial.influence_map.compute(demand)
            reserve_zone = self._spatial.influence_map.reserve_zone(influence)

        # Scatter placement if required — avoid cluster + strategic cells
        constrained: dict[Position, Symbol] = dict(multi_result.all_constrained)
        scatter_count = self._resolve_scatter_count(signature)
        if scatter_count > 0:
            avoid = frozenset(constrained) | frozenset(strategic_cells)
            scatter_positions = self._place_scatters(context, scatter_count, variance, avoid)
            for pos in scatter_positions:
                constrained[pos] = Symbol.S

        # Near-miss placement
        nm_result = self._near_miss_planner.place(
            context, signature, variance,
            avoid=frozenset(constrained) | frozenset(strategic_cells),
            cluster_symbols=frozenset(multi_result.cluster_symbols),
            cluster_positions=multi_result.all_occupied,
        )
        constrained.update(nm_result.constrained_cells)

        propagators = self._select_propagators(
            cluster_groups=cluster_groups,
            near_miss_groups=nm_result.groups or None,
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
            predicted_post_gravity=settle_result,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            planned_explosion=frozenset(multi_result.all_occupied),
            is_terminal=False,
            reserve_zone=reserve_zone,
            # Wild landing position lets PostGravityPropagator count the wild
            # as same-symbol, preventing WFC from forming groups that merge
            # through it into booster-spawning clusters
            predicted_wild_positions=frozenset({booster_landing}),
        )

    def _all_sizes_spawn_wilds(self, size_ranges: tuple[Range, ...]) -> bool:
        """True when every size in every range maps to a Wild spawn."""
        return all(
            self._spawn_eval.booster_for_size(r.min_val) == "W"
            and self._spawn_eval.booster_for_size(r.max_val) == "W"
            for r in size_ranges
        )

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
        cluster_groups: list[tuple[frozenset[Position], Symbol]] | None = None,
        near_miss_groups: list[NearMissGroup] | None = None,
    ) -> list:
        """Select WFC propagators for noise fill around the cluster(s).

        NoClusterPropagator prevents WFC from extending placed clusters.
        ClusterBoundaryPropagator per cluster group forbids the cluster symbol
        at adjacent cells. MaxComponentPropagator/NearMissAwareDeadPropagator
        caps fill components below near-miss size.
        """
        from ...board_filler.propagators import MaxComponentPropagator

        propagators = [
            NoSpecialSymbolPropagator(self._config.symbols),
            NoClusterPropagator(self._config.board.min_cluster_size),
        ]
        if cluster_groups:
            for positions, symbol in cluster_groups:
                propagators.append(ClusterBoundaryPropagator(
                    positions, symbol, self._config.board,
                ))
        nm_fill_cap = self._config.board.min_cluster_size - 2
        if near_miss_groups:
            propagators.append(NearMissAwareDeadPropagator(
                max_component=nm_fill_cap,
                protected_groups=near_miss_groups,
                board_config=self._config.board,
            ))
        else:
            propagators.append(MaxComponentPropagator(nm_fill_cap))
        return propagators
