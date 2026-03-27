"""Booster arm strategy — places a cluster adjacent to a dormant booster.

The cluster must be orthogonally adjacent to the target booster so that
the explosion arms it. If the booster participates in a chain, strategic
cells are planned for the chain spatial arrangement.
"""

from __future__ import annotations

import random

from ...board_filler.propagators import NoClusterPropagator, NoSpecialSymbolPropagator
from ...config.schema import MasterConfig
from ...primitives.board import Position
from ...primitives.symbols import Symbol, SymbolTier
from ..context import BoardContext, DormantBooster
from ..evaluators import ChainEvaluator
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.cluster_builder import ClusterBuilder
from ..services.forward_simulator import ForwardSimulator
from ..services.landing_evaluator import BoosterLandingEvaluator
from ..services.influence_map import DemandSpec
from ..services.seed_planner import SeedPlanner, build_cluster_exclusions
from ..services.spatial_context import StepSpatialContext
from ..services.utility_scorer import ScoringContext
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


class BoosterArmStrategy:
    """Produces a cluster adjacent to a dormant booster to arm it.

    Selects the most urgent booster to arm, grows a cluster next to it,
    and optionally plans chain arrangement in strategic cells.
    """

    __slots__ = (
        "_config", "_forward_sim", "_cluster_builder",
        "_seed_planner", "_chain_eval", "_landing_eval", "_rng",
        "_spatial",
    )

    def __init__(
        self,
        config: MasterConfig,
        forward_sim: ForwardSimulator,
        cluster_builder: ClusterBuilder,
        seed_planner: SeedPlanner,
        chain_eval: ChainEvaluator,
        landing_eval: BoosterLandingEvaluator,
        rng: random.Random,
        spatial: StepSpatialContext | None = None,
    ) -> None:
        self._config = config
        self._forward_sim = forward_sim
        self._cluster_builder = cluster_builder
        self._seed_planner = seed_planner
        self._chain_eval = chain_eval
        self._landing_eval = landing_eval
        self._rng = rng
        self._spatial = spatial

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        # Select the most urgent booster to arm
        target_booster = self._select_booster_to_arm(
            context.dormant_boosters, progress, signature,
        )
        booster_pos = target_booster.position

        # Boundary analysis for merge-aware placement
        boundary = self._cluster_builder.analyze_boundary(context)

        # Select cluster parameters with merge awareness — clamp to available space
        cluster_size = self._cluster_builder.select_size(
            progress, signature, variance, self._rng,
            max_available=len(context.empty_cells),
        )
        cluster_symbol = self._cluster_builder.select_symbol(
            progress, signature, variance, self._rng,
            boundary=boundary, planned_size=cluster_size,
        )
        cluster_tier = SymbolTier.LOW if cluster_symbol.value <= 4 else SymbolTier.HIGH

        # Grow cluster adjacent to the booster — AVOID merge to keep exact size for correct booster trigger
        from ..services.merge_policy import MergePolicy
        result = self._cluster_builder.find_positions(
            context, cluster_size, self._rng, variance,
            symbol=cluster_symbol, boundary=boundary, merge_policy=MergePolicy.AVOID,
            must_be_adjacent_to=frozenset({booster_pos}),
        )
        cluster_positions = result.planned_positions

        # Plan chain arrangement if this booster can initiate a chain
        strategic_cells: dict[Position, Symbol] = {}
        reserve_zone: frozenset[Position] | None = None
        if self._needs_chain(target_booster, progress, signature):
            hypothetical = self._forward_sim.build_hypothetical(
                context.board,
                {pos: cluster_symbol for pos in cluster_positions},
            )
            settle_result = self._forward_sim.simulate_explosion(
                hypothetical, cluster_positions,
            )

            # Score the arming cluster's own secondary spawn viability — if it's
            # large enough to spawn a booster, verify the landing won't obstruct
            # the chain sequence
            _ctx, landing_score = self._landing_eval.evaluate_and_score(
                frozenset(cluster_positions), hypothetical,
                target_booster.booster_type,
            )

            # Prevent arm seeds from merging into the arming cluster
            exclusions = build_cluster_exclusions(
                [(frozenset(cluster_positions), cluster_symbol)],
                self._config.board,
            )

            # Compute utility scores for arm seed placement — gravity alignment
            # and booster adjacency steer seeds toward the arming zone
            utility_scores: dict[Position, float] | None = None
            if self._spatial:
                demand = DemandSpec(
                    centroid=booster_pos,
                    cluster_size=self._config.board.min_cluster_size,
                    booster_type=target_booster.booster_type,
                )
                influence = self._spatial.influence_map.compute(demand)
                reserve_zone = self._spatial.influence_map.reserve_zone(influence)
                scoring_ctx = ScoringContext(
                    influence=influence,
                    gravity_field=self._spatial.gravity_field,
                    demand=demand,
                    cluster_positions=frozenset(cluster_positions),
                    board_config=self._config.board,
                    booster_landing=booster_pos,
                )
                utility_scores = self._spatial.utility_scorer.score_candidates(
                    list(settle_result.empty_positions), scoring_ctx,
                )

            strategic_cells = self._seed_planner.plan_arm_seeds(
                booster_pos, settle_result, variance, self._rng,
                exclusions=exclusions,
                utility_scores=utility_scores,
            )

        constrained = {pos: cluster_symbol for pos in cluster_positions}

        return StepIntent(
            step_type=StepType.BOOSTER_ARM,
            constrained_cells=constrained,
            strategic_cells=strategic_cells,
            expected_cluster_count=Range(1, 1),
            expected_cluster_sizes=[Range(result.total_size, result.total_size)],
            expected_cluster_tier=cluster_tier,
            expected_spawns=[],
            expected_arms=[target_booster.booster_type],
            expected_fires=[],
            wfc_propagators=[
                NoSpecialSymbolPropagator(self._config.symbols),
                NoClusterPropagator(self._config.board.min_cluster_size),
            ],
            wfc_symbol_weights=variance.symbol_weights,
            predicted_post_gravity=None,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            # Arming cluster positions will explode — gates gravity-aware WFC
            planned_explosion=frozenset(cluster_positions),
            is_terminal=False,
            reserve_zone=reserve_zone,
        )

    def _select_booster_to_arm(
        self,
        dormant_boosters: list[DormantBooster],
        progress: ProgressTracker,
        signature: ArchetypeSignature,
    ) -> DormantBooster:
        """Select the most urgent booster to arm.

        Prioritizes boosters whose fire is required by the signature and
        whose remaining fire budget is tightest. Falls back to the oldest
        dormant booster (earliest spawned_step).
        """
        needed_fires = progress.remaining_booster_fires()

        # Boosters whose type still has unmet fire requirements
        urgent = [
            b for b in dormant_boosters
            if b.booster_type in needed_fires
            and needed_fires[b.booster_type].min_val > 0
        ]

        if urgent:
            # Most urgent = fewest remaining steps to fire them all
            return min(urgent, key=lambda b: b.spawned_step)

        # No urgent fires needed — arm the oldest dormant booster
        return min(dormant_boosters, key=lambda b: b.spawned_step)

    def _needs_chain(
        self,
        target_booster: DormantBooster,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
    ) -> bool:
        """Check if this booster needs to trigger a chain reaction."""
        if not self._chain_eval.can_initiate_chain(target_booster.booster_type):
            return False
        return signature.required_chain_depth.min_val > progress.chain_depth_max
