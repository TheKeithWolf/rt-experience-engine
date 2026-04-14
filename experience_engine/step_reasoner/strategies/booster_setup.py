"""Booster setup strategy — arranges boosters for chain spatial requirements.

Works backward from chain geometry to determine where each booster must
be positioned, then builds clusters whose centroids spawn boosters at
those required positions.
"""

from __future__ import annotations

import random

from ...board_filler.propagators import NoClusterPropagator, NoSpecialSymbolPropagator
from ...config.schema import MasterConfig
from ...primitives.board import Position
from ...primitives.booster_rules import BoosterRules
from ...primitives.symbols import Symbol, SymbolTier
from ..context import BoardContext
from ..evaluators import ChainEvaluator, SpawnEvaluator
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.cluster_builder import ClusterBuilder
from ..services.merge_policy import MergePolicy, ClusterPositionResult
from ..services.boundary_analyzer import BoundaryAnalysis
from ..services.forward_simulator import ForwardSimulator
from ..services.landing_evaluator import (
    BoosterLandingEvaluator, compute_reshape_bias,
)
from ..services.seed_planner import SeedPlanner
from ..services.spatial_context import StepSpatialContext
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


class BoosterSetupStrategy:
    """Plans spatial arrangement for booster chains.

    Determines which boosters still need to spawn, finds positions where
    their cluster centroids must land, and builds clusters that produce
    the correct centroid. Uses ChainEvaluator for chain validity and
    SpawnEvaluator for cluster size → booster mapping.
    """

    __slots__ = (
        "_config", "_forward_sim", "_cluster_builder", "_seed_planner",
        "_chain_eval", "_spawn_eval", "_landing_eval", "_booster_rules",
        "_rng", "_spatial",
    )

    def __init__(
        self,
        config: MasterConfig,
        forward_sim: ForwardSimulator,
        cluster_builder: ClusterBuilder,
        seed_planner: SeedPlanner,
        chain_eval: ChainEvaluator,
        spawn_eval: SpawnEvaluator,
        landing_eval: BoosterLandingEvaluator,
        rng: random.Random,
        spatial: StepSpatialContext | None = None,
    ) -> None:
        self._config = config
        self._forward_sim = forward_sim
        self._cluster_builder = cluster_builder
        self._seed_planner = seed_planner
        self._chain_eval = chain_eval
        self._spawn_eval = spawn_eval
        self._landing_eval = landing_eval
        self._booster_rules = BoosterRules(config.boosters, config.board, config.symbols)
        self._rng = rng
        self._spatial = spatial

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        # Determine which booster types still need to spawn for the chain
        missing_boosters = self._determine_missing_boosters(
            context, progress, signature,
        )

        # Boundary analysis for merge-aware placement
        boundary = self._cluster_builder.analyze_boundary(context)

        constrained: dict[Position, Symbol] = {}
        expected_spawns: list[str] = []

        for booster_type in missing_boosters:
            # Determine cluster size needed to spawn this booster type
            size_range = self._spawn_eval.size_range_for_booster(booster_type)
            if size_range is None:
                continue

            cluster_size = self._cluster_builder.select_size(
                progress, signature, variance, self._rng,
                size_range=Range(size_range[0], size_range[1]),
                target_booster=booster_type,
            )

            cluster_symbol = self._cluster_builder.select_symbol(
                progress, signature, variance, self._rng,
                boundary=boundary, planned_size=cluster_size,
            )

            # EXPLOIT merge when survivors can push total size past spawn threshold,
            # AVOID otherwise (exact size needed for correct booster type)
            survivor_count = boundary.acceptable_merge_symbols.get(cluster_symbol, 0)
            merged_total = cluster_size + survivor_count
            booster_for_merged = self._spawn_eval.booster_for_size(merged_total)
            policy = (
                MergePolicy.EXPLOIT
                if booster_for_merged == booster_type and survivor_count > 0
                else MergePolicy.AVOID
            )

            result = self._find_viable_cluster(
                context, cluster_size, cluster_symbol,
                boundary, policy, constrained, variance, booster_type,
            )

            for pos in result.planned_positions:
                constrained[pos] = cluster_symbol
            expected_spawns.append(booster_type)

        # Determine tier from the first cluster symbol placed
        cluster_tier = None
        if constrained:
            first_sym = next(iter(constrained.values()))
            cluster_tier = SymbolTier.LOW if first_sym.value <= 4 else SymbolTier.HIGH

        # No forward sim seeds needed — the chain arrangement IS the plan
        return StepIntent(
            step_type=StepType.CASCADE_CLUSTER,
            constrained_cells=constrained,
            strategic_cells={},
            expected_cluster_count=Range(len(missing_boosters), len(missing_boosters)),
            expected_cluster_sizes=[],
            expected_cluster_tier=cluster_tier,
            expected_spawns=expected_spawns,
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=[
                NoSpecialSymbolPropagator(self._config.symbols),
                NoClusterPropagator(self._config.board.min_cluster_size),
            ],
            wfc_symbol_weights=variance.symbol_weights,
            predicted_post_gravity=None,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            # Setup cluster positions will explode — gates gravity-aware WFC
            planned_explosion=frozenset(constrained),
            is_terminal=False,
        )

    def _find_viable_cluster(
        self,
        context: BoardContext,
        cluster_size: int,
        cluster_symbol: Symbol,
        boundary: BoundaryAnalysis,
        policy: MergePolicy,
        constrained: dict[Position, Symbol],
        variance: VarianceHints,
        booster_type: str,
    ) -> ClusterPositionResult:
        """Place a spawn cluster whose post-gravity landing can support the
        next step's arming cluster.

        Generates candidate shapes by progressively biasing variance toward
        upper, vertically concentrated placements (where centroids stay near
        the refill zone). Scores each via the landing evaluator's composite
        criterion; returns the first candidate meeting
        reasoner.arm_feasibility_threshold, or the best-scoring candidate
        if the retry budget is exhausted.

        The retry budget (reasoner.arm_feasibility_retry_budget) caps how
        many reshape attempts the strategy will try before settling for the
        best available — prevents unbounded spinning on infeasible boards.
        """
        avoid = frozenset(constrained)
        threshold = self._config.reasoner.arm_feasibility_threshold
        budget = self._config.reasoner.arm_feasibility_retry_budget

        best_result: ClusterPositionResult | None = None
        best_score = -1.0

        # attempt 0 is the unbiased placement; each subsequent attempt
        # progressively concentrates vertically via compute_reshape_bias.
        # Total attempts = 1 + budget (original + retries).
        # Reshape bias can over-concentrate to the point where find_positions
        # cannot fit the cluster; ValueError on a later attempt is treated as
        # a failed reshape and doesn't mask an earlier viable result.
        for attempt in range(budget + 1):
            attempt_variance = compute_reshape_bias(
                variance, self._config.board, attempt,
            )
            try:
                result = self._cluster_builder.find_positions(
                    context, cluster_size, self._rng, attempt_variance,
                    symbol=cluster_symbol, boundary=boundary, merge_policy=policy,
                    avoid_positions=avoid,
                )
            except ValueError:
                if best_result is None:
                    raise
                continue
            hypothetical = self._forward_sim.build_hypothetical(
                context.board,
                {pos: cluster_symbol for pos in result.planned_positions},
            )
            _ctx, score = self._landing_eval.evaluate_and_score(
                frozenset(result.planned_positions), hypothetical, booster_type,
            )
            if score > best_score:
                best_result = result
                best_score = score
            if score >= threshold:
                break

        # best_result is never None: first attempt (unbiased) raises on
        # genuine infeasibility, which propagates; subsequent reshape
        # failures are caught above.
        assert best_result is not None
        return best_result

    def _determine_missing_boosters(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
    ) -> list[str]:
        """Identify booster types that still need to spawn for the chain.

        Compares the signature's required booster spawns against what has
        already spawned. Returns types with unmet minimums.
        """
        remaining = progress.remaining_booster_spawns()
        existing_types = {b.booster_type for b in context.dormant_boosters}

        missing: list[str] = []
        for btype, needed_range in remaining.items():
            if needed_range.min_val > 0 and btype not in existing_types:
                missing.append(btype)
        return missing
