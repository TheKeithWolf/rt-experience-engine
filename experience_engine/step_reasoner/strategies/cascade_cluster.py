"""Cascade cluster strategy — general mid-cascade step.

Forms a cluster from empty cells on the settled board using survivor-aware
placement. Analyzes the boundary between empty cells and survivors, ranks
symbols by merge risk, selects a merge policy per symbol, then delegates
position finding to ClusterBuilder's merge-policy handlers.

Forward simulates to verify the placement, then backward-reasons about
seeds for the next step.
"""

from __future__ import annotations

import random

from ...board_filler.propagators import (
    ClusterBoundaryPropagator,
    NoClusterPropagator,
    NoSpecialSymbolPropagator,
)
from ...config.schema import MasterConfig
from ...primitives.board import Position
from ...primitives.cluster_detection import detect_clusters
from ...primitives.symbols import Symbol, SymbolTier, symbols_in_tier
from ..context import BoardContext
from ..evaluators import SpawnEvaluator
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.boundary_analyzer import BoundaryAnalysis
from ..services.cluster_builder import ClusterBuilder
from ..services.forward_simulator import ForwardSimulator
from ..services.merge_policy import ClusterPositionResult, MergePolicy
from ..services.seed_planner import SeedPlanner, build_cluster_exclusions
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


class CascadeClusterStrategy:
    """General cascade step — forms a cluster from empty cells on the settled board.

    Uses boundary analysis to rank symbols by merge risk, selects a merge
    policy per symbol, and delegates position finding to ClusterBuilder.
    Forward simulates to verify placement, backward-reasons to seed next step.
    """

    __slots__ = (
        "_config", "_forward_sim", "_cluster_builder", "_seed_planner",
        "_spawn_eval", "_rng",
    )

    def __init__(
        self,
        config: MasterConfig,
        forward_sim: ForwardSimulator,
        cluster_builder: ClusterBuilder,
        seed_planner: SeedPlanner,
        spawn_eval: SpawnEvaluator,
        rng: random.Random,
    ) -> None:
        self._config = config
        self._forward_sim = forward_sim
        self._cluster_builder = cluster_builder
        self._seed_planner = seed_planner
        self._spawn_eval = spawn_eval
        self._rng = rng

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        tier = self._required_tier(progress, signature)

        available_count = len(context.empty_cells)
        if available_count == 0:
            raise ValueError("No empty cells available for cascade cluster")

        # Boundary analysis — computed once, shared by symbol ranking + position finding
        boundary = self._cluster_builder.analyze_boundary(context)

        # Clamp size to available empty cells
        # Use step-level size constraints when cascade_steps defines them
        step_sizes = progress.current_step_size_ranges()
        target_size = self._cluster_builder.select_size(
            progress, signature, variance, self._rng,
            size_range=step_sizes[0] if step_sizes else None,
            max_available=available_count,
        )

        # Rank symbols: safe first, acceptable-merge second, risky last
        candidates = self._rank_symbols(
            boundary, target_size, progress, signature, variance, tier,
        )

        # Try each symbol with its appropriate merge policy until one succeeds
        result: ClusterPositionResult | None = None
        cluster_symbol: Symbol | None = None

        for symbol in candidates:
            policy = self._determine_merge_policy(
                symbol, target_size, boundary, progress, signature,
            )
            try:
                result = self._cluster_builder.find_positions(
                    context, target_size, self._rng, variance,
                    symbol=symbol, boundary=boundary, merge_policy=policy,
                )
            except ValueError:
                continue

            # Forward simulate to verify cluster detection matches expectations
            hypothetical = self._forward_sim.build_hypothetical(
                context.board,
                {pos: symbol for pos in result.planned_positions},
            )
            detected = detect_clusters(hypothetical, self._config)

            if self._cluster_matches(detected, symbol, result):
                cluster_symbol = symbol
                break

            result = None

        if result is None or cluster_symbol is None:
            raise ValueError(
                f"No valid symbol/position combination for cascade cluster "
                f"with {available_count} empty cells"
            )

        cluster_tier = (
            SymbolTier.LOW if cluster_symbol.value <= 4 else SymbolTier.HIGH
        )

        # Booster spawn based on TOTAL size (includes merged survivors)
        booster_type = self._spawn_eval.booster_for_size(result.total_size)
        expected_spawns = [booster_type] if booster_type else []

        # Backward reasoning: seed the next step if not terminal
        strategic_cells: dict[Position, Symbol] = {}
        if not progress.must_terminate_soon():
            hypothetical = self._forward_sim.build_hypothetical(
                context.board,
                {pos: cluster_symbol for pos in result.planned_positions},
            )
            settle_result = self._forward_sim.simulate_explosion(
                hypothetical, result.planned_positions,
            )
            # Prevent strategic seeds from merging into the cascade cluster
            exclusions = build_cluster_exclusions(
                [(frozenset(result.planned_positions), cluster_symbol)],
                self._config.board,
            )
            strategic_cells = self._seed_planner.plan_generic_seeds(
                settle_result, progress, signature, variance, self._rng,
                exclusions=exclusions,
            )

        constrained = {pos: cluster_symbol for pos in result.planned_positions}

        return StepIntent(
            step_type=StepType.CASCADE_CLUSTER,
            constrained_cells=constrained,
            strategic_cells=strategic_cells,
            expected_cluster_count=Range(1, 1),
            expected_cluster_sizes=[Range(result.total_size, result.total_size)],
            expected_cluster_tier=cluster_tier,
            expected_spawns=expected_spawns,
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=[
                NoSpecialSymbolPropagator(self._config.symbols),
                NoClusterPropagator(self._config.board.min_cluster_size),
                # Defense-in-depth: prevent same-symbol survivors at boundary
                ClusterBoundaryPropagator(
                    frozenset(result.planned_positions),
                    cluster_symbol, self._config.board,
                ),
            ],
            wfc_symbol_weights=variance.symbol_weights,
            predicted_post_gravity=None,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            # Cluster positions will explode — gates gravity-aware WFC mechanisms
            planned_explosion=frozenset(result.planned_positions),
            is_terminal=False,
        )

    # -- Helpers -----------------------------------------------------------

    def _required_tier(
        self,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
    ) -> SymbolTier | None:
        """Resolve the symbol tier for this step — from arc phase or legacy map."""
        # Arc-based: read from current phase
        phase = progress.current_phase()
        if phase is not None:
            return phase.cluster_symbol_tier
        # Legacy: read from step-indexed tier map
        if signature.symbol_tier_per_step:
            return signature.symbol_tier_per_step.get(progress.steps_completed)
        return None

    def _rank_symbols(
        self,
        boundary: BoundaryAnalysis,
        target_size: int,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
        tier: SymbolTier | None,
    ) -> list[Symbol]:
        """Rank symbols: safe first, acceptable-merge second, risky last.

        Within each group, sorted by variance weight (descending) to prefer
        least-used symbols.
        """
        resolved_tier = tier or signature.required_cluster_symbols or SymbolTier.ANY
        all_candidates = list(symbols_in_tier(resolved_tier, self._config.symbols))

        safe: list[Symbol] = []
        acceptable: list[Symbol] = []
        risky: list[Symbol] = []

        for sym in all_candidates:
            if sym in boundary.safe_symbols:
                safe.append(sym)
            elif sym in boundary.acceptable_merge_symbols:
                merged_size = target_size + boundary.acceptable_merge_symbols[sym]
                # Within signature cluster size ranges?
                acceptable_flag = any(
                    r.min_val <= merged_size <= r.max_val
                    for r in signature.required_cluster_sizes
                )
                if acceptable_flag:
                    acceptable.append(sym)
                else:
                    risky.append(sym)
            else:
                safe.append(sym)

        weight_key = lambda s: variance.symbol_weights.get(s, 1.0)
        return (
            sorted(safe, key=weight_key, reverse=True)
            + sorted(acceptable, key=weight_key, reverse=True)
            + sorted(risky, key=weight_key, reverse=True)
        )

    def _determine_merge_policy(
        self,
        symbol: Symbol,
        target_size: int,
        boundary: BoundaryAnalysis,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
    ) -> MergePolicy:
        """Decide how to handle potential merges for this symbol.

        Checks: (1) merge would spawn unwanted booster? → AVOID.
        (2) merge pushes payout outside range? → AVOID.
        (3) signature needs a larger cluster? → EXPLOIT.
        (4) merge within signature range? → ACCEPT.
        Fallback: AVOID.
        """
        if symbol in boundary.safe_symbols:
            return MergePolicy.AVOID

        survivor_count = boundary.acceptable_merge_symbols.get(symbol, 0)
        merged_size = target_size + survivor_count

        # Would merge trigger an unwanted booster spawn?
        booster = self._spawn_eval.booster_for_size(merged_size)
        if booster and not progress.can_still_spawn(booster):
            return MergePolicy.AVOID

        # Would merge push payout outside the archetype's range?
        remaining = progress.remaining_payout_budget()
        estimated_payout = self._spawn_eval.estimate_cluster_payout(
            merged_size, symbol,
        ) if hasattr(self._spawn_eval, 'estimate_cluster_payout') else 0.0
        if remaining.max_val > 0 and estimated_payout > remaining.max_val:
            return MergePolicy.AVOID

        # Merge within signature cluster size ranges? → ACCEPT to conserve empty cells
        within_range = any(
            r.min_val <= merged_size <= r.max_val
            for r in signature.required_cluster_sizes
        )
        if within_range:
            return MergePolicy.ACCEPT

        return MergePolicy.AVOID

    def _cluster_matches(
        self,
        detected: list,
        symbol: Symbol,
        result: ClusterPositionResult,
    ) -> bool:
        """Verify detected clusters include one matching our placement.

        The detected cluster may be a superset of planned positions (if merge
        occurred) — that's expected. Checks symbol match and position overlap.
        """
        all_expected = result.planned_positions | result.merged_survivor_positions
        for cluster in detected:
            core = cluster.positions - cluster.wild_positions
            if cluster.symbol is symbol and result.planned_positions.issubset(core):
                return True
        return False
