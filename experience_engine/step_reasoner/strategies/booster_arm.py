"""Booster arm strategy — places a cluster adjacent to a dormant booster.

The cluster must be orthogonally adjacent to the target booster so that
the explosion arms it. If the booster participates in a chain, strategic
cells are planned for the chain spatial arrangement.
"""

from __future__ import annotations

import random

from ...board_filler.propagators import (
    ClusterBoundaryPropagator,
    NoClusterPropagator,
    NoSpecialSymbolPropagator,
)
from ...config.schema import MasterConfig
from ...planning.region_constraint import region_for_step
from ...primitives.board import Position, orthogonal_neighbors
from ...primitives.symbols import Symbol, tier_of
from ..context import BoardContext, DormantBooster
from ..evaluators import ChainEvaluator
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.boundary_analyzer import BoundaryAnalysis
from ..services.cluster_builder import ClusterBuilder
from ..services.forward_simulator import ForwardSimulator
from ..services.landing_evaluator import BoosterLandingEvaluator
from ..services.influence_map import DemandSpec
from ..services.merge_policy import MergePolicy
from ..services.seed_planner import SeedPlanner, build_cluster_exclusions
from ..services.spatial_context import StepSpatialContext
from ..services.utility_scorer import ScoringContext
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints
from ..services.constraint_helpers import build_spawn_cap_propagator


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

        # A1: Cheap pre-check — abort before find_positions when the
        # refill geometry around the booster cannot support any arming
        # cluster. The probe uses a synthetic test cluster (per the doc),
        # which is a strict lower bound on real placement score because
        # find_positions chooses the cluster shape optimally. We therefore
        # gate on probe == 0.0 only (true geometric impossibility); the
        # post-placement check still enforces the full threshold once the
        # real cluster exists. This earlier reject saves the find_positions
        # search on doomed targets without false-positive rejecting
        # placements that would have succeeded.
        probe_score = self._probe_landing_viability(context, target_booster)
        if probe_score == 0.0:
            raise ValueError(
                f"Pre-placement landing probe for "
                f"{target_booster.booster_type} found no viable refill "
                f"geometry (score 0.0)"
            )

        # Scan for survivor components adjacent to the dormant booster —
        # these existing groups can contribute to the arming cluster
        adjacent_survivors = self._booster_adjacent_survivors(booster_pos, boundary)

        # Build affinity scores: base 1.0 + per-cell bonus from config
        # Counteracts the merge-safety penalty for symbols that can arm
        # the booster with fewer new cells
        affinity_per_cell = self._config.booster_arm.survivor_affinity_per_cell
        affinity_scores: dict[Symbol, float] = {
            sym: 1.0 + count * affinity_per_cell
            for sym, count in adjacent_survivors.items()
        }

        # Clamp size to available space and spawn-safe ceiling so the arm
        # cluster doesn't trigger an unbudgeted booster spawn
        step_sizes = progress.current_step_size_ranges()
        size_range = step_sizes[0] if step_sizes else None
        max_size = len(context.empty_cells)
        spawn_ceiling = self._cluster_builder.spawn_safe_ceiling(progress)
        if spawn_ceiling is not None:
            max_size = min(max_size, spawn_ceiling)

        cluster_size = self._cluster_builder.select_size(
            progress, signature, variance, self._rng,
            size_range=size_range,
            max_available=max_size,
        )
        cluster_symbol = self._cluster_builder.select_symbol(
            progress, signature, variance, self._rng,
            boundary=boundary, planned_size=cluster_size,
            affinity_scores=affinity_scores,
        )
        # Tier of the chosen arm-cluster symbol — drives expected_cluster_tier
        # in the StepIntent so downstream WFC honors the symbol's pay group.
        # Sourced from SymbolConfig (low_tier / high_tier lists) so the
        # boundary moves with the paytable rather than a hardcoded ordinal.
        cluster_tier = tier_of(cluster_symbol, self._config.symbols)

        # Exploit survivors adjacent to the booster — they reduce new cells needed.
        # Avoid otherwise (no useful survivors to leverage).
        survivor_count = adjacent_survivors.get(cluster_symbol, 0)
        merged_total = cluster_size + survivor_count
        # Guard: merged total must not trigger an unwanted booster spawn
        unwanted_spawn = (
            self._cluster_builder._spawn_eval.booster_for_size(merged_total)
            if survivor_count > 0
            else None
        )
        policy = (
            MergePolicy.EXPLOIT
            if survivor_count > 0 and unwanted_spawn is None
            else MergePolicy.AVOID
        )

        # Planning guidance (atlas/trajectory) biases cluster shape toward
        # the pre-validated columns when present; region_for_step returns
        # None for unguided arcs, preserving the historic placement.
        region = region_for_step(progress.guidance, progress.steps_completed)
        result = self._cluster_builder.find_positions(
            context, cluster_size, self._rng, variance,
            symbol=cluster_symbol, boundary=boundary, merge_policy=policy,
            must_be_adjacent_to=frozenset({booster_pos}),
            region=region,
        )
        cluster_positions = result.planned_positions

        # Arm contract enforcement: the dormant booster arms only when a
        # cluster position is orthogonally adjacent. `must_be_adjacent_to`
        # constrains the BFS seed, but under EXPLOIT/ACCEPT merge policies
        # the returned planned_positions may be dominated by survivor cells
        # whose layout loses the adjacency the seed started with. Failing
        # here raises ValueError which cascade_generator catches for a
        # fresh instance reroll — cheaper than letting the arm silently
        # fail and surfacing as booster_fire=0 after validation.
        booster_neighbours = frozenset(
            orthogonal_neighbors(booster_pos, self._config.board)
        )
        if not (cluster_positions & booster_neighbours):
            raise ValueError(
                f"Planned arm cluster for {target_booster.booster_type} at "
                f"{booster_pos} has no orthogonal-adjacent cell — cluster "
                f"would not arm the booster"
            )

        # A6: Always run the forward simulation when there are positions to
        # arm. This unblocks plan_arm_seeds for non-chain rocket/bomb/lb/slb
        # arcs, which previously skipped seed planning and over-constrained
        # the WFC refill on the next cascade step.
        hypothetical = self._forward_sim.build_hypothetical(
            context.board,
            {pos: cluster_symbol for pos in cluster_positions},
        )
        settle_result = self._forward_sim.simulate_explosion(
            hypothetical, cluster_positions,
        )

        # Per-step exclusions applied to seed planning regardless of chain —
        # arm seeds must not merge into the arming cluster.
        exclusions = build_cluster_exclusions(
            [(frozenset(cluster_positions), cluster_symbol)],
            self._config.board,
        )

        strategic_cells: dict[Position, Symbol] = {}
        reserve_zone: frozenset[Position] | None = None
        utility_scores: dict[Position, float] | None = None

        if self._needs_chain(target_booster, progress, signature):
            # Chain-only: post-placement landing-score check (defense in
            # depth — even a probe-passing booster can land on a placement
            # whose specific shape blocks the continuation cluster).
            _ctx, landing_score = self._landing_eval.evaluate_and_score(
                frozenset(cluster_positions), hypothetical,
                target_booster.booster_type,
            )
            if landing_score == 0.0:
                raise ValueError(
                    f"Landing for {target_booster.booster_type} has zero arm "
                    f"feasibility — no reachable refill region for a "
                    f"continuation cluster"
                )

            # Chain-only: utility scoring for seed selection. Gravity
            # alignment + booster adjacency steer seeds toward the arming
            # zone, but only chain steps need this extra signal.
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

        # A6: Always plan arm seeds when refill space exists. Empty
        # settle_result.empty_positions short-circuits inside the planner
        # (returns {}), preserving the prior non-chain behavior in that
        # degenerate case.
        if settle_result.empty_positions:
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
                # Budget-aware spawn cap — prevents the arm cluster's noise
                # fill from forming an additional over-budget booster-spawning
                # component (wild-aware through active_wilds).
                build_spawn_cap_propagator(
                    self._cluster_builder._spawn_eval,  # type: ignore[attr-defined]
                    progress,
                    wild_positions=frozenset(progress.active_wilds),
                ),
                # Prevent WFC from placing the arm-cluster symbol adjacent to
                # the constrained cluster cells — without this, strays merge
                # into the planned cluster and distort its shape so it no
                # longer borders the booster, causing the arm to fail.
                ClusterBoundaryPropagator(
                    frozenset(cluster_positions),
                    cluster_symbol,
                    self._config.board,
                ),
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

    def _probe_landing_viability(
        self,
        context: BoardContext,
        target_booster: DormantBooster,
    ) -> float:
        """A1: Cheap pre-check — score the landing of a synthetic test cluster
        placed at the booster's neighbor cells.

        Reuses the same forward simulation + landing evaluator as the
        post-placement check, so the early reject path applies the same
        scoring rule. Returns 0.0 when the booster lacks even
        min_cluster_size adjacent empties — geometrically impossible to arm.
        """
        booster_pos = target_booster.position
        min_size = self._config.board.min_cluster_size
        # Candidate cells = booster's empty neighbors plus their empty
        # neighbors (BFS one ring out). Mirrors the area where any real
        # arming cluster will end up living.
        empty_set = frozenset(context.empty_cells)
        first_ring = [
            p for p in orthogonal_neighbors(booster_pos, self._config.board)
            if p in empty_set
        ]
        if not first_ring:
            return 0.0
        test_positions: list[Position] = []
        seen: set[Position] = set()
        for seed in first_ring:
            if seed in seen:
                continue
            seen.add(seed)
            test_positions.append(seed)
            if len(test_positions) >= min_size:
                break
            for nbr in orthogonal_neighbors(seed, self._config.board):
                if nbr in empty_set and nbr not in seen and nbr != booster_pos:
                    seen.add(nbr)
                    test_positions.append(nbr)
                    if len(test_positions) >= min_size:
                        break
            if len(test_positions) >= min_size:
                break
        if len(test_positions) < min_size:
            return 0.0
        # The landing evaluator only inspects geometry of the post-explosion
        # refill — the symbol identity doesn't affect the score, so we use
        # the first standard symbol as a stable placeholder.
        placeholder = next(iter(self._config.symbols.standard))
        placeholder_symbol = Symbol[placeholder]
        hypothetical = self._forward_sim.build_hypothetical(
            context.board,
            {pos: placeholder_symbol for pos in test_positions},
        )
        _ctx, score = self._landing_eval.evaluate_and_score(
            frozenset(test_positions),
            hypothetical,
            target_booster.booster_type,
        )
        return score

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

    def _booster_adjacent_survivors(
        self,
        booster_pos: Position,
        boundary: BoundaryAnalysis,
    ) -> dict[Symbol, int]:
        """Survivor cells orthogonally adjacent to the booster, per symbol.

        Reuses BoundaryAnalysis.survivor_components (DRY — no second BFS).
        Only includes components that directly touch the booster position,
        since those are the ones that can contribute to the arming cluster.
        """
        booster_neighbors = frozenset(
            orthogonal_neighbors(booster_pos, self._config.board)
        )
        totals: dict[Symbol, int] = {}
        for symbol, components in boundary.survivor_components.items():
            for comp in components:
                if comp.positions & booster_neighbors:
                    totals[symbol] = totals.get(symbol, 0) + comp.size
        return totals
