"""Initial cluster strategy — step 0 for archetypes that need clusters.

Places cluster(s) on an empty board, forward-simulates gravity to predict
post-explosion board state, then backward-reasons about strategic seed
placements for future cascade steps. Supports multi-cluster archetypes
(e.g., t1_multi_cascade) via ClusterBuilder.build_multi_cluster().
"""

from __future__ import annotations

import random

from ...board_filler.propagators import (
    ClusterBoundaryPropagator, NearMissAwareDeadPropagator, NearMissGroup,
    NoClusterPropagator, NoSpecialSymbolPropagator,
)
from ...config.schema import MasterConfig
from ...planning.region_constraint import landing_for_step, region_for_step
from ...primitives.board import Position
from ...primitives.symbols import Symbol, SymbolTier
from ..context import BoardContext
from ..evaluators import SpawnEvaluator
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
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


class InitialClusterStrategy:
    """Builds the initial board with cluster(s), scatters, near-misses, and strategic seeds.

    This strategy does the most forward simulation because the initial board
    determines the entire cascade trajectory. It reasons about cluster
    placement, booster spawn prediction, and backward seeding for future steps.
    Near-misses are placed via NearMissPlanner when the archetype requires them.
    """

    __slots__ = (
        "_config", "_forward_sim", "_cluster_builder", "_seed_planner",
        "_spawn_eval", "_near_miss_planner", "_landing_eval", "_rng",
        "_spatial",
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
        self._rng = rng
        self._spatial = spatial

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

        # Use step-level sizes for both cap check and cluster building —
        # signature-level sizes may include non-wild-spawning ranges (e.g., 5-6)
        # while step-level sizes (e.g., 7-8) correctly reflect this step's constraints
        step_sizes = progress.current_step_size_ranges()

        # Cap cluster count to Wild spawn budget when every cluster would spawn a Wild
        if self._all_sizes_spawn_wilds(step_sizes):
            wild_budget = progress.remaining_booster_spawns().get("W")
            if wild_budget is not None:
                cluster_count = min(cluster_count, wild_budget.max_val)
        # Planning guidance steers the first-step cluster when the atlas /
        # trajectory planner resolved a configuration for this arc. None for
        # unguided arcs, leaving placement unchanged from pre-guidance behavior.
        region = region_for_step(progress.guidance, progress.steps_completed)
        multi_result = self._cluster_builder.build_multi_cluster(
            context, cluster_count, list(step_sizes),
            progress, signature, variance, self._rng,
            region=region,
        )

        # Derive tier from the first cluster symbol placed
        if not multi_result.cluster_symbols:
            cluster_tier = SymbolTier.ANY
        else:
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
        # Build once — reused for both seed exclusion and propagator construction
        cluster_groups = [
            (r.planned_positions, sym)
            for r, sym in zip(multi_result.clusters, multi_result.cluster_symbols)
        ]
        strategic_cells, reserve_zone, predicted_wild_positions = self._plan_strategic_seeds(
            context, cluster_groups, first_booster,
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
        # group prevents same-symbol survivors at each cluster's edge.
        # Pre-count CSP-committed spawns (CSP pins clusters before WFC but the
        # spawn is recorded only during transition) so the spawn cap blocks
        # WFC from creating ADDITIONAL booster-spawning components beyond
        # what the strategy has already committed to.
        committed_spawns: dict[str, int] = {}
        for btype in expected_spawns:
            committed_spawns[btype] = committed_spawns.get(btype, 0) + 1
        propagators = self._select_propagators(
            cluster_groups=cluster_groups,
            near_miss_groups=nm_result.groups or None,
            progress=progress,
            committed_spawns=committed_spawns,
        )

        # Atlas-derived landing — AtlasQuery already validated armable adjacency
        # for this phase's booster. The first cluster (index 0) is the booster
        # spawner in single-cluster archetypes; propagate to BoosterRules via
        # StepIntent.predicted_landings so post-gravity placement matches the
        # position the atlas filtered for arming-cell count.
        atlas_landing = landing_for_step(progress.guidance, progress.steps_completed)
        predicted_landings = (
            {0: atlas_landing}
            if atlas_landing is not None and expected_spawns
            else None
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
            reserve_zone=reserve_zone,
            predicted_wild_positions=predicted_wild_positions,
            predicted_landings=predicted_landings,
        )

    def _plan_strategic_seeds(
        self,
        context: BoardContext,
        cluster_groups: list[tuple[frozenset[Position], Symbol]],
        booster_type: str | None,
        settle_result,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> tuple[dict[Position, Symbol], frozenset[Position] | None, frozenset[Position] | None]:
        """Backward reasoning — determine what future steps need and seed accordingly.

        Returns (strategic_cells, reserve_zone, predicted_wild_positions).
        """
        if progress.must_terminate_soon():
            return {}, None, None

        # Exclusion zones prevent strategic seeds from merging into clusters —
        # strategic cells are pinned before WFC, so ClusterBoundaryPropagator
        # cannot guard them
        exclusions = build_cluster_exclusions(cluster_groups, self._config.board)

        # Predict where the booster will land. The landing's post-gravity
        # arming viability is enforced by the column-overlap gate below, not
        # by the evaluator's score (which is meaningless on a pre-WFC board).
        if booster_type:
            cluster_positions = frozenset().union(*(g[0] for g in cluster_groups))
            ctx = self._landing_eval.evaluate(
                cluster_positions, context.board, booster_type,
            )
            booster_landing = ctx.landing_position

            if self._next_step_needs_arming_cluster(signature, progress):
                # Gate 1 — column overlap. After WFC fills the board and the
                # cluster explodes, gravity leaves empty cells only at the
                # top of columns that contained cluster positions. The
                # booster lands near the cluster centroid; for an arming
                # cluster to form around it, at least one cluster column
                # must sit within orthogonal-adjacency range (±1) of the
                # landing column. The score-based gate was meaningless at
                # Step 0 (evaluator runs on a pre-WFC, nearly-empty board);
                # the column check is derived from gravity + adjacency rules
                # and works regardless of current board state.
                landing_col = booster_landing.reel
                cluster_cols = ctx.cluster_shape_stats.columns_used
                if not any(abs(c - landing_col) <= 1 for c in cluster_cols):
                    raise ValueError(
                        f"Cluster columns {sorted(cluster_cols)} not within "
                        f"1 of landing column {landing_col} — no post-gravity "
                        f"refill adjacency possible"
                    )

                # Gate 2 — orientation constraint. Data-driven: non-rocket
                # archetypes leave signature.rocket_orientation=None, so the
                # condition short-circuits. Prevents rocket_h_fire from
                # committing to a cluster whose firing direction mismatches
                # the archetype's declared orientation.
                if (
                    signature.rocket_orientation is not None
                    and ctx.cluster_shape_stats.orientation
                        != signature.rocket_orientation
                ):
                    raise ValueError(
                        f"Cluster orientation "
                        f"'{ctx.cluster_shape_stats.orientation}' != "
                        f"required '{signature.rocket_orientation}'"
                    )

                utility_scores, reserve_zone = self._compute_spatial_scores(
                    cluster_positions, booster_landing, booster_type,
                    self._config.board.min_cluster_size, settle_result,
                )
                seeds = self._seed_planner.plan_arm_seeds(
                    booster_landing, settle_result, variance, self._rng,
                    exclusions=exclusions,
                    utility_scores=utility_scores,
                )
                return seeds, reserve_zone, None

        seeds = self._seed_planner.plan_generic_seeds(
            settle_result, progress, signature, variance, self._rng,
            exclusions=exclusions,
        )
        return seeds, None, None

    def _compute_spatial_scores(
        self,
        cluster_positions: frozenset[Position],
        booster_landing: Position,
        booster_type: str,
        next_cluster_size: int,
        settle_result,
    ) -> tuple[dict[Position, float] | None, frozenset[Position] | None]:
        """Compute utility scores and reserve zone using spatial intelligence.

        Returns (utility_scores, reserve_zone). Both are None when spatial
        intelligence is disabled.
        """
        if self._spatial is None:
            return None, None

        demand = DemandSpec(
            centroid=booster_landing,
            cluster_size=next_cluster_size,
            booster_type=booster_type,
        )
        influence = self._spatial.influence_map.compute(demand)
        reserve_zone = self._spatial.influence_map.reserve_zone(influence)

        scoring_ctx = ScoringContext(
            influence=influence,
            gravity_field=self._spatial.gravity_field,
            demand=demand,
            cluster_positions=cluster_positions,
            board_config=self._config.board,
            booster_landing=booster_landing,
        )
        candidates = list(settle_result.empty_positions)
        utility_scores = self._spatial.utility_scorer.score_candidates(
            candidates, scoring_ctx,
        )
        return utility_scores, reserve_zone

    def _next_step_needs_arming_cluster(
        self,
        signature: ArchetypeSignature,
        progress: ProgressTracker,
    ) -> bool:
        """Check if the next step needs to arm a dormant booster.

        Arc-based: peek at the next phase's arms field.
        Legacy: index into cascade_steps[next_step].must_arm_booster.
        """
        # Arc-based path. Uses peek_phase_after_current — not peek_next_phase —
        # because at planning time the current step has not yet been counted
        # toward the current phase's repetitions. The current step will
        # complete (or further) the current phase, so the NEXT step belongs
        # to the phase that follows. peek_next_phase would return the current
        # phase while reps remain, making arming-cluster gates unreachable.
        next_phase = progress.peek_phase_after_current()
        if next_phase is not None:
            return next_phase.arms is not None

        # Legacy path
        next_step = progress.steps_completed + 1
        if signature.cascade_steps and next_step < len(signature.cascade_steps):
            step_spec = signature.cascade_steps[next_step]
            return step_spec.must_arm_booster is not None
        return False

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
        progress: ProgressTracker | None = None,
        committed_spawns: dict[str, int] | None = None,
    ) -> list:
        """Select WFC propagators for noise fill around the cluster(s).

        NoClusterPropagator prevents WFC from extending placed clusters
        or forming new accidental clusters. One ClusterBoundaryPropagator
        per cluster group forbids each cluster's symbol at adjacent cells —
        prevents same-symbol survivors from bordering the refill zone
        after explosion + gravity.
        MaxComponentPropagator caps fill components below near-miss size.
        When near-miss groups are present, NearMissAwareDeadPropagator
        replaces it — same component cap plus isolation at group borders.
        When `progress` is supplied, BoosterSpawnCapPropagator is appended
        so WFC cannot form 7-8 (or larger) components that would spawn an
        out-of-budget booster — guards against the `booster_spawn(W)=N
        outside [X]` failure class.
        """
        from ...board_filler.propagators import MaxComponentPropagator
        from ..services.constraint_helpers import build_spawn_cap_propagator

        propagators = [
            NoSpecialSymbolPropagator(self._config.symbols),
            NoClusterPropagator(self._config.board.min_cluster_size),
        ]
        # Signature-aware spawn cap — only when progress is available so the
        # propagator can read live budget state. Always safe: no-op when
        # every budget has remaining headroom.
        if progress is not None:
            propagators.append(build_spawn_cap_propagator(
                self._spawn_eval,
                progress,
                wild_positions=frozenset(progress.active_wilds),
                committed_spawns=committed_spawns,
            ))
        # Isolate each cluster's boundary — WFC cannot place the cluster symbol
        # adjacent to cluster positions, preventing merge on the next cascade step
        if cluster_groups:
            for positions, symbol in cluster_groups:
                propagators.append(ClusterBoundaryPropagator(
                    positions, symbol, self._config.board,
                ))
        # Cap WFC-placed components below near-miss size. NM groups are pinned
        # as constrained cells before WFC runs, so the cap only affects
        # WFC-placed cells — NearMissAwareDeadPropagator adds isolation at
        # group borders to prevent WFC from merging symbols into NM groups.
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
