"""Cluster builder — selects cluster parameters and finds valid positions.

Encapsulates the "pick size + symbol + positions" pattern used by 5+ strategies.
Size selection respects booster spawn thresholds; symbol selection respects tier
constraints and merge risk; position finding uses BFS growth with merge-policy
dispatch (AVOID / ACCEPT / EXPLOIT).

build_multi_cluster() encapsulates the N-cluster accumulating loop — strategies
call it instead of manually looping select_size + select_symbol + find_positions.

Used by: InitialCluster, CascadeCluster, BoosterArm, BoosterSetup, WildBridge.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable

from ...config.schema import BoardConfig, SymbolConfig
from ...pipeline.protocols import Range, RangeFloat
from ...primitives.board import Position, orthogonal_neighbors
from ...primitives.symbols import Symbol, SymbolTier, symbols_in_tier, tier_of
from ...variance.hints import VarianceHints
from ..context import BoardContext
from ..evaluators import PayoutEstimator, SpawnEvaluator
from ..progress import ProgressTracker

from ...archetypes.registry import ArchetypeSignature
from .boundary_analyzer import BoundaryAnalysis, BoundaryAnalyzer
from .merge_policy import ClusterPositionResult, MergePolicy


@dataclass(frozen=True, slots=True)
class MultiClusterResult:
    """Result of placing N non-overlapping clusters on the same board.

    Strategies receive this from build_multi_cluster() and use it to
    build the StepIntent — all_constrained becomes constrained_cells,
    all_occupied feeds avoid sets for scatters/near-misses, and
    per-cluster data drives expected_cluster_sizes and spawn detection.
    """

    clusters: tuple[ClusterPositionResult, ...]
    # symbol → positions for each cluster, combined
    all_constrained: dict[Position, Symbol]
    # Union of all cluster positions — for avoid sets and explosion simulation
    all_occupied: frozenset[Position]
    # Per-cluster symbols in placement order
    cluster_symbols: tuple[Symbol, ...]
    # Per-cluster sizes in placement order
    cluster_sizes: tuple[int, ...]
    total_payout_estimate: float


class ClusterBuilder:
    """Selects cluster size, symbol, and connected positions.

    All game-rule thresholds come from injected evaluators and config —
    no hardcoded values. Randomized methods accept an rng parameter
    for reproducible results (matching WFC/CSP solver conventions).

    Merge-aware: when a BoundaryAnalysis is provided, symbol selection
    penalizes risky symbols and position selection respects the chosen
    MergePolicy (AVOID/ACCEPT/EXPLOIT via dict dispatch).
    """

    __slots__ = (
        "_spawn_eval", "_payout_eval", "_board_config", "_symbol_config",
        "_boundary_analyzer", "_centipayout_multiplier", "_max_seed_retries",
    )

    def __init__(
        self,
        spawn_evaluator: SpawnEvaluator,
        payout_estimator: PayoutEstimator,
        board_config: BoardConfig,
        symbol_config: SymbolConfig,
        boundary_analyzer: BoundaryAnalyzer | None = None,
        centipayout_multiplier: int = 100,
        max_seed_retries: int = 5,
    ) -> None:
        self._spawn_eval = spawn_evaluator
        self._payout_eval = payout_estimator
        self._board_config = board_config
        self._symbol_config = symbol_config
        self._boundary_analyzer = boundary_analyzer
        # Converts centipayout → bet multiplier for payout-aware symbol scoring
        self._centipayout_multiplier = centipayout_multiplier
        # BFS seed retry limit — retries with different seeds when frontier
        # exhaustion occurs on fragmented available-sets (AVOID merge policy)
        self._max_seed_retries = max_seed_retries

    # -- Public convenience -------------------------------------------------

    def analyze_boundary(
        self,
        context: BoardContext,
    ) -> BoundaryAnalysis:
        """Delegate boundary analysis to the injected BoundaryAnalyzer.

        Strategies call this once per step and pass the result to
        select_symbol() and find_positions().
        """
        if self._boundary_analyzer is None:
            raise RuntimeError("BoundaryAnalyzer not injected — cannot analyze boundary")
        empty = frozenset(context.empty_cells)
        return self._boundary_analyzer.analyze(context, empty)

    # -- Size selection -----------------------------------------------------

    def select_size(
        self,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
        rng: random.Random,
        size_range: Range | None = None,
        target_booster: str | None = None,
        max_available: int | None = None,
    ) -> int:
        """Select cluster size within constraints, weighted by variance preference.

        If target_booster is specified, constrains to the booster's spawn
        threshold range (e.g. "R" → 9-10). If size_range is also provided,
        intersects both ranges. Falls back to the signature's first cluster
        size spec if neither constraint is given.

        max_available clamps the upper bound to the number of empty cells
        on the board — prevents requesting more cells than physically exist.
        """
        resolved = self._resolve_size_range(size_range, target_booster, signature)
        effective_max = min(resolved.max_val, max_available) if max_available is not None else resolved.max_val
        effective_min = min(resolved.min_val, effective_max)
        candidates = list(range(effective_min, effective_max + 1))

        if not candidates:
            raise ValueError(
                f"Empty size range after resolution: {resolved}. "
                f"target_booster={target_booster}, size_range={size_range}"
            )

        # Weight by variance preference — earlier in tuple = least-used = higher priority
        pref = variance.cluster_size_preference
        pref_index = {size: idx for idx, size in enumerate(pref)}
        pref_len = len(pref)
        weights = [
            (pref_len - pref_index[s] if s in pref_index else 1)
            for s in candidates
        ]
        return rng.choices(candidates, weights=weights, k=1)[0]

    def spawn_safe_ceiling(self, progress: ProgressTracker) -> int | None:
        """Largest cluster size that won't trigger an overbudget spawn.

        Returns None if all spawn types are within budget (no cap needed).
        Complements the per-size check in _is_valid_merge (line 526-528) by
        providing an upper bound strategies can pass as max_available.
        """
        ceiling: int | None = None
        for threshold in self._spawn_eval.all_thresholds():
            if not progress.can_still_spawn(threshold.booster):
                candidate = threshold.min_size - 1
                ceiling = min(ceiling, candidate) if ceiling is not None else candidate
        return ceiling

    # -- Symbol selection ---------------------------------------------------

    def select_symbol(
        self,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
        rng: random.Random,
        tier: SymbolTier | None = None,
        boundary: BoundaryAnalysis | None = None,
        planned_size: int | None = None,
        affinity_scores: dict[Symbol, float] | None = None,
    ) -> Symbol:
        """Select cluster symbol within tier, merge-aware and payout-aware.

        Four scoring dimensions combined multiplicatively (no if/elif chains):
        1. Merge safety — penalizes symbols that would merge with survivors
        2. Payout targeting — steers toward symbols that fit the remaining budget
        3. Variance balance — symbol rotation via population-level weights
        4. Affinity — optional caller-provided per-symbol multiplier (default 1.0)

        When boundary is None: simple variance-weighted selection (backward compat).
        When affinity_scores is None: affinity term is 1.0 for all symbols (no-op).
        """
        resolved_tier = tier or signature.required_cluster_symbols or SymbolTier.ANY
        candidates = list(symbols_in_tier(resolved_tier, self._symbol_config))

        if not candidates:
            raise ValueError(f"No symbols found for tier {resolved_tier}")

        remaining_budget = progress.remaining_payout_budget()
        remaining_steps = progress.remaining_cascade_steps()

        scored: list[tuple[Symbol, float]] = []
        for sym in candidates:
            variance_score = variance.symbol_weights.get(sym, 1.0)
            merge_score = self._merge_score(
                sym, boundary, planned_size or 0, progress, signature,
            )
            payout_score = self._payout_score(
                sym, planned_size or 0, remaining_budget, remaining_steps,
            )
            affinity = affinity_scores.get(sym, 1.0) if affinity_scores else 1.0
            scored.append((sym, variance_score * merge_score * payout_score * affinity))

        symbols, weights = zip(*scored)
        return rng.choices(list(symbols), weights=list(weights), k=1)[0]

    def _merge_score(
        self,
        symbol: Symbol,
        boundary: BoundaryAnalysis | None,
        planned_size: int,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
    ) -> float:
        """Score symbol by merge risk — 1.0 is safe, lower is riskier."""
        if boundary is None:
            return 1.0
        if symbol in boundary.safe_symbols:
            return 1.0
        if symbol in boundary.acceptable_merge_symbols:
            merged_size = planned_size + boundary.acceptable_merge_symbols[symbol]
            if self._merged_size_acceptable(merged_size, progress, signature):
                # Merge within range — moderate penalty to prefer safe symbols
                return 0.6
            # Merge exceeds range — heavy penalty (position restriction may save it)
            return 0.1
        # Not in safe or risky — fallback to full weight
        return 1.0

    def _payout_score(
        self,
        symbol: Symbol,
        planned_size: int,
        remaining_budget: RangeFloat,
        remaining_steps: Range,
    ) -> float:
        """Score symbol by how well its payout fits the remaining budget.

        Penalizes cheap symbols when payout_running_low (below floor) and
        expensive symbols when above ceiling. Neutral when on track or when
        no budget constraint exists (remaining_budget.min_val <= 0).
        """
        if remaining_budget.min_val <= 0 and remaining_budget.max_val <= 0:
            return 1.0

        try:
            sym_tier = tier_of(symbol, self._symbol_config)
        except ValueError:
            return 1.0

        # Centipayout estimate for this cluster at this size
        estimated = self._payout_eval.estimate_step_payout(planned_size, sym_tier)
        # Convert to bet multiplier for comparison with remaining_budget
        estimated_mult = estimated / self._centipayout_multiplier

        steps_left = max(1, remaining_steps.max_val)
        target_per_step = remaining_budget.min_val / steps_left if remaining_budget.min_val > 0 else 0
        ceiling_per_step = remaining_budget.max_val / steps_left if remaining_budget.max_val > 0 else float("inf")

        if target_per_step > 0 and estimated_mult < target_per_step * 0.5:
            # Too cheap — we'll fall short of minimum payout
            ratio = estimated_mult / max(target_per_step, 0.01)
            return max(0.05, ratio)

        if ceiling_per_step < float("inf") and estimated_mult > ceiling_per_step * 1.5:
            # Too expensive — we'll overshoot maximum payout
            ratio = ceiling_per_step / max(estimated_mult, 0.01)
            return max(0.05, ratio)

        # On track — slight preference for closer to target
        if target_per_step > 0:
            distance = abs(estimated_mult - target_per_step) / target_per_step
            return max(0.3, 1.0 - distance * 0.5)
        return 1.0

    # -- Position selection -------------------------------------------------

    def find_positions(
        self,
        context: BoardContext,
        size: int,
        rng: random.Random,
        variance: VarianceHints,
        symbol: Symbol | None = None,
        boundary: BoundaryAnalysis | None = None,
        merge_policy: MergePolicy = MergePolicy.AVOID,
        centroid_target: Position | None = None,
        must_be_adjacent_to: frozenset[Position] | None = None,
        avoid_positions: frozenset[Position] | None = None,
    ) -> ClusterPositionResult:
        """Find connected positions for a cluster, merge-aware.

        When boundary is provided, dispatches to the appropriate merge-policy
        handler (AVOID/ACCEPT/EXPLOIT via dict dispatch — no if/elif chains).
        Returns ClusterPositionResult with planned positions plus merge info.

        When boundary is None, falls back to simple BFS growth (backward compat).
        """
        excluded = avoid_positions or frozenset()
        available = self._compute_available_cells(context, excluded)

        # ACCEPT and EXPLOIT place fewer new cells (survivors contribute), so skip
        # the strict size check — the handlers compute reduced_size internally
        if len(available) < size and (boundary is None or merge_policy == MergePolicy.AVOID):
            raise ValueError(
                f"Only {len(available)} cells available, need {size} for cluster"
            )

        # No boundary analysis — simple BFS (backward compat for strategies not yet updated)
        if boundary is None or symbol is None:
            positions = self._bfs_grow(
                available, size, variance, rng, centroid_target, must_be_adjacent_to,
            )
            return ClusterPositionResult(
                planned_positions=positions,
                merged_survivor_positions=frozenset(),
                total_size=len(positions),
                merge_occurred=False,
            )

        # Dict dispatch for merge policies — no if/elif chain (OCP)
        handlers: dict[MergePolicy, Callable[..., ClusterPositionResult]] = {
            MergePolicy.AVOID: self._find_avoiding_merge,
            MergePolicy.ACCEPT: self._find_accepting_merge,
            MergePolicy.EXPLOIT: self._find_exploiting_merge,
        }
        handler = handlers[merge_policy]
        return handler(
            context, size, symbol, available, boundary,
            variance, rng, centroid_target, must_be_adjacent_to,
        )

    # -- Merge-policy handlers (dispatched, not if/elif'd) ------------------

    def _find_avoiding_merge(
        self,
        context: BoardContext,
        size: int,
        symbol: Symbol,
        available: set[Position],
        boundary: BoundaryAnalysis,
        variance: VarianceHints,
        rng: random.Random,
        centroid_target: Position | None,
        must_be_adjacent_to: frozenset[Position] | None,
    ) -> ClusterPositionResult:
        """Build the cluster from safe cells first, include risky only if needed.

        Classifies empty cells into safe (no merge risk for this symbol) and
        risky (adjacent to a survivor of this symbol). Prefers safe cells,
        but falls back to risky cells seeded from the safest position.
        """
        risky_cells = boundary.merge_risk.get(symbol, frozenset())
        safe_cells = available - risky_cells
        risky_available = available & risky_cells

        if len(safe_cells) >= size:
            # Enough safe cells — grow from safe only. Pre-filter to a connected
            # component so BFS doesn't start in a too-small fragment (the AVOID
            # partition can create disconnected safe-cell islands)
            filtered = self._filter_to_viable_component(safe_cells, size, rng)
            positions = self._bfs_grow(
                filtered, size, variance, rng, centroid_target, must_be_adjacent_to,
            )
            return ClusterPositionResult(
                planned_positions=positions,
                merged_survivor_positions=frozenset(),
                total_size=size,
                merge_occurred=False,
            )

        # Not enough safe cells — include risky, seed from safest position.
        # Pre-filter combined set to a viable connected component
        all_available = safe_cells | risky_available
        filtered = self._filter_to_viable_component(all_available, size, rng)
        safe_in_component = safe_cells & filtered
        seed = (
            self._safest_seed(safe_in_component, boundary, symbol, rng)
            if safe_in_component else None
        )
        positions = self._bfs_grow(
            filtered, size, variance, rng, centroid_target, must_be_adjacent_to,
            forced_seed=seed,
        )

        # Check if any risky cells were used — compute actual merge
        used_risky = positions & risky_cells
        if used_risky:
            merged = self._compute_merge(positions, symbol, boundary)
            return ClusterPositionResult(
                planned_positions=positions,
                merged_survivor_positions=merged,
                total_size=len(positions) + len(merged),
                merge_occurred=True,
            )

        return ClusterPositionResult(
            planned_positions=positions,
            merged_survivor_positions=frozenset(),
            total_size=size,
            merge_occurred=False,
        )

    def _find_accepting_merge(
        self,
        context: BoardContext,
        size: int,
        symbol: Symbol,
        available: set[Position],
        boundary: BoundaryAnalysis,
        variance: VarianceHints,
        rng: random.Random,
        centroid_target: Position | None,
        must_be_adjacent_to: frozenset[Position] | None,
    ) -> ClusterPositionResult:
        """Place fewer new cells — survivors contribute to the total cluster size.

        Seeds from contact points (cells adjacent to survivors) to guarantee the
        merge happens. Reduces planned count by the number of survivors that will join.
        """
        survivor_components = boundary.survivor_components.get(symbol, [])
        total_survivors = sum(c.size for c in survivor_components)

        # Survivors fill part of the cluster — place fewer new cells
        reduced_size = max(1, size - total_survivors)

        # Seed from a contact point to ensure merge connectivity
        contact_points: frozenset[Position] = frozenset()
        for comp in survivor_components:
            contact_points = contact_points | comp.contact_points
        seed_candidates = available & contact_points
        forced_seed = rng.choice(list(seed_candidates)) if seed_candidates else None

        if len(available) < reduced_size:
            raise ValueError(
                f"Only {len(available)} cells available, need {reduced_size} "
                f"(reduced from {size} by {total_survivors} survivors)"
            )

        positions = self._bfs_grow(
            available, reduced_size, variance, rng, centroid_target, must_be_adjacent_to,
            forced_seed=forced_seed,
        )

        merged = self._compute_merge(positions, symbol, boundary)
        return ClusterPositionResult(
            planned_positions=positions,
            merged_survivor_positions=merged,
            total_size=len(positions) + len(merged),
            merge_occurred=True,
        )

    def _find_exploiting_merge(
        self,
        context: BoardContext,
        size: int,
        symbol: Symbol,
        available: set[Position],
        boundary: BoundaryAnalysis,
        variance: VarianceHints,
        rng: random.Random,
        centroid_target: Position | None,
        must_be_adjacent_to: frozenset[Position] | None,
    ) -> ClusterPositionResult:
        """Deliberately exploit survivors to push total size past a threshold.

        Places the minimum new cells needed so that planned + survivors >= size.
        Seeds from contact points to guarantee the merge.
        """
        survivor_components = boundary.survivor_components.get(symbol, [])
        total_survivors = sum(c.size for c in survivor_components)
        needed_new = max(1, size - total_survivors)

        # Seed from contact points to guarantee merge
        contact_points: frozenset[Position] = frozenset()
        for comp in survivor_components:
            contact_points = contact_points | comp.contact_points
        seed_candidates = available & contact_points
        forced_seed = rng.choice(list(seed_candidates)) if seed_candidates else None

        if len(available) < needed_new:
            raise ValueError(
                f"Only {len(available)} cells available, need {needed_new} "
                f"to exploit {total_survivors} survivors for target size {size}"
            )

        positions = self._bfs_grow(
            available, needed_new, variance, rng, centroid_target, must_be_adjacent_to,
            forced_seed=forced_seed,
        )

        merged = self._compute_merge(positions, symbol, boundary)
        return ClusterPositionResult(
            planned_positions=positions,
            merged_survivor_positions=merged,
            total_size=len(positions) + len(merged),
            merge_occurred=True,
        )

    # -- Shared merge computation (DRY — used by all three handlers) --------

    def _compute_merge(
        self,
        planned: frozenset[Position],
        symbol: Symbol,
        boundary: BoundaryAnalysis,
    ) -> frozenset[Position]:
        """Compute which survivor cells will merge with the planned cluster.

        A survivor component merges if any of its contact points are in the
        planned position set — the cluster and component share an edge.
        """
        merged: set[Position] = set()
        for comp in boundary.survivor_components.get(symbol, []):
            if comp.contact_points & planned:
                merged |= comp.positions
        return frozenset(merged)

    def _merged_size_acceptable(
        self,
        merged_size: int,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
    ) -> bool:
        """Is the merged cluster size still within the archetype's constraints?

        Checks signature cluster size ranges and whether the merge would
        spawn an unwanted booster (pushing past a spawn threshold).
        """
        for size_range in signature.required_cluster_sizes:
            if size_range.min_val <= merged_size <= size_range.max_val:
                return True

        # Merged size outside all signature ranges — check booster side effects
        booster = self._spawn_eval.booster_for_size(merged_size)
        if booster and not progress.can_still_spawn(booster):
            return False

        return False

    def _safest_seed(
        self,
        safe_cells: set[Position],
        boundary: BoundaryAnalysis,
        symbol: Symbol,
        rng: random.Random,
    ) -> Position:
        """Pick the safe cell farthest from any survivor component of this symbol.

        When the AVOID policy must include risky cells, seeding from the
        safest position grows the cluster away from survivors.
        """
        survivor_positions: set[Position] = set()
        for comp in boundary.survivor_components.get(symbol, []):
            survivor_positions |= comp.positions

        if not survivor_positions:
            return rng.choice(list(safe_cells))

        # Maximize minimum distance to any survivor
        candidates = list(safe_cells)
        distances = [
            min(abs(p.reel - s.reel) + abs(p.row - s.row) for s in survivor_positions)
            for p in candidates
        ]
        # Weight by distance — farther cells preferred
        return rng.choices(candidates, weights=distances, k=1)[0]

    # -- Core BFS growth ----------------------------------------------------

    def _bfs_grow(
        self,
        available: set[Position],
        size: int,
        variance: VarianceHints,
        rng: random.Random,
        centroid_target: Position | None = None,
        must_be_adjacent_to: frozenset[Position] | None = None,
        forced_seed: Position | None = None,
    ) -> frozenset[Position]:
        """BFS growth with seed retry — retries with different seeds on frontier exhaustion.

        When the available-set is fragmented (e.g., AVOID merge policy splits
        safe cells into disconnected islands), the first seed may land in a
        too-small fragment. Retrying with a different seed finds the viable one.
        """
        for attempt in range(self._max_seed_retries):
            result = self._bfs_grow_once(
                available, size, variance, rng,
                centroid_target, must_be_adjacent_to,
                # Only honour forced_seed on the first attempt — subsequent
                # retries must pick a fresh seed to escape the bad fragment
                forced_seed if attempt == 0 else None,
            )
            if result is not None:
                return result
        raise ValueError(
            f"Frontier exhausted after {self._max_seed_retries} seed retries. "
            f"Board lacks enough connected space ({len(available)} available, need {size})."
        )

    def _bfs_grow_once(
        self,
        available: set[Position],
        size: int,
        variance: VarianceHints,
        rng: random.Random,
        centroid_target: Position | None = None,
        must_be_adjacent_to: frozenset[Position] | None = None,
        forced_seed: Position | None = None,
    ) -> frozenset[Position] | None:
        """Single BFS growth attempt. Returns None on frontier exhaustion.

        Core algorithm shared by all merge-policy handlers and the
        backward-compatible no-boundary path. Weighted by spatial_bias.
        """
        # A forced_seed is only honoured when it also satisfies the
        # must_be_adjacent_to contract. This prevents merge-safety pickers
        # (e.g. _safest_seed in _find_avoiding_merge) from bypassing the
        # booster-adjacency requirement and producing clusters that never
        # arm their target booster.
        use_forced = forced_seed is not None and forced_seed in available
        if use_forced and must_be_adjacent_to is not None:
            if not any(
                n in must_be_adjacent_to
                for n in orthogonal_neighbors(forced_seed, self._board_config)
            ):
                use_forced = False

        if use_forced:
            seed = forced_seed
        else:
            seed = self._select_seed(
                available, variance, rng, centroid_target, must_be_adjacent_to,
            )

        result: set[Position] = {seed}
        frontier: deque[Position] = deque()
        self._expand_frontier(seed, result, available, frontier)

        while len(result) < size:
            if not frontier:
                return None
            frontier_list = list(frontier)
            weights = [variance.spatial_bias.get(p, 1.0) for p in frontier_list]
            chosen = rng.choices(frontier_list, weights=weights, k=1)[0]
            frontier.remove(chosen)
            result.add(chosen)
            self._expand_frontier(chosen, result, available, frontier)

        return frozenset(result)

    # -- Connected-component filtering --------------------------------------

    def _find_position_components(self, positions: set[Position]) -> list[set[Position]]:
        """Connected components in a position set via orthogonal adjacency.

        Position-only BFS — no Board required. Uses the same orthogonal_neighbors
        definition as cluster detection so connectivity semantics stay consistent.
        """
        visited: set[Position] = set()
        components: list[set[Position]] = []
        for pos in positions:
            if pos in visited:
                continue
            component: set[Position] = set()
            queue = deque([pos])
            visited.add(pos)
            while queue:
                current = queue.popleft()
                component.add(current)
                for neighbor in orthogonal_neighbors(current, self._board_config):
                    if neighbor in positions and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components

    def _filter_to_viable_component(
        self,
        available: set[Position],
        min_size: int,
        rng: random.Random,
    ) -> set[Position]:
        """Select a connected component >= min_size from a potentially fragmented set.

        When the AVOID merge policy splits safe cells into disconnected islands,
        BFS can start in a too-small fragment and exhaust its frontier. This
        pre-filter picks a viable fragment so BFS always has room to grow.

        Weighted-random selection by size when multiple viable components exist —
        larger fragments are preferred (more maneuvering room) but smaller viable
        ones aren't excluded.
        """
        components = self._find_position_components(available)
        if len(components) <= 1:
            return available
        viable = [c for c in components if len(c) >= min_size]
        if not viable:
            # No fragment large enough — return largest, let seed retry handle it
            return max(components, key=len)
        if len(viable) == 1:
            return viable[0]
        # Prefer larger fragments but allow smaller viable ones
        weights = [len(c) for c in viable]
        return rng.choices(viable, weights=weights, k=1)[0]

    # -- Private helpers ---------------------------------------------------

    def _resolve_size_range(
        self,
        size_range: Range | None,
        target_booster: str | None,
        signature: ArchetypeSignature,
    ) -> Range:
        """Resolve the effective size range from constraints.

        Priority: intersect target_booster range with explicit size_range,
        fall back to signature's first cluster size spec.
        """
        booster_range: Range | None = None
        if target_booster is not None:
            raw = self._spawn_eval.size_range_for_booster(target_booster)
            if raw is not None:
                booster_range = Range(min_val=raw[0], max_val=raw[1])

        if booster_range is not None and size_range is not None:
            return Range(
                min_val=max(booster_range.min_val, size_range.min_val),
                max_val=min(booster_range.max_val, size_range.max_val),
            )
        if booster_range is not None:
            return booster_range
        if size_range is not None:
            return size_range
        if signature.required_cluster_sizes:
            return signature.required_cluster_sizes[0]
        return Range(
            min_val=self._board_config.min_cluster_size,
            max_val=self._board_config.min_cluster_size,
        )

    def _compute_available_cells(
        self,
        context: BoardContext,
        excluded: frozenset[Position],
    ) -> set[Position]:
        """Determine which cells are available for cluster placement.

        On an empty board (initial step), all positions are candidates.
        On a post-settle board, only empty cells are available.
        """
        board = context.board
        all_positions = board.all_positions()

        empty_cells = context.empty_cells
        if len(empty_cells) == len(all_positions):
            candidates = set(all_positions)
        else:
            candidates = set(empty_cells)

        candidates -= excluded
        return candidates

    def _select_seed(
        self,
        available: set[Position],
        variance: VarianceHints,
        rng: random.Random,
        centroid_target: Position | None,
        must_be_adjacent_to: frozenset[Position] | None,
    ) -> Position:
        """Choose the BFS growth starting cell.

        Applies must_be_adjacent_to filter first (if specified), then weights
        by inverse distance to centroid_target and variance.spatial_bias.
        """
        candidates = list(available)

        if must_be_adjacent_to is not None:
            candidates = [
                p for p in candidates
                if any(
                    n in must_be_adjacent_to
                    for n in orthogonal_neighbors(p, self._board_config)
                )
            ]
            if not candidates:
                raise ValueError(
                    "No available cells are adjacent to the required positions"
                )

        if centroid_target is not None:
            weights = [
                variance.spatial_bias.get(p, 1.0)
                / max(1, abs(p.reel - centroid_target.reel) + abs(p.row - centroid_target.row))
                for p in candidates
            ]
        else:
            weights = [variance.spatial_bias.get(p, 1.0) for p in candidates]

        return rng.choices(candidates, weights=weights, k=1)[0]

    def _expand_frontier(
        self,
        pos: Position,
        result: set[Position],
        available: set[Position],
        frontier: deque[Position],
    ) -> None:
        """Add orthogonal neighbors of pos to the frontier if eligible."""
        for neighbor in orthogonal_neighbors(pos, self._board_config):
            if neighbor in available and neighbor not in result and neighbor not in frontier:
                frontier.append(neighbor)

    # -- Multi-cluster placement ------------------------------------------------

    def select_cluster_count(
        self,
        required: Range,
        rng: random.Random,
    ) -> int:
        """Pick cluster count within the signature's required range.

        Returns exact value when min==max. Otherwise uniform random
        within the range — variance hints don't influence count
        because the archetype signature is the primary constraint.
        """
        if required.min_val == required.max_val:
            return required.min_val
        return rng.randint(required.min_val, required.max_val)

    def build_multi_cluster(
        self,
        context: BoardContext,
        count: int,
        size_ranges: list[Range],
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
        rng: random.Random,
        extra_occupied: frozenset[Position] | None = None,
    ) -> MultiClusterResult:
        """Place N non-overlapping, non-merging clusters on the board.

        Each cluster placed modifies the boundary context for the next —
        previously placed positions become occupied and their symbols
        become merge risks via BoundaryAnalyzer.extra_symbols. Reuses
        existing select_size(), select_symbol(), find_positions() per
        iteration (SRP — those methods unchanged).

        Args:
            count: how many clusters to place
            size_ranges: per-cluster size range (reuses last entry if fewer than count)
            extra_occupied: positions already taken (scatters, boosters, etc.)
        """
        if self._boundary_analyzer is None:
            raise RuntimeError("BoundaryAnalyzer not injected — cannot build multi-cluster")

        clusters: list[ClusterPositionResult] = []
        all_constrained: dict[Position, Symbol] = {}
        occupied = extra_occupied or frozenset()
        # Symbols placed so far — BoundaryAnalyzer uses these as extra survivors
        placed_symbols: dict[Position, Symbol] = {}
        cluster_symbols: list[Symbol] = []
        cluster_sizes: list[int] = []
        total_payout = 0.0

        for i in range(count):
            # Boundary analysis treats previously placed clusters as survivors
            effective_empty = frozenset(context.empty_cells) - occupied
            boundary = self._boundary_analyzer.analyze(
                context, effective_empty,
                extra_occupied=occupied,
                extra_symbols=placed_symbols if placed_symbols else None,
            )

            # Size range for this cluster — reuse last range if fewer than count
            size_range = size_ranges[min(i, len(size_ranges) - 1)]

            size = self.select_size(
                progress, signature, variance, rng,
                size_range=size_range,
                max_available=len(effective_empty),
            )

            symbol = self.select_symbol(
                progress, signature, variance, rng,
                boundary=boundary, planned_size=size,
            )

            # Avoid all previously placed cluster positions
            result = self.find_positions(
                context, size, rng, variance,
                symbol=symbol, boundary=boundary,
                merge_policy=MergePolicy.AVOID,
                avoid_positions=occupied,
            )

            # Accumulate — each cluster's positions become off-limits for the next
            clusters.append(result)
            for pos in result.planned_positions:
                all_constrained[pos] = symbol
                placed_symbols[pos] = symbol
            occupied = occupied | result.planned_positions
            cluster_symbols.append(symbol)
            cluster_sizes.append(result.total_size)

            # Payout estimate for budget tracking
            try:
                tier = tier_of(symbol, self._symbol_config)
            except ValueError:
                tier = SymbolTier.LOW
            payout_centipayout = self._payout_eval.estimate_step_payout(
                result.total_size, tier,
            )
            total_payout += payout_centipayout

        return MultiClusterResult(
            clusters=tuple(clusters),
            all_constrained=all_constrained,
            all_occupied=occupied,
            cluster_symbols=tuple(cluster_symbols),
            cluster_sizes=tuple(cluster_sizes),
            total_payout_estimate=total_payout,
        )
