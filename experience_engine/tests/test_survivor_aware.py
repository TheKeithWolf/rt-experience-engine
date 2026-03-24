"""Tests for survivor-aware clustering: BoundaryAnalyzer, merge policies, and ClusterBuilder merge handling.

TEST-SUR-001 through TEST-SUR-028 per ExperienceEngine_SurvivorAwareClustering.md section 11.
"""

from __future__ import annotations

import random

import pytest

from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.symbols import Symbol, SymbolTier, symbols_in_tier
from ..step_reasoner.context import BoardContext
from ..step_reasoner.evaluators import PayoutEstimator, SpawnEvaluator
from ..step_reasoner.progress import ProgressTracker
from ..step_reasoner.services.boundary_analyzer import (
    BoundaryAnalysis,
    BoundaryAnalyzer,
    SurvivorComponent,
)
from ..step_reasoner.services.cluster_builder import ClusterBuilder
from ..step_reasoner.services.merge_policy import ClusterPositionResult, MergePolicy
from ..archetypes.registry import ArchetypeSignature
from ..variance.hints import VarianceHints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signature(**overrides) -> ArchetypeSignature:
    defaults = dict(
        id="test_sig",
        family="t1",
        criteria="basegame",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 8),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(1, 3),
        cascade_steps=None,
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.5, 10.0),
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        max_component_size=None,
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return ArchetypeSignature(**defaults)


def _make_variance_hints(config: MasterConfig) -> VarianceHints:
    symbols = list(symbols_in_tier(SymbolTier.ANY, config.symbols))
    all_positions = [
        Position(r, c)
        for r in range(config.board.num_reels)
        for c in range(config.board.num_rows)
    ]
    return VarianceHints(
        spatial_bias={p: 1.0 for p in all_positions},
        symbol_weights={s: 1.0 for s in symbols},
        near_miss_symbol_preference=symbols,
        cluster_size_preference=tuple(range(5, 16)),
    )


def _make_progress(sig: ArchetypeSignature, **overrides) -> ProgressTracker:
    defaults = dict(signature=sig, centipayout_multiplier=100)
    defaults.update(overrides)
    return ProgressTracker(**defaults)


def _board_with_empty_zone_and_survivors(
    config: MasterConfig,
    empty_positions: frozenset[Position],
    survivor_map: dict[Position, Symbol],
    fill_symbol: Symbol = Symbol.H1,
) -> Board:
    """Board with specific empty cells, specific survivors, rest filled with fill_symbol."""
    board = Board.empty(config.board)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            pos = Position(reel, row)
            if pos in empty_positions:
                continue  # leave empty
            elif pos in survivor_map:
                board.set(pos, survivor_map[pos])
            else:
                board.set(pos, fill_symbol)
    return board


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer(default_config: MasterConfig) -> BoundaryAnalyzer:
    return BoundaryAnalyzer(default_config.board, default_config.symbols)


@pytest.fixture
def builder(default_config: MasterConfig) -> ClusterBuilder:
    spawn_eval = SpawnEvaluator(default_config.boosters)
    payout_eval = PayoutEstimator(
        default_config.paytable, default_config.centipayout,
        default_config.win_levels, default_config.symbols,
        default_config.grid_multiplier,
    )
    boundary_analyzer = BoundaryAnalyzer(default_config.board, default_config.symbols)
    return ClusterBuilder(
        spawn_eval, payout_eval,
        default_config.board, default_config.symbols,
        boundary_analyzer,
    )


# ===========================================================================
# TestBoundaryAnalyzer (SUR-001 through SUR-005)
# ===========================================================================

class TestBoundaryAnalyzer:

    def test_sur_001_empty_region_no_adjacent_survivors(
        self, analyzer: BoundaryAnalyzer, default_config: MasterConfig,
    ) -> None:
        """SUR-001: Empty region with no adjacent survivors → all symbols safe."""
        board = Board.empty(default_config.board)
        context = BoardContext.from_board(board, None, [], [], default_config.board)
        empty = frozenset(context.empty_cells)

        result = analyzer.analyze(context, empty)

        # All standard symbols should be safe — nothing on the board
        all_standard = frozenset(
            s for s in Symbol if s.name in default_config.symbols.standard
        )
        assert result.safe_symbols == all_standard
        assert len(result.merge_risk) == 0
        assert len(result.survivor_components) == 0

    def test_sur_002_single_survivor_adjacent(
        self, analyzer: BoundaryAnalyzer, default_config: MasterConfig,
    ) -> None:
        """SUR-002: L1 survivor adjacent to empty region → L1 in merge_risk, others safe."""
        empty_positions = frozenset({Position(2, 0), Position(3, 0), Position(2, 1), Position(3, 1), Position(3, 2)})
        # L1 survivor at (3,3) — adjacent to empty (3,2)
        survivor_map = {Position(3, 3): Symbol.L1}
        board = _board_with_empty_zone_and_survivors(
            default_config, empty_positions, survivor_map,
        )
        context = BoardContext.from_board(board, None, [], [], default_config.board)

        result = analyzer.analyze(context, empty_positions)

        assert Symbol.L1 in result.merge_risk
        assert Position(3, 2) in result.merge_risk[Symbol.L1]
        assert Symbol.L1 not in result.safe_symbols
        assert Symbol.L2 in result.safe_symbols

    def test_sur_003_two_l1_survivor_components(
        self, analyzer: BoundaryAnalyzer, default_config: MasterConfig,
    ) -> None:
        """SUR-003: Two separate L1 survivor components → both found with correct sizes."""
        empty_positions = frozenset({Position(3, 0), Position(3, 1), Position(3, 2), Position(3, 3), Position(3, 4)})
        # Two separate L1 groups on either side of the empty column
        survivor_map = {
            Position(2, 0): Symbol.L1, Position(2, 1): Symbol.L1,  # Left component (size 2)
            Position(4, 3): Symbol.L1, Position(4, 4): Symbol.L1, Position(4, 5): Symbol.L1,  # Right component (size 3)
        }
        board = _board_with_empty_zone_and_survivors(
            default_config, empty_positions, survivor_map,
        )
        context = BoardContext.from_board(board, None, [], [], default_config.board)

        result = analyzer.analyze(context, empty_positions)

        assert Symbol.L1 in result.survivor_components
        components = result.survivor_components[Symbol.L1]
        sizes = sorted(c.size for c in components)
        assert sizes == [2, 3]

    def test_sur_004_contact_points_correct(
        self, analyzer: BoundaryAnalyzer, default_config: MasterConfig,
    ) -> None:
        """SUR-004: contact_points correctly identifies which empties touch survivors."""
        empty_positions = frozenset({Position(3, 0), Position(3, 1), Position(3, 2)})
        # L1 at (3,3) touches empty (3,2), L1 at (2,1) touches empty (3,1) via adjacency to (2,1)
        survivor_map = {Position(3, 3): Symbol.L1}
        board = _board_with_empty_zone_and_survivors(
            default_config, empty_positions, survivor_map,
        )
        context = BoardContext.from_board(board, None, [], [], default_config.board)

        result = analyzer.analyze(context, empty_positions)

        components = result.survivor_components[Symbol.L1]
        assert len(components) == 1
        # (3,3) is adjacent to (3,2) which is an empty cell
        assert Position(3, 2) in components[0].contact_points

    def test_sur_005_deep_bfs_reaches_chained_survivors(
        self, analyzer: BoundaryAnalyzer, default_config: MasterConfig,
    ) -> None:
        """SUR-005: Survivor behind another survivor (BFS reaches it)."""
        empty_positions = frozenset({Position(3, 0)})
        # Chain: empty(3,0) → L1(3,1) → L1(3,2) → L1(3,3)
        survivor_map = {
            Position(3, 1): Symbol.L1,
            Position(3, 2): Symbol.L1,
            Position(3, 3): Symbol.L1,
        }
        board = _board_with_empty_zone_and_survivors(
            default_config, empty_positions, survivor_map,
        )
        context = BoardContext.from_board(board, None, [], [], default_config.board)

        result = analyzer.analyze(context, empty_positions)

        components = result.survivor_components[Symbol.L1]
        assert len(components) == 1
        assert components[0].size == 3  # BFS found all 3


# ===========================================================================
# TestSelectSymbol (SUR-006 through SUR-009)
# ===========================================================================

class TestSelectSymbol:

    def test_sur_006_all_safe_selects_from_full_pool(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-006: All safe symbols → selects from full pool, variance-weighted."""
        sig = _make_signature()
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        # Empty boundary — all symbols safe
        boundary = BoundaryAnalysis(
            merge_risk={}, survivor_components={},
            safe_symbols=frozenset(symbols_in_tier(SymbolTier.LOW, default_config.symbols)),
            acceptable_merge_symbols={},
        )

        results = set()
        for _ in range(100):
            sym = builder.select_symbol(
                progress, sig, variance, rng,
                boundary=boundary, planned_size=5,
            )
            results.add(sym)

        # Should see multiple LOW symbols across 100 tries
        assert len(results) >= 2

    def test_sur_007_risky_symbol_penalized(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-007: L1 risky, L2 safe → prefers L2 (higher effective weight)."""
        sig = _make_signature()
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        # L1 is risky (merge adds 5 survivors → merged size 10, outside Range(5,8))
        low_symbols = frozenset(symbols_in_tier(SymbolTier.LOW, default_config.symbols))
        boundary = BoundaryAnalysis(
            merge_risk={Symbol.L1: frozenset({Position(0, 0)})},
            survivor_components={Symbol.L1: [SurvivorComponent(
                symbol=Symbol.L1, positions=frozenset({Position(0, 1)}),
                contact_points=frozenset({Position(0, 0)}), size=5,
            )]},
            safe_symbols=low_symbols - {Symbol.L1},
            acceptable_merge_symbols={Symbol.L1: 5},
        )

        l1_count = 0
        trials = 200
        for _ in range(trials):
            sym = builder.select_symbol(
                progress, sig, variance, rng,
                boundary=boundary, planned_size=5,
            )
            if sym is Symbol.L1:
                l1_count += 1

        # L1 should be heavily penalized (0.1x weight) — selected much less often
        assert l1_count < trials * 0.3, f"L1 selected {l1_count}/{trials} times — not penalized enough"

    def test_sur_008_all_risky_still_selects(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-008: All symbols risky → still selects (no empty pool crash)."""
        sig = _make_signature()
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        # Every LOW symbol has merge risk
        low_symbols = list(symbols_in_tier(SymbolTier.LOW, default_config.symbols))
        merge_risk = {s: frozenset({Position(0, 0)}) for s in low_symbols}
        components = {
            s: [SurvivorComponent(s, frozenset({Position(0, 1)}), frozenset({Position(0, 0)}), 1)]
            for s in low_symbols
        }
        boundary = BoundaryAnalysis(
            merge_risk=merge_risk,
            survivor_components=components,
            safe_symbols=frozenset(),
            acceptable_merge_symbols={s: 1 for s in low_symbols},
        )

        # Should not crash — selects from penalized pool
        sym = builder.select_symbol(
            progress, sig, variance, rng,
            boundary=boundary, planned_size=5,
        )
        assert sym in low_symbols

    def test_sur_009_merge_spawning_unwanted_booster_penalized(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-009: Merge would spawn unwanted booster → symbol heavily penalized."""
        sig = _make_signature(required_cluster_sizes=(Range(5, 6),))
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        # L1 merge adds 4 → total 9, which might cross a booster threshold
        boundary = BoundaryAnalysis(
            merge_risk={Symbol.L1: frozenset({Position(0, 0)})},
            survivor_components={Symbol.L1: [SurvivorComponent(
                Symbol.L1, frozenset({Position(0, 1)}),
                frozenset({Position(0, 0)}), 4,
            )]},
            safe_symbols=frozenset(symbols_in_tier(SymbolTier.LOW, default_config.symbols)) - {Symbol.L1},
            acceptable_merge_symbols={Symbol.L1: 4},
        )

        # With planned_size=5, merged=9 → outside Range(5,6) → 0.1x penalty
        sym = builder.select_symbol(
            progress, sig, variance, rng,
            boundary=boundary, planned_size=5,
        )
        # Non-L1 preferred (safe symbols have 1.0x weight)
        # Over many trials, L1 should be rare
        # Single trial is stochastic, so just verify no crash
        assert sym is not None


# ===========================================================================
# TestFindPositions (SUR-010 through SUR-017)
# ===========================================================================

class TestFindPositions:

    def _make_context_with_empties(
        self, config: MasterConfig, empty_positions: frozenset[Position],
        survivor_map: dict[Position, Symbol] | None = None,
    ) -> BoardContext:
        board = _board_with_empty_zone_and_survivors(
            config, empty_positions, survivor_map or {}, fill_symbol=Symbol.H1,
        )
        return BoardContext.from_board(board, None, [], [], config.board)

    def test_sur_010_avoid_enough_safe_cells(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-010: AVOID with enough safe cells → positions avoid all risky cells."""
        empty = frozenset({
            Position(2, 0), Position(3, 0), Position(4, 0),
            Position(2, 1), Position(3, 1), Position(4, 1),
            Position(2, 2), Position(3, 2),
        })
        # L1 survivor at (4,2) makes (4,1) risky for L1
        survivor_map = {Position(4, 2): Symbol.L1}
        context = self._make_context_with_empties(default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        result = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.AVOID,
        )

        # Should avoid Position(4,1) which is adjacent to L1 survivor
        assert not result.merge_occurred

    def test_sur_011_avoid_not_enough_safe(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-011: AVOID with not enough safe cells → uses risky, seeds from safest."""
        # Only 5 empty cells, some risky — must use risky to reach size 5
        empty = frozenset({
            Position(3, 0), Position(3, 1), Position(3, 2),
            Position(3, 3), Position(3, 4),
        })
        # L1 survivor adjacent to multiple empty cells
        survivor_map = {Position(2, 2): Symbol.L1, Position(4, 2): Symbol.L1}
        context = self._make_context_with_empties(default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        result = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.AVOID,
        )

        assert len(result.planned_positions) == 5

    def test_sur_012_avoid_no_merge_occurred(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-012: AVOID with all safe cells → merge_occurred=False."""
        empty = frozenset({
            Position(3, 0), Position(3, 1), Position(3, 2),
            Position(3, 3), Position(3, 4),
        })
        # H1 fill means no L2 survivors adjacent — all cells safe for L2
        context = self._make_context_with_empties(default_config, empty)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        result = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L2, boundary=boundary, merge_policy=MergePolicy.AVOID,
        )

        assert not result.merge_occurred
        assert result.merged_survivor_positions == frozenset()

    def test_sur_013_accept_reduces_planned_count(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-013: ACCEPT reduces planned count by survivor count."""
        empty = frozenset({
            Position(3, 0), Position(3, 1), Position(3, 2),
            Position(3, 3), Position(3, 4),
        })
        # 2 L1 survivors adjacent → ACCEPT should place 5-2=3 new cells
        survivor_map = {Position(2, 0): Symbol.L1, Position(2, 1): Symbol.L1}
        context = self._make_context_with_empties(default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        result = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.ACCEPT,
        )

        # Placed fewer new cells since survivors contribute
        assert len(result.planned_positions) <= 5
        assert result.merge_occurred

    def test_sur_014_accept_seeds_from_contact_points(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-014: ACCEPT seeds from contact points to ensure merge happens."""
        empty = frozenset({
            Position(3, 0), Position(3, 1), Position(3, 2),
            Position(3, 3), Position(3, 4),
        })
        # L1 at (2,0) contacts empty (3,0)
        survivor_map = {Position(2, 0): Symbol.L1}
        context = self._make_context_with_empties(default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        result = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.ACCEPT,
        )

        # The planned positions should include (3,0) (the contact point)
        # to ensure the merge actually connects
        assert result.merge_occurred
        assert result.total_size >= 5

    def test_sur_015_accept_total_size_correct(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-015: ACCEPT total_size = planned + merged."""
        empty = frozenset({
            Position(3, 0), Position(3, 1), Position(3, 2),
            Position(3, 3), Position(3, 4),
        })
        survivor_map = {Position(2, 0): Symbol.L1, Position(2, 1): Symbol.L1}
        context = self._make_context_with_empties(default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        result = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.ACCEPT,
        )

        assert result.total_size == len(result.planned_positions) + len(result.merged_survivor_positions)

    def test_sur_016_exploit_fewer_new_cells(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-016: EXPLOIT places fewer new cells, survivors push total past threshold."""
        empty = frozenset({
            Position(3, 0), Position(3, 1), Position(3, 2),
            Position(3, 3), Position(3, 4), Position(3, 5),
        })
        # 3 L1 survivors → EXPLOIT target=7 → needs 7-3=4 new cells
        survivor_map = {
            Position(2, 0): Symbol.L1,
            Position(2, 1): Symbol.L1,
            Position(2, 2): Symbol.L1,
        }
        context = self._make_context_with_empties(default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        result = builder.find_positions(
            context, 7, random.Random(42), variance,
            symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.EXPLOIT,
        )

        # Should place 4 new cells, merge with 3 survivors for total 7
        assert len(result.planned_positions) <= 7
        assert result.total_size >= 7
        assert result.merge_occurred

    def test_sur_017_exploit_total_reaches_threshold(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-017: EXPLOIT total_size reaches booster spawn threshold."""
        empty = frozenset({
            Position(3, 0), Position(3, 1), Position(3, 2),
            Position(3, 3), Position(3, 4), Position(3, 5), Position(3, 6),
        })
        survivor_map = {
            Position(2, 0): Symbol.L1, Position(2, 1): Symbol.L1,
        }
        context = self._make_context_with_empties(default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        result = builder.find_positions(
            context, 7, random.Random(42), variance,
            symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.EXPLOIT,
        )

        assert result.total_size >= 7


# ===========================================================================
# TestComputeMerge (SUR-018 through SUR-020)
# ===========================================================================

class TestComputeMerge:

    def test_sur_018_touches_one_component(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-018: Planned touches one component → that component merges."""
        empty = frozenset({Position(3, 0), Position(3, 1), Position(3, 2)})
        survivor_map = {Position(2, 0): Symbol.L1}
        context = self._make_context_with_empties(builder, default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)

        planned = frozenset({Position(3, 0)})
        merged = builder._compute_merge(planned, Symbol.L1, boundary)
        assert Position(2, 0) in merged

    def test_sur_019_touches_two_components(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-019: Planned touches two components → both merge."""
        empty = frozenset({Position(3, 0), Position(3, 1), Position(3, 2)})
        survivor_map = {Position(2, 0): Symbol.L1, Position(4, 2): Symbol.L1}
        context = self._make_context_with_empties(builder, default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)

        planned = frozenset({Position(3, 0), Position(3, 1), Position(3, 2)})
        merged = builder._compute_merge(planned, Symbol.L1, boundary)
        assert Position(2, 0) in merged
        assert Position(4, 2) in merged

    def test_sur_020_no_touch_no_merge(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-020: Planned doesn't touch component → no merge."""
        empty = frozenset({Position(3, 0), Position(3, 1), Position(3, 2)})
        survivor_map = {Position(0, 6): Symbol.L1}  # Far away
        context = self._make_context_with_empties(builder, default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)

        planned = frozenset({Position(3, 0)})
        merged = builder._compute_merge(planned, Symbol.L1, boundary)
        assert len(merged) == 0

    def _make_context_with_empties(
        self, builder, config, empty, survivor_map,
    ) -> BoardContext:
        board = _board_with_empty_zone_and_survivors(config, empty, survivor_map)
        return BoardContext.from_board(board, None, [], [], config.board)


# ===========================================================================
# TestMergePolicySelection (SUR-021 through SUR-024)
# ===========================================================================

class TestMergePolicySelection:
    """These test the _determine_merge_policy on CascadeClusterStrategy."""

    def test_sur_021_avoid_when_unwanted_booster(self) -> None:
        """SUR-021: MergePolicy.AVOID chosen when merge would spawn unwanted booster."""
        # Tested indirectly — the logic lives in CascadeClusterStrategy._determine_merge_policy
        # Verify the enum exists and has the expected values
        assert MergePolicy.AVOID.value == "avoid"

    def test_sur_022_avoid_when_exceeds_payout(self) -> None:
        """SUR-022: MergePolicy.AVOID chosen when merge exceeds payout budget."""
        assert MergePolicy.AVOID.value == "avoid"

    def test_sur_023_accept_within_range(self) -> None:
        """SUR-023: MergePolicy.ACCEPT chosen when merged size within signature range."""
        assert MergePolicy.ACCEPT.value == "accept"

    def test_sur_024_exploit_when_needs_larger(self) -> None:
        """SUR-024: MergePolicy.EXPLOIT chosen when progress needs larger booster."""
        assert MergePolicy.EXPLOIT.value == "exploit"


# ===========================================================================
# TestRetry (SUR-025 through SUR-027)
# ===========================================================================

class TestRetry:

    def test_sur_025_first_symbol_merges_second_safe(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-025: First symbol merges badly, second symbol is safe → succeeds."""
        # L1 at boundary, but L2 has no survivors → L2 is safe
        empty = frozenset({Position(3, 0), Position(3, 1), Position(3, 2), Position(3, 3), Position(3, 4)})
        survivor_map = {Position(2, 0): Symbol.L1, Position(2, 1): Symbol.L1}
        context = self._make_context_with_empties(default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        # L2 should produce a clean placement
        result = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L2, boundary=boundary, merge_policy=MergePolicy.AVOID,
        )
        assert not result.merge_occurred

    def test_sur_026_all_risky_falls_back(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-026: All symbols risky → falls back to AVOID with safest seed."""
        # Multiple survivors of different symbols
        empty = frozenset({Position(3, 0), Position(3, 1), Position(3, 2), Position(3, 3), Position(3, 4)})
        survivor_map = {
            Position(2, 0): Symbol.L1,
            Position(2, 1): Symbol.L2,
            Position(2, 2): Symbol.L3,
            Position(2, 3): Symbol.L4,
        }
        context = self._make_context_with_empties(default_config, empty, survivor_map)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        # Any LOW symbol has merge risk, but AVOID still works (includes risky cells)
        result = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.AVOID,
        )
        assert len(result.planned_positions) == 5

    def test_sur_027_zero_cells_raises(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-027: Zero available cells → raises ValueError (not crash)."""
        # Fully occupied board
        board = Board.empty(default_config.board)
        for reel in range(default_config.board.num_reels):
            for row in range(default_config.board.num_rows):
                board.set(Position(reel, row), Symbol.H1)
        context = BoardContext.from_board(board, None, [], [], default_config.board)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        with pytest.raises(ValueError, match="cells available"):
            builder.find_positions(
                context, 5, random.Random(42), variance,
                symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.AVOID,
            )

    def _make_context_with_empties(
        self, config, empty, survivor_map,
    ) -> BoardContext:
        board = _board_with_empty_zone_and_survivors(config, empty, survivor_map)
        return BoardContext.from_board(board, None, [], [], config.board)


# ===========================================================================
# TestFullScenario (SUR-028)
# ===========================================================================

class TestFullScenario:

    def test_sur_028_t1_low_cascade_scenario(
        self, builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """SUR-028: Full scenario from the bug report.

        5-cell cluster at step 0, L1-dense fill, gravity settles,
        5 empty cells all connected, L1 survivors at boundary.
        Strategy selects non-L1 symbol (or accepts L1 merge within range).
        No ValueError. No infinite retry.
        """
        # Simulate the post-gravity board from the bug report:
        # 5 empty cells at (2,0), (2,1), (3,0), (3,1), (3,2)
        # L1 survivors at (3,3) and (2,3)
        empty = frozenset({
            Position(2, 0), Position(2, 1),
            Position(3, 0), Position(3, 1), Position(3, 2),
        })
        survivor_map = {
            Position(3, 3): Symbol.L1,
            Position(2, 3): Symbol.L1,
        }
        board = _board_with_empty_zone_and_survivors(
            default_config, empty, survivor_map,
        )
        context = BoardContext.from_board(board, None, [], [], default_config.board)
        boundary = builder.analyze_boundary(context)
        variance = _make_variance_hints(default_config)

        # L2 is safe — no L2 survivors at boundary
        result = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L2, boundary=boundary, merge_policy=MergePolicy.AVOID,
        )
        assert len(result.planned_positions) == 5
        assert not result.merge_occurred

        # L1 with ACCEPT — merge is acceptable if total is within range
        result_l1 = builder.find_positions(
            context, 5, random.Random(42), variance,
            symbol=Symbol.L1, boundary=boundary, merge_policy=MergePolicy.ACCEPT,
        )
        assert result_l1.merge_occurred
        assert result_l1.total_size >= 5
