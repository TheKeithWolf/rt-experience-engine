"""Tests for Step 4 shared services: ForwardSimulator, ClusterBuilder, SeedPlanner.

TEST-R4-001 through TEST-R4-017 per the implementation spec.
"""

from __future__ import annotations

import inspect
import random

import pytest

from ..archetypes.registry import ArchetypeSignature
from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.gravity import GravityDAG, SettleResult, settle
from ..primitives.symbols import Symbol, SymbolTier, symbols_in_tier, is_standard
from ..step_reasoner.context import BoardContext, DormantBooster
from ..step_reasoner.evaluators import PayoutEstimator, SpawnEvaluator
from ..step_reasoner.progress import ClusterRecord, ProgressTracker
from ..step_reasoner.services.forward_simulator import ForwardSimulator
from ..step_reasoner.services.cluster_builder import ClusterBuilder
from ..step_reasoner.services.seed_planner import (
    ClusterExclusion,
    SeedPlanner,
    build_cluster_exclusions,
)
from ..variance.hints import VarianceHints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signature(**overrides) -> ArchetypeSignature:
    """Build a minimal ArchetypeSignature with sensible defaults."""
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
        max_component_size=None,
        required_cascade_depth=Range(2, 5),
        cascade_steps=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.5, 5.0),
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return ArchetypeSignature(**defaults)


def _make_variance_hints(
    config: MasterConfig,
    symbol_weight_overrides: dict[Symbol, float] | None = None,
) -> VarianceHints:
    """Build VarianceHints with uniform spatial_bias and configurable symbol weights."""
    # Uniform spatial bias across all positions
    spatial_bias: dict[Position, float] = {}
    total = config.board.num_reels * config.board.num_rows
    weight = 1.0 / total
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            spatial_bias[Position(reel, row)] = weight

    # Default uniform symbol weights, with optional overrides
    symbols = symbols_in_tier(SymbolTier.ANY, config.symbols)
    symbol_weights: dict[Symbol, float] = {s: 1.0 for s in symbols}
    if symbol_weight_overrides:
        symbol_weights.update(symbol_weight_overrides)

    return VarianceHints(
        spatial_bias=spatial_bias,
        symbol_weights=symbol_weights,
        near_miss_symbol_preference=symbols,
        cluster_size_preference=tuple(range(5, 16)),
    )


def _make_board_with_cluster(
    config: MasterConfig,
    symbol: Symbol,
    positions: frozenset[Position],
    fill_symbol: Symbol = Symbol.L1,
) -> Board:
    """Build a board with a known cluster at specified positions, rest filled."""
    board = Board.empty(config.board)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            pos = Position(reel, row)
            if pos in positions:
                board.set(pos, symbol)
            else:
                board.set(pos, fill_symbol)
    return board


def _make_progress(signature: ArchetypeSignature, **overrides) -> ProgressTracker:
    """Build a ProgressTracker with sensible defaults for tests."""
    defaults = dict(signature=signature, centipayout_multiplier=100)
    defaults.update(overrides)
    return ProgressTracker(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gravity_dag(default_config: MasterConfig) -> GravityDAG:
    return GravityDAG(default_config.board, default_config.gravity)


@pytest.fixture
def forward_simulator(
    gravity_dag: GravityDAG, default_config: MasterConfig,
) -> ForwardSimulator:
    return ForwardSimulator(gravity_dag, default_config.board, default_config.gravity)


@pytest.fixture
def spawn_evaluator(default_config: MasterConfig) -> SpawnEvaluator:
    return SpawnEvaluator(default_config.boosters)


@pytest.fixture
def payout_estimator(default_config: MasterConfig) -> PayoutEstimator:
    return PayoutEstimator(
        default_config.paytable,
        default_config.centipayout,
        default_config.win_levels,
        default_config.symbols,
        default_config.grid_multiplier,
    )


@pytest.fixture
def cluster_builder(
    spawn_evaluator: SpawnEvaluator,
    payout_estimator: PayoutEstimator,
    default_config: MasterConfig,
) -> ClusterBuilder:
    from ..step_reasoner.services.boundary_analyzer import BoundaryAnalyzer
    boundary_analyzer = BoundaryAnalyzer(default_config.board, default_config.symbols)
    return ClusterBuilder(
        spawn_evaluator, payout_estimator,
        default_config.board, default_config.symbols,
        boundary_analyzer,
        multi_seed_threshold=default_config.solvers.multi_seed_threshold,
        reasoner_config=default_config.reasoner,
    )


@pytest.fixture
def seed_planner(
    forward_simulator: ForwardSimulator, default_config: MasterConfig,
) -> SeedPlanner:
    return SeedPlanner(
        forward_simulator, default_config.board, default_config.symbols,
    )


# ---------------------------------------------------------------------------
# TestForwardSimulator
# ---------------------------------------------------------------------------

class TestForwardSimulator:
    """Gravity reasoning wrapper — hypothetical boards and explosion simulation."""

    def test_build_hypothetical_returns_copy_original_unchanged(
        self, forward_simulator: ForwardSimulator, default_config: MasterConfig,
    ) -> None:
        """TEST-R4-001: build_hypothetical returns copy with placements, original unchanged."""
        board = Board.empty(default_config.board)
        board.set(Position(0, 0), Symbol.L1)

        placements = {Position(3, 3): Symbol.H3, Position(4, 4): Symbol.H2}
        result = forward_simulator.build_hypothetical(board, placements)

        # Copy has the placements
        assert result.get(Position(3, 3)) is Symbol.H3
        assert result.get(Position(4, 4)) is Symbol.H2
        # Copy preserves original data
        assert result.get(Position(0, 0)) is Symbol.L1
        # Original is unchanged
        assert board.get(Position(3, 3)) is None
        assert board.get(Position(4, 4)) is None

    def test_simulate_explosion_matches_gravity_settle(
        self,
        forward_simulator: ForwardSimulator,
        gravity_dag: GravityDAG,
        default_config: MasterConfig,
    ) -> None:
        """TEST-R4-002: simulate_explosion matches gravity.settle() exactly."""
        # Build a fully populated board with a known cluster
        cluster_positions = frozenset({
            Position(0, 6), Position(1, 6), Position(2, 6),
            Position(3, 6), Position(4, 6),
        })
        board = _make_board_with_cluster(
            default_config, Symbol.L2, cluster_positions, fill_symbol=Symbol.H1,
        )

        # Run both paths
        service_result = forward_simulator.simulate_explosion(board, cluster_positions)
        direct_result = settle(gravity_dag, board, cluster_positions, default_config.gravity)

        # Results must be identical
        assert service_result.empty_positions == direct_result.empty_positions
        assert service_result.move_steps == direct_result.move_steps
        # Boards should have the same content
        for reel in range(default_config.board.num_reels):
            for row in range(default_config.board.num_rows):
                pos = Position(reel, row)
                assert service_result.board.get(pos) == direct_result.board.get(pos)

    def test_predict_booster_landing_post_gravity(
        self,
        forward_simulator: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """TEST-R4-003: predict_booster_landing returns correct post-gravity position."""
        # Place a cluster at bottom row — centroid is at (2, 6)
        cluster_positions = frozenset({
            Position(0, 6), Position(1, 6), Position(2, 6),
            Position(3, 6), Position(4, 6),
        })
        board = _make_board_with_cluster(
            default_config, Symbol.L2, cluster_positions, fill_symbol=Symbol.H1,
        )
        centroid = Position(2, 6)

        landing = forward_simulator.predict_booster_landing(
            centroid, board, cluster_positions,
        )

        # Centroid is in the bottom row. After exploding the cluster,
        # symbols above fall down. The booster at (2,6) stays at row 6
        # because it sits at the bottom and nothing pushes it further down.
        assert landing.row == default_config.board.num_rows - 1
        # Column should be the same since straight-down gravity keeps it in place
        assert landing.reel == 2

    def test_predict_symbol_landings_mapping(
        self,
        forward_simulator: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """TEST-R4-004: predict_symbol_landings returns correct pre→post mapping."""
        # Place a cluster in the middle rows, track symbols above that will fall
        cluster_positions = frozenset({
            Position(0, 3), Position(0, 4), Position(0, 5),
            Position(1, 3), Position(1, 4),
        })
        board = _make_board_with_cluster(
            default_config, Symbol.L3, cluster_positions, fill_symbol=Symbol.H1,
        )
        # Put a distinctive symbol at top of column 0 to track
        board.set(Position(0, 0), Symbol.H3)

        # Track position (0,0) — it should fall downward after the explosion
        tracked = frozenset({Position(0, 0)})
        mapping = forward_simulator.predict_symbol_landings(
            board, tracked, cluster_positions,
        )

        # The tracked position should have moved down (gravity pulls it)
        assert Position(0, 0) in mapping
        final_pos = mapping[Position(0, 0)]
        # After exploding 3 cells in column 0 (rows 3,4,5), H3 at (0,0)
        # should fall from row 0 toward the bottom
        assert final_pos.reel == 0
        assert final_pos.row > 0  # It must have fallen


# ---------------------------------------------------------------------------
# TestClusterBuilder
# ---------------------------------------------------------------------------

class TestClusterBuilder:
    """Cluster parameter selection and connected-position finding."""

    def test_select_size_within_signature_range(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """TEST-R4-005: select_size always returns within signature range."""
        sig = _make_signature(required_cluster_sizes=(Range(5, 8),))
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        sizes = [
            cluster_builder.select_size(progress, sig, variance, rng)
            for _ in range(100)
        ]
        assert all(5 <= s <= 8 for s in sizes)

    def test_select_size_target_booster_rocket(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """TEST-R4-006: target_booster='R' constrains size to 9-10."""
        sig = _make_signature()
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        sizes = [
            cluster_builder.select_size(
                progress, sig, variance, rng, target_booster="R",
            )
            for _ in range(100)
        ]
        assert all(s in (9, 10) for s in sizes)

    def test_select_symbol_respects_tier(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """TEST-R4-007: select_symbol with tier=LOW returns only L1-L4."""
        sig = _make_signature()
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        low_symbols = set(symbols_in_tier(SymbolTier.LOW, default_config.symbols))
        high_symbols = set(symbols_in_tier(SymbolTier.HIGH, default_config.symbols))

        for _ in range(100):
            result = cluster_builder.select_symbol(
                progress, sig, variance, rng, tier=SymbolTier.LOW,
            )
            assert result in low_symbols

        for _ in range(100):
            result = cluster_builder.select_symbol(
                progress, sig, variance, rng, tier=SymbolTier.HIGH,
            )
            assert result in high_symbols

    def test_select_symbol_weighted_by_variance(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """TEST-R4-008: symbol_weights bias selection toward heavily weighted symbols."""
        sig = _make_signature(required_cluster_symbols=SymbolTier.HIGH)
        progress = _make_progress(sig)
        # Give H3 a 10x weight advantage over other high-tier symbols
        variance = _make_variance_hints(
            default_config, symbol_weight_overrides={Symbol.H3: 10.0},
        )
        rng = random.Random(42)

        results = [
            cluster_builder.select_symbol(progress, sig, variance, rng)
            for _ in range(500)
        ]
        h3_count = sum(1 for s in results if s is Symbol.H3)
        # With 10x weight among 3 symbols (H1=1, H2=1, H3=10), expected ~83%
        assert h3_count > 200, f"H3 selected only {h3_count}/500 times (expected >200)"

    def test_find_positions_connected_and_correct_size(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """TEST-R4-009: find_positions returns a connected frozenset of correct size."""
        board = Board.empty(default_config.board)
        context = BoardContext.from_board(
            board, None, [], [], default_config.board,
        )
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        result = cluster_builder.find_positions(
            context, size=7, rng=rng, variance=variance,
        )

        assert len(result.planned_positions) == 7
        assert isinstance(result.planned_positions, frozenset)
        # Verify connectivity — BFS from any position should reach all others
        _assert_connected(result.planned_positions, default_config)

    def test_find_positions_must_be_adjacent_to(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """TEST-R4-010: must_be_adjacent_to ensures adjacency to the target set."""
        board = Board.empty(default_config.board)
        context = BoardContext.from_board(
            board, None, [], [], default_config.board,
        )
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)
        target = frozenset({Position(3, 3)})

        for _ in range(20):
            result = cluster_builder.find_positions(
                context, size=5, rng=rng, variance=variance,
                must_be_adjacent_to=target,
            )
            target_neighbors = set(
                orthogonal_neighbors(Position(3, 3), default_config.board)
            )
            # At least one result position must be adjacent to the target
            assert result.planned_positions & target_neighbors, (
                f"No result position neighbors target {target}"
            )

    def test_find_positions_centroid_target_biases(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """TEST-R4-011: centroid_target biases cluster position toward that area."""
        board = Board.empty(default_config.board)
        context = BoardContext.from_board(
            board, None, [], [], default_config.board,
        )
        variance = _make_variance_hints(default_config)
        target = Position(3, 3)

        # Run many times and measure average distance to target
        total_dist_with_target = 0.0
        total_dist_without_target = 0.0
        trials = 50

        for i in range(trials):
            rng = random.Random(42 + i)
            result_with = cluster_builder.find_positions(
                context, size=5, rng=rng, variance=variance,
                centroid_target=target,
            ).planned_positions
            rng2 = random.Random(42 + i + 1000)
            result_without = cluster_builder.find_positions(
                context, size=5, rng=rng2, variance=variance,
            ).planned_positions
            # Average manhattan distance of cluster positions to target
            total_dist_with_target += sum(
                abs(p.reel - target.reel) + abs(p.row - target.row)
                for p in result_with
            ) / len(result_with)
            total_dist_without_target += sum(
                abs(p.reel - target.reel) + abs(p.row - target.row)
                for p in result_without
            ) / len(result_without)

        avg_with = total_dist_with_target / trials
        avg_without = total_dist_without_target / trials
        # Biased placement should be closer to target on average
        assert avg_with < avg_without, (
            f"Biased avg distance ({avg_with:.2f}) should be less than "
            f"unbiased ({avg_without:.2f})"
        )

    def test_find_positions_avoid_positions_excluded(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """TEST-R4-012: avoid_positions are never in the result."""
        board = Board.empty(default_config.board)
        context = BoardContext.from_board(
            board, None, [], [], default_config.board,
        )
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        # Avoid the entire center column
        avoid = frozenset(
            Position(3, row) for row in range(default_config.board.num_rows)
        )
        result = cluster_builder.find_positions(
            context, size=5, rng=rng, variance=variance,
            avoid_positions=avoid,
        )
        assert not (result.planned_positions & avoid), "Result contains avoided positions"

    def test_forced_seed_respects_must_be_adjacent_to(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """Forced seed non-adjacent to the must_be_adjacent_to target is rejected.

        Regression guard for the bug where _find_avoiding_merge passed a
        merge-safe forced_seed that was nowhere near the booster, producing
        clusters that failed to arm the target. The contract is enforced in
        _bfs_grow_once: when must_be_adjacent_to is active, a forced_seed
        that fails adjacency must fall through to _select_seed.
        """
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)
        target = frozenset({Position(3, 3)})
        available = {
            Position(r, c)
            for r in range(default_config.board.num_reels)
            for c in range(default_config.board.num_rows)
        }
        # Corner seed — guaranteed not adjacent to (3,3)
        non_adjacent_forced = Position(0, 0)

        result = cluster_builder._bfs_grow_once(
            available, size=5, variance=variance, rng=rng,
            must_be_adjacent_to=target, forced_seed=non_adjacent_forced,
        )

        assert result is not None
        # The forced seed must not have been honoured — instead _select_seed
        # ran and chose a seed adjacent to the target
        target_neighbors = set(
            orthogonal_neighbors(Position(3, 3), default_config.board)
        )
        assert result & target_neighbors, (
            "Result has no cell adjacent to must_be_adjacent_to target — "
            "adjacency contract violated"
        )

    def test_forced_seed_used_when_adjacent(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """Forced seed adjacent to the must_be_adjacent_to target is honoured.

        Regression guard: adjacency-satisfying forced seeds must not be
        rejected by the new gate. This preserves _find_accepting_merge and
        _find_exploiting_merge behaviour which depend on forced_seed working.
        """
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)
        target = frozenset({Position(3, 3)})
        # Direct orthogonal neighbour of (3,3) — satisfies adjacency
        adjacent_forced = Position(3, 2)
        available = {
            Position(r, c)
            for r in range(default_config.board.num_reels)
            for c in range(default_config.board.num_rows)
        }

        result = cluster_builder._bfs_grow_once(
            available, size=5, variance=variance, rng=rng,
            must_be_adjacent_to=target, forced_seed=adjacent_forced,
        )

        assert result is not None
        assert adjacent_forced in result, (
            "Adjacent forced_seed was not honoured — regression in "
            "merge-accepting/exploiting policies"
        )


# ---------------------------------------------------------------------------
# TestSeedPlanner
# ---------------------------------------------------------------------------

class TestSeedPlanner:
    """Strategic future-step cell placement via backward gravity reasoning."""

    def test_bridge_seeds_adjacent_to_wild_post_gravity(
        self,
        seed_planner: SeedPlanner,
        forward_simulator: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """TEST-R4-013: bridge seeds placed in refill zone target wild adjacency."""
        # Create a board, explode some cells to create a settle result
        board = _make_board_with_cluster(
            default_config, Symbol.L2,
            frozenset({Position(3, 4), Position(3, 5), Position(3, 6),
                       Position(4, 5), Position(4, 6)}),
            fill_symbol=Symbol.H1,
        )
        exploded = frozenset({
            Position(3, 4), Position(3, 5), Position(3, 6),
            Position(4, 5), Position(4, 6),
        })
        settle_result = forward_simulator.simulate_explosion(board, exploded)

        wild_pos = Position(3, 6)  # Wild lands at bottom of column 3
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        seeds = seed_planner.plan_bridge_seeds(
            wild_pos, settle_result, Symbol.L1, count=3,
            variance=variance, rng=rng,
        )

        # Seeds should be placed in refill zone (empty_positions from settle)
        for pos in seeds:
            assert pos in settle_result.empty_positions, (
                f"Seed at {pos} not in empty positions"
            )
        # Seeds should target columns adjacent to the wild position
        wild_neighbor_cols = {
            n.reel for n in orthogonal_neighbors(wild_pos, default_config.board)
        }
        for pos in seeds:
            assert pos.reel in wild_neighbor_cols, (
                f"Seed at {pos} not in a column adjacent to wild at {wild_pos}"
            )

    def test_bridge_seeds_correct_count(
        self,
        seed_planner: SeedPlanner,
        forward_simulator: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """TEST-R4-014: plan_bridge_seeds returns the requested count of seeds."""
        board = _make_board_with_cluster(
            default_config, Symbol.L2,
            frozenset({Position(3, 4), Position(3, 5), Position(3, 6),
                       Position(4, 5), Position(4, 6)}),
            fill_symbol=Symbol.H1,
        )
        exploded = frozenset({
            Position(3, 4), Position(3, 5), Position(3, 6),
            Position(4, 5), Position(4, 6),
        })
        settle_result = forward_simulator.simulate_explosion(board, exploded)

        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        seeds = seed_planner.plan_bridge_seeds(
            Position(3, 6), settle_result, Symbol.L1, count=3,
            variance=variance, rng=rng,
        )
        assert len(seeds) == 3

    def test_arm_seeds_adjacent_to_booster(
        self,
        seed_planner: SeedPlanner,
        forward_simulator: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """TEST-R4-015: arm seeds are placed in columns near the booster."""
        board = _make_board_with_cluster(
            default_config, Symbol.L2,
            frozenset({Position(2, 4), Position(2, 5), Position(2, 6),
                       Position(3, 5), Position(3, 6)}),
            fill_symbol=Symbol.H1,
        )
        exploded = frozenset({
            Position(2, 4), Position(2, 5), Position(2, 6),
            Position(3, 5), Position(3, 6),
        })
        settle_result = forward_simulator.simulate_explosion(board, exploded)

        booster_pos = Position(2, 6)  # Dormant booster at bottom of column 2
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        seeds = seed_planner.plan_arm_seeds(
            booster_pos, settle_result, variance=variance, rng=rng,
        )

        assert len(seeds) > 0, "Expected at least one arm seed"
        # All seeds should be in refill zone
        for pos in seeds:
            assert pos in settle_result.empty_positions
        # Seeds should target columns adjacent to/including the booster
        booster_neighbor_cols = {
            n.reel for n in orthogonal_neighbors(booster_pos, default_config.board)
        }
        booster_neighbor_cols.add(booster_pos.reel)
        for pos in seeds:
            assert pos.reel in booster_neighbor_cols

    def test_generic_seeds_valid_placements(
        self,
        seed_planner: SeedPlanner,
        forward_simulator: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """TEST-R4-016: generic seeds are in empty positions with standard symbols."""
        board = _make_board_with_cluster(
            default_config, Symbol.L2,
            frozenset({Position(0, 4), Position(0, 5), Position(0, 6),
                       Position(1, 5), Position(1, 6)}),
            fill_symbol=Symbol.H1,
        )
        exploded = frozenset({
            Position(0, 4), Position(0, 5), Position(0, 6),
            Position(1, 5), Position(1, 6),
        })
        settle_result = forward_simulator.simulate_explosion(board, exploded)

        sig = _make_signature()
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)
        rng = random.Random(42)

        seeds = seed_planner.plan_generic_seeds(
            settle_result, progress, sig, variance=variance, rng=rng,
        )

        assert len(seeds) > 0
        standard_symbols = set(symbols_in_tier(SymbolTier.ANY, default_config.symbols))
        for pos, sym in seeds.items():
            assert pos in settle_result.empty_positions, (
                f"Seed position {pos} not in refill zone"
            )
            assert sym in standard_symbols, (
                f"Seed symbol {sym} is not a standard symbol"
            )


# ---------------------------------------------------------------------------
# TestDependencyInjection
# ---------------------------------------------------------------------------

class TestDependencyInjection:
    """Verify all services receive dependencies via __init__, not internal construction."""

    def test_all_services_receive_deps_via_init(self) -> None:
        """TEST-R4-017: constructors accept external dependencies, not self-constructing."""
        # ForwardSimulator requires dag, board_config, gravity_config
        fs_params = set(inspect.signature(ForwardSimulator.__init__).parameters.keys())
        assert "dag" in fs_params
        assert "board_config" in fs_params
        assert "gravity_config" in fs_params

        # ClusterBuilder requires spawn_evaluator, payout_estimator, board/symbol config
        cb_params = set(inspect.signature(ClusterBuilder.__init__).parameters.keys())
        assert "spawn_evaluator" in cb_params
        assert "payout_estimator" in cb_params
        assert "board_config" in cb_params
        assert "symbol_config" in cb_params

        # SeedPlanner requires forward_simulator, board/symbol config
        sp_params = set(inspect.signature(SeedPlanner.__init__).parameters.keys())
        assert "forward_simulator" in sp_params
        assert "board_config" in sp_params
        assert "symbol_config" in sp_params


# ---------------------------------------------------------------------------
# TestSeedPlannerExclusions
# ---------------------------------------------------------------------------

class TestSeedPlannerExclusions:
    """Cluster boundary exclusion — strategic seeds must not merge into clusters."""

    def test_generic_seeds_exclude_cluster_boundary(
        self,
        seed_planner: SeedPlanner,
        forward_simulator: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """Seeds matching a cluster symbol are removed from its exclusion zone."""
        # Place an L2 cluster and explode it to get a settle result
        cluster_positions = frozenset({
            Position(3, 4), Position(3, 5), Position(3, 6),
            Position(4, 5), Position(4, 6),
        })
        board = _make_board_with_cluster(
            default_config, Symbol.L2, cluster_positions, fill_symbol=Symbol.H1,
        )
        settle_result = forward_simulator.simulate_explosion(board, cluster_positions)

        sig = _make_signature()
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        # Build exclusion for L2 at cluster_positions
        exclusions = build_cluster_exclusions(
            [(cluster_positions, Symbol.L2)], default_config.board,
        )
        zone = exclusions[0].zone

        # Run many times — no returned seed should place L2 inside the zone
        for i in range(50):
            rng = random.Random(42 + i)
            seeds = seed_planner.plan_generic_seeds(
                settle_result, progress, sig, variance, rng,
                exclusions=exclusions,
            )
            for pos, sym in seeds.items():
                if sym is Symbol.L2:
                    assert pos not in zone, (
                        f"Seed L2 at {pos} falls inside exclusion zone"
                    )

    def test_bridge_seeds_exclude_cluster_boundary(
        self,
        seed_planner: SeedPlanner,
        forward_simulator: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """Bridge seeds with a symbol matching the exclusion are filtered out."""
        cluster_positions = frozenset({
            Position(3, 4), Position(3, 5), Position(3, 6),
            Position(4, 5), Position(4, 6),
        })
        board = _make_board_with_cluster(
            default_config, Symbol.L2, cluster_positions, fill_symbol=Symbol.H1,
        )
        settle_result = forward_simulator.simulate_explosion(board, cluster_positions)

        exclusions = build_cluster_exclusions(
            [(cluster_positions, Symbol.L2)], default_config.board,
        )
        zone = exclusions[0].zone
        variance = _make_variance_hints(default_config)

        # Bridge with L2 — same symbol as exclusion
        rng = random.Random(42)
        seeds = seed_planner.plan_bridge_seeds(
            Position(3, 6), settle_result, Symbol.L2, count=3,
            variance=variance, rng=rng, exclusions=exclusions,
        )
        for pos in seeds:
            assert pos not in zone, (
                f"Bridge seed at {pos} falls inside L2 exclusion zone"
            )

    def test_exclusions_empty_is_noop(
        self,
        seed_planner: SeedPlanner,
        forward_simulator: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """Empty exclusions tuple produces identical output to no exclusions."""
        cluster_positions = frozenset({
            Position(0, 4), Position(0, 5), Position(0, 6),
            Position(1, 5), Position(1, 6),
        })
        board = _make_board_with_cluster(
            default_config, Symbol.L2, cluster_positions, fill_symbol=Symbol.H1,
        )
        settle_result = forward_simulator.simulate_explosion(board, cluster_positions)

        sig = _make_signature()
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        seeds_without = seed_planner.plan_generic_seeds(
            settle_result, progress, sig, variance, rng1,
        )
        seeds_with_empty = seed_planner.plan_generic_seeds(
            settle_result, progress, sig, variance, rng2,
            exclusions=(),
        )
        assert seeds_without == seeds_with_empty

    def test_filter_excluded_preserves_non_matching_symbols(
        self, default_config: MasterConfig,
    ) -> None:
        """Seeds with a different symbol than the exclusion survive even at excluded positions."""
        cluster_positions = frozenset({
            Position(3, 3), Position(3, 4), Position(3, 5),
            Position(4, 4), Position(4, 5),
        })
        exclusions = build_cluster_exclusions(
            [(cluster_positions, Symbol.L2)], default_config.board,
        )
        zone = exclusions[0].zone

        # Place H1 seeds at positions inside the L2 exclusion zone — they should survive
        seeds_in_zone = {pos: Symbol.H1 for pos in list(zone)[:3]}
        result = SeedPlanner._filter_excluded(seeds_in_zone, exclusions)
        assert result == seeds_in_zone, (
            "Non-matching symbols were incorrectly filtered"
        )


# ---------------------------------------------------------------------------
# Helpers for connectivity verification
# ---------------------------------------------------------------------------

def _assert_connected(positions: frozenset[Position], config: MasterConfig) -> None:
    """Assert that all positions form a single connected component via orthogonal adjacency."""
    if len(positions) <= 1:
        return

    visited: set[Position] = set()
    stack = [next(iter(positions))]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in orthogonal_neighbors(current, config.board):
            if neighbor in positions and neighbor not in visited:
                stack.append(neighbor)

    assert visited == positions, (
        f"Positions are not fully connected. "
        f"Reached {len(visited)}/{len(positions)} from BFS."
    )
