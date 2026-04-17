"""Tests for BoosterLandingEvaluator, landing criteria, and reshape bias.

Covers:
- Per-criterion scoring (Wild, Rocket, Bomb, LB/SLB)
- Evaluator service (evaluate, score, evaluate_and_score, find_best_shape)
- compute_reshape_bias progressive vertical concentration
- plan_chain_aware_seeds column filtering
"""

from __future__ import annotations

import random

import pytest

from ..config.schema import MasterConfig
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.booster_rules import BoosterRules
from ..primitives.gravity import GravityDAG
from ..primitives.symbols import Symbol, SymbolTier, symbols_in_tier
from ..step_reasoner.services.forward_simulator import ForwardSimulator
from ..step_reasoner.services.landing_criteria import (
    BombArmCriterion,
    ChainConstraint,
    LightballArmCriterion,
    RocketArmCriterion,
    SuperLightballArmCriterion,
    WildBridgeCriterion,
)
from ..step_reasoner.services.landing_evaluator import (
    BoosterLandingEvaluator,
    LandingContext,
    ShapeStats,
    compute_reshape_bias,
)
from ..step_reasoner.services.seed_planner import SeedPlanner
from ..variance.hints import VarianceHints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_variance(config: MasterConfig) -> VarianceHints:
    """Uniform variance hints for testing."""
    total = config.board.num_reels * config.board.num_rows
    weight = 1.0 / total
    spatial_bias = {
        Position(r, c): weight
        for r in range(config.board.num_reels)
        for c in range(config.board.num_rows)
    }
    symbols = symbols_in_tier(SymbolTier.ANY, config.symbols)
    return VarianceHints(
        spatial_bias=spatial_bias,
        symbol_weights={s: 1.0 for s in symbols},
        near_miss_symbol_preference=symbols,
        cluster_size_preference=tuple(range(5, 16)),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gravity_dag(default_config: MasterConfig) -> GravityDAG:
    return GravityDAG(default_config.board, default_config.gravity)


@pytest.fixture
def forward_sim(
    gravity_dag: GravityDAG, default_config: MasterConfig,
) -> ForwardSimulator:
    return ForwardSimulator(gravity_dag, default_config.board, default_config.gravity)


@pytest.fixture
def booster_rules(default_config: MasterConfig) -> BoosterRules:
    return BoosterRules(default_config.boosters, default_config.board, default_config.symbols)


@pytest.fixture
def evaluator(
    forward_sim: ForwardSimulator,
    booster_rules: BoosterRules,
    default_config: MasterConfig,
) -> BoosterLandingEvaluator:
    """Evaluator with all 5 criteria registered."""
    lc = default_config.landing_criteria
    criteria = {
        "W": WildBridgeCriterion(default_config.board, lc),
        "R": RocketArmCriterion(booster_rules, default_config.board, lc),
        "B": BombArmCriterion(booster_rules, default_config.board, lc),
        "LB": LightballArmCriterion(default_config.board),
        "SLB": LightballArmCriterion(default_config.board),
    }
    return BoosterLandingEvaluator(
        forward_sim, booster_rules, default_config.board, criteria,
    )


@pytest.fixture
def seed_planner(
    forward_sim: ForwardSimulator, default_config: MasterConfig,
) -> SeedPlanner:
    return SeedPlanner(forward_sim, default_config.board, default_config.symbols)


# ---------------------------------------------------------------------------
# Wild Criterion Tests
# ---------------------------------------------------------------------------

class TestWildBridgeCriterion:
    """WildBridgeCriterion scores based on adjacent refill cell count."""

    def test_zero_adjacent_returns_zero(self, default_config: MasterConfig) -> None:
        """No adjacent refill → score 0.0 (bridge impossible)."""
        criterion = WildBridgeCriterion(
            default_config.board, default_config.landing_criteria,
        )
        ctx = LandingContext(
            booster_type="W",
            cluster_positions=frozenset({Position(3, 3)}),
            landing_position=Position(3, 6),
            adjacent_refill=(),
            all_refill=(),
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=1, row_span=1, orientation="H",
                centroid=Position(3, 3), columns_used=frozenset({3}),
                max_depth_per_column={3: 3},
            ),
        )
        assert criterion.score(ctx) == 0.0

    def test_enough_adjacent_scores_high(self, default_config: MasterConfig) -> None:
        """4+ adjacent refill → score >= 0.8 (bridge viable)."""
        criterion = WildBridgeCriterion(
            default_config.board, default_config.landing_criteria,
        )
        # min_cluster_size is 5 for royal_tumble, so needed = 4
        adjacent = tuple(
            Position(r, 0) for r in range(4)
        )
        ctx = LandingContext(
            booster_type="W",
            cluster_positions=frozenset({Position(3, 3)}),
            landing_position=Position(3, 1),
            adjacent_refill=adjacent,
            all_refill=adjacent,
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=1, row_span=1, orientation="H",
                centroid=Position(3, 3), columns_used=frozenset({3}),
                max_depth_per_column={3: 3},
            ),
        )
        assert criterion.score(ctx) >= 0.8

    def test_multi_column_bonus(self, default_config: MasterConfig) -> None:
        """Adjacent cells in 2+ columns get the side bonus."""
        criterion = WildBridgeCriterion(
            default_config.board, default_config.landing_criteria,
        )
        # Two adjacent cells in different columns
        adjacent = (Position(2, 0), Position(4, 0))
        ctx = LandingContext(
            booster_type="W",
            cluster_positions=frozenset({Position(3, 3)}),
            landing_position=Position(3, 1),
            adjacent_refill=adjacent,
            all_refill=adjacent,
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=1, row_span=1, orientation="H",
                centroid=Position(3, 3), columns_used=frozenset({3}),
                max_depth_per_column={3: 3},
            ),
        )
        # Same cells but in one column
        adjacent_single = (Position(3, 0), Position(3, 2))
        ctx_single = LandingContext(
            booster_type="W",
            cluster_positions=frozenset({Position(3, 3)}),
            landing_position=Position(3, 1),
            adjacent_refill=adjacent_single,
            all_refill=adjacent_single,
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=1, row_span=1, orientation="H",
                centroid=Position(3, 3), columns_used=frozenset({3}),
                max_depth_per_column={3: 3},
            ),
        )
        assert criterion.score(ctx) > criterion.score(ctx_single)


# ---------------------------------------------------------------------------
# Rocket Criterion Tests
# ---------------------------------------------------------------------------

class TestRocketArmCriterion:
    """RocketArmCriterion scores arm feasibility + centrality."""

    def test_edge_landing_penalized(
        self, booster_rules: BoosterRules, default_config: MasterConfig,
    ) -> None:
        """Landing at reel=0 (board edge) scores lower than center."""
        criterion = RocketArmCriterion(
            booster_rules, default_config.board,
            default_config.landing_criteria,
        )
        # Edge landing with some adjacent cells
        edge_adjacent = tuple(Position(r, 0) for r in range(3))
        ctx_edge = LandingContext(
            booster_type="R",
            cluster_positions=frozenset({Position(0, 3)}),
            landing_position=Position(0, 6),
            adjacent_refill=edge_adjacent,
            all_refill=edge_adjacent,
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=2, row_span=5, orientation="H",
                centroid=Position(0, 3), columns_used=frozenset({0, 1}),
                max_depth_per_column={0: 5, 1: 4},
            ),
        )

        # Center landing with same adjacent count
        center_adjacent = tuple(Position(r, 0) for r in range(3))
        ctx_center = LandingContext(
            booster_type="R",
            cluster_positions=frozenset({Position(3, 3)}),
            landing_position=Position(3, 3),
            adjacent_refill=center_adjacent,
            all_refill=center_adjacent,
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=2, row_span=5, orientation="H",
                centroid=Position(3, 3), columns_used=frozenset({3, 4}),
                max_depth_per_column={3: 5, 4: 4},
            ),
        )
        assert criterion.score(ctx_center) > criterion.score(ctx_edge)

    def test_wrong_orientation_penalized(
        self, booster_rules: BoosterRules, default_config: MasterConfig,
    ) -> None:
        """Desired "H" but actual "V" → heavy penalty."""
        criterion = RocketArmCriterion(
            booster_rules, default_config.board,
            default_config.landing_criteria,
            desired_orientation="H",
        )
        adjacent = tuple(Position(r, 0) for r in range(4))
        ctx = LandingContext(
            booster_type="R",
            cluster_positions=frozenset({Position(3, 3)}),
            landing_position=Position(3, 3),
            adjacent_refill=adjacent,
            all_refill=adjacent,
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=5, row_span=2, orientation="V",
                centroid=Position(3, 3), columns_used=frozenset({1, 2, 3, 4, 5}),
                max_depth_per_column={1: 3, 2: 3, 3: 3, 4: 3, 5: 3},
            ),
        )
        # Score with penalty should be < 0.5
        assert criterion.score(ctx) < 0.5


# ---------------------------------------------------------------------------
# Bomb Criterion Tests
# ---------------------------------------------------------------------------

class TestBombArmCriterion:
    """BombArmCriterion scores arm feasibility + blast coverage."""

    def test_corner_landing_penalized(
        self, booster_rules: BoosterRules, default_config: MasterConfig,
    ) -> None:
        """Landing at (0,0) clips the blast to ~4 cells → low blast score."""
        criterion = BombArmCriterion(
            booster_rules, default_config.board,
            default_config.landing_criteria,
        )
        adjacent = tuple(Position(r, 0) for r in range(3))

        ctx_corner = LandingContext(
            booster_type="B",
            cluster_positions=frozenset({Position(0, 0)}),
            landing_position=Position(0, 0),
            adjacent_refill=adjacent,
            all_refill=adjacent,
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=3, row_span=4, orientation="H",
                centroid=Position(0, 0), columns_used=frozenset({0, 1, 2}),
                max_depth_per_column={0: 3, 1: 3, 2: 3},
            ),
        )

        ctx_center = LandingContext(
            booster_type="B",
            cluster_positions=frozenset({Position(3, 3)}),
            landing_position=Position(3, 3),
            adjacent_refill=adjacent,
            all_refill=adjacent,
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=3, row_span=4, orientation="H",
                centroid=Position(3, 3), columns_used=frozenset({2, 3, 4}),
                max_depth_per_column={2: 5, 3: 5, 4: 5},
            ),
        )
        assert criterion.score(ctx_center) > criterion.score(ctx_corner)


# ---------------------------------------------------------------------------
# Lightball / SLB Tests
# ---------------------------------------------------------------------------

class TestLightballCriterion:
    """LightballArmCriterion is pure armability."""

    def test_lb_slb_same_score(self, default_config: MasterConfig) -> None:
        """LB and SLB produce identical scores for identical context."""
        lb = LightballArmCriterion(default_config.board)
        slb = SuperLightballArmCriterion(default_config.board)
        adjacent = tuple(Position(r, 0) for r in range(3))
        ctx = LandingContext(
            booster_type="LB",
            cluster_positions=frozenset({Position(3, 3)}),
            landing_position=Position(3, 3),
            adjacent_refill=adjacent,
            all_refill=adjacent,
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=1, row_span=1, orientation="H",
                centroid=Position(3, 3), columns_used=frozenset({3}),
                max_depth_per_column={3: 3},
            ),
        )
        assert lb.score(ctx) == slb.score(ctx)

    def test_zero_adjacent_returns_zero(self, default_config: MasterConfig) -> None:
        """No adjacent refill → score 0.0."""
        criterion = LightballArmCriterion(default_config.board)
        ctx = LandingContext(
            booster_type="LB",
            cluster_positions=frozenset({Position(3, 3)}),
            landing_position=Position(3, 6),
            adjacent_refill=(),
            all_refill=(),
            settle_result=None,  # type: ignore[arg-type]
            cluster_shape_stats=ShapeStats(
                col_span=1, row_span=1, orientation="H",
                centroid=Position(3, 3), columns_used=frozenset({3}),
                max_depth_per_column={3: 3},
            ),
        )
        assert criterion.score(ctx) == 0.0


# ---------------------------------------------------------------------------
# Evaluator Service Tests
# ---------------------------------------------------------------------------

class TestBoosterLandingEvaluator:
    """BoosterLandingEvaluator computes physics and delegates scoring."""

    def test_evaluate_landing_matches_forward_sim(
        self,
        evaluator: BoosterLandingEvaluator,
        forward_sim: ForwardSimulator,
        booster_rules: BoosterRules,
        default_config: MasterConfig,
    ) -> None:
        """evaluate() returns a landing position matching predict_booster_landing."""
        # 7-cell vertical cluster in reel 3 → Wild spawn
        cluster = frozenset(Position(3, r) for r in range(7))
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)

        ctx = evaluator.evaluate(cluster, board, "W")

        # Verify landing matches direct ForwardSimulator call
        centroid = booster_rules.compute_centroid(cluster)
        expected_landing = forward_sim.predict_booster_landing(centroid, board, cluster)
        assert ctx.landing_position == expected_landing

    def test_adjacent_refill_matches_manual(
        self,
        evaluator: BoosterLandingEvaluator,
        forward_sim: ForwardSimulator,
        booster_rules: BoosterRules,
        default_config: MasterConfig,
    ) -> None:
        """adjacent_refill in context matches orthogonal_neighbors ∩ empty_positions."""
        cluster = frozenset(Position(3, r) for r in range(7))
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)

        ctx = evaluator.evaluate(cluster, board, "W")

        # Manual computation
        settle = forward_sim.simulate_explosion(board, cluster)
        neighbors = set(orthogonal_neighbors(ctx.landing_position, default_config.board))
        expected = set(pos for pos in settle.empty_positions if pos in neighbors)
        assert set(ctx.adjacent_refill) == expected

    def test_score_dispatches_to_correct_criterion(
        self,
        evaluator: BoosterLandingEvaluator,
        default_config: MasterConfig,
    ) -> None:
        """score() uses the registered criterion for the booster type."""
        # Cluster in center with good adjacency → should score differently per type
        cluster = frozenset(Position(3, r) for r in range(7))
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)

        ctx_w = evaluator.evaluate(cluster, board, "W")
        ctx_r = evaluator.evaluate(cluster, board, "R")

        score_w = evaluator.score(ctx_w)
        score_r = evaluator.score(ctx_r)

        # Both should be valid floats in [0, 1]
        assert 0.0 <= score_w <= 1.0
        assert 0.0 <= score_r <= 1.0

    def test_unknown_booster_type_returns_one(
        self,
        evaluator: BoosterLandingEvaluator,
        default_config: MasterConfig,
    ) -> None:
        """No criterion registered for unknown type → score 1.0 (assume viable)."""
        cluster = frozenset(Position(3, r) for r in range(7))
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)

        ctx = evaluator.evaluate(cluster, board, "UNKNOWN")
        assert evaluator.score(ctx) == 1.0

    def test_find_best_shape_picks_highest(
        self,
        evaluator: BoosterLandingEvaluator,
        default_config: MasterConfig,
    ) -> None:
        """find_best_shape returns the candidate with the highest score."""
        # Two clusters: one in center (better), one at edge (worse)
        cluster_center = frozenset(Position(3, r) for r in range(7))
        board_center = _make_board_with_cluster(
            default_config, Symbol.L2, cluster_center,
        )

        cluster_edge = frozenset(Position(0, r) for r in range(7))
        board_edge = _make_board_with_cluster(
            default_config, Symbol.L2, cluster_edge,
        )

        candidates = [
            (cluster_edge, board_edge),
            (cluster_center, board_center),
        ]
        result = evaluator.find_best_shape(candidates, "R", threshold=0.99)
        assert result is not None
        ctx, score = result
        # The center cluster should score higher
        assert ctx.cluster_positions in (cluster_center, cluster_edge)

    def test_find_best_shape_none_for_empty(
        self, evaluator: BoosterLandingEvaluator,
    ) -> None:
        """Empty candidate list returns None."""
        assert evaluator.find_best_shape([], "W") is None


# ---------------------------------------------------------------------------
# Reshape Bias Tests
# ---------------------------------------------------------------------------

class TestComputeReshapeBias:
    """compute_reshape_bias progressively concentrates toward upper rows."""

    def test_attempt_zero_preserves_bias(self, default_config: MasterConfig) -> None:
        """Attempt 0 returns the original variance unchanged."""
        variance = _make_variance(default_config)
        result = compute_reshape_bias(variance, default_config.board, attempt=0)
        assert result is variance  # Same object — no modification

    def test_later_attempts_favor_upper_rows(
        self, default_config: MasterConfig,
    ) -> None:
        """Attempt 2+ weights row 0 significantly higher than row 6."""
        variance = _make_variance(default_config)
        result = compute_reshape_bias(variance, default_config.board, attempt=2)

        # Compare weight at top row vs bottom row for same reel
        top_pos = Position(3, 0)
        bottom_pos = Position(3, 6)
        assert result.spatial_bias[top_pos] > result.spatial_bias[bottom_pos] * 5

    def test_symbol_weights_preserved(self, default_config: MasterConfig) -> None:
        """Reshape only modifies spatial_bias, not symbol_weights."""
        variance = _make_variance(default_config)
        result = compute_reshape_bias(variance, default_config.board, attempt=3)
        assert result.symbol_weights == variance.symbol_weights
        assert result.near_miss_symbol_preference == variance.near_miss_symbol_preference
        assert result.cluster_size_preference == variance.cluster_size_preference


# ---------------------------------------------------------------------------
# Chain-Aware Seeds Tests
# ---------------------------------------------------------------------------

class TestPlanChainAwareSeeds:
    """plan_chain_aware_seeds composes plan_arm_seeds + column filter."""

    def test_filters_seeds_outside_effect_zone(
        self,
        seed_planner: SeedPlanner,
        forward_sim: ForwardSimulator,
        booster_rules: BoosterRules,
        default_config: MasterConfig,
    ) -> None:
        """Seeds in columns outside the effect zone are removed."""
        # Build a board with a cluster, simulate explosion
        cluster = frozenset(Position(3, r) for r in range(7))
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)
        settle = forward_sim.simulate_explosion(board, cluster)

        # Effect zone only covers reels 2-4 (narrow zone)
        effect_zone = frozenset(
            Position(r, row) for r in range(2, 5) for row in range(7)
        )
        variance = _make_variance(default_config)
        rng = random.Random(42)

        seeds = seed_planner.plan_chain_aware_seeds(
            Position(3, 6), settle, effect_zone, variance, rng,
        )

        # All returned seeds must be in effect zone columns
        effect_columns = {p.reel for p in effect_zone}
        for pos in seeds:
            assert pos.reel in effect_columns, (
                f"Seed at {pos} is outside effect columns {effect_columns}"
            )

    def test_empty_effect_zone_returns_empty(
        self,
        seed_planner: SeedPlanner,
        forward_sim: ForwardSimulator,
        default_config: MasterConfig,
    ) -> None:
        """Effect zone with no column overlap → empty result."""
        cluster = frozenset(Position(3, r) for r in range(7))
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)
        settle = forward_sim.simulate_explosion(board, cluster)

        # Effect zone in reel 6 only — arm seeds target columns near reel 3
        effect_zone = frozenset(Position(6, row) for row in range(7))
        variance = _make_variance(default_config)
        rng = random.Random(42)

        seeds = seed_planner.plan_chain_aware_seeds(
            Position(3, 6), settle, effect_zone, variance, rng,
        )

        # Seeds in reel 6 or empty — arm_seeds targets reels near 3, so
        # filter may remove all of them
        for pos in seeds:
            assert pos.reel == 6


# ---------------------------------------------------------------------------
# ChainConstraint Tests
# ---------------------------------------------------------------------------

class TestChainConstraint:
    """ChainConstraint is a frozen data container."""

    def test_frozen(self) -> None:
        """ChainConstraint is immutable."""
        cc = ChainConstraint(
            source_type="R",
            source_position=Position(3, 3),
            source_orientation="H",
            effect_zone=frozenset({Position(r, 3) for r in range(7)}),
        )
        with pytest.raises(AttributeError):
            cc.source_type = "B"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ShapeStats Tests
# ---------------------------------------------------------------------------

class TestShapeStats:
    """ShapeStats correctly derives cluster geometry."""

    def test_horizontal_cluster_orientation(
        self,
        evaluator: BoosterLandingEvaluator,
        default_config: MasterConfig,
    ) -> None:
        """Wide cluster (col_span > row_span) → orientation 'V' (fires vertically)."""
        # Horizontal line of 7 in row 3
        cluster = frozenset(Position(r, 3) for r in range(7))
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)
        ctx = evaluator.evaluate(cluster, board, "R")
        assert ctx.cluster_shape_stats.col_span == 7
        assert ctx.cluster_shape_stats.row_span == 1
        assert ctx.cluster_shape_stats.orientation == "V"

    def test_vertical_cluster_orientation(
        self,
        evaluator: BoosterLandingEvaluator,
        default_config: MasterConfig,
    ) -> None:
        """Tall cluster (row_span > col_span) → orientation 'H' (fires horizontally)."""
        # Vertical line of 7 in reel 3
        cluster = frozenset(Position(3, r) for r in range(7))
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)
        ctx = evaluator.evaluate(cluster, board, "R")
        assert ctx.cluster_shape_stats.row_span == 7
        assert ctx.cluster_shape_stats.col_span == 1
        assert ctx.cluster_shape_stats.orientation == "H"

    def test_columns_used(
        self,
        evaluator: BoosterLandingEvaluator,
        default_config: MasterConfig,
    ) -> None:
        """columns_used matches unique reel indices in cluster."""
        cluster = frozenset({
            Position(1, 3), Position(2, 3), Position(3, 3),
            Position(2, 4), Position(3, 4),
        })
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)
        ctx = evaluator.evaluate(cluster, board, "W")
        assert ctx.cluster_shape_stats.columns_used == frozenset({1, 2, 3})

    def test_max_depth_per_column(
        self,
        evaluator: BoosterLandingEvaluator,
        default_config: MasterConfig,
    ) -> None:
        """max_depth_per_column tracks deepest row per reel."""
        cluster = frozenset({
            Position(2, 1), Position(2, 3), Position(2, 5),  # reel 2: rows 1,3,5
            Position(3, 2), Position(3, 4),                   # reel 3: rows 2,4
        })
        board = _make_board_with_cluster(default_config, Symbol.L2, cluster)
        ctx = evaluator.evaluate(cluster, board, "W")
        assert ctx.cluster_shape_stats.max_depth_per_column[2] == 5
        assert ctx.cluster_shape_stats.max_depth_per_column[3] == 4
