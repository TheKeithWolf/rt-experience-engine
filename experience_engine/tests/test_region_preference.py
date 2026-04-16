"""Tests for Step 5: RegionPreferenceFactor + ClusterBuilder region param.

Covers:
- RegionPreferenceFactor scoring inside vs outside the region.
- Defaults to 1.0 when the ScoringContext has no region.
- ClusterBuilder.find_positions biases cluster placement toward viable_columns
  when a RegionConstraint is supplied, and remains unchanged when it is None.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest

from ..config.schema import MasterConfig
from ..planning.region_constraint import RegionConstraint
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..step_reasoner.context import BoardContext
from ..step_reasoner.evaluators import PayoutEstimator, SpawnEvaluator
from ..step_reasoner.services.cluster_builder import ClusterBuilder
from ..step_reasoner.services.utility_scorer import (
    RegionPreferenceFactor,
    ScoringContext,
)
from ..variance.hints import VarianceHints


# ---------------------------------------------------------------------------
# RegionPreferenceFactor
# ---------------------------------------------------------------------------


def _scoring_context(
    region: RegionConstraint | None, board_config
) -> ScoringContext:
    """Minimal ScoringContext — only the fields the factor reads are populated."""
    return ScoringContext(
        influence={},
        gravity_field=MagicMock(),
        demand=MagicMock(),
        cluster_positions=frozenset(),
        board_config=board_config,
        booster_landing=None,
        region=region,
    )


def test_region_factor_returns_one_without_region(
    default_config: MasterConfig,
) -> None:
    factor = RegionPreferenceFactor(falloff=0.5)
    ctx = _scoring_context(region=None, board_config=default_config.board)
    assert factor.evaluate(Position(0, 0), ctx) == 1.0


def test_region_factor_scores_inside_as_one(
    default_config: MasterConfig,
) -> None:
    factor = RegionPreferenceFactor(falloff=0.5)
    region = RegionConstraint(
        viable_columns=frozenset({2, 3}),
        preferred_row_range=None,
    )
    ctx = _scoring_context(region, default_config.board)
    assert factor.evaluate(Position(2, 4), ctx) == 1.0
    assert factor.evaluate(Position(3, 0), ctx) == 1.0


def test_region_factor_applies_falloff_per_column(
    default_config: MasterConfig,
) -> None:
    factor = RegionPreferenceFactor(falloff=0.5)
    region = RegionConstraint(
        viable_columns=frozenset({3}),
        preferred_row_range=None,
    )
    ctx = _scoring_context(region, default_config.board)
    # One column away → falloff^1 = 0.5
    assert factor.evaluate(Position(2, 0), ctx) == pytest.approx(0.5)
    # Three columns away → falloff^3 = 0.125
    assert factor.evaluate(Position(0, 0), ctx) == pytest.approx(0.125)


def test_region_factor_rejects_out_of_bounds_falloff() -> None:
    with pytest.raises(ValueError, match="falloff"):
        RegionPreferenceFactor(falloff=0.0)
    with pytest.raises(ValueError, match="falloff"):
        RegionPreferenceFactor(falloff=1.5)


# ---------------------------------------------------------------------------
# ClusterBuilder region parameter
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_builder(default_config: MasterConfig) -> ClusterBuilder:
    """ClusterBuilder configured with a strong region falloff to make the
    bias visible in a small number of sampled runs."""
    spawn_eval = SpawnEvaluator(default_config.boosters)
    payout_eval = PayoutEstimator(
        default_config.paytable, default_config.centipayout,
        default_config.win_levels, default_config.symbols,
        default_config.grid_multiplier,
    )
    return ClusterBuilder(
        spawn_evaluator=spawn_eval,
        payout_estimator=payout_eval,
        board_config=default_config.board,
        symbol_config=default_config.symbols,
        region_falloff=0.1,
    )


@pytest.fixture
def empty_context(default_config: MasterConfig) -> BoardContext:
    board = Board.empty(default_config.board)
    grid_mults = GridMultiplierGrid(
        default_config.grid_multiplier, default_config.board
    )
    return BoardContext(
        board=board,
        grid_multipliers=grid_mults,
        dormant_boosters=[],
        active_wilds=[],
        _board_config=default_config.board,
    )


def _flat_variance(default_config: MasterConfig) -> VarianceHints:
    """Uniform spatial bias so region falloff is the only weight signal."""
    bias = {
        Position(reel, row): 1.0
        for reel in range(default_config.board.num_reels)
        for row in range(default_config.board.num_rows)
    }
    return VarianceHints(
        spatial_bias=bias,
        symbol_weights={s: 1.0 for s in Symbol},
        near_miss_symbol_preference=(),
        cluster_size_preference=(),
    )


def test_cluster_builder_rejects_invalid_falloff(
    default_config: MasterConfig,
) -> None:
    """Falloff must be in (0.0, 1.0] — zero would collapse all weights."""
    spawn_eval = SpawnEvaluator(default_config.boosters)
    payout_eval = PayoutEstimator(
        default_config.paytable, default_config.centipayout,
        default_config.win_levels, default_config.symbols,
        default_config.grid_multiplier,
    )
    with pytest.raises(ValueError, match="region_falloff"):
        ClusterBuilder(
            spawn_evaluator=spawn_eval,
            payout_estimator=payout_eval,
            board_config=default_config.board,
            symbol_config=default_config.symbols,
            region_falloff=0.0,
        )


def test_find_positions_without_region_uses_full_board(
    cluster_builder: ClusterBuilder,
    empty_context: BoardContext,
    default_config: MasterConfig,
) -> None:
    """region=None must leave existing behavior unchanged — samples should
    land in multiple columns given uniform variance."""
    variance = _flat_variance(default_config)
    rng = random.Random(1234)
    reels_touched = set()
    for _ in range(30):
        result = cluster_builder.find_positions(
            empty_context, size=5, rng=rng, variance=variance,
        )
        reels_touched.update(p.reel for p in result.planned_positions)
    # Uniform weights over 7 reels should spread placements across >1 column.
    assert len(reels_touched) >= 3


def test_find_positions_with_region_biases_toward_viable_columns(
    cluster_builder: ClusterBuilder,
    empty_context: BoardContext,
    default_config: MasterConfig,
) -> None:
    """With a sharp falloff and a narrow viable_columns set, most samples
    must place the majority of their cells in the viable columns."""
    variance = _flat_variance(default_config)
    region = RegionConstraint(
        viable_columns=frozenset({2, 3}),
        preferred_row_range=None,
    )
    rng = random.Random(42)
    inside_fraction_total = 0.0
    trials = 30
    for _ in range(trials):
        result = cluster_builder.find_positions(
            empty_context, size=5, rng=rng, variance=variance, region=region,
        )
        inside = sum(
            1 for p in result.planned_positions if p.reel in region.viable_columns
        )
        inside_fraction_total += inside / 5
    avg_inside = inside_fraction_total / trials
    # Falloff 0.1 makes non-viable columns exponentially discouraged; the
    # sampled majority must land inside the region.
    assert avg_inside >= 0.8
