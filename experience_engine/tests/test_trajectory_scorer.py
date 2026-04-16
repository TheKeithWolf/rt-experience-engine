"""Tests for Step 6: TrajectoryScorer composite math + neutral_fill coverage."""

from __future__ import annotations

import random

import pytest

from ..config.schema import MasterConfig, TrajectoryConfig
from ..primitives.board import Board, Position
from ..primitives.gravity import SettleResult
from ..primitives.symbols import Symbol
from ..trajectory.data_types import TrajectoryWaypoint
from ..trajectory.scorer import ScoredTrajectory, TrajectoryScorer
from ..trajectory.sketch_fill import neutral_fill


_TRAJ_CFG = TrajectoryConfig(
    max_sketch_retries=5,
    waypoint_feasibility_threshold=0.2,
    sketch_feasibility_threshold=0.15,
)


def _waypoint(score: float) -> TrajectoryWaypoint:
    empty_settle = SettleResult(
        board=None, move_steps=(), empty_positions=(),  # type: ignore[arg-type]
    )
    return TrajectoryWaypoint(
        phase_index=0,
        cluster_region=frozenset(),
        cluster_symbol=Symbol.L1,
        booster_type=None,
        booster_spawn_pos=None,
        booster_landing_pos=None,
        settle_result=empty_settle,
        landing_context=None,
        landing_score=score,
        reserve_zone=frozenset(),
        chain_target_pos=None,
        seed_hints={},
    )


def test_scorer_rejects_empty_waypoints() -> None:
    """An arc with no steps cannot be feasible."""
    scorer = TrajectoryScorer(_TRAJ_CFG)
    result = scorer.score(())
    assert result.is_feasible is False
    assert result.composite_score == 0.0


def test_scorer_short_circuits_below_waypoint_threshold() -> None:
    """A single weak waypoint must fail the sketch immediately."""
    scorer = TrajectoryScorer(_TRAJ_CFG)
    waypoints = [_waypoint(0.9), _waypoint(0.1)]  # second is below 0.2
    result = scorer.score(waypoints)
    assert result.is_feasible is False
    assert result.composite_score == 0.0


def test_scorer_feasible_when_composite_clears_threshold() -> None:
    scorer = TrajectoryScorer(_TRAJ_CFG)
    result = scorer.score([_waypoint(0.9), _waypoint(0.8)])
    assert result.is_feasible is True
    assert result.composite_score == pytest.approx(0.72)


def test_scorer_infeasible_when_composite_below_sketch_threshold() -> None:
    cfg = TrajectoryConfig(
        max_sketch_retries=1,
        waypoint_feasibility_threshold=0.1,
        sketch_feasibility_threshold=0.5,
    )
    scorer = TrajectoryScorer(cfg)
    # Each waypoint clears the per-waypoint floor, but 0.6 * 0.6 = 0.36 < 0.5
    result = scorer.score([_waypoint(0.6), _waypoint(0.6)])
    assert result.composite_score == pytest.approx(0.36)
    assert result.is_feasible is False


# ---------------------------------------------------------------------------
# neutral_fill
# ---------------------------------------------------------------------------


def test_neutral_fill_populates_every_empty(default_config: MasterConfig) -> None:
    board = Board.empty(default_config.board)
    empty = [
        Position(reel, row)
        for reel in range(default_config.board.num_reels)
        for row in range(default_config.board.num_rows)
    ]
    standard = tuple(Symbol[name] for name in default_config.symbols.standard)
    weights = {s: 1.0 for s in standard}
    neutral_fill(board, empty, standard, weights, random.Random(0))
    for pos in empty:
        assert board.get(pos) in standard


def test_neutral_fill_skips_already_filled_cells(
    default_config: MasterConfig,
) -> None:
    """Passing an occupied position must be a no-op for that cell."""
    board = Board.empty(default_config.board)
    anchor = Position(0, 0)
    board.set(anchor, Symbol.H1)
    standard = (Symbol.L1, Symbol.L2)
    neutral_fill(
        board,
        [anchor, Position(0, 1)],
        standard,
        {Symbol.L1: 1.0, Symbol.L2: 1.0},
        random.Random(0),
    )
    assert board.get(anchor) == Symbol.H1
    assert board.get(Position(0, 1)) in standard


def test_neutral_fill_rejects_empty_standard_set() -> None:
    board = Board.empty(__import__(
        "games.royal_tumble.experience_engine.config.schema",
        fromlist=["BoardConfig"],
    ).BoardConfig(num_reels=1, num_rows=1, min_cluster_size=2))
    with pytest.raises(ValueError, match="standard_symbols"):
        neutral_fill(board, [Position(0, 0)], (), {}, random.Random(0))


def test_neutral_fill_handles_all_zero_weights(
    default_config: MasterConfig,
) -> None:
    """All-zero weights must fall back to uniform — not crash."""
    board = Board.empty(default_config.board)
    standard = (Symbol.L1, Symbol.L2)
    neutral_fill(
        board,
        [Position(0, 0)],
        standard,
        {Symbol.L1: 0.0, Symbol.L2: 0.0},
        random.Random(0),
    )
    assert board.get(Position(0, 0)) in standard
