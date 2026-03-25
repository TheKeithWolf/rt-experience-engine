"""Tests for StepTransitionSimulator.transition_with_boosters().

Verifies that the booster lifecycle (arm → fire → clear → gravity) is correctly
integrated into the transition pipeline. Tests use a pre-set board with a
dormant rocket adjacent to a cluster that will explode.
"""

from __future__ import annotations

import pytest

from ..boosters.fire_handlers import (
    fire_bomb,
    fire_lightball,
    fire_rocket,
    fire_superlightball,
)
from ..boosters.phase_executor import BoosterPhaseExecutor
from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..pipeline.simulator import StepTransitionSimulator
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.gravity import GravityDAG
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.symbols import Symbol
from ..step_reasoner.progress import ClusterRecord
from ..step_reasoner.results import SpawnRecord, StepResult


def _make_phase_executor(
    tracker: BoosterTracker, config: MasterConfig,
) -> BoosterPhaseExecutor:
    """Build a BoosterPhaseExecutor with real fire handlers."""
    rules = BoosterRules(config.boosters, config.board, config.symbols)
    executor = BoosterPhaseExecutor(tracker, rules, rules.chain_initiators)
    executor.register_fire_handler(Symbol.R, fire_rocket)
    executor.register_fire_handler(Symbol.B, fire_bomb)
    executor.register_fire_handler(Symbol.LB, fire_lightball)
    executor.register_fire_handler(Symbol.SLB, fire_superlightball)
    return executor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gravity_dag(default_config: MasterConfig) -> GravityDAG:
    return GravityDAG(default_config.board, default_config.gravity)


@pytest.fixture
def simulator(
    gravity_dag: GravityDAG, default_config: MasterConfig,
) -> StepTransitionSimulator:
    return StepTransitionSimulator(gravity_dag, default_config)


# ---------------------------------------------------------------------------
# TEST: transition() returns empty booster fire fields (backward compat)
# ---------------------------------------------------------------------------

def test_transition_returns_empty_booster_fields(
    simulator: StepTransitionSimulator,
    default_config: MasterConfig,
) -> None:
    """Plain transition() returns empty booster_fire_records and None gravity."""
    board = Board.empty(default_config.board)
    # Place a 5-cell L1 cluster in row 0 across reels 0-4
    cluster_positions = frozenset(Position(r, 0) for r in range(5))
    for pos in cluster_positions:
        board.set(pos, Symbol.L1)
    # Fill remaining cells so gravity has something to move
    for reel in range(7):
        for row in range(7):
            pos = Position(reel, row)
            if board.get(pos) is None:
                board.set(pos, Symbol.L2)

    tracker = BoosterTracker(default_config.board)
    grid_mults = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)

    step_result = StepResult(
        step_index=0,
        clusters=(
            ClusterRecord(
                symbol=Symbol.L1, size=5, positions=cluster_positions,
                step_index=0, payout=50,
            ),
        ),
        spawns=(),
        fires=(),
        symbol_tier=None,
        step_payout=50,
    )

    result = simulator.transition(board, step_result, tracker, grid_mults)

    # Backward compat: plain transition has empty booster fields
    assert result.booster_fire_records == ()
    assert result.booster_gravity_record is None


# ---------------------------------------------------------------------------
# TEST: transition_with_boosters fires a dormant rocket
# ---------------------------------------------------------------------------

def test_transition_with_boosters_fires_rocket(
    simulator: StepTransitionSimulator,
    default_config: MasterConfig,
) -> None:
    """Dormant rocket adjacent to exploding cluster gets armed and fires."""
    board = Board.empty(default_config.board)

    # Fill entire board with L2 so gravity works
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), Symbol.L2)

    # Place a 5-cell L1 cluster adjacent to where the rocket sits
    # Cluster at reel=2, rows 0-4 (vertical strip)
    cluster_positions = frozenset(Position(2, row) for row in range(5))
    for pos in cluster_positions:
        board.set(pos, Symbol.L1)

    # Place dormant rocket at (3, 2) — adjacent to cluster position (2, 2)
    rocket_pos = Position(3, 2)
    tracker = BoosterTracker(default_config.board)
    tracker.add(Symbol.R, rocket_pos, orientation="H")

    grid_mults = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)
    phase_executor = _make_phase_executor(tracker, default_config)

    step_result = StepResult(
        step_index=1,
        clusters=(
            ClusterRecord(
                symbol=Symbol.L1, size=5, positions=cluster_positions,
                step_index=1, payout=50,
            ),
        ),
        spawns=(),
        fires=(),
        symbol_tier=None,
        step_payout=50,
    )

    result = simulator.transition_with_boosters(
        board, step_result, tracker, grid_mults, phase_executor,
    )

    # Rocket should have fired — booster_fire_records is non-empty
    assert len(result.booster_fire_records) == 1
    fire_rec = result.booster_fire_records[0]
    assert fire_rec.booster_type == "R"
    assert fire_rec.orientation == "H"
    # Horizontal rocket at row 2 clears the entire row (minus immune positions)
    assert fire_rec.affected_count > 0

    # Post-booster gravity should have run
    assert result.booster_gravity_record is not None

    # Board should be settled (no None cells in non-exploded area)
    assert result.board is not None


# ---------------------------------------------------------------------------
# TEST: transition_with_boosters with no armed boosters
# ---------------------------------------------------------------------------

def test_transition_with_boosters_no_armed_no_fire(
    simulator: StepTransitionSimulator,
    default_config: MasterConfig,
) -> None:
    """When no boosters are adjacent to cluster, no fire phase occurs."""
    board = Board.empty(default_config.board)

    # Fill entire board
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), Symbol.L2)

    # Small cluster at (0,0)-(0,4)
    cluster_positions = frozenset(Position(0, row) for row in range(5))
    for pos in cluster_positions:
        board.set(pos, Symbol.L1)

    # Place rocket far from cluster at (6, 6) — not adjacent to any cluster pos
    tracker = BoosterTracker(default_config.board)
    tracker.add(Symbol.R, Position(6, 6), orientation="V")

    grid_mults = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)
    phase_executor = _make_phase_executor(tracker, default_config)

    step_result = StepResult(
        step_index=0,
        clusters=(
            ClusterRecord(
                symbol=Symbol.L1, size=5, positions=cluster_positions,
                step_index=0, payout=50,
            ),
        ),
        spawns=(),
        fires=(),
        symbol_tier=None,
        step_payout=50,
    )

    result = simulator.transition_with_boosters(
        board, step_result, tracker, grid_mults, phase_executor,
    )

    # No adjacent boosters → no fire phase
    assert result.booster_fire_records == ()
    assert result.booster_gravity_record is None
