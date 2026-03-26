"""Tests for StepTransitionSimulator.transition_and_arm() and execute_terminal_booster_phase().

Verifies that:
- transition_and_arm() handles cluster explosion, spawning, gravity, and arming
  (fires are deferred to the post-terminal phase)
- execute_terminal_booster_phase() fires armed boosters on a dead board
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
# TEST: transition_and_arm arms a dormant rocket (fires are deferred)
# ---------------------------------------------------------------------------

def test_transition_and_arm_arms_rocket(
    simulator: StepTransitionSimulator,
    default_config: MasterConfig,
) -> None:
    """Dormant rocket adjacent to exploding cluster gets ARMED but not fired."""
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

    result = simulator.transition_and_arm(
        board, step_result, tracker, grid_mults, phase_executor,
    )

    # Rocket should be ARMED but NOT fired — fires are deferred to post-terminal phase
    assert result.booster_fire_records == ()
    assert result.booster_gravity_record is None

    # Verify the rocket is armed in the tracker
    from ..boosters.state_machine import BoosterState
    all_boosters = tracker.all_boosters()
    rockets = [b for b in all_boosters if b.booster_type is Symbol.R]
    assert len(rockets) == 1
    assert rockets[0].state is BoosterState.ARMED


# ---------------------------------------------------------------------------
# TEST: transition_and_arm with no armed boosters
# ---------------------------------------------------------------------------

def test_freshly_spawned_booster_not_armed_by_source_cluster(
    simulator: StepTransitionSimulator,
    default_config: MasterConfig,
) -> None:
    """A rocket spawned from a size-9 cluster must NOT fire in the same transition.

    Regression test for the immediate-fire bug: freshly-spawned boosters sit at
    their source cluster's centroid, so they are inherently adjacent to the
    cluster positions. The exclusion set prevents arm_adjacent from arming them.
    """
    board = Board.empty(default_config.board)

    # Fill board with L2 background
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), Symbol.L2)

    # Place a 9-cell L1 cluster (3x3 block at reels 1-3, rows 1-3)
    # Size 9 → Rocket spawn per config spawn_thresholds
    cluster_positions = frozenset(
        Position(reel, row)
        for reel in range(1, 4)
        for row in range(1, 4)
    )
    for pos in cluster_positions:
        board.set(pos, Symbol.L1)

    # No pre-existing boosters — the only booster will be spawned during transition
    tracker = BoosterTracker(default_config.board)
    grid_mults = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)
    phase_executor = _make_phase_executor(tracker, default_config)

    step_result = StepResult(
        step_index=0,
        clusters=(
            ClusterRecord(
                symbol=Symbol.L1, size=9, positions=cluster_positions,
                step_index=0, payout=500,
            ),
        ),
        spawns=(),
        fires=(),
        symbol_tier=None,
        step_payout=500,
    )

    result = simulator.transition_and_arm(
        board, step_result, tracker, grid_mults, phase_executor,
    )

    # Rocket should have spawned but NOT fired
    assert len(result.booster_fire_records) == 0

    # Rocket should be DORMANT in tracker
    from ..boosters.state_machine import BoosterState
    all_boosters = tracker.all_boosters()
    rockets = [b for b in all_boosters if b.booster_type is Symbol.R]
    assert len(rockets) == 1
    assert rockets[0].state is BoosterState.DORMANT


def test_transition_and_arm_no_armed_no_fire(
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

    result = simulator.transition_and_arm(
        board, step_result, tracker, grid_mults, phase_executor,
    )

    # No adjacent boosters → no fire phase
    assert result.booster_fire_records == ()
    assert result.booster_gravity_record is None


# ---------------------------------------------------------------------------
# TESTS: execute_terminal_booster_phase()
# ---------------------------------------------------------------------------

def test_terminal_booster_phase_fires_armed_rocket(
    simulator: StepTransitionSimulator,
    default_config: MasterConfig,
) -> None:
    """Armed rocket on a dead board fires and clears its row."""
    board = Board.empty(default_config.board)

    # Fill entire board so the rocket has cells to clear
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), Symbol.L2)

    # Place rocket at (3, 2) and arm it via adjacency to a fake cluster
    rocket_pos = Position(3, 2)
    tracker = BoosterTracker(default_config.board)
    tracker.add(Symbol.R, rocket_pos, orientation="H")
    # Arm by claiming an adjacent position as a cluster position
    tracker.arm_adjacent(frozenset({Position(2, 2)}))

    grid_mults = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)
    phase_executor = _make_phase_executor(tracker, default_config)

    result = simulator.execute_terminal_booster_phase(
        board, tracker, grid_mults, phase_executor,
    )

    # Rocket should have fired
    assert len(result.booster_fire_records) == 1
    fire_rec = result.booster_fire_records[0]
    assert fire_rec.booster_type == "R"
    assert fire_rec.orientation == "H"
    assert fire_rec.affected_count > 0

    # Post-booster gravity should have run
    assert result.booster_gravity_record is not None

    # No spawns in the terminal booster phase
    assert result.spawns == ()


def test_terminal_booster_phase_no_armed_returns_empty(
    simulator: StepTransitionSimulator,
    default_config: MasterConfig,
) -> None:
    """No armed boosters → empty fire records and no booster gravity."""
    board = Board.empty(default_config.board)
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), Symbol.L2)

    # Dormant (not armed) rocket — should NOT fire
    tracker = BoosterTracker(default_config.board)
    tracker.add(Symbol.R, Position(3, 3), orientation="V")

    grid_mults = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)
    phase_executor = _make_phase_executor(tracker, default_config)

    result = simulator.execute_terminal_booster_phase(
        board, tracker, grid_mults, phase_executor,
    )

    assert result.booster_fire_records == ()
    assert result.booster_gravity_record is None


def test_terminal_booster_phase_chain_propagation(
    simulator: StepTransitionSimulator,
    default_config: MasterConfig,
) -> None:
    """Armed rocket's blast hits a dormant bomb, chain-triggering it."""
    board = Board.empty(default_config.board)
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), Symbol.L2)

    tracker = BoosterTracker(default_config.board)

    # Horizontal rocket at (0, 3) — clears row 3. Armed via adjacency.
    # Must be on both tracker and board for the fire handler to see other boosters in path
    rocket_pos = Position(0, 3)
    board.set(rocket_pos, Symbol.R)
    tracker.add(Symbol.R, rocket_pos, orientation="H")
    tracker.arm_adjacent(frozenset({Position(0, 2)}))

    # Dormant bomb at (4, 3) — sits in the rocket's row, will be chain-triggered
    # Must be on both tracker and board for the fire handler to detect it
    bomb_pos = Position(4, 3)
    board.set(bomb_pos, Symbol.B)
    tracker.add(Symbol.B, bomb_pos)

    grid_mults = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)
    phase_executor = _make_phase_executor(tracker, default_config)

    result = simulator.execute_terminal_booster_phase(
        board, tracker, grid_mults, phase_executor,
    )

    # Both rocket and bomb should have fired (chain reaction)
    assert len(result.booster_fire_records) == 2
    fired_types = {rec.booster_type for rec in result.booster_fire_records}
    assert fired_types == {"R", "B"}
