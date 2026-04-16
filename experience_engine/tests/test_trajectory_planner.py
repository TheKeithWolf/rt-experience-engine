"""Tests for Step 8: TrajectoryPlanner orchestration."""

from __future__ import annotations

import random

import pytest

from ..config.schema import MasterConfig, TrajectoryConfig
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.gravity import GravityDAG
from ..primitives.symbols import Symbol, SymbolTier
from ..step_reasoner.evaluators import ChainEvaluator
from ..step_reasoner.services.forward_simulator import ForwardSimulator
from ..step_reasoner.services.landing_evaluator import BoosterLandingEvaluator
from ..step_reasoner.services.landing_criteria import (
    BombArmCriterion,
    LightballArmCriterion,
    RocketArmCriterion,
    WildBridgeCriterion,
)
from ..trajectory.phase_simulators import SketchDependencies
from ..trajectory.planner import TrajectoryPlanner
from ..trajectory.scorer import TrajectoryScorer


@pytest.fixture
def planner(default_config: MasterConfig) -> TrajectoryPlanner:
    dag = GravityDAG(default_config.board, default_config.gravity)
    forward = ForwardSimulator(
        dag, default_config.board, default_config.gravity
    )
    rules = BoosterRules(
        default_config.boosters, default_config.board, default_config.symbols
    )
    criteria = {
        "W": WildBridgeCriterion(default_config.board),
        "R": RocketArmCriterion(rules, default_config.board),
        "B": BombArmCriterion(rules, default_config.board),
        "LB": LightballArmCriterion(default_config.board),
        "SLB": LightballArmCriterion(default_config.board),
    }
    evaluator = BoosterLandingEvaluator(
        forward, rules, default_config.board, criteria
    )
    deps = SketchDependencies(
        config=default_config,
        gravity_dag=dag,
        forward_sim=forward,
        landing_eval=evaluator,
        booster_rules=rules,
        chain_eval=ChainEvaluator(default_config.boosters),
        standard_symbols=tuple(
            Symbol[name] for name in default_config.symbols.standard
        ),
    )
    scorer = TrajectoryScorer(
        TrajectoryConfig(
            max_sketch_retries=1,
            waypoint_feasibility_threshold=0.01,
            sketch_feasibility_threshold=0.01,
        )
    )
    return TrajectoryPlanner(deps, scorer)


def _filled_board(config: MasterConfig) -> Board:
    board = Board.empty(config.board)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            board.set(Position(reel, row), Symbol.L1)
    return board


def _phase(
    size: tuple[int, int],
    spawns: tuple[str, ...] | None = None,
    arms: tuple[str, ...] | None = None,
) -> NarrativePhase:
    return NarrativePhase(
        id="p",
        intent="",
        repetitions=Range(1, 1),
        cluster_count=Range(1, 1),
        cluster_sizes=(Range(*size),),
        cluster_symbol_tier=SymbolTier.ANY,
        spawns=spawns,
        arms=arms,
        fires=None,
        wild_behavior=None,
        ends_when="always",
    )


def _arc(phases: tuple[NarrativePhase, ...]) -> NarrativeArc:
    return NarrativeArc(
        phases=phases,
        payout=RangeFloat(0.0, 100.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )


def test_planner_produces_sketch_for_single_phase_arc(
    planner: TrajectoryPlanner, default_config: MasterConfig
) -> None:
    arc = _arc((_phase((5, 5)),))
    sketch = planner.sketch(arc, _filled_board(default_config), random.Random(0))
    assert sketch.is_feasible is True
    assert len(sketch.waypoints) == 1
    assert sketch.arc is arc


def test_planner_handles_spawn_then_arm_arc(
    planner: TrajectoryPlanner, default_config: MasterConfig
) -> None:
    """Two-phase arc: rocket spawn → arm. The arm phase consumes the dormant
    recorded by the spawn phase; feasibility requires both waypoints."""
    arc = _arc(
        (
            _phase((9, 9), spawns=("R",)),
            _phase((5, 5), arms=("R",)),
        )
    )
    sketch = planner.sketch(arc, _filled_board(default_config), random.Random(0))
    assert len(sketch.waypoints) == 2
    assert sketch.waypoints[0].booster_type == "R"


def test_planner_infeasible_when_arm_has_no_dormant(
    planner: TrajectoryPlanner, default_config: MasterConfig
) -> None:
    """An arm phase without a preceding spawn must short-circuit."""
    arc = _arc((_phase((5, 5), arms=("R",)),))
    sketch = planner.sketch(arc, _filled_board(default_config), random.Random(0))
    assert sketch.is_feasible is False
    assert sketch.waypoints == ()


def test_planner_does_not_mutate_input_board(
    planner: TrajectoryPlanner, default_config: MasterConfig
) -> None:
    """Board.copy() guarantees the caller's board is untouched."""
    arc = _arc((_phase((5, 5)),))
    board = _filled_board(default_config)
    snapshot = [
        board.get(Position(reel, row))
        for reel in range(default_config.board.num_reels)
        for row in range(default_config.board.num_rows)
    ]
    planner.sketch(arc, board, random.Random(0))
    after = [
        board.get(Position(reel, row))
        for reel in range(default_config.board.num_reels)
        for row in range(default_config.board.num_rows)
    ]
    assert snapshot == after
