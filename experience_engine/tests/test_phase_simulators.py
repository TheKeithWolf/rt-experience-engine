"""Tests for Step 7: phase simulator dispatch + per-simulator contracts."""

from __future__ import annotations

import random

import pytest

from ..config.schema import MasterConfig
from ..narrative.arc import NarrativePhase
from ..pipeline.protocols import Range
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
from ..trajectory.phase_simulators import (
    ArmPhaseSimulator,
    CascadePhaseSimulator,
    FirePhaseSimulator,
    SketchDependencies,
    SketchState,
    SpawnPhaseSimulator,
    build_default_phase_registry,
    resolve_simulator,
)


def _filled_board(config: MasterConfig) -> Board:
    board = Board.empty(config.board)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            board.set(Position(reel, row), Symbol.L1)
    return board


@pytest.fixture
def sketch_deps(default_config: MasterConfig) -> SketchDependencies:
    dag = GravityDAG(default_config.board, default_config.gravity)
    forward = ForwardSimulator(
        dag, default_config.board, default_config.gravity
    )
    rules = BoosterRules(
        default_config.boosters, default_config.board, default_config.symbols
    )
    lc = default_config.landing_criteria
    criteria = {
        "W": WildBridgeCriterion(default_config.board, lc),
        "R": RocketArmCriterion(rules, default_config.board, lc),
        "B": BombArmCriterion(rules, default_config.board, lc),
        "LB": LightballArmCriterion(default_config.board),
        "SLB": LightballArmCriterion(default_config.board),
    }
    evaluator = BoosterLandingEvaluator(
        forward, rules, default_config.board, criteria
    )
    chain_eval = ChainEvaluator(default_config.boosters)
    return SketchDependencies(
        config=default_config,
        gravity_dag=dag,
        forward_sim=forward,
        landing_eval=evaluator,
        booster_rules=rules,
        chain_eval=chain_eval,
        standard_symbols=tuple(
            Symbol[name] for name in default_config.symbols.standard
        ),
    )


def _phase(
    size: tuple[int, int],
    spawns: tuple[str, ...] | None = None,
    arms: tuple[str, ...] | None = None,
    fires: tuple[str, ...] | None = None,
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
        fires=fires,
        wild_behavior=None,
        ends_when="always",
    )


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


def test_registry_covers_every_known_shape() -> None:
    """The registry must answer all 8 (spawns, arms, fires) combinations."""
    registry = build_default_phase_registry()
    assert len(registry) == 8


def test_resolver_falls_back_to_cascade_for_unknown_shape() -> None:
    registry = {(False, False, False): CascadePhaseSimulator()}
    phase = _phase((5, 5), spawns=("R",))
    assert isinstance(resolve_simulator(phase, registry), CascadePhaseSimulator)


def test_spawn_phase_picked_for_spawns_key() -> None:
    registry = build_default_phase_registry()
    phase = _phase((9, 9), spawns=("R",))
    assert isinstance(resolve_simulator(phase, registry), SpawnPhaseSimulator)


def test_arm_phase_picked_for_arms_key() -> None:
    registry = build_default_phase_registry()
    phase = _phase((5, 5), arms=("R",))
    assert isinstance(resolve_simulator(phase, registry), ArmPhaseSimulator)


def test_fire_phase_picked_for_fires_key() -> None:
    registry = build_default_phase_registry()
    phase = _phase((5, 5), fires=("R",))
    assert isinstance(resolve_simulator(phase, registry), FirePhaseSimulator)


# ---------------------------------------------------------------------------
# Per-simulator contracts
# ---------------------------------------------------------------------------


def test_cascade_phase_produces_waypoint(
    sketch_deps: SketchDependencies, default_config: MasterConfig
) -> None:
    state = SketchState(board=_filled_board(default_config))
    sim = CascadePhaseSimulator()
    waypoint = sim.simulate(_phase((5, 5)), state, sketch_deps, random.Random(0))
    assert waypoint is not None
    assert waypoint.booster_type is None
    assert len(waypoint.cluster_region) == 5


def test_spawn_phase_assigns_landing_and_tracks_dormant(
    sketch_deps: SketchDependencies, default_config: MasterConfig
) -> None:
    state = SketchState(board=_filled_board(default_config))
    sim = SpawnPhaseSimulator()
    # Size 9 maps to the R (rocket) spawn threshold in default.yaml.
    waypoint = sim.simulate(
        _phase((9, 9), spawns=("R",)), state, sketch_deps, random.Random(0)
    )
    assert waypoint is not None
    assert waypoint.booster_type == "R"
    assert waypoint.booster_landing_pos is not None
    assert waypoint.booster_landing_pos in state.dormant_positions


def test_arm_phase_requires_prior_dormant(
    sketch_deps: SketchDependencies, default_config: MasterConfig
) -> None:
    state = SketchState(board=_filled_board(default_config))
    sim = ArmPhaseSimulator()
    waypoint = sim.simulate(
        _phase((5, 5), arms=("R",)), state, sketch_deps, random.Random(0)
    )
    assert waypoint is None  # No dormant present → refuses to simulate


def test_fire_phase_requires_prior_dormant(
    sketch_deps: SketchDependencies, default_config: MasterConfig
) -> None:
    state = SketchState(board=_filled_board(default_config))
    sim = FirePhaseSimulator()
    waypoint = sim.simulate(
        _phase((5, 5), fires=("R",)), state, sketch_deps, random.Random(0)
    )
    assert waypoint is None
