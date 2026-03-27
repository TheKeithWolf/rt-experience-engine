"""Tests for rl_archive Phase 4 — observation, action space, and environment.

RLA-030 through RLA-043: Environment lifecycle, phase progression,
observation builder, action interpreter, and termination conditions.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from ..config.schema import (
    BoardConfig,
    DescriptorConfig,
    EnvironmentConfig,
    GridMultiplierConfig,
    MasterConfig,
    SymbolConfig,
)
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.symbols import Symbol
from ..rl_archive.action_space import ActionInterpreter, CascadeAction
from ..rl_archive.environment import CascadeEnvironment, StepInfo
from ..rl_archive.observation import CascadeObservation, ObservationBuilder, ObservationEncoder
from ..step_reasoner.context import BoardContext
from ..step_reasoner.intent import StepType
from ..step_reasoner.progress import ProgressTracker
from ..step_reasoner.results import ClusterRecord, SpawnRecord, StepResult
from ..variance.hints import VarianceHints


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_BOARD_CONFIG = BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5)
_GRID_MULT_CONFIG = GridMultiplierConfig(
    initial_value=0, first_hit_value=1, increment=1, cap=100,
    minimum_contribution=1,
)
_SYMBOL_CONFIG = SymbolConfig(
    standard=("L1", "L2", "L3", "L4", "H1", "H2", "H3"),
    low_tier=("L1", "L2", "L3", "L4"),
    high_tier=("H1", "H2", "H3"),
    payout_rank=(
        ("L1", 1), ("L2", 2), ("L3", 3), ("L4", 4),
        ("H1", 5), ("H2", 6), ("H3", 7),
    ),
)
_ENV_CONFIG = EnvironmentConfig(
    max_episode_steps=15,
    invalid_step_penalty=-1.0,
    completion_bonus=5.0,
    failure_penalty=-3.0,
    feasibility_weight=0.4,
    progress_weight=0.3,
)


def _make_phase(
    phase_id: str = "p1",
    reps: Range = Range(1, 3),
    cluster_count: Range = Range(1, 2),
    ends_when: str = "always",
) -> NarrativePhase:
    return NarrativePhase(
        id=phase_id,
        intent="test",
        repetitions=reps,
        cluster_count=cluster_count,
        cluster_sizes=(Range(5, 10),),
        cluster_symbol_tier=None,
        spawns=None,
        arms=None,
        fires=None,
        wild_behavior=None,
        ends_when=ends_when,
    )


def _make_arc(
    phases: tuple[NarrativePhase, ...] | None = None,
) -> NarrativeArc:
    if phases is None:
        phases = (_make_phase(),)
    return NarrativeArc(
        phases=phases,
        payout=RangeFloat(0.0, 10.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )


def _make_step_result(
    step_index: int = 0,
    step_payout: int = 100,
    clusters: tuple[ClusterRecord, ...] | None = None,
) -> StepResult:
    if clusters is None:
        clusters = (
            ClusterRecord(
                symbol=Symbol.L1,
                size=5,
                positions=frozenset(Position(0, i) for i in range(5)),
                step_index=step_index,
                payout=100,
            ),
        )
    return StepResult(
        step_index=step_index,
        clusters=clusters,
        spawns=(),
        fires=(),
        symbol_tier=None,
        step_payout=step_payout,
    )


def _make_obs_builder() -> ObservationBuilder:
    return ObservationBuilder(_SYMBOL_CONFIG, _BOARD_CONFIG)


# ---------------------------------------------------------------------------
# RLA-030: reset() returns observation with first phase
# ---------------------------------------------------------------------------


def test_rla_030_reset_returns_first_phase() -> None:
    """RLA-030: reset() returns observation with current_phase_id = first phase."""
    arc = _make_arc(phases=(_make_phase(phase_id="phase_01"),))
    obs_builder = _make_obs_builder()

    obs = obs_builder.build(
        BoardContext.from_board(
            Board.empty(_BOARD_CONFIG),
            GridMultiplierGrid(_GRID_MULT_CONFIG, _BOARD_CONFIG),
            [], [], _BOARD_CONFIG,
        ),
        ProgressTracker(
            _make_mock_signature(arc),
            100,
        ),
        arc,
    )

    assert obs.current_phase_id == "phase_01"
    assert obs.current_phase_index == 0


# ---------------------------------------------------------------------------
# RLA-031: reset() sets phase_repetition to 0
# ---------------------------------------------------------------------------


def test_rla_031_reset_phase_repetition_zero() -> None:
    """RLA-031: reset() starts with phase_repetition = 0."""
    arc = _make_arc()
    obs_builder = _make_obs_builder()

    progress = ProgressTracker(_make_mock_signature(arc), 100)
    obs = obs_builder.build(
        BoardContext.from_board(
            Board.empty(_BOARD_CONFIG),
            GridMultiplierGrid(_GRID_MULT_CONFIG, _BOARD_CONFIG),
            [], [], _BOARD_CONFIG,
        ),
        progress,
        arc,
    )

    assert obs.phase_repetition == 0


# ---------------------------------------------------------------------------
# RLA-039: ObservationBuilder symbol indices match SymbolConfig.standard order
# ---------------------------------------------------------------------------


def test_rla_039_symbol_indices_match_config() -> None:
    """RLA-039: ObservationBuilder symbol indices match SymbolConfig.standard order."""
    obs_builder = _make_obs_builder()
    board = Board.empty(_BOARD_CONFIG)
    # Place L1 at (0,0) and H3 at (1,0)
    board.set(Position(0, 0), Symbol.L1)
    board.set(Position(1, 0), Symbol.H3)

    grid_mults = GridMultiplierGrid(_GRID_MULT_CONFIG, _BOARD_CONFIG)
    context = BoardContext.from_board(board, grid_mults, [], [], _BOARD_CONFIG)
    arc = _make_arc()
    progress = ProgressTracker(_make_mock_signature(arc), 100)

    obs = obs_builder.build(context, progress, arc)

    # L1 is index 0 in standard ("L1", "L2", ...)
    assert obs.board_symbols[0][0] == 0
    # H3 is index 6 in standard
    assert obs.board_symbols[1][0] == 6


# ---------------------------------------------------------------------------
# RLA-041: Terminal action produces TERMINAL_DEAD intent
# ---------------------------------------------------------------------------


def test_rla_041_terminal_action_produces_terminal_intent() -> None:
    """RLA-041: Terminal action produces StepType.TERMINAL intent."""
    from ..config.schema import BoosterConfig, SpawnThreshold
    from ..primitives.booster_rules import BoosterRules

    booster_config = BoosterConfig(
        spawn_thresholds=(SpawnThreshold(booster="W", min_size=7, max_size=49),),
        spawn_order=("W",),
        rocket_tie_orientation="H",
        bomb_blast_radius=1,
        immune_to_rocket=(),
        immune_to_bomb=(),
        chain_initiators=(),
    )
    # Use MagicMock for cluster_builder since terminal doesn't use it
    mock_builder = MagicMock()

    interpreter = ActionInterpreter(_SYMBOL_CONFIG, _BOARD_CONFIG, mock_builder)
    terminal_action = CascadeAction(
        cluster_symbol_index=0,
        cluster_size=5,
        centroid_col=3,
        centroid_row=3,
        is_terminal=True,
    )

    context = MagicMock()
    progress = MagicMock()
    hints = VarianceHints(
        spatial_bias={}, symbol_weights={},
        near_miss_symbol_preference=(), cluster_size_preference=(),
    )
    rng = random.Random(42)

    intent = interpreter.interpret(terminal_action, context, progress, hints, rng)

    assert intent.step_type == StepType.TERMINAL_DEAD
    assert intent.is_terminal is True
    # ClusterBuilder should not have been called for terminal
    mock_builder.find_positions.assert_not_called()


# ---------------------------------------------------------------------------
# RLA-043: Observation includes correct phases_remaining count
# ---------------------------------------------------------------------------


def test_rla_043_phases_remaining_correct() -> None:
    """RLA-043: Observation includes correct phases_remaining count."""
    phases = (
        _make_phase(phase_id="p1"),
        _make_phase(phase_id="p2"),
        _make_phase(phase_id="p3"),
    )
    arc = _make_arc(phases=phases)
    obs_builder = _make_obs_builder()

    # At phase index 0, phases_remaining = 3 - 0 - 1 = 2
    progress = ProgressTracker(_make_mock_signature(arc), 100)
    obs = obs_builder.build(
        BoardContext.from_board(
            Board.empty(_BOARD_CONFIG),
            GridMultiplierGrid(_GRID_MULT_CONFIG, _BOARD_CONFIG),
            [], [], _BOARD_CONFIG,
        ),
        progress,
        arc,
    )
    assert obs.phases_remaining == 2

    # Advance to phase 1 → phases_remaining = 1
    progress.current_phase_index = 1
    obs2 = obs_builder.build(
        BoardContext.from_board(
            Board.empty(_BOARD_CONFIG),
            GridMultiplierGrid(_GRID_MULT_CONFIG, _BOARD_CONFIG),
            [], [], _BOARD_CONFIG,
        ),
        progress,
        arc,
    )
    assert obs2.phases_remaining == 1


# ---------------------------------------------------------------------------
# RLA-005 (encoder): ObservationEncoder board output shape
# ---------------------------------------------------------------------------


def test_observation_encoder_board_shape() -> None:
    """ObservationEncoder produces correct board tensor shape."""
    encoder = ObservationEncoder(_SYMBOL_CONFIG, _BOARD_CONFIG)
    obs_builder = _make_obs_builder()
    arc = _make_arc()
    progress = ProgressTracker(_make_mock_signature(arc), 100)
    obs = obs_builder.build(
        BoardContext.from_board(
            Board.empty(_BOARD_CONFIG),
            GridMultiplierGrid(_GRID_MULT_CONFIG, _BOARD_CONFIG),
            [], [], _BOARD_CONFIG,
        ),
        progress,
        arc,
    )

    board_tensor = encoder.encode_board(obs)
    # Channels = 7 symbols + 4 (multiplier, empty, wild, booster) = 11
    assert board_tensor.shape == (11, 7, 7)


def test_observation_encoder_scalar_shape() -> None:
    """ObservationEncoder produces correct scalar feature vector."""
    encoder = ObservationEncoder(_SYMBOL_CONFIG, _BOARD_CONFIG)
    obs_builder = _make_obs_builder()
    arc = _make_arc()
    progress = ProgressTracker(_make_mock_signature(arc), 100)
    obs = obs_builder.build(
        BoardContext.from_board(
            Board.empty(_BOARD_CONFIG),
            GridMultiplierGrid(_GRID_MULT_CONFIG, _BOARD_CONFIG),
            [], [], _BOARD_CONFIG,
        ),
        progress,
        arc,
    )

    scalars = encoder.encode_scalars(obs)
    assert scalars.shape == (6,)


# ---------------------------------------------------------------------------
# RLA-006 (action): CascadeAction is frozen
# ---------------------------------------------------------------------------


def test_cascade_action_is_frozen() -> None:
    """CascadeAction is immutable."""
    import dataclasses
    action = CascadeAction(
        cluster_symbol_index=0, cluster_size=5,
        centroid_col=3, centroid_row=3, is_terminal=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.cluster_size = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helper: mock signature with narrative arc
# ---------------------------------------------------------------------------


def _make_mock_signature(arc: NarrativeArc) -> MagicMock:
    """Create a mock ArchetypeSignature with the given narrative arc."""
    from ..pipeline.protocols import Range, RangeFloat

    sig = MagicMock()
    sig.id = "test_archetype"
    sig.family = "test"
    sig.criteria = "basegame"
    sig.narrative_arc = arc
    sig.payout_range = RangeFloat(0.0, 10.0)
    sig.required_cascade_depth = Range(1, 5)
    sig.required_booster_spawns = {}
    sig.required_booster_fires = {}
    sig.required_chain_depth = Range(0, 0)
    return sig
