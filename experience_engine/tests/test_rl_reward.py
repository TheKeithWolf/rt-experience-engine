"""Tests for rl_archive/reward.py — phase-aware reward function.

RLA-050 through RLA-058: Phase match reward, feasibility, terminal
bonus/penalty, weight disabling, and determinism.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ..config.schema import BoardConfig, EnvironmentConfig, RewardConfig
from ..narrative.arc import NarrativePhase
from ..pipeline.protocols import Range
from ..primitives.board import Position
from ..primitives.symbols import Symbol
from ..rl_archive.reward import PhaseRewardComputer
from ..step_reasoner.results import ClusterRecord, FireRecord, SpawnRecord, StepResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BOARD_CONFIG = BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5)
_ENV_CONFIG = EnvironmentConfig(
    max_episode_steps=15,
    invalid_step_penalty=-1.0,
    completion_bonus=5.0,
    failure_penalty=-3.0,
    feasibility_weight=0.4,
    progress_weight=0.3,
)
_REWARD_CONFIG = RewardConfig(
    phase_match_reward=1.0,
    cluster_match_reward=0.5,
    spawn_match_reward=0.5,
    fire_match_reward=2.0,
    wild_behavior_match_reward=0.5,
    feasibility_empty_cell_weight=0.3,
    feasibility_adjacency_weight=0.7,
)


def _make_phase(
    spawns: tuple[str, ...] | None = None,
    fires: tuple[str, ...] | None = None,
    wild_behavior: str | None = None,
    cluster_count: Range = Range(1, 2),
) -> NarrativePhase:
    return NarrativePhase(
        id="p1", intent="test",
        repetitions=Range(1, 3),
        cluster_count=cluster_count,
        cluster_sizes=(Range(5, 10),),
        cluster_symbol_tier=None,
        spawns=spawns, arms=None, fires=fires,
        wild_behavior=wild_behavior,
        ends_when="always",
    )


def _make_step_result(
    clusters: int = 1,
    spawns: tuple[SpawnRecord, ...] = (),
    fires: tuple[FireRecord, ...] = (),
    step_payout: int = 100,
) -> StepResult:
    cluster_records = tuple(
        ClusterRecord(
            symbol=Symbol.L1, size=5,
            positions=frozenset(Position(0, i) for i in range(5)),
            step_index=0, payout=100,
        )
        for _ in range(clusters)
    )
    return StepResult(
        step_index=0, clusters=cluster_records,
        spawns=spawns, fires=fires,
        symbol_tier=None, step_payout=step_payout,
    )


def _make_context(
    active_wilds: list | None = None,
    dormant_boosters: list | None = None,
) -> MagicMock:
    ctx = MagicMock()
    ctx.active_wilds = active_wilds or []
    ctx.dormant_boosters = dormant_boosters or []
    ctx.neighbors_of.return_value = (Position(0, 1), Position(1, 0))
    ctx.empty_neighbors_of.return_value = [Position(0, 1)]
    return ctx


def _make_computer(
    reward_config: RewardConfig | None = None,
) -> PhaseRewardComputer:
    return PhaseRewardComputer(
        reward_config or _REWARD_CONFIG, _ENV_CONFIG, _BOARD_CONFIG,
    )


# ---------------------------------------------------------------------------
# RLA-050: Spawn match reward
# ---------------------------------------------------------------------------


def test_rla_050_spawn_match_reward() -> None:
    """RLA-050: Step matching spawn phase with correct spawn yields spawn_match_reward."""
    phase = _make_phase(spawns=("R",))
    spawn = SpawnRecord(booster_type="R", position=Position(3, 3),
                        source_cluster_index=0, step_index=0)
    step = _make_step_result(spawns=(spawn,))
    computer = _make_computer()

    reward = computer.compute(step, phase, _make_context(), done=False, arc_satisfied=False)
    # Should include spawn_match_reward (0.5 * 1.0 = 0.5) plus cluster_match
    assert reward > 0.0


# ---------------------------------------------------------------------------
# RLA-051: Wild behavior match
# ---------------------------------------------------------------------------


def test_rla_051_wild_behavior_match() -> None:
    """RLA-051: Step matching bridge phase with wild spawn yields wild_behavior_match_reward."""
    phase = _make_phase(wild_behavior="spawn")
    spawn = SpawnRecord(booster_type="W", position=Position(3, 3),
                        source_cluster_index=0, step_index=0)
    step = _make_step_result(spawns=(spawn,))
    computer = _make_computer()

    reward = computer.compute(step, phase, _make_context(), done=False, arc_satisfied=False)
    assert reward > 0.0


# ---------------------------------------------------------------------------
# RLA-052: No phase constraint match yields zero phase reward
# ---------------------------------------------------------------------------


def test_rla_052_no_match_zero_phase_reward() -> None:
    """RLA-052: Step not matching any phase constraint yields zero phase reward."""
    # Phase expects 0 clusters (terminal-like) but step has 1
    phase = _make_phase(cluster_count=Range(0, 0))
    step = _make_step_result(clusters=1)
    context = _make_context()
    # No wilds/boosters → zero feasibility
    context.active_wilds = []
    context.dormant_boosters = []

    computer = _make_computer()
    reward = computer.compute(step, phase, context, done=False, arc_satisfied=False)

    # cluster_count doesn't match, no spawns/fires/wild_behavior constraints
    # Feasibility is 0 since no wilds/boosters
    assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RLA-053: Feasibility = 1.0 when all cells adjacent to wild are empty
# ---------------------------------------------------------------------------


def test_rla_053_feasibility_all_empty() -> None:
    """RLA-053: High feasibility when all cells adjacent to wild are empty."""
    context = _make_context(active_wilds=[Position(3, 3)])
    # All neighbors are empty
    context.neighbors_of.return_value = (Position(2, 3), Position(4, 3))
    context.empty_neighbors_of.return_value = [Position(2, 3), Position(4, 3)]

    computer = _make_computer()
    # Use None phase so only feasibility contributes
    reward = computer.compute(
        _make_step_result(), None, context, done=False, arc_satisfied=False,
    )
    # Feasibility should be positive (empty_fraction=1.0)
    assert reward > 0.0


# ---------------------------------------------------------------------------
# RLA-054: Feasibility = 0.0 when no adjacent cells are empty
# ---------------------------------------------------------------------------


def test_rla_054_feasibility_none_empty() -> None:
    """RLA-054: Zero feasibility when no adjacent cells are empty."""
    context = _make_context(active_wilds=[Position(3, 3)])
    context.neighbors_of.return_value = (Position(2, 3), Position(4, 3))
    context.empty_neighbors_of.return_value = []

    computer = _make_computer()
    # Phase match should still contribute; check feasibility portion
    # With no phase, only feasibility matters
    reward = computer.compute(
        _make_step_result(), None, context, done=False, arc_satisfied=False,
    )
    # adjacency_ratio > 0 (2/49) but empty_fraction = 0
    # reward = (0.3 * 0.0 + 0.7 * (2/49)) * 0.4
    assert reward >= 0.0
    assert reward < 0.1  # Very small from adjacency ratio


# ---------------------------------------------------------------------------
# RLA-055: Completion applies completion_bonus
# ---------------------------------------------------------------------------


def test_rla_055_completion_bonus() -> None:
    """RLA-055: Terminal + arc_satisfied applies completion_bonus."""
    computer = _make_computer()
    reward = computer.compute(
        None, None, _make_context(), done=True, arc_satisfied=True,
    )
    assert reward == pytest.approx(_ENV_CONFIG.completion_bonus)


# ---------------------------------------------------------------------------
# RLA-056: Failure applies failure_penalty
# ---------------------------------------------------------------------------


def test_rla_056_failure_penalty() -> None:
    """RLA-056: Terminal + not arc_satisfied applies failure_penalty."""
    computer = _make_computer()
    reward = computer.compute(
        None, None, _make_context(), done=True, arc_satisfied=False,
    )
    assert reward == pytest.approx(_ENV_CONFIG.failure_penalty)


# ---------------------------------------------------------------------------
# RLA-057: Setting weight to 0.0 disables component
# ---------------------------------------------------------------------------


def test_rla_057_zero_weight_disables() -> None:
    """RLA-057: Setting a reward weight to 0.0 disables that component."""
    # All weights zero except completion bonus
    zero_config = RewardConfig(
        phase_match_reward=0.0,
        cluster_match_reward=0.0,
        spawn_match_reward=0.0,
        fire_match_reward=0.0,
        wild_behavior_match_reward=0.0,
        feasibility_empty_cell_weight=0.0,
        feasibility_adjacency_weight=0.0,
    )
    computer = _make_computer(zero_config)

    phase = _make_phase(spawns=("R",))
    spawn = SpawnRecord(booster_type="R", position=Position(3, 3),
                        source_cluster_index=0, step_index=0)
    step = _make_step_result(spawns=(spawn,))

    reward = computer.compute(step, phase, _make_context(), done=False, arc_satisfied=False)
    # All phase + feasibility weights are zero, not done → zero reward
    assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RLA-058: Reward is deterministic
# ---------------------------------------------------------------------------


def test_rla_058_deterministic() -> None:
    """RLA-058: Reward is deterministic given identical inputs."""
    phase = _make_phase(spawns=("R",))
    spawn = SpawnRecord(booster_type="R", position=Position(3, 3),
                        source_cluster_index=0, step_index=0)
    step = _make_step_result(spawns=(spawn,))
    context = _make_context(active_wilds=[Position(3, 3)])
    computer = _make_computer()

    r1 = computer.compute(step, phase, context, done=False, arc_satisfied=False)
    r2 = computer.compute(step, phase, context, done=False, arc_satisfied=False)
    assert r1 == r2
