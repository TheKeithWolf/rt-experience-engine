"""Tests for the cascade pipeline — Phase 6 deliverables.

Covers TEST-P6-001: PostGravityMapping constraint correctness.
Covers TEST-P6-009 through P6-011: Cascade depth counts winning steps only.

Tests P6-002 through P6-008 have been removed: they depended on the deleted
sequence_planner module.  CascadeInstanceGenerator.generate() now raises
NotImplementedError until the StepReasoner is built.
"""

from __future__ import annotations

import pytest

from ..archetypes.dead import register_dead_archetypes
from ..archetypes.registry import ArchetypeRegistry
from ..archetypes.tier1 import register_cascade_t1_archetypes, register_static_t1_archetypes
from ..config.schema import MasterConfig
from ..pipeline.cascade_generator import CascadeInstanceGenerator
from ..pipeline.step_validator import StepValidator, StepValidationFailed
from ..primitives.board import Board, Position
from ..primitives.gravity import GravityDAG
from ..primitives.symbols import Symbol
from ..spatial_solver.constraints import PostGravityMapping
from ..spatial_solver.data_types import (
    ClusterAssignment,
    SolverContext,
)
from ..step_reasoner.progress import ClusterRecord, ProgressTracker
from ..step_reasoner.results import StepResult
from ..validation.validator import InstanceValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cascade_registry(default_config: MasterConfig) -> ArchetypeRegistry:
    """Registry with dead + all t1 archetypes (static and cascade)."""
    reg = ArchetypeRegistry(default_config)
    register_dead_archetypes(reg)
    register_static_t1_archetypes(reg)
    register_cascade_t1_archetypes(reg)
    return reg


@pytest.fixture
def gravity_dag(default_config: MasterConfig) -> GravityDAG:
    return GravityDAG(default_config.board, default_config.gravity)


@pytest.fixture
def cascade_generator(
    default_config: MasterConfig,
    cascade_registry: ArchetypeRegistry,
    gravity_dag: GravityDAG,
) -> CascadeInstanceGenerator:
    return CascadeInstanceGenerator(
        default_config, cascade_registry, gravity_dag,
    )


@pytest.fixture
def validator(
    default_config: MasterConfig,
    cascade_registry: ArchetypeRegistry,
) -> InstanceValidator:
    return InstanceValidator(default_config, cascade_registry)


# ---------------------------------------------------------------------------
# TEST-P6-001: PostGravityMapping — step 1 positions within predicted empties
# ---------------------------------------------------------------------------


def test_post_gravity_mapping_accepts_valid_positions(
    default_config: MasterConfig,
) -> None:
    """Placements within the allowed set pass the constraint."""
    allowed = frozenset({
        Position(0, 0), Position(0, 1), Position(1, 0), Position(1, 1), Position(2, 0),
    })
    constraint = PostGravityMapping(allowed)

    context = SolverContext(default_config.board)
    # Place a cluster entirely within allowed positions
    context.clusters.append(ClusterAssignment(
        symbol=Symbol.L1,
        positions=frozenset({Position(0, 0), Position(0, 1), Position(1, 0),
                             Position(1, 1), Position(2, 0)}),
        size=5,
    ))
    context.occupied.update(context.clusters[0].positions)
    assert constraint.is_satisfied(context) is True


def test_post_gravity_mapping_rejects_out_of_bounds(
    default_config: MasterConfig,
) -> None:
    """Placements outside the allowed set fail the constraint."""
    allowed = frozenset({Position(0, 0), Position(0, 1), Position(1, 0)})
    constraint = PostGravityMapping(allowed)

    context = SolverContext(default_config.board)
    # Place a cluster with one position outside allowed
    context.clusters.append(ClusterAssignment(
        symbol=Symbol.L1,
        positions=frozenset({Position(0, 0), Position(0, 1), Position(1, 0),
                             Position(1, 1), Position(2, 0)}),
        size=5,
    ))
    assert constraint.is_satisfied(context) is False


def test_post_gravity_mapping_checks_scatter_positions(
    default_config: MasterConfig,
) -> None:
    """Scatter positions outside allowed set fail the constraint."""
    allowed = frozenset({Position(0, 0)})
    constraint = PostGravityMapping(allowed)

    context = SolverContext(default_config.board)
    context.scatter_positions.add(Position(3, 3))  # Not in allowed
    assert constraint.is_satisfied(context) is False


# ---------------------------------------------------------------------------
# Helpers for cascade depth validation tests
# ---------------------------------------------------------------------------


def _make_dead_board(config: MasterConfig) -> Board:
    """Board with alternating symbols that produces zero clusters."""
    board = Board.empty(config.board)
    symbols = [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4]
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            idx = (reel * 2 + row) % len(symbols)
            board.set(Position(reel, row), symbols[idx])
    return board


def _make_winning_step(step_index: int, centipayout: int = 30) -> StepResult:
    """StepResult with one winning cluster — represents a real cascade event."""
    return StepResult(
        step_index=step_index,
        clusters=(
            ClusterRecord(
                symbol=Symbol.L2,
                size=5,
                positions=frozenset({
                    Position(0, 0), Position(0, 1), Position(1, 0),
                    Position(1, 1), Position(2, 0),
                }),
                step_index=step_index,
                payout=centipayout,
            ),
        ),
        spawns=(),
        fires=(),
        symbol_tier=None,
        step_payout=centipayout,
    )


def _make_terminal_step(step_index: int) -> StepResult:
    """StepResult with zero clusters — terminal dead board, not a cascade."""
    return StepResult(
        step_index=step_index,
        clusters=(),
        spawns=(),
        fires=(),
        symbol_tier=None,
        step_payout=0,
    )


# ---------------------------------------------------------------------------
# TEST-P6-009: Cascade depth counts only winning steps (not terminal)
# ---------------------------------------------------------------------------


def test_cascade_depth_counts_only_winning_steps(
    default_config: MasterConfig,
    cascade_registry: ArchetypeRegistry,
) -> None:
    """1 winning step + 1 terminal = cascade depth 1, passes Range(1, 1)."""
    sig = cascade_registry.get("t1_cascade_1")
    step_validator = StepValidator(default_config)
    progress = ProgressTracker(sig, default_config.centipayout.multiplier)

    winning = _make_winning_step(step_index=0, centipayout=30)
    terminal = _make_terminal_step(step_index=1)

    progress.update(winning)
    progress.update(terminal)

    dead_board = _make_dead_board(default_config)
    # Must not raise — cascade depth is 1 (the winning step), not 2 (total steps)
    instance = step_validator.validate_instance(
        [winning, terminal], sig, dead_board, progress,
    )
    assert instance.payout == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# TEST-P6-010: Cascade depth below minimum is rejected
# ---------------------------------------------------------------------------


def test_cascade_depth_below_minimum_rejected(
    default_config: MasterConfig,
    cascade_registry: ArchetypeRegistry,
) -> None:
    """0 winning steps (only terminal) fails Range(1, 1) — no cascade occurred.

    Payout check fires first (0 payout < 0.2 minimum), so we verify
    that validation rejects this degenerate case. The cascade depth
    check would also fail (depth 0 outside [1, 1]) if payout passed.
    """
    sig = cascade_registry.get("t1_cascade_1")
    step_validator = StepValidator(default_config)
    progress = ProgressTracker(sig, default_config.centipayout.multiplier)

    terminal = _make_terminal_step(step_index=0)
    progress.update(terminal)

    dead_board = _make_dead_board(default_config)
    with pytest.raises(StepValidationFailed, match="payout"):
        step_validator.validate_instance(
            [terminal], sig, dead_board, progress,
        )


# ---------------------------------------------------------------------------
# TEST-P6-011: Two winning steps + terminal = cascade depth 2
# ---------------------------------------------------------------------------


def test_cascade_depth_two_winning_steps(
    default_config: MasterConfig,
    cascade_registry: ArchetypeRegistry,
) -> None:
    """2 winning steps + 1 terminal = cascade depth 2, passes Range(2, 2)."""
    sig = cascade_registry.get("t1_cascade_2")
    step_validator = StepValidator(default_config)
    progress = ProgressTracker(sig, default_config.centipayout.multiplier)

    winning_0 = _make_winning_step(step_index=0, centipayout=25)
    winning_1 = _make_winning_step(step_index=1, centipayout=25)
    terminal = _make_terminal_step(step_index=2)

    progress.update(winning_0)
    progress.update(winning_1)
    progress.update(terminal)

    dead_board = _make_dead_board(default_config)
    instance = step_validator.validate_instance(
        [winning_0, winning_1, terminal], sig, dead_board, progress,
    )
    assert instance.payout == pytest.approx(0.5)
