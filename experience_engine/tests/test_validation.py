"""Tests for the validation pipeline.

Covers TEST-P4-015 through P4-018.
"""

from __future__ import annotations

import pytest

from ..archetypes.dead import register_dead_archetypes
from ..archetypes.registry import ArchetypeRegistry
from ..archetypes.tier1 import register_static_t1_archetypes
from ..config.schema import MasterConfig
from ..pipeline.data_types import GeneratedInstance
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..spatial_solver.data_types import SpatialStep, ClusterAssignment
from ..validation.validator import InstanceValidator


@pytest.fixture
def full_registry(default_config: MasterConfig) -> ArchetypeRegistry:
    reg = ArchetypeRegistry(default_config)
    register_dead_archetypes(reg)
    register_static_t1_archetypes(reg)
    return reg


@pytest.fixture
def validator(
    default_config: MasterConfig,
    full_registry: ArchetypeRegistry,
) -> InstanceValidator:
    return InstanceValidator(default_config, full_registry)


def _make_dead_board(config: MasterConfig) -> Board:
    """Create a valid dead board — all standard symbols, no clusters >= 5."""
    board = Board.empty(config.board)
    # Fill with alternating L1/L2/L3/L4 pattern that prevents clusters
    symbols = [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4]
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            # Offset by both reel and row to prevent horizontal and vertical clusters
            idx = (reel * 2 + row) % len(symbols)
            board.set(Position(reel, row), symbols[idx])
    return board


def _make_dead_instance(
    config: MasterConfig,
    archetype_id: str = "dead_empty",
) -> GeneratedInstance:
    board = _make_dead_board(config)
    return GeneratedInstance(
        sim_id=0,
        archetype_id=archetype_id,
        family="dead",
        criteria="0",
        board=board,
        spatial_step=SpatialStep(
            clusters=(), near_misses=(), scatter_positions=frozenset(), boosters=(),
        ),
        payout=0.0,
        centipayout=0,
        win_level=1,
    )


# ---------------------------------------------------------------------------
# TEST-P4-015: Validation passes for correct dead_empty
# ---------------------------------------------------------------------------

def test_valid_dead_empty_passes(
    default_config: MasterConfig,
    validator: InstanceValidator,
) -> None:
    instance = _make_dead_instance(default_config, "dead_empty")
    metrics = validator.validate(instance)
    assert metrics.is_valid, f"Expected valid, got errors: {metrics.validation_errors}"
    assert metrics.payout == 0.0
    assert metrics.cluster_count == 0


# ---------------------------------------------------------------------------
# TEST-P4-016: Validation fails for dead board with accidental cluster
# ---------------------------------------------------------------------------

def test_dead_board_with_cluster_fails(
    default_config: MasterConfig,
    validator: InstanceValidator,
) -> None:
    """A dead board that accidentally contains a cluster should fail validation."""
    board = _make_dead_board(default_config)
    # Force a cluster of 5 L1 symbols in a row
    for col in range(5):
        board.set(Position(col, 0), Symbol.L1)

    instance = GeneratedInstance(
        sim_id=0,
        archetype_id="dead_empty",
        family="dead",
        criteria="0",
        board=board,
        spatial_step=SpatialStep(
            clusters=(), near_misses=(), scatter_positions=frozenset(), boosters=(),
        ),
        payout=0.0,
        centipayout=0,
        win_level=1,
    )
    metrics = validator.validate(instance)
    # Should fail because cluster_count > 0 for a dead archetype
    assert not metrics.is_valid
    assert any("cluster_count" in e for e in metrics.validation_errors)


# ---------------------------------------------------------------------------
# TEST-P4-017: Validation fails for payout outside range
# ---------------------------------------------------------------------------

def test_payout_outside_range_fails(
    default_config: MasterConfig,
    validator: InstanceValidator,
) -> None:
    """A t1 instance with clusters that produce payout outside the signature range should fail."""
    # Create a board with massive cluster payout that exceeds t1_single max (3.0x)
    board = Board.empty(default_config.board)
    # Fill with alternating to prevent other clusters
    symbols = [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4]
    for reel in range(default_config.board.num_reels):
        for row in range(default_config.board.num_rows):
            idx = (reel * 2 + row) % len(symbols)
            board.set(Position(reel, row), symbols[idx])

    # Place a large H3 cluster (15+ symbols = 25.0x payout)
    for reel in range(5):
        for row in range(3):
            board.set(Position(reel, row), Symbol.H3)

    cluster_positions = frozenset(
        Position(reel, row) for reel in range(5) for row in range(3)
    )
    instance = GeneratedInstance(
        sim_id=0,
        archetype_id="t1_single",
        family="t1",
        criteria="basegame",
        board=board,
        spatial_step=SpatialStep(
            clusters=(ClusterAssignment(
                symbol=Symbol.H3,
                positions=cluster_positions,
                size=15,
            ),),
            near_misses=(),
            scatter_positions=frozenset(),
            boosters=(),
        ),
        payout=25.0,
        centipayout=2500,
        win_level=7,
    )
    metrics = validator.validate(instance)
    assert not metrics.is_valid
    assert any("payout" in e for e in metrics.validation_errors)


# ---------------------------------------------------------------------------
# TEST-P4-018: Payout computation — centipayout conversion correct
# ---------------------------------------------------------------------------

def test_centipayout_conversion(
    default_config: MasterConfig,
    validator: InstanceValidator,
) -> None:
    """Verify centipayout conversion: 2.6x → 260, 0.1x → 10, 0.0x → 0."""
    from ..primitives.paytable import Paytable
    paytable = Paytable(
        default_config.paytable,
        default_config.centipayout,
        default_config.win_levels,
    )
    assert paytable.to_centipayout(2.6) == 260
    assert paytable.to_centipayout(0.1) == 10
    assert paytable.to_centipayout(0.0) == 0
