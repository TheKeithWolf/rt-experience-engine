"""TEST-P1-037 through TEST-P1-038: Grid multiplier increment and floor."""

from ..config.schema import MasterConfig
from ..primitives.board import Position
from ..primitives.grid_multipliers import GridMultiplierGrid


def test_p1_037_increment_sequence(default_config: MasterConfig) -> None:
    """TEST-P1-037: Grid multiplier: 0 → 1 → 2 → ... → cap (from config)."""
    grid = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)
    pos = Position(0, 0)
    cap = default_config.grid_multiplier.cap

    # Initially 0 (initial_value)
    assert grid.get(pos) == 0

    # First hit: 0 → 1 (first_hit_value)
    grid.increment(pos)
    assert grid.get(pos) == 1

    # Subsequent hits: +1 each (increment)
    grid.increment(pos)
    assert grid.get(pos) == 2

    grid.increment(pos)
    assert grid.get(pos) == 3

    # Drive to cap
    for _ in range(200):
        grid.increment(pos)
    assert grid.get(pos) == cap


def test_p1_038_minimum_contribution_floor(default_config: MasterConfig) -> None:
    """TEST-P1-038: position_multiplier_sum returns minimum_contribution when all 0."""
    grid = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)

    # All positions at initial_value (0)
    positions = [Position(0, 0), Position(1, 0), Position(2, 0)]
    result = grid.position_multiplier_sum(positions)

    # Sum is 0, but floored to minimum_contribution (1)
    assert result == default_config.grid_multiplier.minimum_contribution
    assert result == 1
