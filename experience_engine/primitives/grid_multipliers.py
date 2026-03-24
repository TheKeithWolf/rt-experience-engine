"""Grid position multiplier tracking.

Each board position has a multiplier that starts at initial_value (0), jumps
to first_hit_value (1) on first cluster participation, then increments by
config.increment on each subsequent hit, up to config.cap.

The position_multiplier_sum floors at minimum_contribution to ensure base
payout always applies even when the grid is inactive (all zeros).
"""

from __future__ import annotations

from collections.abc import Iterable

from ..config.schema import BoardConfig, GridMultiplierConfig
from .board import Position


class GridMultiplierGrid:
    """Mutable 2D grid tracking per-position multiplier values."""

    __slots__ = ("_grid", "_config", "_num_reels", "_num_rows")

    def __init__(self, config: GridMultiplierConfig, board_config: BoardConfig) -> None:
        self._config = config
        self._num_reels = board_config.num_reels
        self._num_rows = board_config.num_rows
        self._grid: list[list[int]] = [
            [config.initial_value] * board_config.num_rows
            for _ in range(board_config.num_reels)
        ]

    def get(self, pos: Position) -> int:
        """Current multiplier value at a position."""
        return self._grid[pos.reel][pos.row]

    def increment(self, pos: Position) -> None:
        """Advance the multiplier at a position by one step, capped at config.cap.

        First hit jumps from initial_value to first_hit_value.
        Subsequent hits add config.increment.
        """
        current = self._grid[pos.reel][pos.row]
        if current == self._config.initial_value:
            self._grid[pos.reel][pos.row] = self._config.first_hit_value
        else:
            self._grid[pos.reel][pos.row] = min(
                current + self._config.increment,
                self._config.cap,
            )

    def position_multiplier_sum(self, positions: Iterable[Position]) -> int:
        """Sum of multipliers at given positions, floored at minimum_contribution.

        The floor ensures base payout always applies even when all grid
        positions are at initial_value (zero).
        """
        total = sum(self._grid[p.reel][p.row] for p in positions)
        return max(total, self._config.minimum_contribution)

    def reset(self) -> None:
        """Reset all positions to initial_value."""
        for reel in range(self._num_reels):
            for row in range(self._num_rows):
                self._grid[reel][row] = self._config.initial_value

    def copy(self) -> GridMultiplierGrid:
        """Deep copy with independent grid state."""
        new = GridMultiplierGrid.__new__(GridMultiplierGrid)
        new._config = self._config
        new._num_reels = self._num_reels
        new._num_rows = self._num_rows
        new._grid = [row[:] for row in self._grid]
        return new
