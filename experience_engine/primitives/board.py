"""Board representation and spatial geometry functions.

Board uses (reel, row) coordinates where reel=column index (0=left),
row index (0=top). The board is mutable — solvers mutate it during
construction. Use Board.copy() for explicit cloning when needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config.schema import BoardConfig
from .symbols import Symbol


@dataclass(frozen=True, slots=True)
class Position:
    """A cell coordinate on the board. reel=column (0=left), row (0=top)."""

    reel: int
    row: int


class Board:
    """Mutable 2D grid of symbols with dimensions from config.

    grid[reel][row] — None indicates an empty cell (pre-fill or post-explosion).
    """

    __slots__ = ("_grid", "_num_reels", "_num_rows")

    def __init__(self, num_reels: int, num_rows: int) -> None:
        self._num_reels = num_reels
        self._num_rows = num_rows
        self._grid: list[list[Symbol | None]] = [
            [None] * num_rows for _ in range(num_reels)
        ]

    @classmethod
    def empty(cls, config: BoardConfig) -> Board:
        """Factory for a board with all cells empty (None)."""
        return cls(config.num_reels, config.num_rows)

    @property
    def num_reels(self) -> int:
        return self._num_reels

    @property
    def num_rows(self) -> int:
        return self._num_rows

    def get(self, pos: Position) -> Symbol | None:
        """Return the symbol at the given position, or None if empty."""
        return self._grid[pos.reel][pos.row]

    def set(self, pos: Position, symbol: Symbol | None) -> None:
        """Assign a symbol (or None) to the given position."""
        self._grid[pos.reel][pos.row] = symbol

    def copy(self) -> Board:
        """Deep copy — new board with independent grid state."""
        new_board = Board(self._num_reels, self._num_rows)
        for reel in range(self._num_reels):
            for row in range(self._num_rows):
                new_board._grid[reel][row] = self._grid[reel][row]
        return new_board

    def board_hash(self) -> int:
        """Content-based hash for deduplication — different boards produce different hashes."""
        return hash(tuple(
            tuple(self._grid[reel])
            for reel in range(self._num_reels)
        ))

    def empty_positions(self) -> list[Position]:
        """All positions where the cell is None, in reel-major order."""
        result: list[Position] = []
        for reel in range(self._num_reels):
            for row in range(self._num_rows):
                if self._grid[reel][row] is None:
                    result.append(Position(reel, row))
        return result

    def all_positions(self) -> list[Position]:
        """All valid positions on the board, in reel-major order."""
        return [
            Position(reel, row)
            for reel in range(self._num_reels)
            for row in range(self._num_rows)
        ]


def is_valid(pos: Position, config: BoardConfig) -> bool:
    """True if position is within board bounds."""
    return 0 <= pos.reel < config.num_reels and 0 <= pos.row < config.num_rows


def orthogonal_neighbors(pos: Position, config: BoardConfig) -> tuple[Position, ...]:
    """Return up/down/left/right neighbors within board bounds."""
    candidates = (
        Position(pos.reel - 1, pos.row),  # left
        Position(pos.reel + 1, pos.row),  # right
        Position(pos.reel, pos.row - 1),  # up
        Position(pos.reel, pos.row + 1),  # down
    )
    return tuple(p for p in candidates if is_valid(p, config))


def manhattan_distance(a: Position, b: Position) -> int:
    """Manhattan (L1) distance between two positions."""
    return abs(a.reel - b.reel) + abs(a.row - b.row)


def euclidean_distance(a: Position, b: Position) -> float:
    """Euclidean (L2) distance between two positions."""
    return math.sqrt((a.reel - b.reel) ** 2 + (a.row - b.row) ** 2)
