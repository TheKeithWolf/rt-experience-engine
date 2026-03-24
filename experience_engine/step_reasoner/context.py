"""Board context — observable state snapshot for the step reasoner.

DormantBooster is a lightweight record of an unfired booster, decoupled from
the full BoosterInstance lifecycle (no state machine dependency).

BoardContext wraps the current board into a query-friendly facade with
cached derived properties. Created fresh each step via from_board() —
treat as read-only after construction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property

from ..config.schema import BoardConfig
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.symbols import Symbol


@dataclass(frozen=True, slots=True)
class DormantBooster:
    """A booster on the board that has not yet fired.

    Lightweight snapshot for the reasoner — decoupled from the full
    BoosterInstance state machine in boosters.tracker.
    """

    booster_type: str          # "R", "B", "LB", "SLB"
    position: Position
    orientation: str | None    # "H"/"V" for rockets, None for others
    spawned_step: int          # Cascade step when this booster was spawned


@dataclass
class BoardContext:
    """Observable board state for the step reasoner.

    NOT frozen — uses @cached_property for lazy derived attributes.
    Created once per step, not hot-path — __dict__ overhead is acceptable.

    Do not mutate the board after cached properties have been accessed;
    the cached values will be stale. Create a new BoardContext instead.
    """

    board: Board
    grid_multipliers: GridMultiplierGrid
    dormant_boosters: list[DormantBooster]
    active_wilds: list[Position]
    _board_config: BoardConfig

    @cached_property
    def empty_cells(self) -> list[Position]:
        """Positions with no symbol — available for gravity refill."""
        return self.board.empty_positions()

    @cached_property
    def surviving_symbols(self) -> dict[Position, Symbol]:
        """All occupied positions and their symbols."""
        result: dict[Position, Symbol] = {}
        for reel in range(self.board.num_reels):
            for row in range(self.board.num_rows):
                pos = Position(reel, row)
                sym = self.board.get(pos)
                if sym is not None:
                    result[pos] = sym
        return result

    @cached_property
    def symbol_counts(self) -> dict[Symbol, int]:
        """Frequency of each symbol currently on the board."""
        return dict(Counter(self.surviving_symbols.values()))

    def neighbors_of(self, pos: Position) -> tuple[Position, ...]:
        """Orthogonal neighbors within board bounds.

        Delegates to primitives.board.orthogonal_neighbors — single
        source of truth for adjacency geometry.
        """
        return orthogonal_neighbors(pos, self._board_config)

    def empty_neighbors_of(self, pos: Position) -> list[Position]:
        """Orthogonal neighbors that are currently empty (None)."""
        return [n for n in self.neighbors_of(pos) if self.board.get(n) is None]

    @classmethod
    def from_board(
        cls,
        board: Board,
        grid_multipliers: GridMultiplierGrid,
        dormant_boosters: list[DormantBooster],
        active_wilds: list[Position],
        board_config: BoardConfig,
    ) -> BoardContext:
        """Factory that captures current board state as a context snapshot."""
        return cls(
            board=board,
            grid_multipliers=grid_multipliers,
            dormant_boosters=dormant_boosters,
            active_wilds=active_wilds,
            _board_config=board_config,
        )
