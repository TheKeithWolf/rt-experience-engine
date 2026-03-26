"""Tests for booster board placement fix — non-wild boosters must occupy their cell.

T-FIX-001 through T-FIX-004 verify that place_booster() writes all booster types
to the board, preventing gravity holes that cascade into payout drift.
"""

from __future__ import annotations

import pytest

from ..boosters.state_machine import BoosterState
from ..boosters.tracker import BoosterTracker
from ..config.schema import BoardConfig, GridMultiplierConfig, MasterConfig
from ..primitives.board import Board, Position
from ..primitives.booster_rules import place_booster
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.symbols import Symbol


# ---------------------------------------------------------------------------
# T-FIX-001: Non-wild booster written to board AND registered in tracker
# ---------------------------------------------------------------------------

class TestPlaceBoosterNonWild:
    """T-FIX-001: place_booster with a non-wild symbol occupies the board cell
    and registers the booster in the tracker as DORMANT."""

    def test_rocket_on_board_and_in_tracker(
        self, default_config: MasterConfig, empty_board: Board,
    ) -> None:
        tracker = BoosterTracker(default_config.board)
        pos = Position(5, 0)

        place_booster(
            Symbol.R, pos, empty_board, tracker,
            orientation="H", source_cluster_index=0,
        )

        assert empty_board.get(pos) is Symbol.R
        instance = tracker.get_at(pos)
        assert instance is not None
        assert instance.booster_type is Symbol.R
        assert instance.state is BoosterState.DORMANT
        assert instance.orientation == "H"

    def test_bomb_on_board_and_in_tracker(
        self, default_config: MasterConfig, empty_board: Board,
    ) -> None:
        tracker = BoosterTracker(default_config.board)
        pos = Position(3, 3)

        place_booster(Symbol.B, pos, empty_board, tracker)

        assert empty_board.get(pos) is Symbol.B
        assert tracker.get_at(pos) is not None


# ---------------------------------------------------------------------------
# T-FIX-002: Wild booster written to board but NOT registered in tracker
# ---------------------------------------------------------------------------

class TestPlaceBoosterWild:
    """T-FIX-002: place_booster with Symbol.W places on board only — wilds are
    regular symbols for cluster purposes and don't need tracker management."""

    def test_wild_on_board_not_in_tracker(
        self, default_config: MasterConfig, empty_board: Board,
    ) -> None:
        tracker = BoosterTracker(default_config.board)
        pos = Position(2, 2)

        place_booster(Symbol.W, pos, empty_board, tracker)

        assert empty_board.get(pos) is Symbol.W
        assert tracker.get_at(pos) is None


# ---------------------------------------------------------------------------
# T-FIX-003: Board has no gravity hole at booster spawn position
# ---------------------------------------------------------------------------

class TestNoGravityHole:
    """T-FIX-003: After place_booster, the spawn position is occupied — gravity
    will not treat it as a vacancy."""

    def test_spawn_position_not_empty(
        self, default_config: MasterConfig, empty_board: Board,
    ) -> None:
        tracker = BoosterTracker(default_config.board)
        pos = Position(5, 0)

        place_booster(
            Symbol.R, pos, empty_board, tracker, orientation="V",
        )

        # The cell must be occupied — not None, which would create a gravity hole
        assert empty_board.get(pos) is not None


# ---------------------------------------------------------------------------
# T-FIX-004: GridMultiplierGrid.nonzero_positions() diagnostic helper
# ---------------------------------------------------------------------------

class TestNonzeroPositions:
    """T-FIX-004: nonzero_positions returns only incremented cells with correct values."""

    def test_empty_grid_returns_nothing(self, default_config: MasterConfig) -> None:
        grid = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)
        assert grid.nonzero_positions() == []

    def test_incremented_positions_returned(self, default_config: MasterConfig) -> None:
        grid = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)

        grid.increment(Position(1, 1))
        grid.increment(Position(3, 4))

        result = grid.nonzero_positions()
        positions = {pos: val for pos, val in result}

        assert len(positions) == 2
        assert positions[Position(1, 1)] == default_config.grid_multiplier.first_hit_value
        assert positions[Position(3, 4)] == default_config.grid_multiplier.first_hit_value

    def test_double_increment_shows_accumulated_value(
        self, default_config: MasterConfig,
    ) -> None:
        grid = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)

        grid.increment(Position(2, 2))
        grid.increment(Position(2, 2))

        result = grid.nonzero_positions()
        positions = {pos: val for pos, val in result}

        expected = (
            default_config.grid_multiplier.first_hit_value
            + default_config.grid_multiplier.increment
        )
        assert positions[Position(2, 2)] == expected
