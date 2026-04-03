"""Tests for EventTracer — board/grid state tracking and visual rendering.

Verifies the tracer correctly renders:
- Side-by-side board+grid view in WIN_INFO
- Board grids at each gravity cascade sub-step
- New event types: boosterSpawnInfo, boosterArmInfo, boosterFireInfo
"""

from __future__ import annotations

import pytest

from ..config.schema import MasterConfig
from ..output.book_record import BookRecord
from ..output.event_types import (
    BOOSTER_ARM_INFO,
    BOOSTER_FIRE_INFO,
    BOOSTER_SPAWN_INFO,
    GRAVITY_SETTLE,
    REVEAL,
    UPDATE_BOARD_MULTIPLIERS,
    WIN_INFO,
)
from ..tracer.tracer import EventTracer


# ---------------------------------------------------------------------------
# Helpers — build minimal event payloads for tracer testing
# ---------------------------------------------------------------------------

def _make_board_grid(symbols: list[list[str]]) -> list[list[dict]]:
    """Build board_grid[reel][row] from a list of symbol name columns."""
    return [[{"name": name} for name in reel] for reel in symbols]


def _make_zero_grid(num_reels: int, num_rows: int) -> list[list[int]]:
    """Build an all-zero grid multiplier matrix."""
    return [[0] * num_rows for _ in range(num_reels)]


def _make_reveal_event(
    board_grid: list[list[dict]],
    board_mults: list[list[int]] | None = None,
    idx: int = 0,
) -> dict:
    event: dict = {
        "index": idx,
        "type": REVEAL,
        "board": board_grid,
        "gameType": "basegame",
    }
    if board_mults is not None:
        event["boardMultipliers"] = board_mults
    else:
        # Default: all zeros matching board dimensions
        event["boardMultipliers"] = _make_zero_grid(len(board_grid), len(board_grid[0]))
    return event


def _make_update_board_multipliers_event(
    changes: list[dict],
    idx: int = 1,
) -> dict:
    return {
        "index": idx,
        "type": UPDATE_BOARD_MULTIPLIERS,
        "boardMultipliers": changes,
    }


def _make_win_info_event(
    wins: list[dict],
    total_win: int = 0,
    idx: int = 2,
) -> dict:
    return {
        "index": idx,
        "type": WIN_INFO,
        "totalWin": total_win,
        "wins": wins,
    }


def _make_gravity_settle_event(
    move_steps: list[list[dict]],
    new_symbols: list[list[dict]],
    idx: int = 5,
) -> dict:
    return {
        "index": idx,
        "type": GRAVITY_SETTLE,
        "moveSteps": move_steps,
        "newSymbols": new_symbols,
    }


def _trace_events(
    tracer: EventTracer,
    events: list[dict],
    criteria: str = "basegame",
) -> str:
    """Build a BookRecord from events and trace to string."""
    book = BookRecord(
        id=0,
        payoutMultiplier=100,
        events=tuple(events),
        criteria=criteria,
        baseGameWins=1.0,
        freeGameWins=0.0,
    )
    return tracer.trace_to_string(book)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 3x3 board for compact tests (reels x rows)
SMALL_SYMBOLS = [
    ["H1", "H2", "H3"],  # reel 0
    ["L1", "L2", "L3"],  # reel 1
    ["H4", "H1", "L1"],  # reel 2
]


@pytest.fixture
def tracer(default_config: MasterConfig) -> EventTracer:
    return EventTracer(default_config)


# ---------------------------------------------------------------------------
# Board/grid state caching
# ---------------------------------------------------------------------------

class TestStateCaching:

    def test_reveal_caches_board_state(
        self, tracer: EventTracer, default_config: MasterConfig,
    ) -> None:
        """REVEAL event should populate _board_state for downstream renderers."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        events = [_make_reveal_event(board_grid)]
        _trace_events(tracer, events)

        assert tracer._board_state
        assert tracer._board_state[0][0]["name"] == "H1"
        assert tracer._board_state[2][2]["name"] == "L1"

    def test_reveal_caches_grid_state_from_board_multipliers(
        self, tracer: EventTracer, default_config: MasterConfig,
    ) -> None:
        """REVEAL event should populate _grid_state from boardMultipliers."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        grid = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        events = [_make_reveal_event(board_grid, board_mults=grid)]
        _trace_events(tracer, events)

        assert tracer._grid_state == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    def test_update_board_multipliers_applies_sparse_delta(
        self, tracer: EventTracer, default_config: MasterConfig,
    ) -> None:
        """updateBoardMultipliers applies sparse changes to cached grid state."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        initial_grid = _make_zero_grid(3, 3)
        events = [
            _make_reveal_event(board_grid, board_mults=initial_grid),
            _make_update_board_multipliers_event([
                {"multiplier": 1, "position": {"reel": 0, "row": 2}},
                {"multiplier": 1, "position": {"reel": 1, "row": 1}},
            ]),
        ]
        _trace_events(tracer, events)

        assert tracer._grid_state[0][2] == 1
        assert tracer._grid_state[1][1] == 1
        assert tracer._grid_state[0][0] == 0  # unchanged

    def test_trace_resets_state(
        self, tracer: EventTracer, default_config: MasterConfig,
    ) -> None:
        """Each trace() call should start with fresh state."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        events = [_make_reveal_event(board_grid)]
        _trace_events(tracer, events)
        assert tracer._board_state  # populated

        # Trace again with no events — state should reset
        _trace_events(tracer, [])
        assert tracer._board_state == []
        assert tracer._grid_state == []


# ---------------------------------------------------------------------------
# WIN INFO side-by-side board + grid
# ---------------------------------------------------------------------------

class TestWinInfoSideBySide:

    def test_win_info_renders_board_with_winners(
        self, tracer: EventTracer,
    ) -> None:
        """WIN INFO should show *XX* markers on winning cluster positions."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        grid = _make_zero_grid(3, 3)

        wins = [{
            "basePayout": 50,
            "clusterPayout": 50,
            "clusterMultiplier": 1,
            "clusterSize": 2,
            "overlay": {"reel": 0, "row": 0},
            "cluster": {
                "cells": [
                    {"symbol": "H1", "reel": 0, "row": 0, "multiplier": 0},
                    {"symbol": "H1", "reel": 2, "row": 1, "multiplier": 0},
                ],
            },
        }]

        events = [
            _make_reveal_event(board_grid, board_mults=grid),
            _make_win_info_event(wins, total_win=50),
        ]
        output = _trace_events(tracer, events)

        # Board side should have winners highlighted
        assert "*H1*" in output

    def test_win_info_renders_grid_with_touched_markers(
        self, tracer: EventTracer,
    ) -> None:
        """WIN INFO grid should show [N] for touched positions, plain N for prior."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        # Grid with some prior values already set
        grid = [[0, 0, 0], [0, 2, 0], [1, 0, 0]]

        # Winner at (2, 0) — should show [1] since grid has value 1 there
        wins = [{
            "basePayout": 10,
            "clusterPayout": 10,
            "clusterMultiplier": 1,
            "clusterSize": 1,
            "overlay": {"reel": 2, "row": 0},
            "cluster": {
                "cells": [
                    {"symbol": "H4", "reel": 2, "row": 0, "multiplier": 1},
                ],
            },
        }]

        events = [
            _make_reveal_event(board_grid, board_mults=grid),
            _make_win_info_event(wins, total_win=10),
        ]
        output = _trace_events(tracer, events)

        # Touched position (2, 0) with value 1 → bracketed [1]
        assert "[ 1]" in output
        # Prior position (1, 1) with value 2 → plain (not bracketed)
        assert " 2 " in output

    def test_win_info_shows_side_by_side_headers(
        self, tracer: EventTracer,
    ) -> None:
        """Side-by-side view should include descriptive headers."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        grid = _make_zero_grid(3, 3)

        wins = [{
            "basePayout": 10,
            "clusterPayout": 10,
            "clusterMultiplier": 1,
            "clusterSize": 1,
            "overlay": {"reel": 0, "row": 0},
            "cluster": {
                "cells": [
                    {"symbol": "H1", "reel": 0, "row": 0, "multiplier": 0},
                ],
            },
        }]

        events = [
            _make_reveal_event(board_grid, board_mults=grid),
            _make_win_info_event(wins, total_win=10),
        ]
        output = _trace_events(tracer, events)

        assert "Board with winners [*]:" in output
        assert "Grid multipliers touched [x]:" in output


# ---------------------------------------------------------------------------
# GRAVITY SETTLE board grids (no explodingSymbols)
# ---------------------------------------------------------------------------

class TestGravitySettle:

    def _setup_board_and_gravity(
        self,
        tracer: EventTracer,
        move_steps: list[list[dict]],
        new_symbols: list[list[dict]],
    ) -> str:
        """Set up a REVEAL + GRAVITY_SETTLE sequence and return trace output."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        events = [
            _make_reveal_event(board_grid),
            _make_gravity_settle_event(move_steps, new_symbols),
        ]
        return _trace_events(tracer, events)

    def test_gravity_pass_shows_moved_symbols(self, tracer: EventTracer) -> None:
        """GRAVITY pass should render board with moved positions highlighted."""
        move_steps = [[
            {"fromCell": {"reel": 0, "row": 1}, "toCell": {"reel": 0, "row": 2}},
            {"fromCell": {"reel": 0, "row": 0}, "toCell": {"reel": 0, "row": 1}},
        ]]
        output = self._setup_board_and_gravity(tracer, move_steps, [[]])

        assert "Pass 1: GRAVITY (2 move(s))" in output
        # Direction arrow for straight down
        assert "↓" in output

    def test_gravity_pass_includes_symbol_names(self, tracer: EventTracer) -> None:
        """GRAVITY move lines should include the symbol name being moved."""
        move_steps = [[
            {"fromCell": {"reel": 0, "row": 1}, "toCell": {"reel": 0, "row": 2}},
        ]]
        output = self._setup_board_and_gravity(tracer, move_steps, [[]])

        # H2 is at (reel=0, row=1) in SMALL_SYMBOLS
        assert "H2" in output

    def test_refill_renders_new_symbols(self, tracer: EventTracer) -> None:
        """REFILL step should list new symbols with spec format."""
        new_symbols = [[{"symbol": "L4", "position": {"reel": 0, "row": 0}}]]
        output = self._setup_board_and_gravity(tracer, [], new_symbols)

        assert "REFILL (1 new symbol(s) from reel strip)" in output
        assert "L4" in output
        assert "(R0,row0)" in output

    def test_settle_renders_final_board(self, tracer: EventTracer) -> None:
        """SETTLE step should render the complete settled board."""
        new_symbols = [[{"symbol": "L4", "position": {"reel": 0, "row": 0}}]]
        output = self._setup_board_and_gravity(tracer, [], new_symbols)

        assert "SETTLE" in output
        lines = output.split("\n")
        settle_idx = next(
            i for i, l in enumerate(lines) if l.strip() == "SETTLE"
        )
        # Board header (column labels) follows the SETTLE line
        assert "R0" in lines[settle_idx + 1]

    def test_settle_updates_board_state(self, tracer: EventTracer) -> None:
        """After GRAVITY_SETTLE, _board_state should reflect the settled board."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        new_symbols = [[{"symbol": "L4", "position": {"reel": 0, "row": 0}}]]
        events = [
            _make_reveal_event(board_grid),
            _make_gravity_settle_event([], new_symbols),
        ]
        _trace_events(tracer, events)

        # Position (0, 0) was refilled with L4
        assert tracer._board_state[0][0]["name"] == "L4"

    def test_multi_pass_gravity(self, tracer: EventTracer) -> None:
        """Multiple gravity passes should each render a labeled step."""
        pass1 = [
            {"fromCell": {"reel": 0, "row": 1}, "toCell": {"reel": 0, "row": 2}},
        ]
        pass2 = [
            {"fromCell": {"reel": 0, "row": 0}, "toCell": {"reel": 0, "row": 1}},
        ]
        output = self._setup_board_and_gravity(tracer, [pass1, pass2], [[]])

        assert "Pass 1: GRAVITY" in output
        assert "Pass 2: GRAVITY" in output


# ---------------------------------------------------------------------------
# Side-by-side formatter
# ---------------------------------------------------------------------------

class TestFormatSideBySide:

    def test_equal_height_columns(self, tracer: EventTracer) -> None:
        left = ["AAA", "BBB"]
        right = ["XXX", "YYY"]
        result = tracer._format_side_by_side(left, right, "Left:", "Right:")

        assert "Left:" in result[0]
        assert "Right:" in result[0]
        assert len(result) == 3  # header + 2 body lines

    def test_unequal_height_pads_shorter(self, tracer: EventTracer) -> None:
        left = ["AAA"]
        right = ["XXX", "YYY", "ZZZ"]
        result = tracer._format_side_by_side(left, right, "L:", "R:")

        # Should have header + 3 body lines (padded to max height)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Grid multiplier formatter touched vs prior
# ---------------------------------------------------------------------------

class TestFormatGridMults:

    def test_no_touched_brackets_all(self, tracer: EventTracer) -> None:
        """Without touched param, all non-zero values get brackets."""
        grid = [[0, 1], [2, 0]]
        lines = tracer._format_grid_mults(grid)
        content = "\n".join(lines)
        assert "[ 1]" in content
        assert "[ 2]" in content

    def test_touched_brackets_only_touched(self, tracer: EventTracer) -> None:
        """With touched param, only touched positions get brackets."""
        grid = [[0, 1], [2, 0]]
        touched = {(0, 1)}  # only (reel=0, row=1) is touched
        lines = tracer._format_grid_mults(grid, touched=touched)
        content = "\n".join(lines)
        # (0, 1) has value 1 and is touched → bracketed
        assert "[ 1]" in content
        # (1, 0) has value 2 but NOT touched → plain
        assert " 2 " in content


# ---------------------------------------------------------------------------
# Booster spawn/arm/fire board-state sync
# ---------------------------------------------------------------------------


class TestBoosterSpawnBoardStateSync:

    def test_spawn_updates_board_state(self, tracer: EventTracer) -> None:
        """boosterSpawnInfo at (1,2) should write symbol into _board_state[1][2]."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        events = [
            _make_reveal_event(board_grid),
            {
                "type": BOOSTER_SPAWN_INFO,
                "index": 1,
                "boosters": [
                    {"symbol": "W", "position": {"reel": 1, "row": 2}},
                ],
            },
        ]
        _trace_events(tracer, events)

        assert tracer._board_state[1][2]["name"] == "W"

    def test_spawn_multiple_types(self, tracer: EventTracer) -> None:
        """boosterSpawnInfo with multiple boosters updates all positions."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        events = [
            _make_reveal_event(board_grid),
            {
                "type": BOOSTER_SPAWN_INFO,
                "index": 1,
                "boosters": [
                    {"symbol": "W", "position": {"reel": 0, "row": 0}},
                    {"symbol": "R", "position": {"reel": 2, "row": 1}},
                ],
            },
        ]
        _trace_events(tracer, events)

        assert tracer._board_state[0][0]["name"] == "W"
        assert tracer._board_state[2][1]["name"] == "R"


class TestBoosterArmRendering:

    def test_arm_info_renders(self, tracer: EventTracer) -> None:
        """boosterArmInfo should render BOOSTER ARM section."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        events = [
            _make_reveal_event(board_grid),
            {
                "type": BOOSTER_ARM_INFO,
                "index": 1,
                "boosters": [
                    {"symbol": "R", "position": {"reel": 3, "row": 2}},
                ],
            },
        ]
        output = _trace_events(tracer, events)

        assert "BOOSTER ARM" in output
        assert "R ARMED" in output


class TestBoosterFireRendering:

    def test_fire_info_renders(self, tracer: EventTracer) -> None:
        """boosterFireInfo should render BOOSTER FIRE section."""
        board_grid = _make_board_grid(SMALL_SYMBOLS)
        events = [
            _make_reveal_event(board_grid),
            {
                "type": BOOSTER_FIRE_INFO,
                "index": 1,
                "boosters": [
                    {
                        "symbol": "R",
                        "clearedCells": [
                            {"symbol": "L2", "position": {"reel": 0, "row": 0}},
                            {"symbol": "L2", "position": {"reel": 0, "row": 1}},
                        ],
                    },
                ],
            },
        ]
        output = _trace_events(tracer, events)

        assert "BOOSTER FIRE" in output
        assert "R cleared 2 cells" in output
