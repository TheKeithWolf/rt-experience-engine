"""Reel strip cursor + refill strategy tests (TEST-REEL-010 through 018)."""

from __future__ import annotations

import random
from pathlib import Path

from ..config.schema import BoardConfig
from ..pipeline.refill_strategy import RefillStrategy
from ..pipeline.reel_refill import ReelStripRefill
from ..primitives.board import Board, Position
from ..primitives.reel_strip import (
    ReelStripCursor,
    load_reel_strip,
    read_circular,
)

BOARD_CONFIG = BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5)
REFERENCE_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "reel_strip.csv"
)


def _strip():
    return load_reel_strip(REFERENCE_CSV, BOARD_CONFIG)


def _all_zero_stops(strip) -> tuple[int, ...]:
    return tuple(0 for _ in range(strip.num_reels))


# ---------------------------------------------------------------------------
# ReelStripCursor
# ---------------------------------------------------------------------------

class TestCursorInitialBoard:

    def test_reel_010_initial_board_no_wrap(self) -> None:
        """TEST-REEL-010: initial_board at stop=0 returns first 7 rows verbatim."""
        strip = _strip()
        cursor = ReelStripCursor(strip, _all_zero_stops(strip), num_rows=7)
        board = cursor.initial_board()

        assert len(board) == 7  # one tuple per reel
        for reel, column in enumerate(board):
            assert column == strip.columns[reel][:7]

    def test_reel_011_initial_board_wraps_at_stop_seven(self) -> None:
        """TEST-REEL-011: stop=7 wraps (indices 7,8,0,1,2,3,4)."""
        strip = _strip()
        stops = tuple(7 for _ in range(strip.num_reels))
        cursor = ReelStripCursor(strip, stops, num_rows=7)
        board = cursor.initial_board()

        for reel, column in enumerate(board):
            col = strip.columns[reel]
            expected = (col[7], col[8], col[0], col[1], col[2], col[3], col[4])
            assert column == expected


class TestCursorRefill:

    def test_reel_012_refill_returns_symbols_above_window(self) -> None:
        """TEST-REEL-012: refill(count=2) at stop=3 → strip indices 2, 1."""
        strip = _strip()
        stops = tuple(3 for _ in range(strip.num_reels))
        cursor = ReelStripCursor(strip, stops, num_rows=7)
        cursor.initial_board()  # stop is independent of initial-board read

        got = cursor.refill(reel=0, count=2)
        col = strip.columns[0]
        # First refill symbol = strip[stop-1] (closest to stop, enters deepest);
        # second = strip[stop-2] (further up the strip).
        assert got == (col[2], col[1])

    def test_reel_013_refill_advances_cursor(self) -> None:
        """TEST-REEL-013: sequential refills draw disjoint strip indices."""
        strip = _strip()
        stops = tuple(3 for _ in range(strip.num_reels))
        cursor = ReelStripCursor(strip, stops, num_rows=7)

        first = cursor.refill(reel=0, count=2)
        second = cursor.refill(reel=0, count=3)
        col = strip.columns[0]
        # cursor after first refill = (3 - 1) - 2 = 0
        # first batch spans indices 2, 1 → returned in that order
        assert first == (col[2], col[1])
        # second batch spans indices 0, -1 (=8), -2 (=7) → returned 0, 8, 7
        assert second == (col[0], col[8], col[7])

    def test_reel_014_cursor_full_wrap(self) -> None:
        """TEST-REEL-014: after strip_length symbols drawn, cursor returns to start."""
        strip = _strip()
        stops = tuple(3 for _ in range(strip.num_reels))
        cursor = ReelStripCursor(strip, stops, num_rows=7)

        drawn_first_lap = cursor.refill(reel=0, count=strip.strip_length)
        drawn_second_lap = cursor.refill(reel=0, count=strip.strip_length)
        # After a full lap the batch repeats (same strip positions, same symbols).
        assert drawn_first_lap == drawn_second_lap


# ---------------------------------------------------------------------------
# ReelStripRefill
# ---------------------------------------------------------------------------

class TestReelStripRefill:

    def _cursor(self, stop: int = 3) -> ReelStripCursor:
        strip = _strip()
        stops = tuple(stop for _ in range(strip.num_reels))
        return ReelStripCursor(strip, stops, num_rows=7)

    def test_reel_015_fill_output_shape(self) -> None:
        """TEST-REEL-015: fill returns tuple[(reel, row, name), ...] matching protocol."""
        refill = ReelStripRefill(self._cursor())
        board = Board.empty(BOARD_CONFIG)
        empties = [Position(0, 0), Position(0, 1)]

        result = refill.fill(board, empties, random.Random(0))

        assert isinstance(result, tuple)
        assert len(result) == 2
        for entry in result:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            reel, row, name = entry
            assert isinstance(reel, int)
            assert isinstance(row, int)
            assert isinstance(name, str)

    def test_reel_016_deepest_empty_gets_first_refill_symbol(self) -> None:
        """TEST-REEL-016: first refill symbol lands at the deepest empty row.

        Physical: the symbol closest to the previous stop is the first to
        cross the window boundary; after k symbols enter, it is at the
        deepest newly-vacant row.
        """
        strip = _strip()
        stops = tuple(3 for _ in range(strip.num_reels))
        cursor = ReelStripCursor(strip, stops, num_rows=7)
        refill = ReelStripRefill(cursor)
        board = Board.empty(BOARD_CONFIG)

        # Empties at rows 0 and 2 on reel 0 — row 2 is deepest.
        empties = [Position(0, 0), Position(0, 2)]
        result = refill.fill(board, empties, random.Random(0))

        col = strip.columns[0]
        # First refill symbol = col[2]; should be assigned to deepest (row 2).
        deepest_entry = next(e for e in result if e[1] == 2)
        top_entry = next(e for e in result if e[1] == 0)
        assert deepest_entry == (0, 2, col[2].name)
        assert top_entry == (0, 0, col[1].name)

    def test_reel_017_satisfies_refill_strategy_protocol(self) -> None:
        """TEST-REEL-017: ReelStripRefill is a RefillStrategy (runtime_checkable)."""
        refill = ReelStripRefill(self._cursor())
        assert isinstance(refill, RefillStrategy)

    def test_reel_018_deterministic_regardless_of_rng(self) -> None:
        """TEST-REEL-018: Identical cursor state → identical refill output, any rng."""
        board = Board.empty(BOARD_CONFIG)
        empties = [Position(0, 0), Position(1, 1), Position(2, 2)]

        refill_a = ReelStripRefill(self._cursor())
        refill_b = ReelStripRefill(self._cursor())

        result_a = refill_a.fill(board, empties, random.Random(0))
        result_b = refill_b.fill(board, empties, random.Random(999))

        assert result_a == result_b
