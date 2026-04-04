"""Tests for the shared formatting package.

Verifies constants, cell styles, board formatter, grid formatter, and layout
produce correct ASCII output matching the spec visual vocabulary.
"""

from __future__ import annotations

import pytest

from ..formatting.board_formatter import format_board_grid
from ..formatting.cells import CellStyle, format_cell, format_empty, format_vacancy
from ..formatting.constants import CELL_BORDER, CELL_CONTENT_WIDTH, CELL_NAME_WIDTH
from ..formatting.grid_formatter import format_grid_mults
from ..formatting.layout import format_side_by_side


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_cell_name_width_matches_symbol_enum(self) -> None:
        """CELL_NAME_WIDTH should equal the longest Symbol name (SLB = 3)."""
        assert CELL_NAME_WIDTH == 3

    def test_cell_content_width_derived(self) -> None:
        """CELL_CONTENT_WIDTH = CELL_NAME_WIDTH + 1 for decoration padding."""
        assert CELL_CONTENT_WIDTH == CELL_NAME_WIDTH + 1

    def test_cell_border_matches_content_width(self) -> None:
        """Border segment should be dashes matching content width + plus sign."""
        assert CELL_BORDER == "----+"
        assert len(CELL_BORDER) == CELL_CONTENT_WIDTH + 1


# ---------------------------------------------------------------------------
# Cell styles
# ---------------------------------------------------------------------------

class TestCellStyles:

    @pytest.mark.parametrize("name,style,expected", [
        ("H2", CellStyle.REGULAR, "  H2"),    # space + right-aligned in 3
        ("H2", CellStyle.WINNER, "*H2*"),      # * + right-aligned in 2 + *
        ("RH", CellStyle.SPAWNED, "*RH*"),     # * + right-aligned in 2 + *
        ("RH", CellStyle.ARMED, "[RH]"),       # [ + right-aligned in 2 + ]
        ("S", CellStyle.WINNER, "* S*"),        # * + right-aligned in 2 + *
        ("SLB", CellStyle.REGULAR, " SLB"),    # space + right-aligned in 3
    ])
    def test_format_cell_styles(
        self, name: str, style: CellStyle, expected: str,
    ) -> None:
        result = format_cell(name, style)
        assert result == expected
        assert len(result) == CELL_CONTENT_WIDTH

    def test_format_vacancy(self) -> None:
        """Vacancy should render as [  ] (bracketed empty)."""
        result = format_vacancy()
        assert result == "[  ]"
        assert len(result) == CELL_CONTENT_WIDTH

    def test_format_empty(self) -> None:
        """Empty should render as blank space."""
        result = format_empty()
        assert len(result) == CELL_CONTENT_WIDTH


# ---------------------------------------------------------------------------
# Board formatter
# ---------------------------------------------------------------------------

class TestBoardFormatter:

    def test_bordered_output(self) -> None:
        """Board should have +----+ borders between rows."""
        def resolve(reel: int, row: int) -> tuple[str, CellStyle]:
            return "H2", CellStyle.REGULAR

        lines = format_board_grid(2, 2, resolve)
        content = "\n".join(lines)
        assert "+----+----+" in content

    def test_column_headers(self) -> None:
        """Board should have R0, R1, ... column headers."""
        def resolve(reel: int, row: int) -> tuple[str, CellStyle]:
            return "H1", CellStyle.REGULAR

        lines = format_board_grid(3, 1, resolve)
        header = lines[0]
        assert "R0" in header
        assert "R1" in header
        assert "R2" in header

    def test_row_labels(self) -> None:
        """Each row should be labeled with 'row N'."""
        def resolve(reel: int, row: int) -> tuple[str, CellStyle]:
            return "L1", CellStyle.REGULAR

        lines = format_board_grid(2, 3, resolve)
        content = "\n".join(lines)
        assert "row 0" in content
        assert "row 1" in content
        assert "row 2" in content

    def test_winner_highlighting(self) -> None:
        """Winner cells should use *XX* notation."""
        def resolve(reel: int, row: int) -> tuple[str, CellStyle]:
            if reel == 0 and row == 0:
                return "H1", CellStyle.WINNER
            return "L2", CellStyle.REGULAR

        lines = format_board_grid(2, 2, resolve)
        content = "\n".join(lines)
        assert "*H1*" in content
        assert "  L2" in content

    def test_empty_board(self) -> None:
        """Zero-dimension board should return placeholder."""
        def resolve(reel: int, row: int) -> tuple[str, CellStyle]:
            return "", CellStyle.EMPTY

        lines = format_board_grid(0, 0, resolve)
        assert lines == ["  (empty board)"]


# ---------------------------------------------------------------------------
# Grid formatter
# ---------------------------------------------------------------------------

class TestGridFormatter:

    def test_bordered_output(self) -> None:
        """Grid should have +----+ borders."""
        grid = [[1, 0], [0, 2]]
        lines = format_grid_mults(grid, 2, 2)
        content = "\n".join(lines)
        assert "+----+" in content

    def test_bracketed_all_when_no_distinction(self) -> None:
        """Without touched/updated, all non-zero values get brackets."""
        grid = [[0, 1], [2, 0]]
        lines = format_grid_mults(grid, 2, 2)
        content = "\n".join(lines)
        assert "[ 1]" in content
        assert "[ 2]" in content

    def test_touched_vs_untouched(self) -> None:
        """Touched positions get brackets, untouched get plain format."""
        grid = [[0, 1], [2, 0]]
        touched = {(0, 1)}
        lines = format_grid_mults(grid, 2, 2, touched=touched)
        content = "\n".join(lines)
        # Touched: bracketed
        assert "[ 1]" in content
        # Untouched: plain (no brackets around 2)
        assert content.count("[") == content.count("[ 1]")

    def test_updated_style(self) -> None:
        """Updated positions get asterisk style *N*."""
        grid = [[0, 1], [2, 0]]
        updated = {(0, 1)}
        lines = format_grid_mults(grid, 2, 2, updated=updated)
        content = "\n".join(lines)
        # Updated: asterisked
        assert "*" in content

    def test_empty_grid(self) -> None:
        """Empty grid returns placeholder."""
        lines = format_grid_mults([], 0, 0)
        assert lines == ["  (empty grid)"]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

class TestLayout:

    def test_side_by_side_headers(self) -> None:
        result = format_side_by_side(["AAA"], ["BBB"], "Left:", "Right:")
        assert "Left:" in result[0]
        assert "Right:" in result[0]

    def test_side_by_side_body_count(self) -> None:
        left = ["A1", "A2"]
        right = ["B1", "B2"]
        result = format_side_by_side(left, right, "L:", "R:")
        # Header + 2 body lines
        assert len(result) == 3

    def test_side_by_side_pads_shorter(self) -> None:
        left = ["A"]
        right = ["B1", "B2", "B3"]
        result = format_side_by_side(left, right, "L:", "R:")
        # Header + 3 body lines (padded)
        assert len(result) == 4
