"""Bordered ASCII board grid — single renderer for all consumers.

The formatter owns geometry (headers, borders, row labels). The caller owns
semantics (what each cell looks like) via the CellResolver callback — Strategy
pattern that replaces three independent _format_board() implementations.
"""

from __future__ import annotations

from collections.abc import Callable

from .cells import CellStyle, format_cell
from .constants import CELL_BORDER, CELL_CONTENT_WIDTH

# Callback: given (reel, row), return (display_name, CellStyle)
CellResolver = Callable[[int, int], tuple[str, CellStyle]]


def format_board_grid(
    num_reels: int,
    num_rows: int,
    resolve_cell: CellResolver,
    *,
    row_label: str = "row",
) -> list[str]:
    """Render any board as a bordered ASCII grid.

    The caller supplies resolve_cell(reel, row) which returns the display name
    and style for each position. This function handles geometry: headers,
    borders, row labels. Output matches spec's +----+ bordered format.
    """
    if num_reels == 0 or num_rows == 0:
        return ["  (empty board)"]

    lines: list[str] = []

    # Column headers — spacing derived from cell content width
    gap = " " * (CELL_CONTENT_WIDTH - 2)
    header = " " * (CELL_CONTENT_WIDTH + 1) + gap.join(f"R{r}" for r in range(num_reels))
    lines.append(header)

    # Top border
    border_row = "+" + CELL_BORDER * num_reels
    lines.append(border_row)

    for row in range(num_rows):
        # Cell content row
        cells: list[str] = []
        for reel in range(num_reels):
            name, style = resolve_cell(reel, row)
            cells.append(format_cell(name, style))

        line = "|" + "|".join(cells) + f"|  {row_label} {row}"
        lines.append(line)

        # Row separator border
        lines.append(border_row)

    return lines
