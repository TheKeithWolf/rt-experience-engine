"""Grid multiplier ASCII rendering — bordered table with three cell styles.

Spec visual vocabulary for multiplier cells:
- | 1  | — active from prior step (non-zero, not touched/updated)
- |[1] | — hit this step (bracketed, used in WIN_INFO touched positions)
- |*1* | — updated this step (asterisked, used in updateBoardMultipliers delta)
- |    | — zero / empty
"""

from __future__ import annotations

from .cells import CellStyle, format_cell
from .constants import CELL_BORDER, CELL_CONTENT_WIDTH


def format_grid_mults(
    grid: list[list[int]],
    num_reels: int,
    num_rows: int,
    *,
    touched: set[tuple[int, int]] | None = None,
    updated: set[tuple[int, int]] | None = None,
) -> list[str]:
    """Format grid multiplier matrix as bordered ASCII table.

    touched: positions hit this step — shown as [N] (bracketed)
    updated: positions changed this step — shown as *N* (asterisked)
    Neither: non-zero values shown as plain N (active from prior)
    Zero values: empty cell
    When touched is None and updated is None, all non-zero use bracket style.
    """
    if not grid:
        return ["  (empty grid)"]

    lines: list[str] = []

    # Column headers — same spacing as board formatter
    gap = " " * (CELL_CONTENT_WIDTH - 2)
    header = " " * (CELL_CONTENT_WIDTH + 1) + gap.join(f"R{r}" for r in range(num_reels))
    lines.append(header)

    # Top border
    border_row = "+" + CELL_BORDER * num_reels
    lines.append(border_row)

    # No distinction requested — bracket all non-zero values
    no_distinction = touched is None and updated is None

    for row in range(num_rows):
        cells: list[str] = []
        for reel in range(num_reels):
            val = grid[reel][row] if reel < len(grid) and row < len(grid[reel]) else 0

            if val > 0 and (no_distinction or (touched is not None and (reel, row) in touched)):
                # Bracketed — hit this step or no distinction requested
                cells.append(format_cell(str(val), CellStyle.ARMED))
            elif val > 0 and updated is not None and (reel, row) in updated:
                # Asterisked — updated this step (updateBoardMultipliers delta)
                cells.append(format_cell(str(val), CellStyle.WINNER))
            elif val > 0:
                # Plain — active from a prior step, not touched/updated this step
                cells.append(format_cell(str(val), CellStyle.REGULAR))
            else:
                # Empty — zero value
                cells.append(format_cell("", CellStyle.EMPTY))

        line = "|" + "|".join(cells) + f"|  row {row}"
        lines.append(line)

        # Row separator border
        lines.append(border_row)

    return lines
