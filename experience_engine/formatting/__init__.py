"""Shared ASCII formatting — single source of truth for board/grid rendering.

Zero knowledge of events or tracer state. Consumers import specific functions.
"""

from .board_formatter import CellResolver, format_board_grid
from .cells import CellStyle, format_cell, format_empty, format_vacancy
from .constants import (
    CELL_BORDER,
    CELL_CONTENT_WIDTH,
    CELL_NAME_WIDTH,
    MAJOR_SEP,
    MINOR_SEP,
)
from .grid_formatter import format_grid_mults
from .layout import format_side_by_side

__all__ = [
    "CELL_BORDER",
    "CELL_CONTENT_WIDTH",
    "CELL_NAME_WIDTH",
    "CellResolver",
    "CellStyle",
    "MAJOR_SEP",
    "MINOR_SEP",
    "format_board_grid",
    "format_cell",
    "format_empty",
    "format_grid_mults",
    "format_side_by_side",
    "format_vacancy",
]
