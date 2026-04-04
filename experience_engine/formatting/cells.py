"""Cell decoration vocabulary — closed enum replaces scattered if/else chains.

Each CellStyle member defines (left_char, right_char) wrapping the symbol name.
Adding a new visual style requires only a new enum member — no existing code changes.
Matches RoyalTumble_ExperienceEngine_Tracer.md visual vocabulary.
"""

from enum import Enum

from .constants import CELL_CONTENT_WIDTH, CELL_NAME_WIDTH


class CellStyle(Enum):
    """Visual vocabulary for board cell decoration.

    Each member defines (left_char, right_char) wrapping the symbol name.
    Some styles share the same visual delimiters but carry different semantic
    meaning (e.g. SPAWNED vs WINNER both use asterisks, but represent
    different game states in the booster lifecycle).
    """

    REGULAR = (" ", " ")   # | H2 |
    WINNER = ("*", "*")    # |*H2*|
    SPAWNED = ("*", "*")   # |*RH*|  (newly spawned booster, highlighted)
    ARMED = ("[", "]")     # |[RH]|
    VACANCY = ("[", "]")   # |[  ]|
    EMPTY = (" ", " ")     # |    |


def format_cell(name: str, style: CellStyle) -> str:
    """Format a single cell with decoration. Width derived from constants.

    Returns exactly CELL_CONTENT_WIDTH chars. For decorated styles (non-space
    delimiters), name is right-aligned in CELL_NAME_WIDTH-1 to fit both
    left and right chars within CELL_CONTENT_WIDTH. For undecorated styles,
    name is right-aligned in CELL_NAME_WIDTH with left padding.
    """
    left, right = style.value
    # Decorated styles: left + name + right must fit in CELL_CONTENT_WIDTH
    # e.g. *H2* = 4 chars: * + H2(right-aligned in 2) + *
    # Undecorated styles: left + name fills CELL_CONTENT_WIDTH
    # e.g.  H2  = 4 chars: space + H2(right-aligned in 3)
    if left.strip():
        # Decorated — shrink name field to fit right delimiter
        inner_width = CELL_CONTENT_WIDTH - 2  # space for left + right
        return f"{left}{name:>{inner_width}}{right}"
    else:
        # Undecorated — left padding + right-aligned name fills width
        return f"{left}{name:>{CELL_NAME_WIDTH}}"


def format_vacancy() -> str:
    """Vacancy cell — exploded/cleared position shown as [  ]."""
    return format_cell("  ", CellStyle.VACANCY)


def format_empty() -> str:
    """Empty cell — no symbol, blank space."""
    return format_cell("  ", CellStyle.EMPTY)
