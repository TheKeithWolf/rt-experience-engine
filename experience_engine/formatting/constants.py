"""ASCII geometry constants derived from game config — single source of truth.

All grid rendering across the codebase derives cell widths, borders, and
separators from these values. Changing the Symbol enum (e.g. adding a longer
name) automatically updates all formatting.
"""

from ..primitives.symbols import Symbol

# Derived from longest symbol name (e.g. "SLB") — controls all grid geometry
CELL_NAME_WIDTH: int = max(len(s.name) for s in Symbol)

# Total content width inside | delimiters — name + 1 char padding for decoration
CELL_CONTENT_WIDTH: int = CELL_NAME_WIDTH + 1

# Row border segment per cell — produces "----+" for CELL_CONTENT_WIDTH=4
CELL_BORDER: str = "-" * CELL_CONTENT_WIDTH + "+"

# Section separators matching spec visual vocabulary
MAJOR_SEP: str = "=" * 60
MINOR_SEP: str = "-" * 60
