"""Side-by-side column layout — pure geometry, no domain coupling.

Used by WIN_INFO to show board next to grid, and by REVEAL for
side-by-side board + multipliers display.
"""

from __future__ import annotations


def format_side_by_side(
    left_lines: list[str],
    right_lines: list[str],
    left_header: str,
    right_header: str,
    *,
    gap: int = 4,
) -> list[str]:
    """Paste two column blocks side by side with headers and a gap.

    Pads shorter block with blank lines to match the taller block's height.
    """
    left_width = max((len(line) for line in left_lines), default=0)
    left_width = max(left_width, len(left_header))
    spacer = " " * gap

    lines: list[str] = []
    lines.append(f"{left_header:<{left_width}}{spacer}{right_header}")

    max_rows = max(len(left_lines), len(right_lines))
    for i in range(max_rows):
        l_line = left_lines[i] if i < len(left_lines) else ""
        r_line = right_lines[i] if i < len(right_lines) else ""
        lines.append(f"{l_line:<{left_width}}{spacer}{r_line}")

    return lines
