"""Circular reel strip primitive — conventional-slot-style board population.

A reel strip is a CSV where each column is a reel (left-to-right) and each
row is a strip position. On spin, each reel picks a stop index; the visible
board window is read downward from that stop, wrapping circularly to the
top of the column when it reaches the end. When cascades explode cells and
gravity creates vacancies, refill symbols are drawn by advancing the per-reel
cursor upward (above the initial window), also wrapping circularly.

SRP: this module owns the immutable strip data, the single wrapping function
(`read_circular`), and the per-spin cursor state. The cursor is the only
mutable object — the strip itself is frozen and shared across spins.

Why one wrapping function: modular arithmetic on strip indices is identical
for both the initial window read and every refill call. Keeping it in
`read_circular` means no duplicated `i % strip_length` elsewhere.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..config.schema import BoardConfig, ConfigValidationError
from .symbols import Symbol, symbol_from_name


@dataclass(frozen=True, slots=True)
class ReelStrip:
    """Immutable circular reel strip loaded from CSV.

    columns[reel][i] = Symbol at strip index i for that reel.
    All columns share the same strip_length (validated at load).
    Access wraps circularly via read_circular (no direct indexing).
    """

    columns: tuple[tuple[Symbol, ...], ...]
    num_reels: int
    strip_length: int


def load_reel_strip(path: Path | str, board_config: BoardConfig) -> ReelStrip:
    """Parse a reel-strip CSV and validate against board dimensions.

    CSV layout: rows are strip positions, columns are reels (column count
    must equal board_config.num_reels). All columns must be equal length and
    at least num_rows deep (otherwise the initial board window is undefined).
    """
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]

    if not rows:
        raise ConfigValidationError(
            "reel_strip.csv_path", f"empty CSV: {csv_path}",
        )

    num_reels = board_config.num_reels
    # Per-reel symbol lists — built by transposing the row-major CSV.
    columns: list[list[Symbol]] = [[] for _ in range(num_reels)]
    for row_index, row in enumerate(rows):
        if len(row) != num_reels:
            raise ConfigValidationError(
                "reel_strip.csv_path",
                f"row {row_index} has {len(row)} columns, expected {num_reels}",
            )
        for reel_index, cell in enumerate(row):
            columns[reel_index].append(symbol_from_name(cell.strip()))

    strip_length = len(columns[0])
    for reel_index, column in enumerate(columns):
        if len(column) != strip_length:
            raise ConfigValidationError(
                "reel_strip.csv_path",
                f"reel {reel_index} length {len(column)} differs from reel 0 length {strip_length}",
            )

    if strip_length < board_config.num_rows:
        raise ConfigValidationError(
            "reel_strip.csv_path",
            f"strip_length {strip_length} < num_rows {board_config.num_rows}",
        )

    return ReelStrip(
        columns=tuple(tuple(col) for col in columns),
        num_reels=num_reels,
        strip_length=strip_length,
    )


def read_circular(
    strip: ReelStrip, reel: int, start: int, count: int,
) -> tuple[Symbol, ...]:
    """Read `count` symbols from a reel starting at `start`, wrapping circularly.

    This is the single location for modular arithmetic over strip indices —
    both the initial board read (downward from stop) and every refill call
    (upward from cursor, count provided as positive, caller advances start
    by subtracting count) route through here.
    """
    length = strip.strip_length
    column = strip.columns[reel]
    # Normalize `start` so negative cursors (from upward advancement) wrap too.
    return tuple(column[(start + offset) % length] for offset in range(count))


class ReelStripCursor:
    """Mutable per-reel cursor into a circular reel strip.

    Created once per spin with the initial stop positions. The initial board
    window consumes num_rows symbols per reel (downward from stop). The
    refill cursor starts one position *above* the stop (upward), and each
    refill call reads `count` symbols upward then advances the cursor by
    `count`. All wrapping delegates to `read_circular` — the cursor stores
    only an integer index per reel.

    Upward direction: the next symbol to appear at the top of the board
    comes from strip index (stop - 1), then (stop - 2), and so on. We read
    upward by reading `count` symbols ending at the current cursor, which is
    equivalent to reading from `cursor - count + 1` forward. Keeping the
    indexing expressed through `read_circular` avoids duplicating modular
    arithmetic here.
    """

    __slots__ = ("_strip", "_cursors", "_num_rows")

    def __init__(
        self,
        strip: ReelStrip,
        stop_positions: tuple[int, ...],
        num_rows: int,
    ) -> None:
        if len(stop_positions) != strip.num_reels:
            raise ValueError(
                f"stop_positions has {len(stop_positions)} entries, "
                f"expected {strip.num_reels}",
            )
        self._strip = strip
        self._num_rows = num_rows
        # Refill cursor starts one above the top of the visible window.
        # The top of the window is `stop` (strip index for board row 0);
        # the next upward symbol is at (stop - 1), which is this cursor's
        # initial value. `read_circular` handles negative values via modulo.
        self._cursors: list[int] = [stop - 1 for stop in stop_positions]

    def initial_board(self) -> tuple[tuple[Symbol, ...], ...]:
        """Read the initial num_rows window for each reel from its stop.

        The cursor's current value is `stop - 1`; the window starts one
        position below the cursor (the stop itself). Returning
        `read_circular(..., start=cursor + 1, count=num_rows)` keeps all
        modular arithmetic inside `read_circular`.
        """
        return tuple(
            read_circular(
                self._strip,
                reel=reel,
                start=cursor + 1,
                count=self._num_rows,
            )
            for reel, cursor in enumerate(self._cursors)
        )

    def refill(self, reel: int, count: int) -> tuple[Symbol, ...]:
        """Draw next `count` symbols for a reel, advancing cursor upward.

        Returned order: first element falls to the *top* of the refill
        region (the highest empty row). The refill strategy assigns these
        top-down: the first symbol here → the lowest row index among the
        empty cells (which becomes the top of the board after gravity).

        The batch spans cursor, cursor-1, ..., cursor-(count-1). We express
        this as a single read from `cursor - count + 1` going forward,
        reversed so that index `cursor` is first in the returned tuple.
        """
        if count <= 0:
            return ()
        start = self._cursors[reel] - count + 1
        batch = read_circular(self._strip, reel=reel, start=start, count=count)
        # Advance cursor upward by `count` positions for the next refill.
        self._cursors[reel] -= count
        return tuple(reversed(batch))
