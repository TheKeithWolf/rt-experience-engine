"""Reel-strip refill strategy.

Adapts `ReelStripCursor` to the `RefillStrategy` protocol so the existing
gravity-record / event-stream machinery consumes reel-strip refills without
special-casing. The strip is the sole source of symbols — the `rng`
parameter is accepted for protocol compatibility and intentionally unused.

Physical model: when a reel advances by k strip positions, the k new
symbols enter at the top of the visible window. The *first* new symbol
(closest to the previous stop) ends up at the *deepest* newly-vacant row;
the *last* (furthest up the strip) ends up at row 0. This module groups
empty positions by reel, sorts each reel's empties row-descending, and
zips with the cursor's refill batch so that index 0 of the batch lands at
the deepest empty row and subsequent symbols stack upward.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from ..primitives.board import Board, Position
from ..primitives.reel_strip import ReelStripCursor


class ReelStripRefill:
    """Refill strategy that draws symbols from a reel strip cursor.

    Satisfies the `RefillStrategy` protocol. One cursor is owned per spin;
    callers construct a fresh refill each spin from a freshly-seeded cursor.
    """

    __slots__ = ("_cursor",)

    def __init__(self, cursor: ReelStripCursor) -> None:
        self._cursor = cursor

    def fill(
        self,
        board: Board,
        empty_positions: Iterable[Position],
        rng: random.Random,
    ) -> tuple[tuple[int, int, str], ...]:
        by_reel: dict[int, list[Position]] = defaultdict(list)
        for pos in empty_positions:
            by_reel[pos.reel].append(pos)

        if not by_reel:
            return ()

        result: list[tuple[int, int, str]] = []
        # Iterate reels in ascending order so the output is deterministic
        # across refill calls — event-stream consumers compare these tuples.
        for reel in sorted(by_reel):
            positions = by_reel[reel]
            # Deepest empty row (highest row index) receives the first
            # refill symbol — the one just above the previous stop on the
            # strip. Symbols further up the strip stack above it.
            positions.sort(key=lambda p: -p.row)
            symbols = self._cursor.refill(reel, len(positions))
            for pos, sym in zip(positions, symbols):
                result.append((pos.reel, pos.row, sym.name))

        return tuple(result)
