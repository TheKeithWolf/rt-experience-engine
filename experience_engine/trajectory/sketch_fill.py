"""Lightweight neutral fill used between trajectory planner waypoints.

Distinct from TerminalRefill and ClusterSeekingRefill: those perform full
retry / adjacency scoring during real generation. The sketch planner needs
*approximate* continuity between steps — accurate enough to feed the next
settle(), cheap enough to run dozens of times per sketch attempt. No
adjacency scoring, no boundary checks, no retries.

Callers are expected to pass the standard symbol set from config.symbols.
The filler never selects wilds, scatters, or boosters — those are placed
deliberately by the planner, never as neutral background.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol


def neutral_fill(
    board: Board,
    empty_positions: Iterable[Position],
    standard_symbols: tuple[Symbol, ...],
    symbol_weights: dict[Symbol, float],
    rng: random.Random,
) -> None:
    """Fill the given positions with weighted-random standard symbols.

    Mutates `board` in place — matches the convention used by the existing
    refill strategies. Positions already occupied on the board are skipped
    so the function is safe to call with a superset of actual empties.

    Symbols absent from symbol_weights default to weight 1.0 — letting
    callers omit neutral symbols from their variance hints without crashing
    the sketch path.
    """
    if not standard_symbols:
        raise ValueError("standard_symbols must not be empty")
    weights = [max(0.0, symbol_weights.get(s, 1.0)) for s in standard_symbols]
    if not any(w > 0.0 for w in weights):
        # All weights zero would brick rng.choices; fall back to uniform so
        # the sketch still progresses rather than exception-ing out.
        weights = [1.0] * len(standard_symbols)
    for pos in empty_positions:
        if board.get(pos) is not None:
            continue
        chosen = rng.choices(standard_symbols, weights=weights, k=1)[0]
        board.set(pos, chosen)
