"""Adjacency-based symbol scoring for refill strategies.

Shared utility consumed by ClusterSeekingRefill — extracted here because it
couples board state + position + config but belongs to neither strategy
exclusively. Pure function with no mutable state.
"""

from __future__ import annotations

from ..config.schema import BoardConfig
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.symbols import Symbol


def score_symbols_by_adjacency(
    board: Board,
    position: Position,
    standard_symbols: tuple[Symbol, ...],
    board_config: BoardConfig,
    adjacency_boost: float,
    depth_scale: float,
) -> dict[Symbol, float]:
    """Per-symbol weight at *position* based on adjacent same-symbol count.

    For each orthogonal neighbor holding a standard symbol, that symbol's
    weight increases by ``adjacency_boost * (1 + neighbor.row * depth_scale)``.
    Deeper neighbors contribute more, biasing cluster formation toward the
    bottom of the board where gravity accumulates symbols.

    Returns ``{symbol: weight}`` with base weight 1.0 for all symbols.
    Symbols not present on any neighbor keep their base weight, ensuring
    every standard symbol remains selectable.
    """
    # Every standard symbol starts equally likely before adjacency bias
    weights: dict[Symbol, float] = {sym: 1.0 for sym in standard_symbols}
    standard_set = frozenset(standard_symbols)

    for neighbor in orthogonal_neighbors(position, board_config):
        neighbor_sym = board.get(neighbor)
        # Non-standard symbols (wilds, scatters, boosters) have no symbol
        # identity to bias toward — skip them
        if neighbor_sym is not None and neighbor_sym in standard_set:
            weights[neighbor_sym] += adjacency_boost * (
                1.0 + neighbor.row * depth_scale
            )

    return weights
