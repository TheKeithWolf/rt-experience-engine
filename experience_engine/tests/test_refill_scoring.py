"""Adjacency scoring tests for refill strategies.

TEST-REFILL-001 through TEST-REFILL-004.
"""

from __future__ import annotations

from ..config.schema import BoardConfig
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..pipeline.refill_scoring import score_symbols_by_adjacency

# Small 3x3 board for focused unit tests
BOARD_CONFIG = BoardConfig(num_reels=3, num_rows=3, min_cluster_size=5)
STANDARD_SYMBOLS = (Symbol.L1, Symbol.L2, Symbol.L3)


def _make_board(placements: dict[tuple[int, int], Symbol]) -> Board:
    """Build a 3x3 board with specified placements, rest empty."""
    board = Board.empty(BOARD_CONFIG)
    for (reel, row), sym in placements.items():
        board.set(Position(reel, row), sym)
    return board


def test_refill_001_base_weight_when_no_neighbors_match() -> None:
    """TEST-REFILL-001: All symbols get base weight 1.0 when no neighbors match."""
    # Center cell (1,1) surrounded by empty cells
    board = _make_board({})
    weights = score_symbols_by_adjacency(
        board, Position(1, 1), STANDARD_SYMBOLS,
        BOARD_CONFIG, adjacency_boost=3.0, depth_scale=0.3,
    )
    for sym in STANDARD_SYMBOLS:
        assert weights[sym] == 1.0


def test_refill_002_one_adjacent_neighbor_boosts_weight() -> None:
    """TEST-REFILL-002: One adjacent L1 neighbor boosts L1 weight by expected amount."""
    # L1 at (1, 2) — neighbor below center (1, 1)
    neighbor_row = 2
    board = _make_board({(1, neighbor_row): Symbol.L1})
    weights = score_symbols_by_adjacency(
        board, Position(1, 1), STANDARD_SYMBOLS,
        BOARD_CONFIG, adjacency_boost=3.0, depth_scale=0.3,
    )
    expected_l1 = 1.0 + 3.0 * (1.0 + neighbor_row * 0.3)
    assert weights[Symbol.L1] == expected_l1
    # Other symbols keep base weight
    assert weights[Symbol.L2] == 1.0
    assert weights[Symbol.L3] == 1.0


def test_refill_003_two_neighbors_compound_additively() -> None:
    """TEST-REFILL-003: Two adjacent L1 neighbors at different depths compound."""
    # L1 at (1, 0) above center and (1, 2) below center
    board = _make_board({(1, 0): Symbol.L1, (1, 2): Symbol.L1})
    weights = score_symbols_by_adjacency(
        board, Position(1, 1), STANDARD_SYMBOLS,
        BOARD_CONFIG, adjacency_boost=3.0, depth_scale=0.3,
    )
    # Each neighbor adds independently: boost * (1 + row * depth_scale)
    boost_from_row_0 = 3.0 * (1.0 + 0 * 0.3)
    boost_from_row_2 = 3.0 * (1.0 + 2 * 0.3)
    expected_l1 = 1.0 + boost_from_row_0 + boost_from_row_2
    assert weights[Symbol.L1] == expected_l1


def test_refill_004_non_standard_symbols_ignored() -> None:
    """TEST-REFILL-004: Non-standard symbols on the board (wilds, scatters) are ignored."""
    # Wild at (1, 0), Scatter at (0, 1) — neighbors of center (1, 1)
    board = _make_board({(1, 0): Symbol.W, (0, 1): Symbol.S})
    weights = score_symbols_by_adjacency(
        board, Position(1, 1), STANDARD_SYMBOLS,
        BOARD_CONFIG, adjacency_boost=3.0, depth_scale=0.3,
    )
    # All standard symbols keep base weight — specials contribute nothing
    for sym in STANDARD_SYMBOLS:
        assert weights[sym] == 1.0
