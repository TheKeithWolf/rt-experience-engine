"""TEST-P1-010 through TEST-P1-014: Board, Position, and geometry functions."""

from ..config.schema import MasterConfig
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.symbols import Symbol


def test_p1_010_corner_has_2_neighbors(default_config: MasterConfig) -> None:
    """TEST-P1-010: Position (0,0) has 2 orthogonal neighbors."""
    neighbors = orthogonal_neighbors(Position(0, 0), default_config.board)
    assert len(neighbors) == 2
    assert Position(1, 0) in neighbors  # right
    assert Position(0, 1) in neighbors  # down


def test_p1_011_center_has_4_neighbors(default_config: MasterConfig) -> None:
    """TEST-P1-011: Position (3,3) has 4 orthogonal neighbors."""
    neighbors = orthogonal_neighbors(Position(3, 3), default_config.board)
    assert len(neighbors) == 4


def test_p1_012_board_dimensions_match_config(default_config: MasterConfig) -> None:
    """TEST-P1-012: Board dimensions match config (7x7)."""
    board = Board.empty(default_config.board)
    assert board.num_reels == 7
    assert board.num_rows == 7


def test_p1_013_empty_board_has_49_empty_positions(empty_board: Board) -> None:
    """TEST-P1-013: Board.empty() has 49 empty positions."""
    assert len(empty_board.empty_positions()) == 49


def test_p1_014_different_boards_produce_different_hashes(
    default_config: MasterConfig,
) -> None:
    """TEST-P1-014: board_hash produces different hashes for different boards."""
    board_a = Board.empty(default_config.board)
    board_b = Board.empty(default_config.board)

    # Initially identical
    assert board_a.board_hash() == board_b.board_hash()

    # Modify one
    board_b.set(Position(0, 0), Symbol.L1)
    assert board_a.board_hash() != board_b.board_hash()
