"""Shared test fixtures for Experience Engine Phase 1 tests."""

from pathlib import Path

import pytest

from ..config.loader import load_config
from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol

# Path to the default config YAML shipped with the engine
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


@pytest.fixture(scope="session")
def default_config() -> MasterConfig:
    """Load the default config once per test session."""
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def empty_board(default_config: MasterConfig) -> Board:
    """A 7x7 board with all cells empty (None)."""
    return Board.empty(default_config.board)


@pytest.fixture
def sample_board(default_config: MasterConfig) -> Board:
    """A handcrafted 7x7 board with known cluster patterns.

    Layout (reel=col, row increases downward):
    - Row 0: L1 L1 L1 L1 L1 H1 H2  → 5 connected L1 (cluster!)
    - Row 1: L2 L2 L3 L3 L4 H1 H2
    - Row 2: L3 L2 L3 L4 L4 H2 H3
    - Row 3: L4 L4 L4 L4 L1 L1 L1
    - Row 4: H1 H1 H1 H1 H2 H2 H3
    - Row 5: H2 H3 H1 L1 L2 L3 L4
    - Row 6: H3 H3 H3 H3 H3 L1 L2  → 5 connected H3 (cluster!)
    """
    board = Board.empty(default_config.board)
    layout = [
        # reel 0
        [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4, Symbol.H1, Symbol.H2, Symbol.H3],
        # reel 1
        [Symbol.L1, Symbol.L2, Symbol.L2, Symbol.L4, Symbol.H1, Symbol.H3, Symbol.H3],
        # reel 2
        [Symbol.L1, Symbol.L3, Symbol.L3, Symbol.L4, Symbol.H1, Symbol.H1, Symbol.H3],
        # reel 3
        [Symbol.L1, Symbol.L3, Symbol.L4, Symbol.L4, Symbol.H1, Symbol.L1, Symbol.H3],
        # reel 4
        [Symbol.L1, Symbol.L4, Symbol.L4, Symbol.L1, Symbol.H2, Symbol.L2, Symbol.H3],
        # reel 5
        [Symbol.H1, Symbol.H1, Symbol.H2, Symbol.L1, Symbol.H2, Symbol.L3, Symbol.L1],
        # reel 6
        [Symbol.H2, Symbol.H2, Symbol.H3, Symbol.L1, Symbol.H3, Symbol.L4, Symbol.L2],
    ]
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), layout[reel][row])
    return board
