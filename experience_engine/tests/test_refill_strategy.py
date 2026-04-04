"""Refill strategy tests.

TEST-REFILL-010 through TEST-REFILL-016: ClusterSeekingRefill
TEST-REFILL-020 through TEST-REFILL-025: TerminalRefill
"""

from __future__ import annotations

import random
from collections import Counter
from unittest.mock import patch

import pytest

from ..config.schema import BoardConfig, RefillConfig
from ..primitives.board import Board, Position
from ..primitives.cluster_detection import max_component_size
from ..primitives.symbols import Symbol
from ..pipeline.refill_strategy import (
    ClusterSeekingRefill,
    TerminalRefill,
    RefillStrategy,
)

# 5x5 board — large enough for meaningful cluster tests while staying fast
BOARD_CONFIG = BoardConfig(num_reels=5, num_rows=5, min_cluster_size=5)
STANDARD_NAMES = ("L1", "L2", "L3", "L4", "H1", "H2", "H3")
STANDARD_SYMBOLS = tuple(Symbol[n] for n in STANDARD_NAMES)
REFILL_CONFIG = RefillConfig(
    adjacency_boost=3.0, depth_scale=0.3, terminal_max_retries=10,
)


def _make_board(placements: dict[tuple[int, int], Symbol]) -> Board:
    """Build a board with specified placements, rest empty."""
    board = Board.empty(BOARD_CONFIG)
    for (reel, row), sym in placements.items():
        board.set(Position(reel, row), sym)
    return board


# ---------------------------------------------------------------------------
# ClusterSeekingRefill
# ---------------------------------------------------------------------------

class TestClusterSeekingRefill:

    def _make_strategy(self) -> ClusterSeekingRefill:
        return ClusterSeekingRefill(BOARD_CONFIG, STANDARD_NAMES, REFILL_CONFIG)

    def test_refill_010_bottom_up_fill_order(self) -> None:
        """TEST-REFILL-010: Cell at row N is filled before cell at row N-1."""
        strategy = self._make_strategy()
        # Empty cells at different rows in same reel
        empties = [Position(0, 1), Position(0, 3), Position(0, 0)]
        board = _make_board({})
        result = strategy.fill(board, empties, random.Random(42))

        # Bottom-up: row 3, then row 1, then row 0
        rows = [entry[1] for entry in result]
        assert rows == [3, 1, 0]

    def test_refill_011_adjacent_symbol_bias(self) -> None:
        """TEST-REFILL-011: L1 selected significantly more with L1 neighbor."""
        strategy = self._make_strategy()
        # L1 neighbors surrounding (2, 2)
        placements = {(1, 2): Symbol.L1, (3, 2): Symbol.L1, (2, 1): Symbol.L1}
        l1_count = 0
        trials = 200

        for seed in range(trials):
            board = _make_board(placements)
            result = strategy.fill(board, [Position(2, 2)], random.Random(seed))
            if result[0][2] == "L1":
                l1_count += 1

        # With 3 L1 neighbors and boost=3.0, L1 should be selected much more
        # than 1/7 (uniform ~14%). Expect > 40% conservatively.
        assert l1_count / trials > 0.40

    def test_refill_012_depth_bias(self) -> None:
        """TEST-REFILL-012: Deeper L1 neighbor produces higher L1 frequency."""
        strategy = self._make_strategy()
        trials = 300

        # Single L1 neighbor at row 0 (shallow)
        shallow_count = sum(
            1 for seed in range(trials)
            if strategy.fill(
                _make_board({(1, 0): Symbol.L1}),
                [Position(0, 0)], random.Random(seed),
            )[0][2] == "L1"
        )

        # Single L1 neighbor at row 4 (deep)
        deep_count = sum(
            1 for seed in range(trials)
            if strategy.fill(
                _make_board({(1, 4): Symbol.L1}),
                [Position(0, 4)], random.Random(seed),
            )[0][2] == "L1"
        )

        # Deeper neighbor should produce higher bias
        assert deep_count > shallow_count

    def test_refill_013_chain_propagation(self) -> None:
        """TEST-REFILL-013: Symbol placed at row 2 influences row 1 scoring."""
        strategy = self._make_strategy()
        # L1 at row 3 — when row 2 gets filled bottom-up, L1 bias at row 2
        # makes L1 more likely, which then biases row 1 toward L1 as well
        placements = {(0, 3): Symbol.L1}
        l1_at_row1 = 0
        trials = 200

        for seed in range(trials):
            board = _make_board(placements)
            result = strategy.fill(
                board, [Position(0, 1), Position(0, 2)], random.Random(seed),
            )
            # Row 2 filled first (bottom-up), then row 1
            row1_entry = next(e for e in result if e[1] == 1)
            if row1_entry[2] == "L1":
                l1_at_row1 += 1

        # Chain effect should produce bias at row 1 above uniform (~14%)
        assert l1_at_row1 / trials > 0.20

    def test_refill_014_return_type(self) -> None:
        """TEST-REFILL-014: Return type is tuple[tuple[int, int, str], ...]."""
        strategy = self._make_strategy()
        board = _make_board({})
        result = strategy.fill(board, [Position(0, 0), Position(1, 1)], random.Random(0))
        assert isinstance(result, tuple)
        for entry in result:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            assert isinstance(entry[0], int)
            assert isinstance(entry[1], int)
            assert isinstance(entry[2], str)

    def test_refill_015_deterministic_with_same_seed(self) -> None:
        """TEST-REFILL-015: Same seed produces identical results."""
        strategy = self._make_strategy()
        empties = [Position(r, c) for r in range(3) for c in range(3)]
        board = _make_board({(3, 3): Symbol.L1, (4, 4): Symbol.H1})

        result_a = strategy.fill(board, empties, random.Random(12345))
        result_b = strategy.fill(board, empties, random.Random(12345))
        assert result_a == result_b

    def test_refill_016_all_symbols_in_standard_set(self) -> None:
        """TEST-REFILL-016: All returned symbols are in standard_symbols."""
        strategy = self._make_strategy()
        empties = [Position(r, c) for r in range(5) for c in range(5)]
        result = strategy.fill(_make_board({}), empties, random.Random(42))
        standard_names = set(STANDARD_NAMES)
        for _, _, name in result:
            assert name in standard_names

    def test_protocol_compliance(self) -> None:
        """ClusterSeekingRefill satisfies RefillStrategy protocol."""
        assert isinstance(self._make_strategy(), RefillStrategy)


# ---------------------------------------------------------------------------
# TerminalRefill
# ---------------------------------------------------------------------------

class TestTerminalRefill:

    def _make_strategy(self) -> TerminalRefill:
        return TerminalRefill(
            BOARD_CONFIG, STANDARD_NAMES,
            BOARD_CONFIG.min_cluster_size, REFILL_CONFIG,
        )

    def test_refill_020_no_cluster_after_refill(self) -> None:
        """TEST-REFILL-020: No component >= min_cluster_size after refill (100 seeds)."""
        strategy = self._make_strategy()

        for seed in range(100):
            # Start with a partially filled board — some symbols already placed
            placements = {
                (0, 0): Symbol.L1, (0, 1): Symbol.L1,
                (1, 0): Symbol.L2, (1, 1): Symbol.L2,
            }
            board = _make_board(placements)
            empties = board.empty_positions()
            result = strategy.fill(board, empties, random.Random(seed))

            # Apply refill entries to a fresh copy for validation
            check_board = _make_board(placements)
            for reel, row, name in result:
                check_board.set(Position(reel, row), Symbol[name])

            # Verify no standard symbol forms a cluster
            for sym in STANDARD_SYMBOLS:
                component = max_component_size(
                    check_board, sym, BOARD_CONFIG,
                )
                assert component < BOARD_CONFIG.min_cluster_size, (
                    f"seed={seed}: {sym.name} component={component}"
                )

    def test_refill_021_fallback_terminates(self) -> None:
        """TEST-REFILL-021: Worst-case board forces fallback — still terminates."""
        # Board pre-filled with 4 L1s in an L-shape adjacent to the empty cell,
        # making L1 always create a cluster of 5. Fallback should pick another symbol.
        config = RefillConfig(
            adjacency_boost=3.0, depth_scale=0.3, terminal_max_retries=1,
        )
        strategy = TerminalRefill(
            BOARD_CONFIG, STANDARD_NAMES,
            BOARD_CONFIG.min_cluster_size, config,
        )
        # 4 L1s surrounding (2, 2): any L1 at (2,2) would create a cluster of 5
        placements = {
            (1, 2): Symbol.L1, (3, 2): Symbol.L1,
            (2, 1): Symbol.L1, (2, 3): Symbol.L1,
        }
        board = _make_board(placements)
        result = strategy.fill(board, [Position(2, 2)], random.Random(0))

        assert len(result) == 1
        # Should NOT be L1 (would create cluster of 5)
        assert result[0][2] != "L1"

    def test_refill_022_return_type(self) -> None:
        """TEST-REFILL-022: Return type matches tuple[tuple[int, int, str], ...]."""
        strategy = self._make_strategy()
        board = _make_board({})
        result = strategy.fill(board, [Position(0, 0), Position(1, 1)], random.Random(0))
        assert isinstance(result, tuple)
        for entry in result:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            assert isinstance(entry[0], int)
            assert isinstance(entry[1], int)
            assert isinstance(entry[2], str)

    def test_refill_023_deterministic_with_same_seed(self) -> None:
        """TEST-REFILL-023: Same seed produces identical results."""
        strategy = self._make_strategy()
        empties = [Position(r, c) for r in range(3) for c in range(3)]
        board = _make_board({(3, 3): Symbol.L1})

        result_a = strategy.fill(board, empties, random.Random(99))
        result_b = strategy.fill(board, empties, random.Random(99))
        assert result_a == result_b

    def test_refill_024_all_symbols_in_standard_set(self) -> None:
        """TEST-REFILL-024: All returned symbols are in standard_symbols."""
        strategy = self._make_strategy()
        empties = [Position(r, c) for r in range(5) for c in range(5)]
        result = strategy.fill(_make_board({}), empties, random.Random(42))
        standard_names = set(STANDARD_NAMES)
        for _, _, name in result:
            assert name in standard_names

    def test_refill_025_reuses_max_component_size(self) -> None:
        """TEST-REFILL-025: TerminalRefill uses max_component_size from cluster_detection."""
        # Verify the strategy module imports from cluster_detection, not a
        # reimplementation — checked via source module attribute
        import games.royal_tumble.experience_engine.pipeline.refill_strategy as mod
        assert mod.max_component_size.__module__ == (
            "games.royal_tumble.experience_engine.primitives.cluster_detection"
        )

    def test_protocol_compliance(self) -> None:
        """TerminalRefill satisfies RefillStrategy protocol."""
        assert isinstance(self._make_strategy(), RefillStrategy)
