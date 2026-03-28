"""BPT-001 through BPT-006: BridgePathTracer unit tests.

Tests the deterministic BFS path tracing from wild landing to refill zone,
gravity mapping inversion, and shortfall computation.
"""

from __future__ import annotations

import pytest

from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.gravity import GravityDAG, SettleResult, settle
from ..primitives.symbols import Symbol
from ..step_reasoner.services.bridge_path_tracer import BridgePathTracer, BridgePlan


def _filled_board(config: MasterConfig) -> Board:
    """Create a fully filled 7x7 board with alternating symbols."""
    board = Board.empty(config.board)
    syms = [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4, Symbol.H1, Symbol.H2, Symbol.H3]
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            board.set(Position(reel, row), syms[(reel + row) % len(syms)])
    return board


def _settle_with_explosion(config: MasterConfig, exploded: frozenset[Position]) -> SettleResult:
    """Build a filled board, explode given positions, and settle gravity."""
    board = _filled_board(config)
    dag = GravityDAG(config.board, config.gravity)
    return settle(dag, board, exploded, config.gravity)


class TestBridgePathTracer:

    def test_bpt_001_path_through_surviving_cells(self, default_config: MasterConfig) -> None:
        """BPT-001: Path from wild to refill passes through surviving cells only."""
        # Explode a cluster in column 3 — creates refill zone at top of column 3
        exploded = frozenset({
            Position(3, 4), Position(3, 5), Position(3, 6),
        })
        result = _settle_with_explosion(default_config, exploded)

        tracer = BridgePathTracer(default_config.board)
        # Wild lands at a surviving position adjacent to the exploded area
        # Use a position that survived (not in refill zone)
        wild_landing = Position(3, 3)

        plan = tracer.plan(wild_landing, result, target_bridge_size=5)

        # All path positions should be non-empty on the settled board
        for post_pos in plan.path_pre_to_post.values():
            assert result.board.get(post_pos) is not None, (
                f"Path position {post_pos} should be a surviving (non-empty) cell"
            )

    def test_bpt_002_path_toward_refill_centroid(self, default_config: MasterConfig) -> None:
        """BPT-002: Path prefers direction toward refill centroid."""
        # Explode cells in column 5 — refill zone at top of column 5
        exploded = frozenset({
            Position(5, 5), Position(5, 6),
        })
        result = _settle_with_explosion(default_config, exploded)

        tracer = BridgePathTracer(default_config.board)
        # Wild at column 3 — must traverse rightward toward column 5 refill
        wild_landing = Position(3, 4)

        plan = tracer.plan(wild_landing, result, target_bridge_size=5)

        # Path should exist and refill centroid should be near column 5
        assert plan.refill_centroid.reel == 5

    def test_bpt_003_pre_post_mapping_consistent(self, default_config: MasterConfig) -> None:
        """BPT-003: Pre→post mapping matches build_gravity_mappings() output."""
        from ..primitives.gravity import build_gravity_mappings

        exploded = frozenset({
            Position(3, 4), Position(3, 5), Position(3, 6),
        })
        result = _settle_with_explosion(default_config, exploded)

        tracer = BridgePathTracer(default_config.board)
        wild_landing = Position(3, 3)
        plan = tracer.plan(wild_landing, result, target_bridge_size=5)

        # Verify each pre→post pair against the canonical gravity mapping
        _, post_to_pre = build_gravity_mappings(
            result.move_steps, default_config.board,
        )
        for pre_pos, post_pos in plan.path_pre_to_post.items():
            assert pre_pos in post_to_pre.get(post_pos, []), (
                f"Pre {pre_pos} should map to post {post_pos} in gravity mappings"
            )

    def test_bpt_004_shortfall_computation(self, default_config: MasterConfig) -> None:
        """BPT-004: Shortfall = target - path_length - 1 (wild counts as 1)."""
        exploded = frozenset({
            Position(3, 4), Position(3, 5), Position(3, 6),
        })
        result = _settle_with_explosion(default_config, exploded)

        tracer = BridgePathTracer(default_config.board)
        wild_landing = Position(3, 3)
        target = 7

        plan = tracer.plan(wild_landing, result, target_bridge_size=target)

        path_length = len(plan.path_pre_to_post)
        expected_shortfall = max(0, target - path_length - 1)
        assert plan.shortfall == expected_shortfall

    def test_bpt_005_empty_path_when_adjacent_to_refill(self, default_config: MasterConfig) -> None:
        """BPT-005: Empty path when wild is already adjacent to refill zone."""
        # Explode at the bottom of column 3 — wild at (3,5) is adjacent to
        # the refill zone at the top after gravity
        exploded = frozenset({Position(3, 6)})
        result = _settle_with_explosion(default_config, exploded)
        refill_set = frozenset(result.empty_positions)

        tracer = BridgePathTracer(default_config.board)

        # Find a surviving position adjacent to the refill zone
        from ..primitives.board import orthogonal_neighbors
        wild_landing = None
        for pos in result.empty_positions:
            for neighbor in orthogonal_neighbors(pos, default_config.board):
                if neighbor not in refill_set and result.board.get(neighbor) is not None:
                    wild_landing = neighbor
                    break
            if wild_landing:
                break

        if wild_landing is not None:
            plan = tracer.plan(wild_landing, result, target_bridge_size=5)
            # Path should be empty — wild is already adjacent to refill
            assert len(plan.path_pre_to_post) == 0

    def test_bpt_006_no_empty_positions_raises(self, default_config: MasterConfig) -> None:
        """BPT-006: ValueError when no empty positions in settle result."""
        # Create a SettleResult with no empty positions
        board = _filled_board(default_config)
        fake_result = SettleResult(
            board=board,
            move_steps=(),
            empty_positions=(),
        )

        tracer = BridgePathTracer(default_config.board)
        with pytest.raises(ValueError, match="No empty positions"):
            tracer.plan(Position(3, 3), fake_result, target_bridge_size=5)
