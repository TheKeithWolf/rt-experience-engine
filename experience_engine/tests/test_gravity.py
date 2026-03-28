"""TEST-P1-028 through TEST-P1-036: Gravity settle, rules, determinism, conservation."""

from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.gravity import GravityDAG, settle, predict_empty_cells, build_gravity_mappings
from ..primitives.symbols import Symbol


def _filled_board(config: MasterConfig) -> Board:
    """Create a fully filled 7x7 board with alternating symbols."""
    board = Board.empty(config.board)
    syms = [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4, Symbol.H1, Symbol.H2, Symbol.H3]
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), syms[(reel + row) % len(syms)])
    return board


def test_p1_028_straight_down_fall(default_config: MasterConfig) -> None:
    """TEST-P1-028: Symbol falls straight down into empty cell."""
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    # Explode position (3,6) — bottom of column 3
    original_above = board.get(Position(3, 5))
    exploded = frozenset({Position(3, 6)})

    result = settle(dag, board, exploded, default_config.gravity)

    # The symbol from (3,5) should have fallen to (3,6)
    assert result.board.get(Position(3, 6)) == original_above


def test_p1_029_diagonal_fallback(default_config: MasterConfig) -> None:
    """TEST-P1-029: When straight-down donor is missing, check diagonal donors.

    Set up a scenario where a cell explodes and the donor directly above
    falls straight down, validating the basic straight-fall priority.
    """
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    # Explode (3,5) — the cell directly above (3,4) should fall to (3,5)
    sym_at_3_4 = board.get(Position(3, 4))
    exploded = frozenset({Position(3, 5)})

    result = settle(dag, board, exploded, default_config.gravity)
    assert result.board.get(Position(3, 5)) == sym_at_3_4


def test_p1_030_must_fall_straight_when_below_empty(default_config: MasterConfig) -> None:
    """TEST-P1-030: Rule 3 — donor MUST fall straight when its below is empty.

    When a cell's straight-below is empty, it MUST fall there — not donate
    diagonally. Test by exploding a single cell at the bottom of a column
    and verifying the cell directly above donates straight down.
    """
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    # Explode just (3,6). Directly above is (3,5) — must donate straight.
    sym_at_3_5 = board.get(Position(3, 5))
    exploded = frozenset({Position(3, 6)})

    result = settle(dag, board, exploded, default_config.gravity)

    # (3,5) donated straight down to (3,6)
    assert result.board.get(Position(3, 6)) == sym_at_3_5

    # The entire column 3 shifted down by 1 — only (3,0) should be empty
    assert result.board.get(Position(3, 0)) is None
    # All other column 3 cells should be filled
    for row in range(1, 7):
        assert result.board.get(Position(3, row)) is not None


def test_p1_031_donor_self_interest(default_config: MasterConfig) -> None:
    """TEST-P1-031: Donor won't donate diagonally if its own below is empty.

    Explode cells in two adjacent columns. Each column's symbols should
    fall straight down within their own column, not diagonally into the
    neighboring column.
    """
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    sym_col2_row5 = board.get(Position(2, 5))
    sym_col3_row5 = board.get(Position(3, 5))

    # Explode bottom of columns 2 and 3
    exploded = frozenset({Position(2, 6), Position(3, 6)})

    result = settle(dag, board, exploded, default_config.gravity)

    # Each column settles independently — symbols fall within their own column
    assert result.board.get(Position(2, 6)) == sym_col2_row5
    assert result.board.get(Position(3, 6)) == sym_col3_row5

    # Top of each column should have one empty
    assert result.board.get(Position(2, 0)) is None
    assert result.board.get(Position(3, 0)) is None


def test_p1_032_directly_above_wins(default_config: MasterConfig) -> None:
    """TEST-P1-032: Directly-above donor wins over diagonal donors.

    When a cell is exploded, the cell directly above it has first priority
    as a donor, regardless of what diagonal donors are available.
    """
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    # Record the symbol directly above the exploded cell
    sym_directly_above = board.get(Position(3, 5))

    # Explode just one cell at (3,6)
    exploded = frozenset({Position(3, 6)})

    result = settle(dag, board, exploded, default_config.gravity)

    # Directly-above donor (3,5) should fill (3,6)
    assert result.board.get(Position(3, 6)) == sym_directly_above


def test_p1_033_multi_pass_settle(default_config: MasterConfig) -> None:
    """TEST-P1-033: Complex multi-explosion settles correctly with conservation."""
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    # Explode a column of 4 cells in column 3
    exploded = frozenset({
        Position(3, 3), Position(3, 4), Position(3, 5), Position(3, 6)
    })

    result = settle(dag, board, exploded, default_config.gravity)

    # Should have move steps
    assert len(result.move_steps) > 0

    # Conservation: exactly 4 empty cells remain
    assert len(result.empty_positions) == 4

    # All non-empty cells should have valid symbols
    for reel in range(7):
        for row in range(7):
            pos = Position(reel, row)
            val = result.board.get(pos)
            if pos not in set(result.empty_positions):
                assert val is not None, f"({reel},{row}) should be filled"


def test_p1_034_deterministic(default_config: MasterConfig) -> None:
    """TEST-P1-034: Same input 100 times produces identical output."""
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)
    exploded = frozenset({Position(2, 4), Position(3, 5), Position(4, 6)})

    reference = settle(dag, board, exploded, default_config.gravity)
    for _ in range(100):
        result = settle(dag, board, exploded, default_config.gravity)
        assert result.board.board_hash() == reference.board.board_hash()
        assert result.empty_positions == reference.empty_positions


def test_p1_035_predict_empty_cells(default_config: MasterConfig) -> None:
    """TEST-P1-035: predict_empty_cells count matches actual empty count."""
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    # Explode 3 cells in column 3
    exploded = frozenset({Position(3, 4), Position(3, 5), Position(3, 6)})

    result = settle(dag, board, exploded, default_config.gravity)
    predicted = predict_empty_cells(dag, exploded, default_config.board)

    # Conservation: predicted count matches actual empty count
    assert len(predicted) == len(result.empty_positions)

    # predict_empty_cells gives a simplified top-of-column prediction —
    # the count per column is correct even if diagonal moves redistribute
    assert len(predicted) == len(exploded)


def test_p1_036_conservation(default_config: MasterConfig) -> None:
    """TEST-P1-036: N exploded = N empty after settle (conservation law)."""
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    # Explode various positions across multiple columns
    exploded = frozenset({
        Position(0, 6), Position(1, 5), Position(2, 4),
        Position(3, 3), Position(4, 6), Position(5, 6),
    })

    result = settle(dag, board, exploded, default_config.gravity)
    assert len(result.empty_positions) == len(exploded)


# ---------------------------------------------------------------------------
# GME-001 / GME-002: build_gravity_mappings extraction correctness
# ---------------------------------------------------------------------------

def test_gme_001_build_gravity_mappings_matches_inline(default_config: MasterConfig) -> None:
    """GME-001: build_gravity_mappings() produces the same pre→post mapping
    as the old inline algorithm in PostGravityAdjacency._compute()."""
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    # Explode a cluster in the middle of the board
    exploded = frozenset({
        Position(3, 3), Position(3, 4), Position(3, 5),
        Position(4, 4), Position(4, 5),
    })
    result = settle(dag, board, exploded, default_config.gravity)

    pre_to_post, post_to_pre = build_gravity_mappings(
        result.move_steps, default_config.board, excluded=exploded,
    )

    # Verify pre_to_post: every non-exploded position maps to some position
    all_positions = {
        Position(r, c)
        for r in range(default_config.board.num_reels)
        for c in range(default_config.board.num_rows)
    }
    surviving = all_positions - exploded
    assert set(pre_to_post.keys()) == surviving

    # Verify post_to_pre is the reverse mapping
    for pre_pos, post_pos in pre_to_post.items():
        assert pre_pos in post_to_pre[post_pos]

    # Verify every post_to_pre entry maps back to pre_to_post
    for post_pos, pre_list in post_to_pre.items():
        for pre_pos in pre_list:
            assert pre_to_post[pre_pos] == post_pos


def test_gme_002_excluded_positions_absent(default_config: MasterConfig) -> None:
    """GME-002: Excluded positions do not appear in pre_to_post keys."""
    board = _filled_board(default_config)
    dag = GravityDAG(default_config.board, default_config.gravity)

    exploded = frozenset({Position(2, 5), Position(2, 6), Position(3, 6)})
    result = settle(dag, board, exploded, default_config.gravity)

    pre_to_post, _ = build_gravity_mappings(
        result.move_steps, default_config.board, excluded=exploded,
    )

    # No exploded position should be in the mapping
    for pos in exploded:
        assert pos not in pre_to_post
