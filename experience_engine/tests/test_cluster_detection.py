"""TEST-P1-020 through TEST-P1-027: Cluster detection — BFS, wild-aware, exclusions."""

from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.cluster_detection import (
    Cluster,
    detect_clusters,
    detect_components,
    max_component_size,
)
from ..primitives.symbols import Symbol


def _make_board_with(config: MasterConfig, placements: dict[Position, Symbol]) -> Board:
    """Helper: create a board and place specific symbols, fill rest with L1/L2 pattern
    that avoids unintended clusters."""
    board = Board.empty(config.board)
    # Fill with an alternating pattern that prevents clusters of 5+
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            # Alternate between 4 symbols to prevent any cluster >= 5
            idx = (reel * 3 + row) % 4
            syms = [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4]
            board.set(Position(reel, row), syms[idx])
    # Apply specific placements
    for pos, sym in placements.items():
        board.set(pos, sym)
    return board


def test_p1_020_bfs_finds_size_5_cluster(default_config: MasterConfig) -> None:
    """TEST-P1-020: BFS finds size-5 L1 cluster on handcrafted board."""
    board = Board.empty(default_config.board)
    # Fill with alternating to avoid accidental clusters
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), [Symbol.L2, Symbol.L3, Symbol.L4, Symbol.H1][(reel * 3 + row) % 4])

    # Place 5 connected L1 in a horizontal line at row 0
    for reel in range(5):
        board.set(Position(reel, 0), Symbol.L1)

    clusters = detect_clusters(board, default_config)
    l1_clusters = [c for c in clusters if c.symbol is Symbol.L1]
    assert len(l1_clusters) == 1
    assert l1_clusters[0].size == 5


def test_p1_021_bfs_returns_empty_for_max_component_4(default_config: MasterConfig) -> None:
    """TEST-P1-021: BFS returns empty for board with max component 4."""
    board = Board.empty(default_config.board)
    # Fill with pattern ensuring no component reaches 5
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4][(reel * 3 + row) % 4])

    clusters = detect_clusters(board, default_config)
    assert len(clusters) == 0


def test_p1_022_bfs_excludes_scatter_and_booster(default_config: MasterConfig) -> None:
    """TEST-P1-022: BFS excludes scatter and booster symbols from clusters."""
    board = Board.empty(default_config.board)
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), Symbol.L2)

    # Place a line of 5 scatters — should NOT form a cluster
    for reel in range(5):
        board.set(Position(reel, 3), Symbol.S)

    # Place a line of 5 rockets — should NOT form a cluster
    for reel in range(5):
        board.set(Position(reel, 4), Symbol.R)

    clusters = detect_clusters(board, default_config)
    scatter_clusters = [c for c in clusters if c.symbol is Symbol.S]
    rocket_clusters = [c for c in clusters if c.symbol is Symbol.R]
    assert len(scatter_clusters) == 0
    assert len(rocket_clusters) == 0


def test_p1_023_wild_bridges_two_groups(default_config: MasterConfig) -> None:
    """TEST-P1-023: Wild bridges two L1 groups into one cluster (C-COMPAT-2)."""
    board = Board.empty(default_config.board)
    # Prevent accidental clusters
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), [Symbol.L2, Symbol.L3, Symbol.L4, Symbol.H1][(reel * 3 + row) % 4])

    # Group A: 3 L1 at (0,0), (1,0), (2,0)
    board.set(Position(0, 0), Symbol.L1)
    board.set(Position(1, 0), Symbol.L1)
    board.set(Position(2, 0), Symbol.L1)
    # Wild at (3,0) bridges to Group B
    board.set(Position(3, 0), Symbol.W)
    # Group B: 3 L1 at (4,0), (5,0), (6,0)
    board.set(Position(4, 0), Symbol.L1)
    board.set(Position(5, 0), Symbol.L1)
    board.set(Position(6, 0), Symbol.L1)

    clusters = detect_clusters(board, default_config)
    l1_clusters = [c for c in clusters if c.symbol is Symbol.L1]
    assert len(l1_clusters) == 1
    # 6 L1 + 1 wild = 7
    assert l1_clusters[0].size == 7
    assert len(l1_clusters[0].wild_positions) == 1


def test_p1_024_wild_multi_participation(default_config: MasterConfig) -> None:
    """TEST-P1-024: Wild adjacent to both L1 and H2 groups participates in both (R-WILD-5)."""
    board = Board.empty(default_config.board)
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), [Symbol.L3, Symbol.L4, Symbol.H1, Symbol.H3][(reel * 3 + row) % 4])

    # 4 L1 in column 0: (0,0)-(0,3)
    for row in range(4):
        board.set(Position(0, row), Symbol.L1)
    # Wild at (0,4)
    board.set(Position(0, 4), Symbol.W)
    # 4 H2 in column 0: (0,5)-(0,6) and extending to (1,5)-(1,6)
    board.set(Position(0, 5), Symbol.H2)
    board.set(Position(0, 6), Symbol.H2)
    board.set(Position(1, 5), Symbol.H2)
    board.set(Position(1, 6), Symbol.H2)

    clusters = detect_clusters(board, default_config)
    l1_clusters = [c for c in clusters if c.symbol is Symbol.L1]
    h2_clusters = [c for c in clusters if c.symbol is Symbol.H2]

    # L1: 4 standard + 1 wild = 5 (meets min)
    assert len(l1_clusters) == 1
    assert l1_clusters[0].size == 5

    # H2: 4 standard + 1 wild = 5 (meets min)
    assert len(h2_clusters) == 1
    assert h2_clusters[0].size == 5

    # The wild participates in both
    wild_pos = Position(0, 4)
    assert wild_pos in l1_clusters[0].wild_positions
    assert wild_pos in h2_clusters[0].wild_positions


def test_p1_025_wild_no_standard_neighbors_no_cluster(default_config: MasterConfig) -> None:
    """TEST-P1-025: Wild + no standard neighbors → no cluster."""
    board = Board.empty(default_config.board)
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), Symbol.S)  # All scatters

    # Place a wild surrounded by scatters
    board.set(Position(3, 3), Symbol.W)

    clusters = detect_clusters(board, default_config)
    assert len(clusters) == 0


def test_p1_026_wild_does_not_bridge_to_scatter(default_config: MasterConfig) -> None:
    """TEST-P1-026: Wild does not bridge to scatter — scatters excluded from clusters."""
    board = Board.empty(default_config.board)
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), [Symbol.L2, Symbol.L3, Symbol.L4, Symbol.H1][(reel * 3 + row) % 4])

    # 3 L1, 1 wild, 3 scatters in a line
    board.set(Position(0, 0), Symbol.L1)
    board.set(Position(1, 0), Symbol.L1)
    board.set(Position(2, 0), Symbol.L1)
    board.set(Position(3, 0), Symbol.W)
    board.set(Position(4, 0), Symbol.S)
    board.set(Position(5, 0), Symbol.S)
    board.set(Position(6, 0), Symbol.S)

    clusters = detect_clusters(board, default_config)
    # L1: only 3 + 1 wild = 4, below min_cluster_size of 5
    l1_clusters = [c for c in clusters if c.symbol is Symbol.L1]
    assert len(l1_clusters) == 0


def test_p1_027_max_component_size_with_extra(default_config: MasterConfig) -> None:
    """TEST-P1-027: max_component_size correct with extra placements."""
    board = Board.empty(default_config.board)
    for reel in range(7):
        for row in range(7):
            board.set(Position(reel, row), Symbol.L2)

    # Place 3 L1 in a row
    board.set(Position(0, 0), Symbol.L1)
    board.set(Position(1, 0), Symbol.L1)
    board.set(Position(2, 0), Symbol.L1)

    # Without extra: max component = 3
    assert max_component_size(board, Symbol.L1, default_config.board) == 3

    # With extra position adjacent: max component = 4
    extra = frozenset({Position(3, 0)})
    assert max_component_size(board, Symbol.L1, default_config.board, extra=extra) == 4
