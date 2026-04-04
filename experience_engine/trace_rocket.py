"""Trace a rocket booster's full lifecycle: DORMANT -> ARMED -> FIRED.

Standalone CLI that builds a curated 7x7 board, steps through the rocket
lifecycle using real engine primitives, and renders ASCII visuals at each step.
Demonstrates the complete booster state machine for a horizontal rocket.

Usage:
    python -m games.royal_tumble.experience_engine.trace_rocket
"""

from __future__ import annotations

import sys
from pathlib import Path

from .boosters.fire_handlers import fire_rocket
from .boosters.phase_executor import BoosterPhaseExecutor
from .boosters.state_machine import BoosterState
from .boosters.tracker import BoosterTracker
from .config.loader import load_config
from .config.schema import MasterConfig
from .primitives.board import Board, Position
from .primitives.booster_rules import BoosterRules, place_booster
from .primitives.cluster_detection import detect_clusters
from .primitives.symbols import Symbol, is_booster

# Resolved relative to this file
_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

from .formatting.board_formatter import format_board_grid
from .formatting.cells import CellStyle
from .formatting.constants import MAJOR_SEP as _MAJOR_SEP, MINOR_SEP as _MINOR_SEP

# Rocket glyph for board display — orientation-dependent
_ROCKET_GLYPH: dict[str, str] = {"H": "R>", "V": "R^"}


# ---------------------------------------------------------------------------
# Board formatting — delegates to shared formatting package
# ---------------------------------------------------------------------------

def _format_board(
    board: Board,
    highlights: frozenset[Position] = frozenset(),
    rocket_pos: Position | None = None,
    rocket_orient: str | None = None,
) -> list[str]:
    """Render board as bordered ASCII table via shared formatter.

    - Normal cell:      | H2 |
    - Highlighted cell:  |*H2*|
    - Vacancy (None):    |[  ]|
    - Rocket cell:       | R> | (H) or | R^ | (V)
    """
    def resolve_cell(reel: int, row: int) -> tuple[str, CellStyle]:
        pos = Position(reel, row)
        sym = board.get(pos)

        if sym is None:
            return "  ", CellStyle.VACANCY

        # Rocket with orientation glyph
        if pos == rocket_pos and rocket_orient is not None:
            glyph = _ROCKET_GLYPH.get(rocket_orient, f" {sym.name}")
            if pos in highlights:
                return glyph, CellStyle.WINNER
            return glyph, CellStyle.REGULAR

        if pos in highlights:
            return sym.name, CellStyle.WINNER
        if is_booster(sym):
            return sym.name, CellStyle.REGULAR

        return sym.name, CellStyle.REGULAR

    return format_board_grid(board.num_reels, board.num_rows, resolve_cell)


def _print_board(
    board: Board,
    highlights: frozenset[Position] = frozenset(),
    rocket_pos: Position | None = None,
    rocket_orient: str | None = None,
) -> None:
    """Print board to stdout."""
    for line in _format_board(board, highlights, rocket_pos, rocket_orient):
        print(line)


def _print_booster_state(tracker: BoosterTracker) -> None:
    """Print all tracked boosters with state, position, and orientation."""
    boosters = tracker.all_boosters()
    if not boosters:
        print("  Boosters: (none)")
        return
    for b in sorted(boosters, key=lambda x: (x.position.row, x.position.reel)):
        orient_str = f" orientation={b.orientation}" if b.orientation else ""
        print(
            f"  {b.booster_type.name} at ({b.position.reel},{b.position.row})"
            f" {b.state.value}{orient_str}"
        )


def _print_step(num: int, title: str, description: str) -> None:
    """Print step header with separator."""
    print()
    print(_MINOR_SEP)
    print(f"  STEP {num}: {title}")
    print(f"  {description}")
    print(_MINOR_SEP)


# ---------------------------------------------------------------------------
# Curated board layouts
# ---------------------------------------------------------------------------

def _build_initial_board(config: MasterConfig) -> Board:
    """Build 7x7 board with a 9-cell L1 cluster guaranteed to spawn a rocket.

    The L1 cluster spans reel 3 (all 7 rows) plus (2,2) and (4,2) — connected
    via (3,2). Shape is taller than wide (row_span=7, col_span=3) so the rocket
    fires horizontally. Remaining cells use varied symbols to avoid unintended
    clusters >= 5.
    """
    # layout[reel][row] — each inner list is one column, top to bottom
    layout: list[list[Symbol]] = [
        # reel 0
        [Symbol.H2, Symbol.L4, Symbol.H1, Symbol.L3, Symbol.H3, Symbol.L2, Symbol.H1],
        # reel 1
        [Symbol.L3, Symbol.H3, Symbol.L2, Symbol.H2, Symbol.L4, Symbol.H1, Symbol.L3],
        # reel 2 — L1 at row 2 bridges to the reel-3 column
        [Symbol.L4, Symbol.L2, Symbol.L1, Symbol.L4, Symbol.H2, Symbol.L3, Symbol.H2],
        # reel 3 — full L1 column (7 cells)
        [Symbol.L1, Symbol.L1, Symbol.L1, Symbol.L1, Symbol.L1, Symbol.L1, Symbol.L1],
        # reel 4 — L1 at row 2 bridges to the reel-3 column
        [Symbol.H3, Symbol.H2, Symbol.L1, Symbol.L2, Symbol.L3, Symbol.H2, Symbol.L4],
        # reel 5
        [Symbol.L2, Symbol.L3, Symbol.H3, Symbol.H1, Symbol.L2, Symbol.L4, Symbol.H3],
        # reel 6
        [Symbol.H1, Symbol.H1, Symbol.L4, Symbol.L3, Symbol.H1, Symbol.H3, Symbol.L2],
    ]

    board = Board.empty(config.board)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            board.set(Position(reel, row), layout[reel][row])
    return board


def _build_post_gravity_board(config: MasterConfig) -> Board:
    """Build the board state after gravity settles and reels refill.

    Places a 5-cell H1 arming cluster at {(4,3),(4,4),(4,5),(5,4),(5,5)}.
    Position (4,3) is orthogonal neighbor of the rocket at (3,3), triggering
    the DORMANT -> ARMED transition. No other cluster >= 5 is present.
    """
    layout: list[list[Symbol]] = [
        # reel 0
        [Symbol.H2, Symbol.L4, Symbol.H1, Symbol.L3, Symbol.H3, Symbol.L2, Symbol.H1],
        # reel 1
        [Symbol.L3, Symbol.H3, Symbol.L2, Symbol.H2, Symbol.L4, Symbol.H1, Symbol.L3],
        # reel 2 — refilled after explosion
        [Symbol.L4, Symbol.L2, Symbol.H3, Symbol.L4, Symbol.H2, Symbol.L3, Symbol.H2],
        # reel 3 — rocket at row 3, rest refilled
        [Symbol.L2, Symbol.H3, Symbol.L4, Symbol.L3, Symbol.L3, Symbol.H2, Symbol.L4],
        # reel 4 — H1 arming cluster at rows 3, 4, 5
        [Symbol.H3, Symbol.H2, Symbol.L2, Symbol.H1, Symbol.H1, Symbol.H1, Symbol.L4],
        # reel 5 — H1 arming cluster at rows 4, 5
        [Symbol.L2, Symbol.L3, Symbol.H3, Symbol.L2, Symbol.H1, Symbol.H1, Symbol.H3],
        # reel 6
        [Symbol.H1, Symbol.L4, Symbol.L4, Symbol.L3, Symbol.L2, Symbol.H3, Symbol.L2],
    ]

    board = Board.empty(config.board)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            board.set(Position(reel, row), layout[reel][row])
    return board


# ---------------------------------------------------------------------------
# Lifecycle trace
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Step through the rocket booster lifecycle with ASCII visuals."""
    config = load_config(_CONFIG_PATH)
    rules = BoosterRules(config.boosters, config.board, config.symbols)

    print(_MAJOR_SEP)
    print("  ROCKET LIFECYCLE TRACE")
    print("  DORMANT -> ARMED -> FIRED")
    print(_MAJOR_SEP)

    # -----------------------------------------------------------------------
    # Step 1: Initial board with curated 9-cell L1 cluster
    # -----------------------------------------------------------------------
    board = _build_initial_board(config)

    # Pre-compute the L1 cluster positions for highlighting
    l1_positions = frozenset(
        Position(reel, row)
        for reel in range(config.board.num_reels)
        for row in range(config.board.num_rows)
        if board.get(Position(reel, row)) is Symbol.L1
    )

    _print_step(1, "INITIAL BOARD",
                f"{len(l1_positions)} L1 symbols form a connected cluster (marked with *)")
    _print_board(board, highlights=l1_positions)

    # -----------------------------------------------------------------------
    # Step 2: Cluster detection — confirm the 9-cell L1 cluster
    # -----------------------------------------------------------------------
    clusters = detect_clusters(board, config)
    l1_cluster = next(c for c in clusters if c.symbol is Symbol.L1)
    all_cluster_positions = l1_cluster.positions | l1_cluster.wild_positions

    _print_step(2, "CLUSTER DETECTION",
                f"Detected {len(clusters)} cluster(s) meeting min size {config.board.min_cluster_size}")
    for i, c in enumerate(clusters):
        pos_str = ", ".join(
            f"({p.reel},{p.row})" for p in sorted(c.positions, key=lambda p: (p.row, p.reel))
        )
        print(f"  cluster {i}: {c.symbol.name} x{c.size} at [{pos_str}]")

    booster_type = rules.booster_type_for_size(l1_cluster.size)
    print(f"\n  Booster type for size {l1_cluster.size}: "
          f"{booster_type.name if booster_type else 'None'} (Rocket)")

    # -----------------------------------------------------------------------
    # Step 3: Rocket spawn — compute centroid, orientation, place on board
    # -----------------------------------------------------------------------
    centroid = rules.compute_centroid(all_cluster_positions)
    orientation = rules.compute_rocket_orientation(all_cluster_positions)

    tracker = BoosterTracker(config.board)
    place_booster(Symbol.R, centroid, board, tracker,
                  orientation=orientation, source_cluster_index=0)

    _print_step(3, "ROCKET SPAWN (DORMANT)",
                f"Centroid at ({centroid.reel},{centroid.row}), orientation={orientation}")
    _print_board(board, rocket_pos=centroid, rocket_orient=orientation)
    print()
    _print_booster_state(tracker)

    # -----------------------------------------------------------------------
    # Step 4: Cluster explosion — clear cluster cells except rocket
    # -----------------------------------------------------------------------
    exploded_positions: set[Position] = set()
    for pos in all_cluster_positions:
        if pos != centroid:
            board.set(pos, None)
            exploded_positions.add(pos)

    _print_step(4, "CLUSTER EXPLOSION",
                f"{len(exploded_positions)} cluster cells cleared, rocket preserved at ({centroid.reel},{centroid.row})")
    _print_board(board, rocket_pos=centroid, rocket_orient=orientation)

    # -----------------------------------------------------------------------
    # Step 5: Post-gravity board (simulated) — new symbols fill vacancies
    # -----------------------------------------------------------------------
    board = _build_post_gravity_board(config)
    # Rocket stays at its position — overwrite the layout's placeholder
    board.set(centroid, Symbol.R)

    _print_step(5, "POST-GRAVITY BOARD",
                "Board after gravity settle and reel refill (simulated)")
    _print_board(board, rocket_pos=centroid, rocket_orient=orientation)
    print()
    _print_booster_state(tracker)

    # -----------------------------------------------------------------------
    # Step 6: Arming cluster detection — find cluster adjacent to rocket
    # -----------------------------------------------------------------------
    arming_clusters = detect_clusters(board, config)
    if not arming_clusters:
        print("  ERROR: No arming cluster detected — board layout may need adjustment")
        return 1

    arming_cluster = arming_clusters[0]
    arming_positions = arming_cluster.positions | arming_cluster.wild_positions

    _print_step(6, "ARMING CLUSTER DETECTION",
                f"{arming_cluster.symbol.name} x{arming_cluster.size} cluster found")
    _print_board(board, highlights=arming_positions,
                 rocket_pos=centroid, rocket_orient=orientation)

    # Show which position is adjacent to the rocket
    from .primitives.board import orthogonal_neighbors
    rocket_neighbors = set(orthogonal_neighbors(centroid, config.board))
    adjacent_cells = arming_positions & rocket_neighbors
    for adj in adjacent_cells:
        print(f"  ({adj.reel},{adj.row}) is orthogonal neighbor of rocket at "
              f"({centroid.reel},{centroid.row})")

    # -----------------------------------------------------------------------
    # Step 7: Rocket armed — transition DORMANT -> ARMED
    # -----------------------------------------------------------------------
    armed_list = tracker.arm_adjacent(arming_positions)

    _print_step(7, "ROCKET ARMED",
                f"{len(armed_list)} booster(s) armed by adjacent cluster")
    _print_board(board, highlights=arming_positions,
                 rocket_pos=centroid, rocket_orient=orientation)
    print()
    _print_booster_state(tracker)

    # -----------------------------------------------------------------------
    # Step 8: Booster phase — fire the armed rocket
    # -----------------------------------------------------------------------
    executor = BoosterPhaseExecutor(tracker, rules, rules.chain_initiators)
    executor.register_fire_handler(Symbol.R, fire_rocket)

    fire_results = executor.execute_booster_phase(board)

    _print_step(8, "BOOSTER PHASE (FIRE)",
                f"Rocket fires {'HORIZONTALLY across row' if orientation == 'H' else 'VERTICALLY down reel'} "
                f"{centroid.row if orientation == 'H' else centroid.reel}")

    if fire_results:
        result = fire_results[0]
        # Show the rocket's full path — includes rocket position + affected
        path = rules.rocket_path(centroid, orientation)
        path_str = " ".join(
            f"({p.reel},{p.row})" for p in sorted(path, key=lambda p: (p.row, p.reel))
        )
        print(f"  Path: {path_str}")
        print(f"  Cleared: {len(result.affected_positions)} cells")

        # Check for immune symbols that survived
        immune_in_path = [
            p for p in path
            if p != centroid and board.get(p) is not None
            and board.get(p) in rules.immune_to_rocket
        ]
        if immune_in_path:
            for ip in immune_in_path:
                print(f"  Immune: {board.get(ip).name} at ({ip.reel},{ip.row}) survived")
        else:
            print("  Immune: none in path")

        # Highlight the fire path on the board
        fire_highlights = frozenset(result.affected_positions | {centroid})
        print()
        _print_board(board, highlights=fire_highlights,
                     rocket_pos=centroid, rocket_orient=orientation)
    print()
    _print_booster_state(tracker)

    # -----------------------------------------------------------------------
    # Step 9: Post-fire board — clear affected positions and rocket
    # -----------------------------------------------------------------------
    if fire_results:
        for pos in fire_results[0].affected_positions:
            board.set(pos, None)
    # Rocket is consumed by firing
    board.set(centroid, None)

    _print_step(9, "POST-FIRE BOARD",
                "Cleared cells shown as vacancies, rocket consumed")
    _print_board(board)

    print()
    print(_MAJOR_SEP)
    print("  LIFECYCLE COMPLETE: DORMANT -> ARMED -> FIRED")
    print(_MAJOR_SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
