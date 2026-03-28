"""Deterministic pull-model gravity system.

Symbols fall into empty cells created by cluster explosions. The GravityDAG
is precomputed once at engine init and reused across all spins.

Pull-model rules (from GravityDesign.md):
- Rule 1: Prefer straight-down fall (directly above donor)
- Rule 2: Diagonal fallback when straight-down is blocked (down-left before down-right)
- Rule 3: A donor MUST fall straight if its below cell is empty — no diagonal donation
- Rule 4: Directly-above donor wins when multiple donors compete
- Rule 5: Multi-pass iterations until stable
- Rule 6: Conservation — N explosions = N empty cells after settle
- Rule 7: Wilds and boosters fall with gravity, NOT refilled

Donor self-interest: a potential diagonal donor that has its own straight-down
empty target will not donate diagonally.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import BoardConfig, GravityConfig
from .board import Board, Position


@dataclass(frozen=True, slots=True)
class SettleResult:
    """Result of a gravity settle operation."""

    board: Board
    # Moves grouped by pass — each move is (source, destination)
    move_steps: tuple[tuple[tuple[Position, Position], ...], ...]
    # Positions needing refill after settle (empty cells at top of columns)
    empty_positions: tuple[Position, ...]


class GravityDAG:
    """Precomputed gravity donor priority graph.

    For each cell position, stores an ordered tuple of potential donor
    positions, sorted by priority (straight-down first, then diagonals).
    Constructed once per game config, immutable after init.
    """

    __slots__ = ("_donors", "_board_config", "_gravity_config")

    def __init__(self, board_config: BoardConfig, gravity_config: GravityConfig) -> None:
        self._board_config = board_config
        self._gravity_config = gravity_config

        # Precompute donor lookup: for each empty cell, which cells could donate?
        # Donor priorities are (dx, dy) offsets FROM the empty cell TO the donor.
        # (0, -1) = directly above, (1, -1) = above-right, (-1, -1) = above-left
        self._donors: dict[Position, tuple[Position, ...]] = {}
        for reel in range(board_config.num_reels):
            for row in range(board_config.num_rows):
                pos = Position(reel, row)
                donors: list[Position] = []
                for dx, dy in gravity_config.donor_priorities:
                    donor_reel = reel + dx
                    donor_row = row + dy
                    if (0 <= donor_reel < board_config.num_reels
                            and 0 <= donor_row < board_config.num_rows):
                        donors.append(Position(donor_reel, donor_row))
                self._donors[pos] = tuple(donors)

    @property
    def board_config(self) -> BoardConfig:
        return self._board_config

    @property
    def gravity_config(self) -> GravityConfig:
        return self._gravity_config

    def donors_for(self, pos: Position) -> tuple[Position, ...]:
        """Ordered donor candidates for the given empty cell position."""
        return self._donors.get(pos, ())


def settle(
    dag: GravityDAG,
    board: Board,
    exploded: frozenset[Position],
    config: GravityConfig,
) -> SettleResult:
    """Apply gravity to a board after positions have been exploded.

    Deterministic: same input always produces same output.
    Multi-pass bottom-up pull model until stable or max_settle_passes reached.
    """
    board = board.copy()
    board_config = dag.board_config

    # Mark exploded positions as empty
    for pos in exploded:
        board.set(pos, None)

    all_move_steps: list[tuple[tuple[Position, Position], ...]] = []

    for _ in range(config.max_settle_passes):
        pass_moves: list[tuple[Position, Position]] = []

        # Track which cells have already donated this pass to prevent double-moves
        donated_this_pass: set[Position] = set()

        # Scan bottom-up (highest row first), left-to-right within each row
        for row in range(board_config.num_rows - 1, -1, -1):
            for reel in range(board_config.num_reels):
                pos = Position(reel, row)
                if board.get(pos) is not None:
                    continue  # Not empty — nothing to pull

                # Try donors in priority order
                for donor_pos in dag.donors_for(pos):
                    if donor_pos in donated_this_pass:
                        continue
                    if board.get(donor_pos) is None:
                        continue  # Donor is also empty

                    is_straight_down = (donor_pos.reel == pos.reel)

                    if not is_straight_down:
                        # Rule 3 + self-interest: donor must NOT donate diagonally
                        # if its own straight-below cell is empty
                        straight_below = Position(donor_pos.reel, donor_pos.row + 1)
                        if (straight_below.row < board_config.num_rows
                                and board.get(straight_below) is None):
                            continue

                    # This donor can donate — execute the move
                    board.set(pos, board.get(donor_pos))
                    board.set(donor_pos, None)
                    pass_moves.append((donor_pos, pos))
                    donated_this_pass.add(donor_pos)
                    break  # This empty cell is now filled

        if not pass_moves:
            break  # Stable — no moves this pass

        all_move_steps.append(tuple(pass_moves))

    # Collect remaining empty positions
    empty = tuple(
        Position(reel, row)
        for reel in range(board_config.num_reels)
        for row in range(board_config.num_rows)
        if board.get(Position(reel, row)) is None
    )

    return SettleResult(
        board=board,
        move_steps=tuple(all_move_steps),
        empty_positions=empty,
    )


def build_gravity_mappings(
    move_steps: tuple[tuple[tuple[Position, Position], ...], ...],
    board_config: BoardConfig,
    excluded: frozenset[Position] = frozenset(),
) -> tuple[dict[Position, Position], dict[Position, list[Position]]]:
    """Build pre→post and post→pre gravity position mappings.

    Tracks each non-excluded cell through all gravity move passes to find
    its final post-gravity position. Used by PostGravityAdjacency (for
    virtual adjacency) and BridgePathTracer (for backward path mapping).

    Returns:
        (pre_to_post, post_to_pre) — bidirectional mapping dicts.
        pre_to_post maps each surviving pre-gravity position to its
        post-gravity destination. post_to_pre maps each post-gravity
        position to all pre-gravity origins that land there.
    """
    all_positions = [
        Position(reel, row)
        for reel in range(board_config.num_reels)
        for row in range(board_config.num_rows)
    ]

    # Every surviving cell starts mapped to itself
    pre_to_post: dict[Position, Position] = {
        pos: pos for pos in all_positions
        if pos not in excluded
    }

    # Chase each cell through gravity passes — A→B in pass 1, B→C in
    # pass 2 means A's final position is C
    for pass_moves in move_steps:
        move_map: dict[Position, Position] = {}
        for source, dest in pass_moves:
            move_map[source] = dest

        for pre_pos in pre_to_post:
            current_post = pre_to_post[pre_pos]
            if current_post in move_map:
                pre_to_post[pre_pos] = move_map[current_post]

    # Reverse mapping — post-gravity position to all pre-gravity origins
    post_to_pre: dict[Position, list[Position]] = {}
    for pre_pos, post_pos in pre_to_post.items():
        if post_pos not in post_to_pre:
            post_to_pre[post_pos] = []
        post_to_pre[post_pos].append(pre_pos)

    return pre_to_post, post_to_pre


def predict_empty_cells(
    dag: GravityDAG,
    exploded: frozenset[Position],
    board_config: BoardConfig,
) -> list[Position]:
    """Predict which cells will be empty after gravity settle.

    By the conservation law (Rule 6), the count equals len(exploded).
    Empty cells float to the top of each column proportional to how many
    cells were exploded in that column (accounting for diagonal movement
    redistributing across columns).

    This is a simplified prediction — for exact results, use settle().
    The prediction counts explosions per column and marks that many top
    rows as empty per column.
    """
    # Count explosions per column (reel)
    per_column: dict[int, int] = {}
    for pos in exploded:
        per_column[pos.reel] = per_column.get(pos.reel, 0) + 1

    result: list[Position] = []
    for reel in range(board_config.num_reels):
        count = per_column.get(reel, 0)
        # Empty cells bubble to the top of the column
        for row in range(count):
            result.append(Position(reel, row))

    return result
