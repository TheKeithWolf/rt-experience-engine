"""Post-gravity virtual adjacency — maps pre-gravity cell positions to their
post-gravity neighbors for gravity-aware WFC constraint propagation.

After a cluster explodes and gravity settles, cells shift downward. Two cells
that are NOT adjacent pre-gravity may become adjacent post-gravity. This module
computes that virtual adjacency graph once, and both PostGravityPropagator and
GravityGroupComputer consume it (DRY — single settle() call, two consumers).

Uses settle() from primitives/gravity.py as the single source of truth for
gravity simulation — no reimplementation.
"""

from __future__ import annotations

from ..config.schema import BoardConfig, GravityConfig
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.gravity import GravityDAG, SettleResult, settle


class PostGravityAdjacency:
    """Virtual adjacency graph: which cells will be orthogonally adjacent
    after gravity settles following a planned explosion.

    Constructed once per fill_step() call. The virtual_neighbors() method
    returns pre-gravity coordinates mapped through post-gravity adjacency —
    the WFC solver works in pre-gravity space but needs to know what will
    be adjacent after gravity.
    """

    __slots__ = ("_virtual_adj", "_settle_result")

    def __init__(
        self,
        board: Board,
        planned_explosion: frozenset[Position],
        gravity_dag: GravityDAG,
        gravity_config: GravityConfig,
        board_config: BoardConfig,
    ) -> None:
        # Single settle() call — reuses the gravity engine's deterministic simulation
        self._settle_result = settle(
            gravity_dag, board, planned_explosion, gravity_config,
        )

        self._virtual_adj = self._compute(
            board_config, planned_explosion,
        )

    @property
    def settle_result(self) -> SettleResult:
        """The gravity settle result used to build this adjacency graph."""
        return self._settle_result

    def virtual_neighbors(self, pos: Position) -> list[Position]:
        """Pre-gravity positions that will be orthogonally adjacent to pos
        after gravity settles. Returns empty list for positions not tracked."""
        return self._virtual_adj.get(pos, [])

    def _compute(
        self,
        board_config: BoardConfig,
        planned_explosion: frozenset[Position],
    ) -> dict[Position, list[Position]]:
        """Build the virtual adjacency graph from settle result.

        Algorithm:
        1. Build pre_to_post mapping by tracking each cell through all
           gravity move passes (chained: A→B pass1, B→C pass2 ⇒ A maps to C)
        2. Exploded cells are removed — they don't map to post-gravity positions
        3. Build reverse post_to_pre mapping
        4. For each pre-gravity cell, find its post-gravity position, compute
           orthogonal neighbors in post-gravity space, map back to pre-gravity
        """
        all_positions = [
            Position(reel, row)
            for reel in range(board_config.num_reels)
            for row in range(board_config.num_rows)
        ]

        # Step 1: Track each cell's position through all gravity passes
        # Start: every cell maps to itself
        pre_to_post: dict[Position, Position] = {
            pos: pos for pos in all_positions
            if pos not in planned_explosion
        }

        # Process move passes — each pass contains (source, dest) tuples
        # A cell that moved in pass 1 to position B, then B moves to C in
        # pass 2, means the cell's final position is C
        for pass_moves in self._settle_result.move_steps:
            # Build source→dest for this pass
            move_map: dict[Position, Position] = {}
            for source, dest in pass_moves:
                move_map[source] = dest

            # Update pre_to_post: if a cell's current post position was moved,
            # follow the chain to the new position
            for pre_pos in pre_to_post:
                current_post = pre_to_post[pre_pos]
                if current_post in move_map:
                    pre_to_post[pre_pos] = move_map[current_post]

        # Step 2: Build reverse mapping (post → pre)
        post_to_pre: dict[Position, list[Position]] = {}
        for pre_pos, post_pos in pre_to_post.items():
            if post_pos not in post_to_pre:
                post_to_pre[post_pos] = []
            post_to_pre[post_pos].append(pre_pos)

        # Step 3: For each pre-gravity cell, find post-gravity neighbors
        # and map them back to pre-gravity coordinates
        virtual_adj: dict[Position, list[Position]] = {}
        for pre_pos, post_pos in pre_to_post.items():
            neighbors_pre: list[Position] = []
            for post_neighbor in orthogonal_neighbors(post_pos, board_config):
                # Map post-gravity neighbor back to pre-gravity positions
                pre_neighbors = post_to_pre.get(post_neighbor, [])
                for pre_n in pre_neighbors:
                    if pre_n != pre_pos:
                        neighbors_pre.append(pre_n)
            virtual_adj[pre_pos] = neighbors_pre

        return virtual_adj
