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
from ..primitives.gravity import GravityDAG, SettleResult, build_gravity_mappings, settle


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

        Uses build_gravity_mappings() for pre↔post position tracking, then
        maps orthogonal neighbors through the post-gravity coordinate space
        back to pre-gravity coordinates for WFC constraint propagation.
        """
        pre_to_post, post_to_pre = build_gravity_mappings(
            self._settle_result.move_steps, board_config,
            excluded=planned_explosion,
        )

        # For each pre-gravity cell, find post-gravity neighbors
        # and map them back to pre-gravity coordinates
        virtual_adj: dict[Position, list[Position]] = {}
        for pre_pos, post_pos in pre_to_post.items():
            neighbors_pre: list[Position] = []
            for post_neighbor in orthogonal_neighbors(post_pos, board_config):
                pre_neighbors = post_to_pre.get(post_neighbor, [])
                for pre_n in pre_neighbors:
                    if pre_n != pre_pos:
                        neighbors_pre.append(pre_n)
            virtual_adj[pre_pos] = neighbors_pre

        return virtual_adj
