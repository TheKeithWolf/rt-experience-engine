"""Per-cell gravity flow vectors derived from GravityDAG donor priorities.

Precomputed once at engine init, reused across all steps and instances.
Stateless query interface — never mutates board or DAG state.

The flow vector at each cell points toward the weighted centroid of its
receiver positions (cells that list this cell as a donor). Edge columns
with diagonal donors produce nonzero horizontal components, which the
column-matching heuristic in SeedPlanner previously ignored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...primitives.board import Position

if TYPE_CHECKING:
    from ...config.schema import BoardConfig
    from ...primitives.gravity import GravityDAG


class GravityFieldService:
    """Per-cell gravity flow vectors derived from GravityDAG donor priorities.

    Precomputed once at engine init, reused across all steps and instances.
    Stateless query interface — never mutates board or DAG state.
    """

    __slots__ = ("_vectors",)

    def __init__(self, dag: GravityDAG, board_config: BoardConfig) -> None:
        self._vectors: dict[Position, tuple[float, float]] = (
            self._precompute(dag, board_config)
        )

    def flow_vector(self, pos: Position) -> tuple[float, float]:
        """Unit vector of net gravity pull direction at this cell.

        Returns (0, 0) for bottom-row cells (no downward movement possible).
        """
        return self._vectors.get(pos, (0.0, 0.0))

    def alignment_score(self, pos: Position, target: Position) -> float:
        """Cosine similarity between gravity flow at pos and the direction to target.

        Returns 0.0–1.0. Higher means gravity at pos naturally flows toward target.
        Used by UtilityScorer's GravityAlignmentFactor to replace column-matching
        in SeedPlanner — a seed placed where gravity aligns with the demand centroid
        is more likely to arrive at the right position after settle.
        """
        vx, vy = self._vectors.get(pos, (0.0, 0.0))
        dx = target.reel - pos.reel
        dy = target.row - pos.row
        mag_v = (vx * vx + vy * vy) ** 0.5
        mag_d = (dx * dx + dy * dy) ** 0.5
        if mag_v < 1e-9 or mag_d < 1e-9:
            return 0.0
        dot = vx * dx + vy * dy
        return max(0.0, dot / (mag_v * mag_d))

    @staticmethod
    def _precompute(
        dag: GravityDAG, board_config: BoardConfig,
    ) -> dict[Position, tuple[float, float]]:
        """Compute net flow vector per cell from DAG donor relationships.

        For each cell, the flow vector points from the cell toward the
        weighted centroid of its receiver positions (cells it donates to).
        A cell's receivers are all cells that list it in their donors_for().
        """
        # Build reverse map: for each cell, which cells does it donate to?
        receivers: dict[Position, list[Position]] = {}
        for reel in range(board_config.num_reels):
            for row in range(board_config.num_rows):
                pos = Position(reel, row)
                receivers[pos] = []

        # Scan all cells and record reverse donor→receiver relationships
        for reel in range(board_config.num_reels):
            for row in range(board_config.num_rows):
                receiver = Position(reel, row)
                for donor in dag.donors_for(receiver):
                    receivers[donor].append(receiver)

        # Compute flow vector: centroid of receivers relative to donor position
        vectors: dict[Position, tuple[float, float]] = {}
        for pos, recv_list in receivers.items():
            if not recv_list:
                vectors[pos] = (0.0, 0.0)
                continue
            rx_sum = sum(r.reel - pos.reel for r in recv_list)
            ry_sum = sum(r.row - pos.row for r in recv_list)
            mag = (rx_sum * rx_sum + ry_sum * ry_sum) ** 0.5
            if mag < 1e-9:
                # Multiple receivers cancel out — default to pure downward
                vectors[pos] = (0.0, 1.0)
            else:
                vectors[pos] = (rx_sum / mag, ry_sum / mag)

        return vectors
