"""Bridge path tracer — deterministic BFS from wild landing to refill zone.

Traces the shortest path of surviving cells between a Wild's post-gravity
landing position and the refill zone boundary, then maps those post-gravity
positions back to pre-gravity coordinates for bridge symbol placement.

Used by: InitialWildBridgeStrategy (exclusive consumer).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ...config.schema import BoardConfig
from ...primitives.board import Position, orthogonal_neighbors
from ...primitives.gravity import SettleResult, build_gravity_mappings


@dataclass(frozen=True, slots=True)
class BridgePlan:
    """Result of backward path tracing — what InitialWildBridgeStrategy
    needs to set up the board for WildBridgeStrategy.

    path_pre_to_post maps pre-gravity positions to their post-gravity
    destinations along the traced path. InitialWildBridgeStrategy assigns
    bridge_symbol to each key (pre-gravity coordinate).

    shortfall is how many additional cells WildBridgeStrategy must place
    from refill. Derived: target_size - len(path) - 1 (wild counts as 1).

    refill_centroid is the post-gravity center of mass of the refill zone,
    used for reserve zone computation by spatial intelligence.
    """

    path_pre_to_post: dict[Position, Position]
    shortfall: int
    refill_centroid: Position


class BridgePathTracer:
    """Traces the bridge path from Wild landing to refill zone.

    Injected into InitialWildBridgeStrategy. Stateless — each call
    produces a fresh BridgePlan from the inputs.
    """

    __slots__ = ("_board_config",)

    def __init__(self, board_config: BoardConfig) -> None:
        self._board_config = board_config

    def plan(
        self,
        wild_landing: Position,
        settle_result: SettleResult,
        target_bridge_size: int,
    ) -> BridgePlan:
        """Trace path and compute the bridge plan.

        Steps (matching design doc Step -1):
        1. Compute refill centroid from settle_result.empty_positions
        2. BFS from wild_landing toward refill centroid through
           surviving (non-empty) cells on settle_result.board
        3. Build pre↔post gravity mappings from settle_result.move_steps
        4. Map path positions back to pre-gravity cell positions
        5. Compute shortfall = target_size - path_length - 1
        """
        if not settle_result.empty_positions:
            raise ValueError(
                "No empty positions in settle result — cannot trace bridge path"
            )

        refill_centroid = _compute_centroid(settle_result.empty_positions)

        # BFS through surviving cells from wild toward refill zone boundary
        path = self._trace_path(wild_landing, settle_result, refill_centroid)

        # Map post-gravity path positions back to pre-gravity coordinates
        _, post_to_pre = build_gravity_mappings(
            settle_result.move_steps, self._board_config,
        )

        path_pre_to_post: dict[Position, Position] = {}
        for post_pos in path:
            pre_origins = post_to_pre.get(post_pos, [])
            if pre_origins:
                # Use the first pre-gravity origin — deterministic since
                # build_gravity_mappings iterates in consistent reel/row order
                path_pre_to_post[pre_origins[0]] = post_pos

        # Wild itself counts as 1 toward the bridge target size
        shortfall = max(0, target_bridge_size - len(path) - 1)

        return BridgePlan(
            path_pre_to_post=path_pre_to_post,
            shortfall=shortfall,
            refill_centroid=refill_centroid,
        )

    def _trace_path(
        self,
        wild_landing: Position,
        settle_result: SettleResult,
        refill_centroid: Position,
    ) -> list[Position]:
        """BFS from wild through surviving cells to refill-zone boundary.

        Returns the ordered path of surviving cell positions (post-gravity
        coordinates, excluding the wild itself). The path ends at the first
        cell adjacent to the refill zone.

        Neighbors are sorted by manhattan distance to refill_centroid for
        deterministic tie-breaking — ensures reproducible paths.
        """
        refill_set = frozenset(settle_result.empty_positions)
        board = settle_result.board

        # Track visited positions and parent pointers for path reconstruction
        visited: set[Position] = {wild_landing}
        parent: dict[Position, Position] = {}
        queue: deque[Position] = deque([wild_landing])
        target: Position | None = None

        while queue and target is None:
            current = queue.popleft()

            # Sort neighbors by manhattan distance to centroid for deterministic ordering
            neighbors = sorted(
                orthogonal_neighbors(current, self._board_config),
                key=lambda p: _manhattan(p, refill_centroid),
            )

            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                # Only traverse non-empty surviving cells (not refill zone)
                if neighbor in refill_set:
                    # Current cell is adjacent to refill — it's the path endpoint
                    target = current
                    break

                if board.get(neighbor) is not None:
                    parent[neighbor] = current
                    queue.append(neighbor)

        if target is None:
            # Wild is already adjacent to refill or no path exists
            return []

        # Reconstruct path from wild to target (excluding wild itself)
        path: list[Position] = []
        node = target
        while node != wild_landing:
            path.append(node)
            node = parent[node]
        path.reverse()

        return path


def _manhattan(a: Position, b: Position) -> int:
    """Manhattan distance between two board positions."""
    return abs(a.reel - b.reel) + abs(a.row - b.row)


def _compute_centroid(positions: tuple[Position, ...]) -> Position:
    """Compute the integer centroid of a set of positions.

    Uses integer division — the centroid is the nearest grid cell to
    the geometric center of the refill zone.
    """
    total = len(positions)
    reel_sum = sum(p.reel for p in positions)
    row_sum = sum(p.row for p in positions)
    return Position(reel_sum // total, row_sum // total)
