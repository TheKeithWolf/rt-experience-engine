"""Gravity-group collapse ordering — resolves WFC cells in post-gravity groups.

Standard WFC uses global min-entropy cell selection. When gravity will
rearrange the board after explosion, cells that will be adjacent post-gravity
should be resolved together to prevent contradictions. This module partitions
empty cells into connected components based on virtual adjacency, then
collapses the largest group first (most constrained = most likely to conflict).

Reuses PostGravityAdjacency from gravity_adjacency.py — computed once,
consumed by both this module and PostGravityPropagator (DRY).
"""

from __future__ import annotations

from collections import deque

from ..primitives.board import Position
from .cell_state import CellState
from .gravity_adjacency import PostGravityAdjacency


class GravityGroupComputer:
    """Partitions empty cells into connected components via virtual adjacency.

    Groups are sorted largest-first so the most constrained region is
    resolved first — larger groups have more interdependencies and benefit
    most from early resolution.
    """

    __slots__ = ("_virtual_adjacency",)

    def __init__(self, virtual_adjacency: PostGravityAdjacency) -> None:
        self._virtual_adjacency = virtual_adjacency

    def compute_groups(
        self, empty_cells: set[Position]
    ) -> list[list[Position]]:
        """Partition empty cells into connected components via virtual adjacency.

        Returns groups sorted by size descending — largest group first.
        All cells in empty_cells are accounted for (no orphans).
        """
        remaining = set(empty_cells)
        groups: list[list[Position]] = []

        while remaining:
            # BFS from an arbitrary unvisited cell
            seed = next(iter(remaining))
            group: list[Position] = []
            queue: deque[Position] = deque([seed])
            remaining.discard(seed)

            while queue:
                pos = queue.popleft()
                group.append(pos)
                for neighbor in self._virtual_adjacency.virtual_neighbors(pos):
                    if neighbor in remaining:
                        remaining.discard(neighbor)
                        queue.append(neighbor)

            groups.append(group)

        # Largest group first — most constrained region resolved first
        groups.sort(key=len, reverse=True)
        return groups


class GravityAwareEntropySelector:
    """Selects the next WFC cell to collapse using gravity-group ordering.

    Iterates groups in order (largest first). Within each group, picks the
    uncollapsed cell with minimum entropy. Falls back to global min-entropy
    when all groups are fully collapsed (handles cells added after grouping).

    Uses __slots__ — called on every WFC iteration (hot path).
    """

    __slots__ = ("_groups",)

    def __init__(self, groups: list[list[Position]]) -> None:
        self._groups = groups

    def select_next(
        self, cells: dict[Position, CellState]
    ) -> Position | None:
        """Select the next cell to collapse.

        Priority: iterate groups largest-first, within each group pick
        the uncollapsed cell with minimum entropy. Returns None when
        all cells are collapsed.
        """
        # Group-ordered selection — resolves post-gravity neighbors together
        for group in self._groups:
            best: Position | None = None
            best_entropy = float("inf")
            for pos in group:
                cs = cells.get(pos)
                if cs is None or cs.collapsed:
                    continue
                if cs.entropy < best_entropy:
                    best_entropy = cs.entropy
                    best = pos
            if best is not None:
                return best

        # Fallback: global min-entropy for cells not in any group
        # (e.g., cells that were pinned after grouping)
        fallback: Position | None = None
        fallback_entropy = float("inf")
        for pos, cs in cells.items():
            if not cs.collapsed and cs.entropy < fallback_entropy:
                fallback_entropy = cs.entropy
                fallback = pos

        return fallback
