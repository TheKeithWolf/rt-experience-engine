"""Merge policy definitions for survivor-aware cluster placement.

When a cluster is placed in an empty region on a settled board, it may merge
with surviving symbols of the same type at the boundary. MergePolicy tells
the ClusterBuilder HOW to handle this: avoid the merge, accept it, or exploit
it for a larger cluster.

ClusterPositionResult carries the placement decision plus merge information
so strategies can use total_size (not just planned count) for spawn detection
and payout estimation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ...primitives.board import Position
from ...primitives.symbols import Symbol


class MergePolicy(enum.Enum):
    """How the strategy wants the ClusterBuilder to handle survivor merges.

    AVOID: Build the cluster without touching survivors if possible.
    ACCEPT: Allow merges and place fewer new cells (survivors contribute to cluster size).
    EXPLOIT: Deliberately merge to push total size past a booster spawn threshold.
    """

    AVOID = "avoid"
    ACCEPT = "accept"
    EXPLOIT = "exploit"


@dataclass(frozen=True, slots=True)
class ClusterPositionResult:
    """Position selection result with full merge information.

    Strategies use total_size (not len(planned_positions)) for spawn detection
    and payout estimation — the detected cluster will include both planned
    and merged positions.
    """

    # Cells the strategy fills with the cluster symbol → StepIntent.constrained_cells
    planned_positions: frozenset[Position]

    # Existing board cells that will become part of the cluster (already correct symbol).
    # NOT filled — the strategy needs these for size tracking only.
    merged_survivor_positions: frozenset[Position]

    # planned + merged — the actual cluster size BFS will detect
    total_size: int

    # True if any survivors join the cluster (strategy may need to adjust
    # expected_spawns or payout based on the larger true cluster size)
    merge_occurred: bool
