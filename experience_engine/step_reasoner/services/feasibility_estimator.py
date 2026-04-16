"""Feasibility estimator — fast reject when the next phase cannot fit on the board.

Checks only the immediately-next phase's minimum cluster footprint against
the current empty-cell pool. Each phase has its own explosion → settle →
refill cycle, so subsequent phases operate on a refill zone that doesn't
exist yet — summing across phases would double-count cells.

Bridge phases receive special handling: they reuse existing wild + reachable
same-symbol cells, so only a shortfall (bounded by min_cluster_size) is needed
rather than the full cluster footprint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...config.schema import BoardConfig
from ..context import BoardContext
from ..progress import ProgressTracker

if TYPE_CHECKING:
    from ...narrative.arc import NarrativePhase


class FeasibilityEstimator:
    """Estimates whether the next narrative phase can fit on the board.

    Injected with board config at construction to derive bridge-phase cell
    floors from ``min_cluster_size``. Call ``estimate()`` each step.
    """

    __slots__ = ("_min_cluster_size",)

    def __init__(self, board_config: BoardConfig) -> None:
        self._min_cluster_size = board_config.min_cluster_size

    def estimate(
        self,
        progress: ProgressTracker,
        context: BoardContext,
    ) -> bool:
        """Return True if the next phase's cell requirement fits current empties.

        Only the immediately-next phase is checked. Each phase has its own
        explosion → settle → refill cycle, so subsequent phases get their own
        empty-cell pool.
        """
        arc = progress.signature.narrative_arc
        if arc is None:
            return True

        remaining = arc.phases[progress.current_phase_index:]
        if not remaining:
            return True

        required = self._required_cells_for_phase(remaining[0], context)
        return required <= len(context.empty_cells)

    def _required_cells_for_phase(
        self,
        phase: NarrativePhase,
        context: BoardContext,
    ) -> int:
        """Minimum empty cells the phase needs to place its cluster.

        Bridge phases reuse existing wild + reachable same-symbol cells and
        place only a shortfall; min_cluster_size provides the maneuvering
        room WildBridgeStrategy needs without overstating demand.
        """
        if not phase.cluster_sizes:
            return 0

        # Bridge reuse only applies when a wild is already on the board
        if phase.wild_behavior == "bridge" and context.active_wilds:
            return self._min_cluster_size

        return phase.cluster_sizes[0].min_val * phase.cluster_count.min_val
