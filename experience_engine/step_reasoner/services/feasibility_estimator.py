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
        """Return True if both the immediately-next phase fits AND no
        future phase exceeds the steady-state refill floor.

        The next-phase check uses the *current* empty-cell pool (each phase
        has its own explosion → settle → refill cycle, so we cannot sum
        across phases). The bottleneck check (A2) scans all remaining
        phases and rejects if any single phase would need more cells than
        the steady-state refill floor can plausibly deliver — catching dead
        ends two or three phases out without the Defect-1 regression of
        summing requirements.
        """
        arc = progress.signature.narrative_arc
        if arc is None:
            return True

        remaining = arc.phases[progress.current_phase_index:]
        if not remaining:
            return True

        if not self._single_phase_ok(remaining[0], context):
            return False
        return self._estimate_bottleneck(remaining, context)

    def _single_phase_ok(
        self,
        phase: NarrativePhase,
        context: BoardContext,
    ) -> bool:
        """Immediate-next-phase fit on the current empty-cell pool."""
        required = self._required_cells_for_phase(phase, context)
        return required <= len(context.empty_cells)

    def _estimate_bottleneck(
        self,
        remaining: tuple[NarrativePhase, ...],
        context: BoardContext,
    ) -> bool:
        """A2: scan all remaining phases for a single-phase bottleneck.

        Rejects when any phase's cell requirement exceeds the steady-state
        empties floor — the current empty pool plus the refill that a
        cluster of min_cluster_size (the smallest game-rule-allowed cluster)
        would deliver on a single explosion. This is **not** a sum across
        phases; it's still a single-phase comparison, just applied to the
        worst phase, not only the next one.
        """
        worst = max(
            self._required_cells_for_phase(phase, context)
            for phase in remaining
        )
        # Steady-state floor: one min_cluster_size explosion frees that many
        # cells per cycle, on top of the current empty pool.
        steady_state_floor = len(context.empty_cells) + self._min_cluster_size
        return worst <= steady_state_floor

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
