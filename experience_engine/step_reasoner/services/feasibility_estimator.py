"""Feasibility estimator — fast reject of arcs whose remaining phases cannot
fit on the board.

Sums the minimum cluster footprint across every phase still to run, compared
against the empty-cell pool. If the requirement exceeds what the board can
provide, the assessor short-circuits to terminal-dead rather than burning
the retry budget on impossible configurations.

Conservative by design — uses each phase's minimum size × minimum count so
only truly infeasible arcs are rejected.
"""

from __future__ import annotations

from ..context import BoardContext
from ..progress import ProgressTracker


class FeasibilityEstimator:
    """Estimates whether the remaining narrative phases can fit on the board.

    One instance per registry build (stateless); call `estimate()` each step.
    Consumes only the progress tracker and current board context — no rule
    lookups, no rng. Returns True when feasible, False when the minimum
    cluster footprint exceeds the empty-cell pool.
    """

    __slots__ = ()

    def estimate(
        self,
        progress: ProgressTracker,
        context: BoardContext,
    ) -> bool:
        """Return True if every remaining phase can still fit on the board."""
        arc = progress.signature.narrative_arc
        if arc is None:
            # Legacy archetypes without an arc — no phase-level sizing to check
            return True

        remaining_phases = arc.phases[progress.current_phase_index:]
        if not remaining_phases:
            return True

        required = 0
        for phase in remaining_phases:
            # cluster_sizes may be empty for terminal/spec phases — skip those
            if not phase.cluster_sizes:
                continue
            min_size = phase.cluster_sizes[0].min_val
            min_count = phase.cluster_count.min_val
            required += min_size * min_count

        return required <= len(context.empty_cells)
