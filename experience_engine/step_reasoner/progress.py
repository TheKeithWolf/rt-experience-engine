"""Progress tracking — cumulative cascade state against archetype targets.

ClusterRecord is an immutable snapshot of one detected cluster.
ProgressTracker accumulates step results and provides query methods
the reasoner uses to decide what happens next.

Pure state + queries — no game logic. Budget computations use
evaluators; this module only tracks what has happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..archetypes.registry import ArchetypeSignature
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol, SymbolTier, is_wild
from .context import DormantBooster

if TYPE_CHECKING:
    from ..config.schema import BoardConfig
    from .results import StepResult


@dataclass(frozen=True, slots=True)
class ClusterRecord:
    """Immutable record of one detected cluster from a completed step."""

    symbol: Symbol
    size: int
    positions: frozenset[Position]
    step_index: int
    # Centipayout contribution from this cluster
    payout: int


@dataclass
class ProgressTracker:
    """Mutable accumulator tracking cascade progress against archetype targets.

    Created at cascade start from an ArchetypeSignature. Updated after each
    step via update(). The reasoner queries remaining_*() and must_*()
    methods to decide the next step's intent.
    """

    signature: ArchetypeSignature
    # Centipayout multiplier from config — converts centipayout ↔ bet multiplier
    centipayout_multiplier: int

    steps_completed: int = 0
    clusters_produced: list[ClusterRecord] = field(default_factory=list)
    # Booster type name → spawn/fire count
    boosters_spawned: dict[str, int] = field(default_factory=dict)
    boosters_fired: dict[str, int] = field(default_factory=dict)
    chain_depth_max: int = 0
    dormant_boosters: list[DormantBooster] = field(default_factory=list)
    active_wilds: list[Position] = field(default_factory=list)
    # Running centipayout total across all steps
    cumulative_payout: int = 0
    scatter_positions: list[Position] = field(default_factory=list)
    # Step index → symbol tier observed (for narrative arc tracking)
    symbol_tiers_by_step: dict[int, SymbolTier] = field(default_factory=dict)

    # Phase-tracking state for NarrativeArc-based archetypes.
    # Tracks which phase we're in and how many repetitions of the current phase
    # have been consumed. Used by is_satisfied(), current_step_size_ranges(),
    # and the assessor to query phase-level constraints.
    current_phase_index: int = 0
    current_phase_repetitions: int = 0
    # Number of repetitions completed per phase (indexed by phase position)
    phases_completed: list[int] = field(default_factory=list)

    # -- Query methods (read-only) -----------------------------------------

    @staticmethod
    def _remaining_range(required: Range, done: int) -> Range:
        """Compute how much of a Range budget remains after *done* units spent.

        Clamps both min and max to zero so overproduction yields Range(0, 0)
        instead of crashing Range.__post_init__ with a negative max_val.
        """
        remaining_min = max(0, required.min_val - done)
        remaining_max = max(remaining_min, required.max_val - done)
        return Range(min_val=remaining_min, max_val=remaining_max)

    def remaining_cascade_steps(self) -> Range:
        """How many more cascade steps are needed (min) / allowed (max)."""
        return self._remaining_range(
            self.signature.required_cascade_depth, self.steps_completed,
        )

    def remaining_booster_spawns(self) -> dict[str, Range]:
        """Per booster type: how many more spawns are needed/allowed."""
        return {
            btype: self._remaining_range(sig_range, self.boosters_spawned.get(btype, 0))
            for btype, sig_range in self.signature.required_booster_spawns.items()
        }

    def remaining_booster_fires(self) -> dict[str, Range]:
        """Per booster type: how many more fires are needed/allowed."""
        return {
            btype: self._remaining_range(sig_range, self.boosters_fired.get(btype, 0))
            for btype, sig_range in self.signature.required_booster_fires.items()
        }

    def remaining_payout_budget(self) -> RangeFloat:
        """Payout range still available, in bet-multiplier units."""
        spent = self.cumulative_payout / self.centipayout_multiplier
        remaining_min = max(0.0, self.signature.payout_range.min_val - spent)
        remaining_max = max(remaining_min, self.signature.payout_range.max_val - spent)
        return RangeFloat(min_val=remaining_min, max_val=remaining_max)

    def must_terminate_soon(self) -> bool:
        """True if at most one cascade step remains (max)."""
        return self.remaining_cascade_steps().max_val <= 1

    def can_still_spawn(self, booster_type: str) -> bool:
        """True if the signature allows more spawns of this booster type."""
        remaining = self.remaining_booster_spawns()
        if booster_type not in remaining:
            return False
        return remaining[booster_type].max_val > 0

    def needs_to_fire(self, booster_type: str) -> bool:
        """True if the signature still requires fires of this booster type."""
        remaining = self.remaining_booster_fires()
        if booster_type not in remaining:
            return False
        return remaining[booster_type].min_val > 0

    def is_satisfied(self) -> bool:
        """True when all cascade-phase minimums are met — ready to terminate.

        For arc-based archetypes: all non-skippable phases have met their
        minimum repetitions. For legacy archetypes: cascade_steps completion.
        Both check cascade depth, booster spawns, and payout budget.
        Booster fires are validated at instance level after the post-terminal
        booster phase — they do not gate cascade termination.
        """
        arc = self.signature.narrative_arc

        if arc is not None:
            # Arc-based: verify all phases with min > 0 have been completed
            for phase_i, phase in enumerate(arc.phases):
                reps_done = (
                    self.current_phase_repetitions
                    if phase_i == self.current_phase_index
                    else (self.phases_completed[phase_i]
                          if phase_i < len(self.phases_completed) else 0)
                )
                if reps_done < phase.repetitions.min_val:
                    return False
        elif (self.signature.cascade_steps is not None
                and self.steps_completed < len(self.signature.cascade_steps)):
            # Legacy: all mandatory cascade steps must have been executed
            return False

        # Cascade depth minimum met
        if self.remaining_cascade_steps().min_val > 0:
            return False

        # All booster spawn minimums met
        for remaining in self.remaining_booster_spawns().values():
            if remaining.min_val > 0:
                return False

        # Payout minimum met
        if self.remaining_payout_budget().min_val > 0:
            return False

        return True

    def current_step_size_ranges(self) -> tuple[Range, ...]:
        """Effective cluster size ranges for the current step.

        For arc-based archetypes: reads from the current phase's cluster_sizes.
        For legacy archetypes: reads from cascade_steps[step_idx].
        Falls back to signature-level sizes in all cases.
        """
        arc = self.signature.narrative_arc

        # Arc-based: read from the current phase
        if arc is not None:
            if self.current_phase_index < len(arc.phases):
                phase_sizes = arc.phases[self.current_phase_index].cluster_sizes
                if phase_sizes:
                    return phase_sizes
            return self.signature.required_cluster_sizes

        # Legacy: read from cascade_steps
        if self.signature.cascade_steps is not None:
            step_idx = self.steps_completed
            if step_idx < len(self.signature.cascade_steps):
                step_sizes = self.signature.cascade_steps[step_idx].cluster_sizes
                if step_sizes is not None:
                    return step_sizes

        return self.signature.required_cluster_sizes

    def current_phase(self):
        """Return the current NarrativePhase, or None if no arc or past end."""
        arc = self.signature.narrative_arc
        if arc is not None and self.current_phase_index < len(arc.phases):
            return arc.phases[self.current_phase_index]
        return None

    def peek_next_phase(self):
        """Return the next NarrativePhase (after current), or None if at end.

        If the current phase is still repeating (reps < max), the "next step"
        is still in the same phase — return the current phase. Otherwise return
        the phase after current.
        """
        arc = self.signature.narrative_arc
        if arc is None:
            return None
        phase = self.current_phase()
        if phase is not None and self.current_phase_repetitions < phase.repetitions.max_val:
            return phase
        next_idx = self.current_phase_index + 1
        if next_idx < len(arc.phases):
            return arc.phases[next_idx]
        return None

    def advance_phase(self) -> None:
        """Advance to the next phase, recording repetitions completed.

        Called when the current phase's transition predicate fires or max
        repetitions are reached.
        """
        # Extend phases_completed to cover up to current_phase_index
        while len(self.phases_completed) <= self.current_phase_index:
            self.phases_completed.append(0)
        self.phases_completed[self.current_phase_index] = self.current_phase_repetitions
        self.current_phase_index += 1
        self.current_phase_repetitions = 0

    # -- Mutation method (called once per step) -----------------------------

    def update(self, step_result: StepResult) -> None:
        """Advance progress from a completed step.

        Called by the simulation loop after execution and validation.
        Increments counters, appends records, and maintains dormant
        booster list (spawns added, fires removed by position).
        """
        self.steps_completed += 1

        # Cluster tracking
        for cluster in step_result.clusters:
            self.clusters_produced.append(cluster)
            self.cumulative_payout += cluster.payout

        # Booster spawn tracking — route Wilds to active_wilds, others to dormant
        for spawn in step_result.spawns:
            self.boosters_spawned[spawn.booster_type] = (
                self.boosters_spawned.get(spawn.booster_type, 0) + 1
            )
            if spawn.booster_type == "W":
                # Wilds are passive board symbols, not dormant boosters
                self.active_wilds.append(spawn.position)
            else:
                self.dormant_boosters.append(
                    DormantBooster(
                        booster_type=spawn.booster_type,
                        position=spawn.position,
                        orientation=None,
                        spawned_step=spawn.step_index,
                    )
                )

        # Booster fire tracking — remove from dormant list by position
        fired_positions: set[Position] = set()
        for fire in step_result.fires:
            self.boosters_fired[fire.booster_type] = (
                self.boosters_fired.get(fire.booster_type, 0) + 1
            )
            fired_positions.add(fire.position)

        if fired_positions:
            self.dormant_boosters = [
                b for b in self.dormant_boosters
                if b.position not in fired_positions
            ]

        # Narrative arc — record symbol tier for this step
        if step_result.symbol_tier is not None:
            self.symbol_tiers_by_step[step_result.step_index] = step_result.symbol_tier

        # Phase tracking — increment repetitions for arc-based archetypes
        if self.signature.narrative_arc is not None:
            self.current_phase_repetitions += 1

    def sync_active_wilds(self, board: Board, board_config: BoardConfig) -> None:
        """Rebuild active_wilds from actual board state.

        Called after transition — handles both consumption (wilds exploded by
        clusters) and gravity position drift in one pass. Board is the single
        source of truth; no duplicate tracking needed.
        """
        self.active_wilds = [
            Position(reel, row)
            for reel in range(board_config.num_reels)
            for row in range(board_config.num_rows)
            if (sym := board.get(Position(reel, row))) is not None and is_wild(sym)
        ]
