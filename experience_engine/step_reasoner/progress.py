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
from ..primitives.board import Position
from ..primitives.symbols import Symbol, SymbolTier
from .context import DormantBooster

if TYPE_CHECKING:
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

    # -- Query methods (read-only) -----------------------------------------

    def remaining_cascade_steps(self) -> Range:
        """How many more cascade steps are needed (min) / allowed (max)."""
        sig = self.signature.required_cascade_depth
        remaining_min = max(0, sig.min_val - self.steps_completed)
        remaining_max = max(remaining_min, sig.max_val - self.steps_completed)
        return Range(min_val=remaining_min, max_val=remaining_max)

    def remaining_booster_spawns(self) -> dict[str, Range]:
        """Per booster type: how many more spawns are needed/allowed."""
        result: dict[str, Range] = {}
        for btype, sig_range in self.signature.required_booster_spawns.items():
            done = self.boosters_spawned.get(btype, 0)
            result[btype] = Range(
                min_val=max(0, sig_range.min_val - done),
                max_val=sig_range.max_val - done,
            )
        return result

    def remaining_booster_fires(self) -> dict[str, Range]:
        """Per booster type: how many more fires are needed/allowed."""
        result: dict[str, Range] = {}
        for btype, sig_range in self.signature.required_booster_fires.items():
            done = self.boosters_fired.get(btype, 0)
            result[btype] = Range(
                min_val=max(0, sig_range.min_val - done),
                max_val=sig_range.max_val - done,
            )
        return result

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
        """True when all signature minimums are met — ready to terminate.

        Checks cascade depth, booster spawns/fires, and payout budget.
        """
        # Cascade depth minimum met
        if self.remaining_cascade_steps().min_val > 0:
            return False

        # All booster spawn minimums met
        for remaining in self.remaining_booster_spawns().values():
            if remaining.min_val > 0:
                return False

        # All booster fire minimums met
        for remaining in self.remaining_booster_fires().values():
            if remaining.min_val > 0:
                return False

        # Payout minimum met
        if self.remaining_payout_budget().min_val > 0:
            return False

        return True

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

        # Booster spawn tracking — add to dormant list
        for spawn in step_result.spawns:
            self.boosters_spawned[spawn.booster_type] = (
                self.boosters_spawned.get(spawn.booster_type, 0) + 1
            )
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
