"""Step assessment — pure derivation of board situation for strategy selection.

StepAssessment is a frozen snapshot of every decision-relevant fact about
the current cascade step. StepAssessor produces it from observable state
(BoardContext, ProgressTracker) plus the archetype's structural constraints.

No game logic lives here — only situation analysis. The selector reads the
assessment to pick the right strategy; the strategy reads the assessment
to calibrate its intent.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..archetypes.registry import ArchetypeSignature, TerminalNearMissSpec
from ..config.schema import ReasonerConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Position
from ..primitives.symbols import SymbolTier
from .context import BoardContext, DormantBooster
from .evaluators import ChainEvaluator, PayoutEstimator, SpawnEvaluator
from .progress import ProgressTracker


@dataclass(frozen=True, slots=True)
class StepAssessment:
    """Frozen snapshot of every decision-relevant fact about the current step.

    All fields are derived from BoardContext, ProgressTracker, and the
    ArchetypeSignature — no new game logic. The selector and strategies
    read these fields instead of reaching into raw state.
    """

    steps_remaining: Range
    is_first_step: bool
    must_terminate_now: bool
    should_terminate_soon: bool
    # Booster type → minimum remaining spawns still needed
    needs_booster_spawn: dict[str, int]
    needs_chain: bool
    needs_wild_bridge: bool
    payout_remaining: RangeFloat
    dormant_boosters_to_arm: tuple[DormantBooster, ...]
    wilds_available_for_bridge: tuple[Position, ...]
    required_tier_this_step: SymbolTier | None
    terminal_near_misses_required: TerminalNearMissSpec | None
    dormant_boosters_must_survive: tuple[str, ...] | None
    signature_is_dead_family: bool
    booster_needs_arming_soon: bool
    payout_running_low: bool
    payout_running_high: bool


class StepAssessor:
    """Derives a StepAssessment from current board state and progress.

    Pure derivation — no side effects, no mutation. Injected evaluators
    provide rule lookups; all thresholds come from ReasonerConfig.
    """

    __slots__ = ("_spawn_eval", "_chain_eval", "_payout_eval", "_config")

    def __init__(
        self,
        spawn_evaluator: SpawnEvaluator,
        chain_evaluator: ChainEvaluator,
        payout_estimator: PayoutEstimator,
        config: ReasonerConfig,
    ) -> None:
        self._spawn_eval = spawn_evaluator
        self._chain_eval = chain_evaluator
        self._payout_eval = payout_estimator
        self._config = config

    def assess(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
    ) -> StepAssessment:
        """Derive a complete assessment from observable state.

        Delegates to ProgressTracker query methods for remaining budgets
        and computes derived flags (urgency, pacing) from config thresholds.
        """
        steps_remaining = progress.remaining_cascade_steps()
        is_first_step = progress.steps_completed == 0

        # Booster needs — only types with unmet minimums
        needs_spawn = {
            btype: remaining.min_val
            for btype, remaining in progress.remaining_booster_spawns().items()
            if remaining.min_val > 0
        }

        # Chain depth still unmet
        needs_chain = signature.required_chain_depth.min_val > progress.chain_depth_max

        # Wild bridge — explicit cascade_steps directive or inferred from board state
        needs_wild_bridge = self._infer_needs_wild_bridge(
            context, progress, signature,
        )

        payout_remaining = progress.remaining_payout_budget()

        # Symbol tier — from NarrativeArc phase or legacy symbol_tier_per_step
        phase = progress.current_phase()
        if phase is not None:
            required_tier = phase.cluster_symbol_tier
        elif signature.symbol_tier_per_step is not None:
            required_tier = signature.symbol_tier_per_step.get(progress.steps_completed)
        else:
            required_tier = None

        # Dormant boosters that must survive to the terminal board
        dormant_must_survive = (
            tuple(signature.dormant_boosters_on_terminal)
            if signature.dormant_boosters_on_terminal is not None
            else None
        )

        dormant_boosters = tuple(context.dormant_boosters)

        # Arming urgency — dormant boosters exist and remaining steps are tight
        booster_needs_arming_soon = self._compute_arming_urgency(
            dormant_boosters, steps_remaining,
        )

        # Payout pacing — how much of the budget has been spent vs config thresholds
        payout_low, payout_high = self._compute_payout_pacing(
            signature, payout_remaining,
        )

        return StepAssessment(
            steps_remaining=steps_remaining,
            is_first_step=is_first_step,
            must_terminate_now=(
                steps_remaining.max_val <= 0
                or (steps_remaining.min_val <= 0 and progress.is_satisfied())
            ),
            should_terminate_soon=steps_remaining.max_val <= 1,
            needs_booster_spawn=needs_spawn,
            needs_chain=needs_chain,
            needs_wild_bridge=needs_wild_bridge,
            payout_remaining=payout_remaining,
            dormant_boosters_to_arm=dormant_boosters,
            wilds_available_for_bridge=tuple(context.active_wilds),
            required_tier_this_step=required_tier,
            terminal_near_misses_required=signature.terminal_near_misses,
            dormant_boosters_must_survive=dormant_must_survive,
            signature_is_dead_family=signature.family == "dead",
            booster_needs_arming_soon=booster_needs_arming_soon,
            payout_running_low=payout_low,
            payout_running_high=payout_high,
        )

    def _infer_needs_wild_bridge(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
    ) -> bool:
        """True when this step should form a wild bridge.

        Two sources:
        1. Explicit: cascade_steps[current_step].wild_behavior == "bridge"
        2. Inferred: active wilds on board + unmet spawn needs that benefit
           from wild adjacency (any non-wild booster still needed)
        """
        # Arc-based: read wild_behavior from current phase
        phase = progress.current_phase()
        if phase is not None:
            if phase.wild_behavior == "bridge":
                return True
        else:
            # Legacy: read from cascade_steps
            current_step = progress.steps_completed
            if signature.cascade_steps is not None:
                if current_step < len(signature.cascade_steps):
                    step_spec = signature.cascade_steps[current_step]
                    if step_spec.wild_behavior == "bridge":
                        return True

        # Inferred: wilds exist and there are still non-wild booster spawns needed
        if not context.active_wilds:
            return False
        remaining_spawns = progress.remaining_booster_spawns()
        return any(
            btype != "W" and remaining.min_val > 0
            for btype, remaining in remaining_spawns.items()
        )

    def _compute_arming_urgency(
        self,
        dormant_boosters: tuple[DormantBooster, ...],
        steps_remaining: Range,
    ) -> bool:
        """True when dormant boosters exist and steps are too tight to defer arming.

        Dormant boosters must be armed before the cascade ends — fires happen
        automatically in the post-terminal booster phase. Urgency triggers when
        remaining steps are within the arming horizon.
        """
        if not dormant_boosters:
            return False
        return steps_remaining.max_val <= self._config.arming_urgency_horizon + 1

    def _compute_payout_pacing(
        self,
        signature: ArchetypeSignature,
        payout_remaining: RangeFloat,
    ) -> tuple[bool, bool]:
        """Compute (payout_running_low, payout_running_high) flags.

        Compares fraction of budget spent against config thresholds.
        Zero-budget signatures (dead family) always return (False, False).
        """
        max_budget = signature.payout_range.max_val
        if max_budget <= 0.0:
            return False, False

        # How much has been spent = max_budget minus what's left
        spent = max_budget - payout_remaining.max_val
        spent_fraction = spent / max_budget

        running_low = spent_fraction < self._config.payout_low_fraction
        running_high = spent_fraction > self._config.payout_high_fraction
        return running_low, running_high
