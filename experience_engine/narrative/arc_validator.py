"""Arc validator — validates a completed trajectory against a NarrativeArc.

Uses greedy linear phase matching: walk phases left-to-right, consuming step
records. Each step is checked against the current phase's constraints. The
phase advances when max repetitions are reached OR the transition predicate
fires. Returns a list of error strings (empty = valid).

Replaces the per-step constraint loop in validation/validator.py that zipped
CascadeStepConstraints with instance step records by index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..pipeline.protocols import Range
from ..primitives.symbols import SymbolTier, tier_of
from .arc import NarrativeArc, NarrativePhase
from .derivation import DerivedConstraints
from .transitions import TransitionPredicate, _CONTEXT_DEPENDENT_RULES

if TYPE_CHECKING:
    from ..config.schema import SymbolConfig
    from ..pipeline.data_types import CascadeStepRecord
    from ..step_reasoner.context import BoardContext


class _StepRecordAdapter:
    """Lightweight adapter from CascadeStepRecord to the shape transition predicates expect.

    Transition predicates access ``len(result.clusters)`` and ``len(result.fires)``
    — this adapter exposes those as tuple-like attributes without constructing
    full StepResult/ClusterRecord/FireRecord objects.
    """

    __slots__ = ("clusters", "fires")

    def __init__(self, rec: CascadeStepRecord) -> None:
        self.clusters = rec.clusters
        self.fires = rec.booster_fire_records


class NarrativeArcValidator:
    """Validates a completed trajectory against a NarrativeArc phase grammar.

    Injected with transition rules and symbol config at construction time.
    Stateless — a single instance can validate many trajectories.
    """

    __slots__ = ("_transition_rules", "_symbol_config")

    def __init__(
        self,
        transition_rules: dict[str, TransitionPredicate],
        symbol_config: SymbolConfig,
    ) -> None:
        self._transition_rules = transition_rules
        self._symbol_config = symbol_config

    def validate(
        self,
        step_records: tuple[CascadeStepRecord, ...],
        arc: NarrativeArc,
        derived: DerivedConstraints,
        board_contexts: tuple[BoardContext, ...] | None = None,
    ) -> list[str]:
        """Validate a trajectory against the arc. Returns error messages (empty = valid).

        board_contexts is optional — transition predicates that inspect board state
        (e.g., "no_bridges") require it. If omitted, transition predicates receive
        None and rules relying on board context will always return True.
        """
        errors: list[str] = []

        # Phase matching state
        phase_idx = 0
        phase_reps = 0

        # Accumulate booster counts across all steps
        total_spawn_types: dict[str, int] = {}
        total_fire_types: dict[str, int] = {}
        total_payout = 0.0

        for step_i, step_rec in enumerate(step_records):
            # Accumulate global counters
            for btype in step_rec.booster_spawn_types:
                total_spawn_types[btype] = total_spawn_types.get(btype, 0) + 1
            for fire_rec in step_rec.booster_fire_records:
                total_fire_types[fire_rec.booster_type] = (
                    total_fire_types.get(fire_rec.booster_type, 0) + 1
                )
            total_payout += step_rec.step_payout

            # Try to match step against current (or next) phase
            matched = False
            while phase_idx < len(arc.phases):
                phase = arc.phases[phase_idx]
                ctx = board_contexts[step_i] if board_contexts else None

                if self._step_matches_phase(step_rec, phase, errors, step_i):
                    phase_reps += 1
                    matched = True

                    # Check if phase should advance
                    if phase_reps >= phase.repetitions.max_val:
                        phase_idx += 1
                        phase_reps = 0
                    elif self._evaluate_transition(phase, step_rec, ctx):
                        # Transition predicate fired before max reps
                        if phase_reps >= phase.repetitions.min_val:
                            phase_idx += 1
                            phase_reps = 0
                    break
                else:
                    # Step doesn't match current phase — try advancing if min met
                    if phase_reps >= phase.repetitions.min_val:
                        phase_idx += 1
                        phase_reps = 0
                        # Re-check this step against the next phase (loop continues)
                    else:
                        errors.append(
                            f"Step {step_i}: phase '{phase.id}' minimum repetitions "
                            f"({phase.repetitions.min_val}) not met — only {phase_reps} completed"
                        )
                        matched = True  # Error recorded, don't retry
                        break

            if not matched and phase_idx >= len(arc.phases):
                errors.append(
                    f"Step {step_i}: no phase left to match — "
                    f"trajectory has more steps than the arc allows"
                )

        # After all steps consumed: remaining phases must be skippable
        for remaining_idx in range(phase_idx, len(arc.phases)):
            remaining = arc.phases[remaining_idx]
            # Current phase in progress — check if its reps are sufficient
            if remaining_idx == phase_idx and phase_reps > 0:
                if phase_reps < remaining.repetitions.min_val:
                    errors.append(
                        f"Phase '{remaining.id}' ended with {phase_reps} repetitions, "
                        f"minimum is {remaining.repetitions.min_val}"
                    )
                continue
            if remaining.repetitions.min_val > 0:
                errors.append(
                    f"Phase '{remaining.id}' was never reached but requires "
                    f"at least {remaining.repetitions.min_val} repetitions"
                )

        # Global checks
        self._check_global_constraints(
            arc, derived, total_payout, total_spawn_types, total_fire_types, errors,
        )

        return errors

    def _step_matches_phase(
        self,
        step_rec: CascadeStepRecord,
        phase: NarrativePhase,
        errors: list[str],
        step_i: int,
    ) -> bool:
        """Check if a step record satisfies all per-step constraints of a phase.

        Field-iteration over the phase's frozen fields — no branching on phase ID,
        archetype name, or booster type.
        """
        cluster_count = len(step_rec.clusters)

        # Cluster count
        if not phase.cluster_count.contains(cluster_count):
            return False

        # Cluster sizes — each cluster must fit at least one size range
        if phase.cluster_sizes:
            for cluster in step_rec.clusters:
                if not any(sr.contains(cluster.size) for sr in phase.cluster_sizes):
                    return False
        elif cluster_count > 0:
            # Empty cluster_sizes tuple only valid when cluster count is 0
            return False

        # Symbol tier constraint
        if phase.cluster_symbol_tier is not None:
            for cluster in step_rec.clusters:
                try:
                    actual_tier = tier_of(cluster.symbol, self._symbol_config)
                except ValueError:
                    continue  # Non-standard symbols skip tier check
                if phase.cluster_symbol_tier is not SymbolTier.ANY:
                    if actual_tier != phase.cluster_symbol_tier:
                        return False

        # Spawn check — step must include all required booster types
        if phase.spawns is not None:
            step_spawn_set = set(step_rec.booster_spawn_types)
            for required in phase.spawns:
                if required not in step_spawn_set:
                    return False

        # Arm check — step must include arm events for required types
        if phase.arms is not None:
            # Arms are tracked as spawns that transition to armed state;
            # in the step record, armed boosters appear in booster_fire_records
            # with chain_triggered=False, or in spawn_types if arming == spawning.
            # For now, arm matching checks spawn types (arming produces a dormant booster).
            step_spawn_set = set(step_rec.booster_spawn_types)
            for required in phase.arms:
                if required not in step_spawn_set:
                    return False

        # Fire check — step must include fire events for required types
        if phase.fires is not None:
            step_fire_set = {fr.booster_type for fr in step_rec.booster_fire_records}
            for required in phase.fires:
                if required not in step_fire_set:
                    return False

        return True

    def _evaluate_transition(
        self,
        phase: NarrativePhase,
        step_rec: CascadeStepRecord,
        ctx: BoardContext | None,
    ) -> bool:
        """Evaluate the phase's transition predicate.

        If board context is unavailable, transition predicates that require it
        default to True (conservative — allow advancement).
        """
        predicate = self._transition_rules.get(phase.ends_when)
        if predicate is None:
            return True  # Unknown rule defaults to always-advance

        # Build a lightweight adapter from CascadeStepRecord to match the
        # TransitionPredicate(StepResult, BoardContext) signature.
        step_result = _StepRecordAdapter(step_rec)

        # Board-context-dependent predicates can't evaluate without context —
        # default to True (conservative, allow phase advancement).
        # Context-independent predicates (always, no_clusters, booster_fired)
        # can still evaluate with ctx=None since they ignore the context arg.
        if ctx is None and phase.ends_when in _CONTEXT_DEPENDENT_RULES:
            return True

        return predicate(step_result, ctx)  # type: ignore[arg-type]

    def _check_global_constraints(
        self,
        arc: NarrativeArc,
        derived: DerivedConstraints,
        total_payout: float,
        total_spawns: dict[str, int],
        total_fires: dict[str, int],
        errors: list[str],
    ) -> None:
        """Check arc-wide outcome constraints after all steps are consumed."""

        # Payout range
        if not arc.payout.contains(total_payout):
            errors.append(
                f"Total payout {total_payout:.2f} outside arc range "
                f"[{arc.payout.min_val}, {arc.payout.max_val}]"
            )

        # Booster spawn counts
        for btype, required_range in derived.booster_spawns.items():
            actual = total_spawns.get(btype, 0)
            if not required_range.contains(actual):
                errors.append(
                    f"Booster '{btype}' spawns: {actual} outside "
                    f"[{required_range.min_val}, {required_range.max_val}]"
                )

        # Booster fire counts
        for btype, required_range in derived.booster_fires.items():
            actual = total_fires.get(btype, 0)
            if not required_range.contains(actual):
                errors.append(
                    f"Booster '{btype}' fires: {actual} outside "
                    f"[{required_range.min_val}, {required_range.max_val}]"
                )

        # Chain depth
        if not arc.required_chain_depth.contains(0):
            # Chain depth validation requires instance-level chain tracking —
            # deferred to InstanceValidator which already tracks this.
            pass
