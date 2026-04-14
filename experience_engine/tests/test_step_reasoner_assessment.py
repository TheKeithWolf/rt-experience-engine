"""Tests for Step 5: Assessment, Strategy Selection, and Registry.

TEST-R5-001 through TEST-R5-019 per the implementation spec.
"""

from __future__ import annotations

import dataclasses

import pytest

from ..archetypes.registry import (
    ArchetypeSignature,
    CascadeStepConstraint,
    TerminalNearMissSpec,
)
from ..config.schema import MasterConfig, ReasonerConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol, SymbolTier
from ..step_reasoner.assessor import StepAssessment, StepAssessor
from ..step_reasoner.context import BoardContext, DormantBooster
from ..step_reasoner.evaluators import ChainEvaluator, PayoutEstimator, SpawnEvaluator
from ..step_reasoner.progress import ProgressTracker
from ..step_reasoner.registry import StrategyRegistry, build_default_registry
from ..step_reasoner.selector import (
    DEFAULT_SELECTION_RULES,
    SelectionRule,
    StrategySelector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signature(**overrides) -> ArchetypeSignature:
    """Build a minimal ArchetypeSignature with sensible defaults."""
    defaults = dict(
        id="test_sig",
        family="t1",
        criteria="basegame",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 8),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        required_cascade_depth=Range(2, 5),
        cascade_steps=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.5, 5.0),
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return ArchetypeSignature(**defaults)


def _make_reasoner_config(**overrides) -> ReasonerConfig:
    """Build a ReasonerConfig with sensible defaults."""
    defaults = dict(
        payout_low_fraction=0.3,
        payout_high_fraction=0.85,
        arming_urgency_horizon=1,
        terminal_dead_default_max_component=4,
        max_forward_simulations_per_step=10,
        max_strategic_cells_per_step=16,
        lookahead_depth=2,
        survivor_affinity_per_cell=2.0,
        min_booster_landing_score=0.1,
    )
    defaults.update(overrides)
    return ReasonerConfig(**defaults)


def _make_assessor(config: MasterConfig) -> StepAssessor:
    """Build a StepAssessor from config with real evaluators."""
    spawn_eval = SpawnEvaluator(config.boosters)
    chain_eval = ChainEvaluator(config.boosters)
    payout_eval = PayoutEstimator(
        config.paytable, config.centipayout, config.win_levels,
        config.symbols, config.grid_multiplier,
    )
    reasoner_config = config.reasoner
    assert reasoner_config is not None, "default.yaml must include reasoner section"
    return StepAssessor(spawn_eval, chain_eval, payout_eval, reasoner_config)


def _make_context(
    board: Board,
    config: MasterConfig,
    dormant_boosters: list[DormantBooster] | None = None,
    active_wilds: list[Position] | None = None,
) -> BoardContext:
    """Build a BoardContext from fixtures."""
    from ..primitives.grid_multipliers import GridMultiplierGrid

    return BoardContext.from_board(
        board=board,
        grid_multipliers=GridMultiplierGrid(config.grid_multiplier, config.board),
        dormant_boosters=dormant_boosters or [],
        active_wilds=active_wilds or [],
        board_config=config.board,
    )


def _make_assessment(**overrides) -> StepAssessment:
    """Build a StepAssessment with all-default fields for selector tests.

    Defaults represent a mid-cascade non-terminal step with no special
    conditions — falls through to the default "cascade_cluster" strategy.
    """
    defaults = dict(
        steps_remaining=Range(2, 3),
        is_first_step=False,
        must_terminate_now=False,
        should_terminate_soon=False,
        needs_booster_spawn={},
        needs_chain=False,
        needs_wild_bridge=False,
        payout_remaining=RangeFloat(1.0, 3.0),
        dormant_boosters_to_arm=(),
        wilds_available_for_bridge=(),
        required_tier_this_step=None,
        terminal_near_misses_required=None,
        dormant_boosters_must_survive=None,
        signature_is_dead_family=False,
        booster_needs_arming_soon=False,
        payout_running_low=False,
        payout_running_high=False,
        next_phase_is_wild_bridge=False,
    )
    defaults.update(overrides)
    return StepAssessment(**defaults)


# ---------------------------------------------------------------------------
# TEST-R5-001: StepAssessment is frozen
# ---------------------------------------------------------------------------

class TestStepAssessment:

    def test_assessment_is_frozen(self) -> None:
        """R5-001: Assigning to a field on a frozen StepAssessment raises an error."""
        assessment = _make_assessment()
        with pytest.raises(dataclasses.FrozenInstanceError):
            assessment.is_first_step = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TEST-R5-002 through R5-006: StepAssessor
# ---------------------------------------------------------------------------

class TestStepAssessor:

    def test_first_step_detected(
        self, empty_board: Board, default_config: MasterConfig,
    ) -> None:
        """R5-002: assess() sets is_first_step=True when steps_completed=0."""
        assessor = _make_assessor(default_config)
        sig = _make_signature()
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        context = _make_context(empty_board, default_config)

        assessment = assessor.assess(context, progress, sig)
        assert assessment.is_first_step is True

    def test_must_terminate_at_max_depth(
        self, empty_board: Board, default_config: MasterConfig,
    ) -> None:
        """R5-003: must_terminate_now=True when at max cascade depth."""
        assessor = _make_assessor(default_config)
        sig = _make_signature(required_cascade_depth=Range(2, 3))
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        # Advance past max depth
        progress.steps_completed = 3
        context = _make_context(empty_board, default_config)

        assessment = assessor.assess(context, progress, sig)
        assert assessment.must_terminate_now is True

    def test_booster_arming_urgency(
        self, empty_board: Board, default_config: MasterConfig,
    ) -> None:
        """R5-004: booster_needs_arming_soon=True when dormant booster + tight steps."""
        assessor = _make_assessor(default_config)
        sig = _make_signature(
            required_cascade_depth=Range(2, 3),
            required_booster_fires={"R": Range(1, 1)},
        )
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        # 2 steps done out of max 3 → 1 remaining, 1 fire needed → 0 slack
        progress.steps_completed = 2
        dormant = [DormantBooster("R", Position(3, 3), "H", 0)]
        context = _make_context(empty_board, default_config, dormant_boosters=dormant)

        assessment = assessor.assess(context, progress, sig)
        assert assessment.booster_needs_arming_soon is True

    def test_needs_wild_bridge_from_cascade_steps(
        self, empty_board: Board, default_config: MasterConfig,
    ) -> None:
        """R5-005: needs_wild_bridge=True when cascade_steps specifies bridge."""
        assessor = _make_assessor(default_config)
        bridge_step = CascadeStepConstraint(
            cluster_count=Range(1, 1),
            cluster_sizes=(Range(5, 8),),
            cluster_symbol_tier=SymbolTier.LOW,
            must_spawn_booster=None,
            must_arm_booster=None,
            wild_behavior="bridge",
        )
        sig = _make_signature(
            cascade_steps=(bridge_step,),
            required_cascade_depth=Range(1, 3),
        )
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        # Step 0 has wild_behavior="bridge"
        wilds = [Position(2, 2)]
        context = _make_context(empty_board, default_config, active_wilds=wilds)

        assessment = assessor.assess(context, progress, sig)
        assert assessment.needs_wild_bridge is True

    def test_payout_running_high(
        self, empty_board: Board, default_config: MasterConfig,
    ) -> None:
        """R5-006: payout_running_high=True when 90% of budget spent."""
        assessor = _make_assessor(default_config)
        sig = _make_signature(payout_range=RangeFloat(0.0, 10.0))
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        # Spend 9.0x out of 10.0x max → 90% > 85% threshold
        progress.cumulative_payout = 900
        context = _make_context(empty_board, default_config)

        assessment = assessor.assess(context, progress, sig)
        assert assessment.payout_running_high is True
        assert assessment.payout_running_low is False

    def test_payout_running_low(
        self, empty_board: Board, default_config: MasterConfig,
    ) -> None:
        """R5-006b: payout_running_low=True when under 30% of budget spent."""
        assessor = _make_assessor(default_config)
        sig = _make_signature(payout_range=RangeFloat(0.0, 10.0))
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        # Spend 1.0x out of 10.0x max → 10% < 30% threshold
        progress.cumulative_payout = 100
        context = _make_context(empty_board, default_config)

        assessment = assessor.assess(context, progress, sig)
        assert assessment.payout_running_low is True
        assert assessment.payout_running_high is False

    def test_dead_family_zero_budget_no_pacing_flags(
        self, empty_board: Board, default_config: MasterConfig,
    ) -> None:
        """R5-007: Dead family (0 budget) → both payout flags False."""
        assessor = _make_assessor(default_config)
        sig = _make_signature(
            family="dead", criteria="0",
            payout_range=RangeFloat(0.0, 0.0),
            required_cascade_depth=Range(0, 0),
        )
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        context = _make_context(empty_board, default_config)

        assessment = assessor.assess(context, progress, sig)
        assert assessment.signature_is_dead_family is True
        assert assessment.payout_running_low is False
        assert assessment.payout_running_high is False


# ---------------------------------------------------------------------------
# TEST-R5-008 through R5-014: StrategySelector
# ---------------------------------------------------------------------------

class TestStrategySelector:

    def test_terminal_dead_selected(self) -> None:
        """R5-008: must_terminate_now → 'terminal_dead'."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        assessment = _make_assessment(must_terminate_now=True)
        assert selector.select(assessment) == "terminal_dead"

    def test_terminal_near_miss_wins_over_terminal_dead(self) -> None:
        """R5-009: must_terminate_now + near_miss spec → 'terminal_near_miss'."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        nm_spec = TerminalNearMissSpec(count=Range(1, 2), symbol_tier=SymbolTier.LOW)
        assessment = _make_assessment(
            must_terminate_now=True,
            terminal_near_misses_required=nm_spec,
        )
        assert selector.select(assessment) == "terminal_near_miss"

    def test_initial_dead_over_initial_cluster(self) -> None:
        """R5-010: first step + dead family → 'initial_dead'."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        assessment = _make_assessment(
            is_first_step=True, signature_is_dead_family=True,
        )
        assert selector.select(assessment) == "initial_dead"

    def test_initial_cluster_for_non_dead(self) -> None:
        """R5-011: first step + non-dead + non-bridge → 'initial_cluster'."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        assessment = _make_assessment(is_first_step=True)
        assert selector.select(assessment) == "initial_cluster"

    def test_initial_wild_bridge_selected(self) -> None:
        """R5-NEW: first step + next_phase_is_wild_bridge → 'initial_wild_bridge'."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        assessment = _make_assessment(
            is_first_step=True, next_phase_is_wild_bridge=True,
        )
        assert selector.select(assessment) == "initial_wild_bridge"

    def test_initial_wild_bridge_over_initial_cluster(self) -> None:
        """initial_wild_bridge (89) beats initial_cluster (88) when both match first_step."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        # Both is_first_step rules match, but wild_bridge has higher priority
        assessment = _make_assessment(
            is_first_step=True, next_phase_is_wild_bridge=True,
        )
        assert selector.select(assessment) == "initial_wild_bridge"

        # Without bridge flag, initial_cluster catches it
        assessment_no_bridge = _make_assessment(is_first_step=True)
        assert selector.select(assessment_no_bridge) == "initial_cluster"

    def test_booster_arm_selected(self) -> None:
        """R5-012: booster_needs_arming_soon → 'booster_arm'."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        assessment = _make_assessment(booster_needs_arming_soon=True)
        assert selector.select(assessment) == "booster_arm"

    def test_wild_bridge_selected(self) -> None:
        """R5-013: wilds + needs_wild_bridge → 'wild_bridge'."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        assessment = _make_assessment(
            wilds_available_for_bridge=(Position(2, 2),),
            needs_wild_bridge=True,
        )
        assert selector.select(assessment) == "wild_bridge"

    def test_default_cascade_cluster(self) -> None:
        """R5-014: no special conditions → 'cascade_cluster' (default)."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        assessment = _make_assessment()
        assert selector.select(assessment) == "cascade_cluster"

    def test_higher_priority_wins(self) -> None:
        """R5-015: when multiple rules match, highest priority wins."""
        selector = StrategySelector(DEFAULT_SELECTION_RULES)
        # Both terminal_dead (99) and booster_arm (80) would match
        assessment = _make_assessment(
            must_terminate_now=True,
            booster_needs_arming_soon=True,
        )
        assert selector.select(assessment) == "terminal_dead"


# ---------------------------------------------------------------------------
# TEST-R5-016 through R5-019: StrategyRegistry
# ---------------------------------------------------------------------------

class _MockStrategy:
    """Minimal strategy that satisfies the StepStrategy protocol."""

    def plan_step(self, context, progress, signature, variance):
        """Stub — returns None for testing purposes."""
        return None  # type: ignore[return-value]


class TestStrategyRegistry:

    def test_register_and_get(self) -> None:
        """R5-016: register + get round-trips correctly."""
        registry = StrategyRegistry()
        strategy = _MockStrategy()
        registry.register("test_strategy", strategy)
        assert registry.get("test_strategy") is strategy

    def test_duplicate_raises_value_error(self) -> None:
        """R5-017: Registering the same name twice raises ValueError."""
        registry = StrategyRegistry()
        registry.register("dup", _MockStrategy())
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dup", _MockStrategy())

    def test_unknown_raises_key_error(self) -> None:
        """R5-018: Looking up an unregistered name raises KeyError."""
        registry = StrategyRegistry()
        with pytest.raises(KeyError, match="Unknown strategy"):
            registry.get("nonexistent")

    def test_build_default_registry_returns_all_strategies(
        self, default_config,
    ) -> None:
        """R5-019: build_default_registry registers all 9 strategies (Step 6 complete)."""
        from ..primitives.gravity import GravityDAG
        from ..step_reasoner.evaluators import (
            ChainEvaluator, PayoutEstimator, SpawnEvaluator,
        )

        gravity_dag = GravityDAG(default_config.board, default_config.gravity)
        spawn_eval = SpawnEvaluator(default_config.boosters)
        chain_eval = ChainEvaluator(default_config.boosters)
        payout_eval = PayoutEstimator(
            default_config.paytable, default_config.centipayout,
            default_config.win_levels, default_config.symbols,
            default_config.grid_multiplier,
        )
        registry = build_default_registry(
            config=default_config,
            gravity_dag=gravity_dag,
            spatial_solver=None,  # type: ignore[arg-type]
            spawn_evaluator=spawn_eval,
            chain_evaluator=chain_eval,
            payout_evaluator=payout_eval,
        )
        expected = sorted([
            "booster_arm", "booster_setup", "cascade_cluster",
            "initial_cluster", "initial_dead", "initial_wild_bridge",
            "terminal_dead", "terminal_near_miss", "wild_bridge",
        ])
        assert registry.available() == expected
