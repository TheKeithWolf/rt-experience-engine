"""Tests for Step 3 data types: StepIntent, BoardContext, ProgressTracker.

TEST-R3-001 through TEST-R3-015 per the implementation spec.
"""

from __future__ import annotations

import dataclasses

import pytest

from ..archetypes.registry import ArchetypeSignature
from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.symbols import Symbol, SymbolTier
from ..step_reasoner.context import BoardContext, DormantBooster
from ..step_reasoner.intent import StepIntent, StepType
from ..step_reasoner.progress import ClusterRecord, ProgressTracker
from ..step_reasoner.results import FireRecord, SpawnRecord, StepResult


# ---------------------------------------------------------------------------
# Helpers — minimal archetype signatures for testing progress tracking
# ---------------------------------------------------------------------------

def _make_signature(**overrides) -> ArchetypeSignature:
    """Build a minimal ArchetypeSignature with sensible defaults.

    Override any field via kwargs. Defaults create a simple t1-like
    archetype with 1-3 cascade steps and modest payout range.
    """
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


def _make_step_result(
    step_index: int = 0,
    clusters: tuple[ClusterRecord, ...] = (),
    spawns: tuple[SpawnRecord, ...] = (),
    fires: tuple[FireRecord, ...] = (),
    symbol_tier: SymbolTier | None = SymbolTier.LOW,
    step_payout: int = 100,
) -> StepResult:
    """Build a StepResult with sensible defaults for tests."""
    return StepResult(
        step_index=step_index,
        clusters=clusters,
        spawns=spawns,
        fires=fires,
        symbol_tier=symbol_tier,
        step_payout=step_payout,
    )


# ---------------------------------------------------------------------------
# TEST-R3-001: StepIntent is frozen (immutable after creation)
# ---------------------------------------------------------------------------

class TestStepIntent:

    def test_step_intent_is_frozen(self) -> None:
        """R3-001: Assigning to a field on a frozen StepIntent raises an error."""
        intent = StepIntent(
            step_type=StepType.INITIAL,
            constrained_cells={},
            strategic_cells={},
            expected_cluster_count=Range(1, 1),
            expected_cluster_sizes=[Range(5, 5)],
            expected_cluster_tier=SymbolTier.LOW,
            expected_spawns=[],
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=[],
            wfc_symbol_weights={},
            predicted_post_gravity=None,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            is_terminal=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            intent.step_type = StepType.TERMINAL_DEAD  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TEST-R3-002: StepType has all 6 variants
# ---------------------------------------------------------------------------

    def test_step_type_has_six_variants(self) -> None:
        """R3-002: StepType enum has exactly 6 members."""
        assert len(StepType) == 6
        expected = {
            "INITIAL", "CASCADE_CLUSTER", "BOOSTER_ARM",
            "BOOSTER_FIRE", "TERMINAL_DEAD", "TERMINAL_NEAR_MISS",
        }
        assert {m.name for m in StepType} == expected


# ---------------------------------------------------------------------------
# TEST-R3-003 through R3-006: BoardContext
# ---------------------------------------------------------------------------

class TestBoardContext:

    def test_board_context_empty_cells(
        self, empty_board: Board, default_config: MasterConfig,
    ) -> None:
        """R3-003: empty_cells matches board.empty_positions()."""
        ctx = BoardContext.from_board(
            board=empty_board,
            grid_multipliers=_make_grid_mults(default_config),
            dormant_boosters=[],
            active_wilds=[],
            board_config=default_config.board,
        )
        assert ctx.empty_cells == empty_board.empty_positions()

    def test_board_context_symbol_counts(
        self, sample_board: Board, default_config: MasterConfig,
    ) -> None:
        """R3-004: symbol_counts is accurate for a known board layout."""
        ctx = BoardContext.from_board(
            board=sample_board,
            grid_multipliers=_make_grid_mults(default_config),
            dormant_boosters=[],
            active_wilds=[],
            board_config=default_config.board,
        )
        counts = ctx.symbol_counts
        # sample_board is 7x7 = 49 cells, all occupied
        total = sum(counts.values())
        assert total == 49
        # Verify a few known counts from the conftest layout
        # Row 0 has 5 L1 + 1 H1 + 1 H2 (reel-major: reels 0-4 at row 0 are L1)
        assert counts.get(Symbol.L1) is not None
        assert counts[Symbol.L1] > 0

    def test_board_context_neighbors_of(
        self, empty_board: Board, default_config: MasterConfig,
    ) -> None:
        """R3-005: neighbors_of delegates to orthogonal_neighbors correctly."""
        ctx = BoardContext.from_board(
            board=empty_board,
            grid_multipliers=_make_grid_mults(default_config),
            dormant_boosters=[],
            active_wilds=[],
            board_config=default_config.board,
        )
        pos = Position(3, 3)
        assert ctx.neighbors_of(pos) == orthogonal_neighbors(pos, default_config.board)

    def test_board_context_from_board_factory(
        self, sample_board: Board, default_config: MasterConfig,
    ) -> None:
        """R3-006: Factory produces a valid context with all fields populated."""
        dormant = [DormantBooster("R", Position(1, 1), "H", 0)]
        wilds = [Position(2, 2)]
        ctx = BoardContext.from_board(
            board=sample_board,
            grid_multipliers=_make_grid_mults(default_config),
            dormant_boosters=dormant,
            active_wilds=wilds,
            board_config=default_config.board,
        )
        assert ctx.board is sample_board
        assert ctx.dormant_boosters == dormant
        assert ctx.active_wilds == wilds


# ---------------------------------------------------------------------------
# TEST-R3-007 through R3-015: ProgressTracker
# ---------------------------------------------------------------------------

class TestProgressTracker:

    def test_progress_tracker_initial_zero_state(self) -> None:
        """R3-007: Fresh tracker has zero state across all counters."""
        tracker = ProgressTracker(
            signature=_make_signature(),
            centipayout_multiplier=100,
        )
        assert tracker.steps_completed == 0
        assert tracker.clusters_produced == []
        assert tracker.boosters_spawned == {}
        assert tracker.boosters_fired == {}
        assert tracker.cumulative_payout == 0

    def test_remaining_cascade_steps_fresh(self) -> None:
        """R3-008: 0 done, range(2,5) → Range(2,5)."""
        tracker = ProgressTracker(
            signature=_make_signature(required_cascade_depth=Range(2, 5)),
            centipayout_multiplier=100,
        )
        remaining = tracker.remaining_cascade_steps()
        assert remaining == Range(2, 5)

    def test_remaining_cascade_steps_partial(self) -> None:
        """R3-009: 3 done, range(2,5) → Range(0,2)."""
        tracker = ProgressTracker(
            signature=_make_signature(required_cascade_depth=Range(2, 5)),
            centipayout_multiplier=100,
        )
        tracker.steps_completed = 3
        remaining = tracker.remaining_cascade_steps()
        assert remaining == Range(0, 2)

    def test_remaining_payout_budget(self) -> None:
        """R3-010: After spending 1000 centipayout (=10.0x at multiplier=100),
        budget shrinks from (5.0, 50.0) to (0.0, 40.0).
        """
        tracker = ProgressTracker(
            signature=_make_signature(payout_range=RangeFloat(5.0, 50.0)),
            centipayout_multiplier=100,
        )
        tracker.cumulative_payout = 1000  # 1000 / 100 = 10.0x spent
        budget = tracker.remaining_payout_budget()
        # min clamps to 0: 5.0 - 10.0 = -5.0 → 0.0
        assert budget.min_val == pytest.approx(0.0)
        assert budget.max_val == pytest.approx(40.0)

    def test_must_terminate_soon(self) -> None:
        """R3-011: True when max remaining steps <= 1."""
        tracker = ProgressTracker(
            signature=_make_signature(required_cascade_depth=Range(2, 3)),
            centipayout_multiplier=100,
        )
        # 2 done out of max 3 → 1 remaining → must terminate soon
        tracker.steps_completed = 2
        assert tracker.must_terminate_soon() is True

        # 1 done → 2 remaining → not yet
        tracker.steps_completed = 1
        assert tracker.must_terminate_soon() is False

    def test_is_satisfied_all_met(self) -> None:
        """R3-012: Returns True when all minimums are satisfied."""
        tracker = ProgressTracker(
            signature=_make_signature(
                required_cascade_depth=Range(2, 5),
                required_booster_spawns={"R": Range(1, 2)},
                required_booster_fires={"R": Range(1, 1)},
                payout_range=RangeFloat(0.5, 5.0),
            ),
            centipayout_multiplier=100,
        )
        tracker.steps_completed = 3
        tracker.boosters_spawned = {"R": 1}
        tracker.boosters_fired = {"R": 1}
        tracker.cumulative_payout = 100  # 1.0x — within (0.5, 5.0)
        assert tracker.is_satisfied() is True

    def test_is_satisfied_missing_fire(self) -> None:
        """R3-013: Returns False when a required booster fire is missing."""
        tracker = ProgressTracker(
            signature=_make_signature(
                required_cascade_depth=Range(2, 5),
                required_booster_fires={"R": Range(1, 1)},
                payout_range=RangeFloat(0.0, 50.0),
            ),
            centipayout_multiplier=100,
        )
        tracker.steps_completed = 3
        # No fires recorded
        assert tracker.is_satisfied() is False

    def test_update_increments_counters(self) -> None:
        """R3-014: update() advances steps, clusters, and payout."""
        tracker = ProgressTracker(
            signature=_make_signature(),
            centipayout_multiplier=100,
        )
        cluster = ClusterRecord(
            symbol=Symbol.L1,
            size=5,
            positions=frozenset({Position(0, 0), Position(1, 0), Position(2, 0),
                                  Position(3, 0), Position(4, 0)}),
            step_index=0,
            payout=50,
        )
        result = _make_step_result(
            step_index=0,
            clusters=(cluster,),
            step_payout=50,
        )
        tracker.update(result)

        assert tracker.steps_completed == 1
        assert len(tracker.clusters_produced) == 1
        assert tracker.clusters_produced[0] is cluster
        assert tracker.cumulative_payout == 50

    def test_update_removes_fired_from_dormant(self) -> None:
        """R3-015: Fired booster is removed from the dormant list."""
        tracker = ProgressTracker(
            signature=_make_signature(),
            centipayout_multiplier=100,
        )
        rocket_pos = Position(3, 3)
        tracker.dormant_boosters = [
            DormantBooster("R", rocket_pos, "H", 0),
            DormantBooster("B", Position(5, 5), None, 0),
        ]
        fire = FireRecord(
            booster_type="R",
            position=rocket_pos,
            affected_count=7,
            chain_triggered=False,
            step_index=1,
        )
        result = _make_step_result(step_index=1, fires=(fire,))
        tracker.update(result)

        # Rocket removed, bomb survives
        assert len(tracker.dormant_boosters) == 1
        assert tracker.dormant_boosters[0].booster_type == "B"
        assert tracker.boosters_fired == {"R": 1}


# ---------------------------------------------------------------------------
# ProgressTracker — cascade_steps enforcement
# ---------------------------------------------------------------------------

class TestProgressTrackerCascadeSteps:
    """Tests for cascade_steps-aware behaviour in ProgressTracker."""

    def _make_two_step_constraints(self):
        from ..archetypes.registry import CascadeStepConstraint
        step0 = CascadeStepConstraint(
            cluster_count=Range(1, 2), cluster_sizes=(Range(7, 8),),
            cluster_symbol_tier=None, must_spawn_booster="W",
            must_arm_booster=None, must_fire_booster=None,
            wild_behavior="spawn",
        )
        step1 = CascadeStepConstraint(
            cluster_count=Range(1, 1), cluster_sizes=(Range(5, 6),),
            cluster_symbol_tier=None, must_spawn_booster=None,
            must_arm_booster=None, must_fire_booster=None,
            wild_behavior="bridge",
        )
        return step0, step1

    def test_is_satisfied_false_when_cascade_steps_incomplete(self) -> None:
        """is_satisfied() returns False when mandatory cascade_steps remain."""
        step0, step1 = self._make_two_step_constraints()
        tracker = ProgressTracker(
            signature=_make_signature(
                required_cascade_depth=Range(1, 3),
                required_booster_spawns={"W": Range(1, 1)},
                payout_range=RangeFloat(0.4, 15.0),
                cascade_steps=(step0, step1),
            ),
            centipayout_multiplier=100,
        )
        # Simulate step 0 complete — depth/spawns/payout all met, but step 1 not done
        tracker.steps_completed = 1
        tracker.boosters_spawned = {"W": 1}
        tracker.cumulative_payout = 180
        assert tracker.is_satisfied() is False

    def test_current_step_size_ranges_uses_step_constraint(self) -> None:
        """Returns cascade_steps[n].cluster_sizes when defined."""
        step0, step1 = self._make_two_step_constraints()
        tracker = ProgressTracker(
            signature=_make_signature(
                required_cluster_sizes=(Range(5, 8),),
                cascade_steps=(step0, step1),
            ),
            centipayout_multiplier=100,
        )
        # Step 0 — should use step0's tighter sizes
        assert tracker.current_step_size_ranges() == (Range(7, 8),)
        # Step 1 — should use step1's sizes
        tracker.steps_completed = 1
        assert tracker.current_step_size_ranges() == (Range(5, 6),)

    def test_current_step_size_ranges_falls_back_past_steps(self) -> None:
        """Returns signature.required_cluster_sizes when step index exceeds cascade_steps."""
        from ..archetypes.registry import CascadeStepConstraint
        step0 = CascadeStepConstraint(
            cluster_count=Range(1, 2), cluster_sizes=(Range(7, 8),),
            cluster_symbol_tier=None, must_spawn_booster="W",
            must_arm_booster=None, must_fire_booster=None,
            wild_behavior="spawn",
        )
        tracker = ProgressTracker(
            signature=_make_signature(
                required_cluster_sizes=(Range(5, 8),),
                cascade_steps=(step0,),
            ),
            centipayout_multiplier=100,
        )
        # Past the single cascade_step — should fall back to signature sizes
        tracker.steps_completed = 1
        assert tracker.current_step_size_ranges() == (Range(5, 8),)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _make_grid_mults(config: MasterConfig):
    """Build a GridMultiplierGrid from the default config."""
    from ..primitives.grid_multipliers import GridMultiplierGrid
    return GridMultiplierGrid(config.grid_multiplier, config.board)
