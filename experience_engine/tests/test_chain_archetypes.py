"""Phase 11 tests — complex compositions & deep chains.

TEST-P11-001 through TEST-P11-007 covering chain family archetypes
and multi-booster type widening.
"""

from __future__ import annotations

import pytest

from ..archetypes.bomb import register_bomb_archetypes
from ..archetypes.chain import register_chain_archetypes
from ..archetypes.dead import register_dead_archetypes
from ..archetypes.registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    CascadeStepConstraint,
    SignatureValidationError,
)
from ..archetypes.rocket import register_rocket_archetypes
from ..boosters.fire_handlers import fire_bomb, fire_lightball, fire_rocket, fire_superlightball
from ..boosters.phase_executor import BoosterFireResult, BoosterPhaseExecutor
from ..boosters.state_machine import BoosterInstance, BoosterState
from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.symbols import Symbol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rules(config: MasterConfig) -> BoosterRules:
    return BoosterRules(config.boosters, config.board, config.symbols)


def _make_board_with_symbols(
    config: MasterConfig,
    placements: dict[Position, Symbol],
) -> Board:
    """Create a board with specified symbols, rest L1."""
    board = Board.empty(config.board)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            pos = Position(reel, row)
            board.set(pos, placements.get(pos, Symbol.L1))
    return board


def _make_rocket(
    pos: Position,
    orientation: str,
    state: BoosterState = BoosterState.ARMED,
) -> BoosterInstance:
    return BoosterInstance(
        booster_type=Symbol.R,
        position=pos,
        state=state,
        orientation=orientation,
        source_cluster_index=0,
    )


def _make_bomb(
    pos: Position,
    state: BoosterState = BoosterState.ARMED,
) -> BoosterInstance:
    return BoosterInstance(
        booster_type=Symbol.B,
        position=pos,
        state=state,
        orientation=None,
        source_cluster_index=0,
    )


def _make_lb(
    pos: Position,
    state: BoosterState = BoosterState.ARMED,
) -> BoosterInstance:
    return BoosterInstance(
        booster_type=Symbol.LB,
        position=pos,
        state=state,
        orientation=None,
        source_cluster_index=0,
    )


def _make_slb(
    pos: Position,
    state: BoosterState = BoosterState.ARMED,
) -> BoosterInstance:
    return BoosterInstance(
        booster_type=Symbol.SLB,
        position=pos,
        state=state,
        orientation=None,
        source_cluster_index=0,
    )


def _chain_registry(config: MasterConfig) -> ArchetypeRegistry:
    """Register chain family + dependencies needed for validation."""
    reg = ArchetypeRegistry(config)
    register_dead_archetypes(reg)
    register_rocket_archetypes(reg)
    register_bomb_archetypes(reg)
    register_chain_archetypes(reg)
    return reg


# ===========================================================================
# Registration tests
# ===========================================================================

class TestChainRegistration:
    """Chain family archetype registration and CONTRACT-SIG validation."""

    def test_register_all_5_chain_archetypes(
        self, default_config: MasterConfig,
    ) -> None:
        """All 5 chain archetypes register without errors."""
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)

        chain_family = reg.get_family("chain")
        assert len(chain_family) == 5

    def test_chain_archetype_ids(
        self, default_config: MasterConfig,
    ) -> None:
        """Registered IDs match spec."""
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)

        expected_ids = {
            "chain_2_mixed",
            "chain_3_plus",
            "multi_booster_parallel",
            "cascade_to_booster_to_cascade",
            "booster_phase_multi",
        }
        chain_ids = {sig.id for sig in reg.get_family("chain")}
        assert chain_ids == expected_ids

    def test_chain_family_is_basegame(
        self, default_config: MasterConfig,
    ) -> None:
        """All chain archetypes use basegame criteria."""
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)

        for sig in reg.get_family("chain"):
            assert sig.criteria == "basegame", f"{sig.id} has criteria={sig.criteria}"

    def test_contract_sig7_chain_depth_requires_two_types(
        self, default_config: MasterConfig,
    ) -> None:
        """CONTRACT-SIG-7: chain_depth.min > 0 requires at least 2 booster fire types."""
        chain_sigs = {
            sig.id: sig
            for sig in ArchetypeRegistry(default_config).get_family("chain")
            if register_chain_archetypes(ArchetypeRegistry(default_config)) is None  # noqa: side effect
        }
        # Re-register to get actual sigs
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)
        for sig in reg.get_family("chain"):
            if sig.required_chain_depth.min_val > 0:
                # Must have 2+ booster fire types
                assert len(sig.required_booster_fires) >= 2, (
                    f"{sig.id}: chain_depth.min={sig.required_chain_depth.min_val} "
                    f"but only {len(sig.required_booster_fires)} fire types"
                )


# ===========================================================================
# Multi-booster type widening tests
# ===========================================================================

class TestTypeWidening:
    """CascadeStepConstraint with tuple booster fields."""

    def test_tuple_must_spawn_accepted(self) -> None:
        """Multi-booster tuple form is accepted by frozen dataclass."""
        step = CascadeStepConstraint(
            cluster_count=Range(2, 4),
            cluster_sizes=(Range(9, 12),),
            cluster_symbol_tier=None,
            must_spawn_booster=("R", "B"),
            must_arm_booster=None,
        )
        assert step.must_spawn_booster == ("R", "B")

    def test_single_string_still_works(self) -> None:
        """Legacy single-string form still accepted."""
        step = CascadeStepConstraint(
            cluster_count=Range(1, 2),
            cluster_sizes=(Range(11, 12),),
            cluster_symbol_tier=None,
            must_spawn_booster="B",
            must_arm_booster=None,
        )
        assert step.must_spawn_booster == "B"


# ===========================================================================
# Booster phase executor chain tests (TEST-P11-001 to P11-003)
# ===========================================================================

class TestDeepChain:
    """TEST-P11-001: Deep chain R→B→R depth 3 (chain_depth=2)."""

    def test_r_b_r_chain_produces_3_fires(
        self, default_config: MasterConfig,
    ) -> None:
        """R fires H across row, hits B, B fires blast, hits second R.

        R1 at (0,3) H-fire → hits B at (4,3) → B blast → hits R2 at (4,4).
        Three boosters fire total. chain_depth = 2 (two chain-triggered targets).
        """
        rules = _make_rules(default_config)
        r1_pos = Position(0, 3)
        b_pos = Position(4, 3)
        r2_pos = Position(4, 4)

        placements = {
            r1_pos: Symbol.R,
            b_pos: Symbol.B,
            r2_pos: Symbol.R,
        }
        board = _make_board_with_symbols(default_config, placements)

        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.R, r1_pos, orientation="H", source_cluster_index=0)
        tracker.add(Symbol.B, b_pos, orientation=None, source_cluster_index=1)
        tracker.add(Symbol.R, r2_pos, orientation="V", source_cluster_index=2)

        # Arm R1 — B and R2 will be chain-triggered
        tracker.arm_adjacent(frozenset({Position(1, 3)}))

        executor = BoosterPhaseExecutor(
            tracker, rules, rules.chain_initiators,
        )
        executor.register_fire_handler(Symbol.R, fire_rocket)
        executor.register_fire_handler(Symbol.B, fire_bomb)

        results = executor.execute_booster_phase(board)

        # 3 boosters fire: R1 (armed), B (chain), R2 (chain)
        assert len(results) == 3
        fired_types = [r.booster.booster_type for r in results]
        assert fired_types == [Symbol.R, Symbol.B, Symbol.R]

        # chain_depth = count of chain-triggered targets = 2
        chain_targets_count = sum(
            1 for r in results
            if tracker.get_at(r.booster.position) is not None
            and tracker.get_at(r.booster.position).state is BoosterState.CHAIN_TRIGGERED
        )
        # B and R2 are chain-triggered (2 targets)
        b_instance = tracker.get_at(b_pos)
        r2_instance = tracker.get_at(r2_pos)
        assert b_instance is not None and b_instance.state is BoosterState.CHAIN_TRIGGERED
        assert r2_instance is not None and r2_instance.state is BoosterState.CHAIN_TRIGGERED


class TestBombToSlbChain:
    """TEST-P11-002: B→SLB chain — bomb triggers superlightball."""

    def test_bomb_chains_into_slb(
        self, default_config: MasterConfig,
    ) -> None:
        """Bomb fires, SLB in blast zone gets chain-triggered and fires."""
        rules = _make_rules(default_config)
        b_pos = Position(3, 3)
        slb_pos = Position(4, 3)  # within Manhattan distance 1 of bomb

        # SLB needs at least 2 distinct standard symbols — alternate L1 and L2
        placements: dict[Position, Symbol] = {
            b_pos: Symbol.B,
            slb_pos: Symbol.SLB,
        }
        # Fill half the board with L2 so SLB has 2 symbol types to target
        for reel in range(default_config.board.num_reels):
            for row in range(default_config.board.num_rows):
                pos = Position(reel, row)
                if pos not in placements:
                    placements[pos] = Symbol.L2 if row % 2 == 0 else Symbol.L1
        board = _make_board_with_symbols(default_config, placements)

        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.B, b_pos, orientation=None, source_cluster_index=0)
        tracker.add(Symbol.SLB, slb_pos, orientation=None, source_cluster_index=1)

        # Arm bomb — SLB will be chain-triggered
        tracker.arm_adjacent(frozenset({Position(2, 3)}))

        executor = BoosterPhaseExecutor(
            tracker, rules, rules.chain_initiators,
        )
        executor.register_fire_handler(Symbol.B, fire_bomb)
        executor.register_fire_handler(Symbol.SLB, fire_superlightball)

        results = executor.execute_booster_phase(board)

        # B fires, chains into SLB
        assert len(results) == 2
        assert results[0].booster.booster_type is Symbol.B
        assert results[1].booster.booster_type is Symbol.SLB

        # SLB fires and clears symbols (target_symbols populated)
        assert len(results[1].target_symbols) == 2

        # SLB is chain-triggered state
        slb_instance = tracker.get_at(slb_pos)
        assert slb_instance is not None
        assert slb_instance.state is BoosterState.CHAIN_TRIGGERED


class TestLbCannotInitiateChain:
    """TEST-P11-003: LB→R rejected — lightball cannot initiate chains."""

    def test_lb_fires_but_does_not_chain_rocket(
        self, default_config: MasterConfig,
    ) -> None:
        """LB fires (clearing symbols), R at a cleared position is NOT chain-triggered.

        LB is not in chain_initiators — its fire result has empty chain_targets.
        The BoosterPhaseExecutor skips chain propagation for non-initiator types.
        """
        rules = _make_rules(default_config)
        lb_pos = Position(3, 3)
        r_pos = Position(0, 3)  # R is on the board but won't be chained

        placements: dict[Position, Symbol] = {
            lb_pos: Symbol.LB,
            r_pos: Symbol.R,
        }
        board = _make_board_with_symbols(default_config, placements)

        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.LB, lb_pos, orientation=None, source_cluster_index=0)
        tracker.add(Symbol.R, r_pos, orientation="H", source_cluster_index=1)

        # Arm LB — R should NOT be triggered
        tracker.arm_adjacent(frozenset({Position(2, 3)}))

        executor = BoosterPhaseExecutor(
            tracker, rules, rules.chain_initiators,
        )
        executor.register_fire_handler(Symbol.LB, fire_lightball)
        executor.register_fire_handler(Symbol.R, fire_rocket)

        results = executor.execute_booster_phase(board)

        # Only LB fires — R stays dormant (not armed, not chain-triggered)
        assert len(results) == 1
        assert results[0].booster.booster_type is Symbol.LB
        # LB chain_targets is always empty
        assert results[0].chain_targets == ()

        # Rocket remains dormant — LB cannot initiate chains
        r_instance = tracker.get_at(r_pos)
        assert r_instance is not None
        assert r_instance.state is BoosterState.DORMANT


# ===========================================================================
# Archetype signature validation tests (TEST-P11-004 to P11-005)
# ===========================================================================

class TestCascadeToBoosterSignature:
    """TEST-P11-005: cascade_to_booster_to_cascade has growth pattern."""

    def test_early_steps_are_small_clusters(
        self, default_config: MasterConfig,
    ) -> None:
        """First steps have t1-level cluster sizes (5-6)."""
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)
        sig = reg.get("cascade_to_booster_to_cascade")

        # Step 0 and step 1 should have small clusters
        assert sig.cascade_steps[0].cluster_sizes[0].max_val <= 6
        assert sig.cascade_steps[1].cluster_sizes[0].max_val <= 6

    def test_later_step_reaches_booster_threshold(
        self, default_config: MasterConfig,
    ) -> None:
        """A later step reaches cluster size ≥ 9 (booster spawn threshold)."""
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)
        sig = reg.get("cascade_to_booster_to_cascade")

        # Find the step that spawns a booster
        booster_steps = [
            step for step in sig.cascade_steps
            if step.must_spawn_booster is not None
        ]
        assert len(booster_steps) >= 1
        # That step must have cluster size reaching booster threshold (9+)
        spawn_step = booster_steps[0]
        assert spawn_step.cluster_sizes[0].min_val >= 9

    def test_no_scatters(
        self, default_config: MasterConfig,
    ) -> None:
        """Focus on organic cascade — zero scatters."""
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)
        sig = reg.get("cascade_to_booster_to_cascade")

        assert sig.required_scatter_count.max_val == 0


# ===========================================================================
# Dead state and payout tests (TEST-P11-006, TEST-P11-007)
# ===========================================================================

class TestChainArchetypeConstraints:
    """TEST-P11-006 and P11-007: all chain archetypes have valid constraints."""

    def test_all_chain_archetypes_registered_and_valid(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P11-006: All 5 chain archetypes pass CONTRACT-SIG validation."""
        reg = _chain_registry(default_config)
        chain_family = reg.get_family("chain")
        # All 5 registered without validation errors
        assert len(chain_family) == 5

    def test_all_have_positive_cascade_depth(
        self, default_config: MasterConfig,
    ) -> None:
        """All chain archetypes require cascade_depth >= 2 (not static)."""
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)

        for sig in reg.get_family("chain"):
            assert sig.required_cascade_depth.min_val >= 2, (
                f"{sig.id} has cascade_depth.min={sig.required_cascade_depth.min_val}"
            )

    def test_payout_ranges_positive(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P11-007: All payout ranges have min > 0 (chain archetypes produce wins)."""
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)

        for sig in reg.get_family("chain"):
            assert sig.payout_range.min_val > 0.0, (
                f"{sig.id} payout_range.min={sig.payout_range.min_val}"
            )
            assert sig.payout_range.max_val >= sig.payout_range.min_val, (
                f"{sig.id} payout_range invalid: "
                f"{sig.payout_range.min_val} > {sig.payout_range.max_val}"
            )

    def test_chain_archetypes_do_not_trigger_freespin(
        self, default_config: MasterConfig,
    ) -> None:
        """Chain family is basegame — no freespin triggers."""
        reg = ArchetypeRegistry(default_config)
        register_chain_archetypes(reg)

        for sig in reg.get_family("chain"):
            assert sig.triggers_freespin is False
            assert sig.reaches_wincap is False
