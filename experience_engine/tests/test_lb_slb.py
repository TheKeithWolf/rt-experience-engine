"""Phase 10 tests — lightball & superlightball fire handlers, chain reception,
archetype registration, pipeline integration, and diagnostics.

TEST-P10-001 through TEST-P10-007 covering all Phase 10 deliverables.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ..archetypes.lb import register_lb_archetypes
from ..archetypes.registry import ArchetypeRegistry
from ..archetypes.slb import register_slb_archetypes
from ..boosters.fire_handlers import fire_lightball, fire_superlightball
from ..boosters.phase_executor import BoosterPhaseExecutor
from ..boosters.state_machine import BoosterInstance, BoosterState
from ..boosters.tracker import BoosterTracker
from ..boosters.fire_handlers import fire_rocket
from ..config.schema import MasterConfig
from ..diagnostics.engine import DiagnosticsEngine
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.symbols import Symbol
from ..validation.metrics import InstanceMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rules(config: MasterConfig) -> BoosterRules:
    return BoosterRules(config.boosters, config.board, config.symbols)


def _make_board_with_symbols(
    config: MasterConfig,
    placements: dict[Position, Symbol],
    default: Symbol = Symbol.L1,
) -> Board:
    """Create a board with specified symbols, rest filled with default."""
    board = Board.empty(config.board)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            pos = Position(reel, row)
            board.set(pos, placements.get(pos, default))
    return board


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


# ---------------------------------------------------------------------------
# TEST-P10-001: LB targets most abundant, tiebreaker highest payout_rank
# ---------------------------------------------------------------------------

class TestLBTargetingWithTiebreaker:
    """TEST-P10-001: When multiple symbols tie for most abundant, the one
    with the highest payout_rank (from config) is selected."""

    def test_lb_targets_most_abundant(self, default_config: MasterConfig) -> None:
        """LB targets the symbol with the highest count on the board."""
        rules = _make_rules(default_config)
        # Board (49 cells): 25 L1, 14 L2, 10 L3 — L1 is clearly most abundant
        placements: dict[Position, Symbol] = {}
        positions = [
            Position(reel, row)
            for reel in range(default_config.board.num_reels)
            for row in range(default_config.board.num_rows)
        ]
        for pos in positions[:25]:
            placements[pos] = Symbol.L1
        for pos in positions[25:39]:
            placements[pos] = Symbol.L2
        for pos in positions[39:]:
            placements[pos] = Symbol.L3

        board = _make_board_with_symbols(default_config, placements)
        lb = _make_lb(Position(0, 0))

        result = fire_lightball(lb, board, rules)

        # LB should target L1 (most abundant with 25 instances)
        assert result.target_symbols == ("L1",)
        assert len(result.affected_positions) == 25

    def test_lb_tiebreaker_by_payout_rank(self, default_config: MasterConfig) -> None:
        """When two symbols tie for abundance, higher payout_rank wins."""
        rules = _make_rules(default_config)

        # Board (49 cells): 20 L1, 20 H3, 9 L3
        # L1 (payout_rank=1) ties H3 (payout_rank=7) at 20 each
        # H3 should win the tiebreaker
        placements: dict[Position, Symbol] = {}
        positions = [
            Position(reel, row)
            for reel in range(default_config.board.num_reels)
            for row in range(default_config.board.num_rows)
        ]
        for pos in positions[:20]:
            placements[pos] = Symbol.L1
        for pos in positions[20:40]:
            placements[pos] = Symbol.H3
        for pos in positions[40:]:
            placements[pos] = Symbol.L3

        board = _make_board_with_symbols(default_config, placements)
        lb = _make_lb(Position(0, 0))

        result = fire_lightball(lb, board, rules)

        # H3 should win the tiebreaker (payout_rank 7 > L1's 1)
        assert result.target_symbols == ("H3",)
        assert len(result.affected_positions) == 20


# ---------------------------------------------------------------------------
# TEST-P10-002: LB clears ALL of target, does NOT chain-trigger
# ---------------------------------------------------------------------------

class TestLBClearsAllNoChain:
    """TEST-P10-002: LB clears every instance of the target symbol and
    produces no chain targets (it cannot initiate chains)."""

    def test_lb_clears_all_target_symbols(self, default_config: MasterConfig) -> None:
        """All positions of the most abundant symbol appear in affected_positions."""
        rules = _make_rules(default_config)

        # Board (49 cells): 25 L2 (most abundant), 24 L1
        placements: dict[Position, Symbol] = {}
        positions = [
            Position(reel, row)
            for reel in range(default_config.board.num_reels)
            for row in range(default_config.board.num_rows)
        ]
        l2_positions = set()
        for pos in positions[:25]:
            placements[pos] = Symbol.L2
            l2_positions.add(pos)
        # Remaining 24 → L1

        board = _make_board_with_symbols(default_config, placements, default=Symbol.L1)
        lb = _make_lb(Position(3, 3))

        result = fire_lightball(lb, board, rules)

        # Every L2 position should be affected
        assert result.affected_positions == frozenset(l2_positions)

    def test_lb_no_chain_targets(self, default_config: MasterConfig) -> None:
        """LB chain_targets is always empty — it cannot initiate chains."""
        rules = _make_rules(default_config)

        # Place a bomb adjacent to where L1 will be cleared
        placements: dict[Position, Symbol] = {
            Position(3, 3): Symbol.B,  # bomb sitting on the board
        }
        board = _make_board_with_symbols(default_config, placements, default=Symbol.L1)
        lb = _make_lb(Position(0, 0))

        result = fire_lightball(lb, board, rules)

        # LB cannot initiate chains, even if boosters are at affected positions
        assert result.chain_targets == ()


# ---------------------------------------------------------------------------
# TEST-P10-003: SLB targets two most abundant, increments grid mults
# ---------------------------------------------------------------------------

class TestSLBTargetsTwo:
    """TEST-P10-003: SLB targets the two most abundant standard symbols,
    and grid multipliers at cleared positions are incremented."""

    def test_slb_targets_two_most_abundant(self, default_config: MasterConfig) -> None:
        """SLB targets the two symbols with the highest counts."""
        rules = _make_rules(default_config)

        # Board: 15 L1, 12 H1, 10 L2, 12 L3
        placements: dict[Position, Symbol] = {}
        positions = [
            Position(reel, row)
            for reel in range(default_config.board.num_reels)
            for row in range(default_config.board.num_rows)
        ]
        for pos in positions[:15]:
            placements[pos] = Symbol.L1
        for pos in positions[15:27]:
            placements[pos] = Symbol.H1
        for pos in positions[27:37]:
            placements[pos] = Symbol.L2
        for pos in positions[37:]:
            placements[pos] = Symbol.L3

        board = _make_board_with_symbols(default_config, placements)
        slb = _make_slb(Position(0, 0))

        result = fire_superlightball(slb, board, rules)

        # L1 (15) and H1 (12) should be the two most abundant
        assert set(result.target_symbols) == {"L1", "H1"}
        assert len(result.affected_positions) == 15 + 12
        assert result.chain_targets == ()

    def test_slb_grid_multiplier_increment(self, default_config: MasterConfig) -> None:
        """Grid multipliers increment at all positions cleared by SLB, respecting cap."""
        rules = _make_rules(default_config)
        grid_mults = GridMultiplierGrid(
            default_config.grid_multiplier, default_config.board,
        )

        # Simple board: 20 L1, rest L2
        placements: dict[Position, Symbol] = {}
        positions = [
            Position(reel, row)
            for reel in range(default_config.board.num_reels)
            for row in range(default_config.board.num_rows)
        ]
        for pos in positions[:20]:
            placements[pos] = Symbol.L1
        for pos in positions[20:]:
            placements[pos] = Symbol.L2

        board = _make_board_with_symbols(default_config, placements)
        slb = _make_slb(Position(0, 0))

        result = fire_superlightball(slb, board, rules)

        # Simulate the cascade_generator's SLB grid mult increment
        for pos in result.affected_positions:
            grid_mults.increment(pos)

        # Cleared positions should now have the first_hit_value from config
        first_hit = default_config.grid_multiplier.first_hit_value
        for pos in result.affected_positions:
            assert grid_mults.get(pos) == first_hit

        # Non-cleared positions should still be at initial_value
        initial = default_config.grid_multiplier.initial_value
        non_cleared = frozenset(
            Position(reel, row)
            for reel in range(default_config.board.num_reels)
            for row in range(default_config.board.num_rows)
        ) - result.affected_positions
        for pos in non_cleared:
            assert grid_mults.get(pos) == initial


# ---------------------------------------------------------------------------
# TEST-P10-004: Chain — rocket triggers dormant LB
# ---------------------------------------------------------------------------

class TestChainRocketTriggersLB:
    """TEST-P10-004: A rocket in its path hits a dormant LB, which then fires
    via the phase executor's chain logic."""

    def test_rocket_chains_into_lb(self, default_config: MasterConfig) -> None:
        """Rocket fire hits dormant LB → LB chain-fires and clears a symbol type."""
        rules = _make_rules(default_config)

        # Board: mostly L1, with L2 as second most abundant
        placements: dict[Position, Symbol] = {}
        positions = [
            Position(reel, row)
            for reel in range(default_config.board.num_reels)
            for row in range(default_config.board.num_rows)
        ]
        # 30 L1 (most abundant), rest L2
        for pos in positions[:30]:
            placements[pos] = Symbol.L1
        for pos in positions[30:]:
            placements[pos] = Symbol.L2

        # Place LB at (5, 3) — in the path of a horizontal rocket at row 3
        placements[Position(5, 3)] = Symbol.LB
        # Place rocket at (0, 3) — H orientation, fires across row 3
        placements[Position(0, 3)] = Symbol.R

        board = _make_board_with_symbols(default_config, placements)

        # Set up tracker with armed rocket and dormant LB
        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.R, Position(0, 3), "H", 0)
        tracker.add(Symbol.LB, Position(5, 3), None, 1)

        # Arm the rocket (LB stays dormant)
        rocket = tracker.get_at(Position(0, 3))
        tracker._boosters[Position(0, 3)] = BoosterInstance(
            booster_type=Symbol.R,
            position=Position(0, 3),
            state=BoosterState.ARMED,
            orientation="H",
            source_cluster_index=0,
        )

        # Execute booster phase with real handlers
        executor = BoosterPhaseExecutor(
            tracker, rules, rules.chain_initiators,
        )
        executor.register_fire_handler(Symbol.R, fire_rocket)
        executor.register_fire_handler(Symbol.LB, fire_lightball)

        results = executor.execute_booster_phase(board)

        # Should have 2 fire results: rocket then chain-triggered LB
        assert len(results) == 2
        assert results[0].booster.booster_type is Symbol.R
        assert results[1].booster.booster_type is Symbol.LB

        # LB was chain-triggered (not directly armed)
        lb_result = results[1]
        assert lb_result.booster.state is BoosterState.CHAIN_TRIGGERED

        # LB should have targeted L1 (most abundant) and produced target_symbols
        assert lb_result.target_symbols == ("L1",)
        assert len(lb_result.affected_positions) > 0


# ---------------------------------------------------------------------------
# TEST-P10-005: Full pipeline — lb_fire_high targets HIGH symbol
# ---------------------------------------------------------------------------

class TestLBArchetypeRegistration:
    """TEST-P10-005: LB archetypes register successfully and lb_fire_high
    specifies lb_target_tier=HIGH."""

    def test_lb_archetypes_register(self, default_config: MasterConfig) -> None:
        """All 4 LB archetypes register without errors."""
        registry = ArchetypeRegistry(default_config)
        register_lb_archetypes(registry)

        assert registry.get("lb_fire_low") is not None
        assert registry.get("lb_fire_high") is not None
        assert registry.get("lb_cascade") is not None
        assert registry.get("lb_chain_triggered") is not None

    def test_lb_fire_high_targets_high_tier(self, default_config: MasterConfig) -> None:
        """lb_fire_high archetype constrains lb_target_tier to HIGH."""
        registry = ArchetypeRegistry(default_config)
        register_lb_archetypes(registry)

        sig = registry.get("lb_fire_high")
        from ..primitives.symbols import SymbolTier
        assert sig.lb_target_tier is SymbolTier.HIGH

    def test_lb_family_criteria(self, default_config: MasterConfig) -> None:
        """All LB archetypes have family='lb' and criteria='basegame'."""
        registry = ArchetypeRegistry(default_config)
        register_lb_archetypes(registry)

        for arch_id in ("lb_fire_low", "lb_fire_high", "lb_cascade", "lb_chain_triggered"):
            sig = registry.get(arch_id)
            assert sig.family == "lb"
            assert sig.criteria == "basegame"


# ---------------------------------------------------------------------------
# TEST-P10-006: Full pipeline — slb_cascade elevated multipliers
# ---------------------------------------------------------------------------

class TestSLBArchetypeRegistration:
    """TEST-P10-006: SLB archetypes register successfully and slb_cascade
    has extended cascade depth for multiplier buildup."""

    def test_slb_archetypes_register(self, default_config: MasterConfig) -> None:
        """All 4 SLB archetypes register without errors."""
        registry = ArchetypeRegistry(default_config)
        register_slb_archetypes(registry)

        assert registry.get("slb_fire") is not None
        assert registry.get("slb_cascade") is not None
        assert registry.get("slb_chain_triggered") is not None
        assert registry.get("slb_multiplier_stack") is not None

    def test_slb_cascade_deep_cascade(self, default_config: MasterConfig) -> None:
        """slb_cascade has cascade_depth 3-5 for multiplier accumulation."""
        registry = ArchetypeRegistry(default_config)
        register_slb_archetypes(registry)

        sig = registry.get("slb_cascade")
        assert sig.required_cascade_depth.min_val >= 3
        assert sig.required_cascade_depth.max_val <= 5

    def test_slb_family_criteria(self, default_config: MasterConfig) -> None:
        """All SLB archetypes have family='slb' and criteria='basegame'."""
        registry = ArchetypeRegistry(default_config)
        register_slb_archetypes(registry)

        for arch_id in ("slb_fire", "slb_cascade", "slb_chain_triggered", "slb_multiplier_stack"):
            sig = registry.get(arch_id)
            assert sig.family == "slb"
            assert sig.criteria == "basegame"

    def test_slb_no_lb_target_tier(self, default_config: MasterConfig) -> None:
        """SLB archetypes do not use lb_target_tier — they always target top 2."""
        registry = ArchetypeRegistry(default_config)
        register_slb_archetypes(registry)

        for arch_id in ("slb_fire", "slb_cascade", "slb_chain_triggered", "slb_multiplier_stack"):
            sig = registry.get(arch_id)
            assert sig.lb_target_tier is None


# ---------------------------------------------------------------------------
# TEST-P10-007: Diagnostics — lb_target_distribution balanced
# ---------------------------------------------------------------------------

class TestDiagnosticsLBTargetDistribution:
    """TEST-P10-007: DiagnosticsEngine computes lb_target_distribution
    and slb_target_distribution from InstanceMetrics."""

    def test_lb_target_distribution(self, default_config: MasterConfig) -> None:
        """lb_target_distribution tracks which symbols LB targets across instances."""
        engine = DiagnosticsEngine(default_config)

        # Create 200 mock metrics with varied LB targets
        symbols = ["L1", "L2", "L3", "L4", "H1", "H2", "H3"]
        metrics_list: list[InstanceMetrics] = []
        for i in range(200):
            target = symbols[i % len(symbols)]
            metrics_list.append(InstanceMetrics(
                archetype_id="lb_fire_low",
                family="lb",
                criteria="basegame",
                sim_id=i,
                payout=5.0,
                centipayout=500,
                win_level=4,
                cluster_count=1,
                cluster_sizes=(13,),
                cluster_symbols=("L1",),
                scatter_count=0,
                near_miss_count=0,
                near_miss_symbols=(),
                max_component_size=4,
                is_valid=True,
                validation_errors=(),
                lb_target_symbol=target,
            ))

        report = engine.analyze(tuple(metrics_list))

        # lb_target_distribution should have entries for all 7 symbols
        assert len(report.lb_target_distribution) == 7
        # Each symbol should appear roughly 28-29 times (200/7)
        for sym, count in report.lb_target_distribution.items():
            assert count >= 28, f"{sym} appeared only {count} times"
            assert count <= 29, f"{sym} appeared {count} times"

    def test_slb_target_distribution(self, default_config: MasterConfig) -> None:
        """slb_target_distribution tracks SLB target symbols."""
        engine = DiagnosticsEngine(default_config)

        metrics_list: list[InstanceMetrics] = []
        for i in range(100):
            metrics_list.append(InstanceMetrics(
                archetype_id="slb_fire",
                family="slb",
                criteria="basegame",
                sim_id=i,
                payout=10.0,
                centipayout=1000,
                win_level=5,
                cluster_count=1,
                cluster_sizes=(15,),
                cluster_symbols=("L1",),
                scatter_count=0,
                near_miss_count=0,
                near_miss_symbols=(),
                max_component_size=4,
                is_valid=True,
                validation_errors=(),
                slb_target_symbols=("L1", "H1"),
            ))

        report = engine.analyze(tuple(metrics_list))

        # slb_target_distribution should show L1 and H1 each targeted 100 times
        assert report.slb_target_distribution.get("L1", 0) == 100
        assert report.slb_target_distribution.get("H1", 0) == 100

    def test_empty_lb_slb_distributions(self, default_config: MasterConfig) -> None:
        """When no LB/SLB instances exist, distributions are empty."""
        engine = DiagnosticsEngine(default_config)

        metrics_list = [InstanceMetrics(
            archetype_id="dead_empty",
            family="dead",
            criteria="0",
            sim_id=0,
            payout=0.0,
            centipayout=0,
            win_level=0,
            cluster_count=0,
            cluster_sizes=(),
            cluster_symbols=(),
            scatter_count=0,
            near_miss_count=0,
            near_miss_symbols=(),
            max_component_size=3,
            is_valid=True,
            validation_errors=(),
        )]

        report = engine.analyze(tuple(metrics_list))
        assert report.lb_target_distribution == {}
        assert report.slb_target_distribution == {}
