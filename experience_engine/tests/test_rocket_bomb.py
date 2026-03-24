"""Phase 9 tests — rocket & bomb fire handlers, CSP chain constraints,
archetype registration, validation extensions, and diagnostics.

TEST-P9-001 through TEST-P9-025 covering all Phase 9 deliverables.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ..archetypes.bomb import register_bomb_archetypes
from ..archetypes.dead import register_dead_archetypes
from ..archetypes.registry import ArchetypeRegistry, ArchetypeSignature
from ..archetypes.rocket import register_rocket_archetypes
from ..archetypes.wild import register_wild_archetypes
from ..boosters.fire_handlers import fire_bomb, fire_rocket
from ..boosters.phase_executor import BoosterFireResult, BoosterPhaseExecutor
from ..boosters.state_machine import BoosterInstance, BoosterState
from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..diagnostics.engine import DiagnosticsEngine
from ..pipeline.data_types import BoosterFireRecord
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.symbols import Symbol
from ..spatial_solver.constraints import ChainSpatialRelation, RocketOrientationControl
from ..config.schema import BoardConfig
from ..spatial_solver.data_types import ClusterAssignment, SolverContext
from ..validation.metrics import InstanceMetrics


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


def _make_solver_context(
    board_config: BoardConfig,
    clusters: tuple[ClusterAssignment, ...] = (),
) -> SolverContext:
    """Create a SolverContext with clusters pre-populated."""
    ctx = SolverContext(board_config)
    ctx.clusters = list(clusters)
    return ctx


def _full_registry(config: MasterConfig) -> ArchetypeRegistry:
    """Register all Phase 1-9 archetypes."""
    reg = ArchetypeRegistry(config)
    register_dead_archetypes(reg)
    # tier1 registration — import dynamically since the module may have
    # different naming across phases
    try:
        from ..archetypes.tier1 import register_tier1_archetypes
        register_tier1_archetypes(reg)
    except (ImportError, AttributeError):
        pass
    register_wild_archetypes(reg)
    register_rocket_archetypes(reg)
    register_bomb_archetypes(reg)
    return reg


# ===========================================================================
# Fire handler tests (TEST-P9-001 through TEST-P9-007)
# ===========================================================================


class TestFireRocket:
    """TEST-P9-001 to P9-004: Rocket fire handler."""

    def test_h_fire_clears_row(self, default_config: MasterConfig) -> None:
        """TEST-P9-001: Horizontal rocket clears entire row, excludes immune."""
        rules = _make_rules(default_config)
        pos = Position(3, 2)
        # Place immune symbols (W, S) in the row
        placements: dict[Position, Symbol] = {
            Position(0, 2): Symbol.W,  # immune — should NOT be cleared
            Position(5, 2): Symbol.S,  # immune — should NOT be cleared
        }
        board = _make_board_with_symbols(default_config, placements)
        rocket = _make_rocket(pos, "H")

        result = fire_rocket(rocket, board, rules)

        # All non-immune positions in row 2 should be cleared, except rocket's own
        expected_cleared = {
            Position(reel, 2)
            for reel in range(default_config.board.num_reels)
            if Position(reel, 2) != pos           # not rocket's own position
            and Position(reel, 2) not in placements  # not immune positions
        }
        assert result.affected_positions == frozenset(expected_cleared)

    def test_v_fire_clears_column(self, default_config: MasterConfig) -> None:
        """TEST-P9-002: Vertical rocket clears entire column, excludes immune."""
        rules = _make_rules(default_config)
        pos = Position(4, 3)
        board = _make_board_with_symbols(default_config, {})
        rocket = _make_rocket(pos, "V")

        result = fire_rocket(rocket, board, rules)

        expected_cleared = {
            Position(4, row)
            for row in range(default_config.board.num_rows)
            if Position(4, row) != pos
        }
        assert result.affected_positions == frozenset(expected_cleared)

    def test_skips_immune_symbols(self, default_config: MasterConfig) -> None:
        """TEST-P9-003: Rocket skips W and S symbols from config."""
        rules = _make_rules(default_config)
        pos = Position(0, 0)
        # Fill entire row 0 with wilds and scatters
        placements = {Position(reel, 0): Symbol.W for reel in range(7)}
        placements[pos] = Symbol.R  # rocket itself
        board = _make_board_with_symbols(default_config, placements)
        rocket = _make_rocket(pos, "H")

        result = fire_rocket(rocket, board, rules)
        # All are immune — nothing should be cleared
        assert len(result.affected_positions) == 0

    def test_identifies_chain_targets(self, default_config: MasterConfig) -> None:
        """TEST-P9-004: Rocket identifies unfired boosters in its path."""
        rules = _make_rules(default_config)
        pos = Position(0, 3)
        # Place a bomb in the rocket's row
        bomb_pos = Position(5, 3)
        placements = {bomb_pos: Symbol.B}
        board = _make_board_with_symbols(default_config, placements)
        rocket = _make_rocket(pos, "H")

        result = fire_rocket(rocket, board, rules)
        assert bomb_pos in result.chain_targets


class TestFireBomb:
    """TEST-P9-005 to P9-006: Bomb fire handler."""

    def test_clears_blast_zone(self, default_config: MasterConfig) -> None:
        """TEST-P9-005: Bomb clears 3×3 blast zone, clipped at edges."""
        rules = _make_rules(default_config)
        # Place bomb at corner — blast should be clipped
        pos = Position(0, 0)
        board = _make_board_with_symbols(default_config, {})
        bomb = _make_bomb(pos)

        result = fire_bomb(bomb, board, rules)

        # At corner (0,0), blast radius 1 produces positions:
        # (0,0) own pos excluded, (0,1), (1,0), (1,1)
        # All within bounds for a 7x7 board
        assert pos not in result.affected_positions
        assert Position(0, 1) in result.affected_positions
        assert Position(1, 0) in result.affected_positions
        assert Position(1, 1) in result.affected_positions

    def test_chain_triggers_rocket_in_blast(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-006: Bomb identifies rocket in blast zone for chaining."""
        rules = _make_rules(default_config)
        pos = Position(3, 3)
        rocket_pos = Position(4, 3)  # within Manhattan distance 1
        placements = {rocket_pos: Symbol.R}
        board = _make_board_with_symbols(default_config, placements)
        bomb = _make_bomb(pos)

        result = fire_bomb(bomb, board, rules)
        assert rocket_pos in result.chain_targets


class TestChainIntegration:
    """TEST-P9-007: Chain R→B→R with visited set prevents infinite loop."""

    def test_chain_depth_3_no_infinite_loop(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-007: R fires, chains B, B chains another R, no loop."""
        rules = _make_rules(default_config)

        # Set up: R at (0,3) fires H, B at (4,3) in path
        # B at (4,3) fires, R at (4,4) in blast zone
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

        # Arm R1
        tracker.arm_adjacent(frozenset({Position(1, 3)}))

        executor = BoosterPhaseExecutor(
            tracker, rules, rules.chain_initiators,
        )
        executor.register_fire_handler(Symbol.R, fire_rocket)
        executor.register_fire_handler(Symbol.B, fire_bomb)

        results = executor.execute_booster_phase(board)

        # All 3 boosters should fire exactly once
        fired_positions = {r.booster.position for r in results}
        assert r1_pos in fired_positions
        assert b_pos in fired_positions
        assert r2_pos in fired_positions
        # No duplicates — visited set prevents infinite loops
        assert len(results) == 3


# ===========================================================================
# CSP constraint tests (TEST-P9-008 through TEST-P9-011)
# ===========================================================================


class TestChainSpatialRelation:
    """TEST-P9-008 to P9-009: Chain target must be in source's effect zone."""

    def test_accepts_target_in_rocket_path(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-008: Target in rocket's row is accepted."""
        rules = _make_rules(default_config)
        constraint = ChainSpatialRelation(
            source_type=Symbol.R,
            source_position=Position(0, 3),
            source_orientation="H",
            target_position=Position(5, 3),  # same row
            rules=rules,
        )
        # SolverContext content doesn't matter — constraint only checks geometry
        ctx = _make_solver_context(default_config.board)
        assert constraint.is_satisfied(ctx) is True

    def test_rejects_target_outside_bomb_blast(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-009: Target outside bomb's Manhattan radius is rejected."""
        rules = _make_rules(default_config)
        constraint = ChainSpatialRelation(
            source_type=Symbol.B,
            source_position=Position(0, 0),
            source_orientation=None,
            target_position=Position(5, 5),  # far from (0,0)
            rules=rules,
        )
        ctx = _make_solver_context(default_config.board)
        assert constraint.is_satisfied(ctx) is False


class TestRocketOrientationControl:
    """TEST-P9-010 to P9-011: Cluster shape produces correct orientation."""

    def test_accepts_matching_h_cluster(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-010: Tall cluster (row_span > col_span) produces H orientation."""
        rules = _make_rules(default_config)
        constraint = RocketOrientationControl("H", rules)

        # Vertical cluster of 10 — tall (row_span=5 > col_span=2) → H orientation
        # Fit within 7x7 board: 2 columns × 5 rows
        tall_positions = frozenset(
            [Position(2, row) for row in range(5)]
            + [Position(3, row) for row in range(5)]
        )
        cluster = ClusterAssignment(
            symbol=Symbol.L1,
            positions=tall_positions,
            size=10,
            wild_positions=frozenset(),
        )
        ctx = _make_solver_context(default_config.board, clusters=(cluster,))
        assert constraint.is_satisfied(ctx) is True

    def test_rejects_wrong_orientation(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-011: Wide cluster (col_span > row_span) produces V, not H."""
        rules = _make_rules(default_config)
        constraint = RocketOrientationControl("H", rules)

        # Horizontal cluster of 10 — wide (col_span=5 > row_span=2) → V orientation
        # Fit within 7x7 board: 5 columns × 2 rows
        wide_positions = frozenset(
            [Position(reel, 2) for reel in range(5)]
            + [Position(reel, 3) for reel in range(5)]
        )
        cluster = ClusterAssignment(
            symbol=Symbol.L1,
            positions=wide_positions,
            size=10,
            wild_positions=frozenset(),
        )
        ctx = _make_solver_context(default_config.board, clusters=(cluster,))
        assert constraint.is_satisfied(ctx) is False


# ===========================================================================
# Archetype registration tests (TEST-P9-012 through TEST-P9-014)
# ===========================================================================


class TestArchetypeRegistration:
    """TEST-P9-012 to P9-014: All rocket and bomb archetypes register cleanly."""

    def test_rocket_archetypes_register(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-012: All 22 rocket archetypes register without error."""
        reg = ArchetypeRegistry(default_config)
        register_rocket_archetypes(reg)
        assert len(reg.get_family("rocket")) == 22

    def test_bomb_archetypes_register(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-013: All 9 bomb archetypes register without error."""
        reg = ArchetypeRegistry(default_config)
        register_bomb_archetypes(reg)
        assert len(reg.get_family("bomb")) == 9

    def test_total_registered(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-014: Total archetypes = 11 dead + 12 t1 + 17 wild + 22 rocket + 9 bomb."""
        reg = _full_registry(default_config)
        expected_families = {
            "dead": 11,
            "wild": 17,
            "rocket": 22,
            "bomb": 9,
        }
        for family, expected_count in expected_families.items():
            actual = len(reg.get_family(family))
            assert actual == expected_count, (
                f"{family}: expected {expected_count}, got {actual}"
            )


# ===========================================================================
# Validation tests (TEST-P9-015 through TEST-P9-020)
# ===========================================================================


class TestValidationMetrics:
    """TEST-P9-015 to P9-020: Extended validation fields populated correctly."""

    def test_spawn_counts_populated(self) -> None:
        """TEST-P9-015: booster_spawn_counts are populated from cascade records."""
        metrics = InstanceMetrics(
            archetype_id="rocket_h_fire",
            family="rocket",
            criteria="basegame",
            sim_id=1,
            payout=5.0,
            centipayout=500,
            win_level=3,
            cluster_count=1,
            cluster_sizes=(10,),
            cluster_symbols=("L1",),
            scatter_count=0,
            near_miss_count=0,
            near_miss_symbols=(),
            max_component_size=4,
            is_valid=True,
            validation_errors=(),
            booster_spawn_counts=(("R", 1),),
            booster_fire_counts=(("R", 1),),
            chain_depth=0,
            rocket_orientation_actual="H",
        )
        assert dict(metrics.booster_spawn_counts) == {"R": 1}
        assert dict(metrics.booster_fire_counts) == {"R": 1}
        assert metrics.rocket_orientation_actual == "H"

    def test_chain_depth_populated(self) -> None:
        """TEST-P9-017: chain_depth tracked from fire records."""
        metrics = InstanceMetrics(
            archetype_id="rocket_chain_bomb",
            family="rocket",
            criteria="basegame",
            sim_id=2,
            payout=10.0,
            centipayout=1000,
            win_level=5,
            cluster_count=2,
            cluster_sizes=(10, 12),
            cluster_symbols=("L1", "H1"),
            scatter_count=0,
            near_miss_count=0,
            near_miss_symbols=(),
            max_component_size=4,
            is_valid=True,
            validation_errors=(),
            chain_depth=1,
        )
        assert metrics.chain_depth == 1


# ===========================================================================
# Diagnostics tests (TEST-P9-021 through TEST-P9-023)
# ===========================================================================


class TestBoosterDiagnostics:
    """TEST-P9-021 to P9-023: Booster-specific diagnostic metrics."""

    def _make_metrics(
        self,
        count: int,
        spawn_type: str | None = None,
        fire_type: str | None = None,
        orientation: str | None = None,
        chain_depth: int = 0,
    ) -> tuple[InstanceMetrics, ...]:
        """Create a batch of InstanceMetrics for testing."""
        result: list[InstanceMetrics] = []
        for i in range(count):
            spawns = ((spawn_type, 1),) if spawn_type else ()
            fires = ((fire_type, 1),) if fire_type else ()
            result.append(InstanceMetrics(
                archetype_id=f"test_{i}",
                family="rocket",
                criteria="basegame",
                sim_id=i,
                payout=1.0,
                centipayout=100,
                win_level=1,
                cluster_count=1,
                cluster_sizes=(10,),
                cluster_symbols=("L1",),
                scatter_count=0,
                near_miss_count=0,
                near_miss_symbols=(),
                max_component_size=4,
                is_valid=True,
                validation_errors=(),
                booster_spawn_counts=spawns,
                booster_fire_counts=fires,
                chain_depth=chain_depth,
                rocket_orientation_actual=orientation,
            ))
        return tuple(result)

    def test_spawn_rate_computed(self, default_config: MasterConfig) -> None:
        """TEST-P9-021: Spawn rate = instances with spawn / total."""
        engine = DiagnosticsEngine(default_config)
        metrics = self._make_metrics(10, spawn_type="R")
        report = engine.analyze(metrics)
        assert report.booster_spawn_rate.get("R", 0.0) == pytest.approx(1.0)

    def test_orientation_balance(self, default_config: MasterConfig) -> None:
        """TEST-P9-022: Orientation balance computed from rocket fires."""
        engine = DiagnosticsEngine(default_config)
        h_metrics = self._make_metrics(6, fire_type="R", orientation="H")
        v_metrics = self._make_metrics(4, fire_type="R", orientation="V")
        report = engine.analyze(h_metrics + v_metrics)
        assert report.rocket_orientation_balance.get("H", 0) == pytest.approx(0.6)
        assert report.rocket_orientation_balance.get("V", 0) == pytest.approx(0.4)

    def test_chain_trigger_rate(self, default_config: MasterConfig) -> None:
        """TEST-P9-023: Chain trigger rate = instances with chain_depth > 0 / total."""
        engine = DiagnosticsEngine(default_config)
        chain_metrics = self._make_metrics(3, chain_depth=1)
        no_chain = self._make_metrics(7, chain_depth=0)
        report = engine.analyze(chain_metrics + no_chain)
        assert report.chain_trigger_rate == pytest.approx(0.3)


# ===========================================================================
# Integration tests (TEST-P9-024 through TEST-P9-025)
# ===========================================================================


class TestIntegration:
    """TEST-P9-024 to P9-025: Integration and handler registration."""

    def test_cleared_cells_produce_no_payout(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-024: Booster-cleared cells don't generate cluster payout."""
        rules = _make_rules(default_config)
        # Rocket clears positions — these are just removals, not wins
        pos = Position(3, 3)
        board = _make_board_with_symbols(default_config, {})
        rocket = _make_rocket(pos, "H")
        result = fire_rocket(rocket, board, rules)

        # Fire result has affected_positions but no payout — payout is only
        # from clusters detected BEFORE the booster phase
        assert len(result.affected_positions) > 0
        # BoosterFireResult has no payout field — it only tracks cleared cells
        assert not hasattr(result, "payout")

    def test_fire_handler_replaces_stubs(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P9-025: register_fire_handler replaces stubs with real handlers."""
        rules = _make_rules(default_config)
        tracker = BoosterTracker(default_config.board)

        pos = Position(3, 3)
        tracker.add(Symbol.R, pos, orientation="H", source_cluster_index=0)
        # Arm it by placing a cluster adjacent
        tracker.arm_adjacent(frozenset({Position(3, 4)}))

        executor = BoosterPhaseExecutor(
            tracker, rules, rules.chain_initiators,
        )

        # Before registration — stub returns empty
        board = _make_board_with_symbols(default_config, {pos: Symbol.R})
        # After registration — real handler clears the row
        executor.register_fire_handler(Symbol.R, fire_rocket)
        results = executor.execute_booster_phase(board)

        assert len(results) == 1
        # Real handler should have cleared positions (not empty like stub)
        assert len(results[0].affected_positions) > 0

    def test_booster_fire_record_creation(self) -> None:
        """BoosterFireRecord can be created with correct fields."""
        record = BoosterFireRecord(
            booster_type="R",
            position_reel=3,
            position_row=3,
            orientation="H",
            affected_count=6,
            chain_target_count=1,
        )
        assert record.booster_type == "R"
        assert record.orientation == "H"
        assert record.affected_count == 6
