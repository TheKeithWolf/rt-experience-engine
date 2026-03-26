"""Phase 8 tests — booster state machine, tracker, phase executor, ASP, CSP, spawn.

TEST-P8-001 through TEST-P8-011 covering all Phase 8 deliverables.
"""

from __future__ import annotations

import pytest

from ..archetypes.registry import ArchetypeSignature
from ..boosters.phase_executor import BoosterFireResult, BoosterPhaseExecutor
from ..boosters.state_machine import (
    BoosterInstance,
    BoosterState,
    InvalidTransition,
    transition,
)
from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.booster_rules import BoosterRules
from ..primitives.symbols import Symbol, symbol_from_name
from ..spatial_solver.constraints import BoosterArmAdjacency, BoosterCentroidPlacement
from ..spatial_solver.data_types import (
    BoosterPlacement,
    ClusterAssignment,
    SolverContext,
)
from ..variance.hints import VarianceHints


def _uniform_hints(config: MasterConfig) -> VarianceHints:
    """Create uniform variance hints for testing — no preference bias."""
    num_reels = config.board.num_reels
    num_rows = config.board.num_rows
    total = num_reels * num_rows
    spatial = {
        Position(r, c): 1.0 / total
        for r in range(num_reels)
        for c in range(num_rows)
    }
    symbols = {sym: 1.0 for sym in Symbol if sym.name in config.symbols.standard}
    return VarianceHints(
        spatial_bias=spatial,
        symbol_weights=symbols,
        near_miss_symbol_preference=(),
        cluster_size_preference=(5, 6),
    )


# ---------------------------------------------------------------------------
# TEST-P8-001: State machine valid/invalid transitions
# ---------------------------------------------------------------------------

class TestBoosterStateMachine:
    """TEST-P8-001: Valid transitions accepted, invalid rejected."""

    def test_dormant_to_armed(self) -> None:
        inst = BoosterInstance(
            booster_type=Symbol.R, position=Position(3, 3),
            state=BoosterState.DORMANT, orientation="H", source_cluster_index=0,
        )
        result = transition(inst, BoosterState.ARMED)
        assert result.state is BoosterState.ARMED
        assert result.position == inst.position
        assert result.orientation == "H"

    def test_armed_to_fired(self) -> None:
        inst = BoosterInstance(
            booster_type=Symbol.B, position=Position(1, 2),
            state=BoosterState.ARMED, orientation=None, source_cluster_index=1,
        )
        result = transition(inst, BoosterState.FIRED)
        assert result.state is BoosterState.FIRED

    def test_dormant_to_chain_triggered(self) -> None:
        inst = BoosterInstance(
            booster_type=Symbol.LB, position=Position(5, 5),
            state=BoosterState.DORMANT, orientation=None, source_cluster_index=2,
        )
        result = transition(inst, BoosterState.CHAIN_TRIGGERED)
        assert result.state is BoosterState.CHAIN_TRIGGERED

    def test_armed_to_chain_triggered(self) -> None:
        inst = BoosterInstance(
            booster_type=Symbol.R, position=Position(0, 0),
            state=BoosterState.ARMED, orientation="V", source_cluster_index=0,
        )
        result = transition(inst, BoosterState.CHAIN_TRIGGERED)
        assert result.state is BoosterState.CHAIN_TRIGGERED

    def test_fired_to_armed_raises(self) -> None:
        inst = BoosterInstance(
            booster_type=Symbol.R, position=Position(3, 3),
            state=BoosterState.FIRED, orientation="H", source_cluster_index=0,
        )
        with pytest.raises(InvalidTransition):
            transition(inst, BoosterState.ARMED)

    def test_chain_triggered_to_fired_raises(self) -> None:
        inst = BoosterInstance(
            booster_type=Symbol.B, position=Position(1, 1),
            state=BoosterState.CHAIN_TRIGGERED, orientation=None, source_cluster_index=0,
        )
        with pytest.raises(InvalidTransition):
            transition(inst, BoosterState.FIRED)

    def test_dormant_to_fired_raises(self) -> None:
        """Cannot skip arming — must go through ARMED or CHAIN_TRIGGERED."""
        inst = BoosterInstance(
            booster_type=Symbol.R, position=Position(2, 2),
            state=BoosterState.DORMANT, orientation="H", source_cluster_index=0,
        )
        with pytest.raises(InvalidTransition):
            transition(inst, BoosterState.FIRED)

    def test_fired_to_chain_triggered_raises(self) -> None:
        inst = BoosterInstance(
            booster_type=Symbol.R, position=Position(2, 2),
            state=BoosterState.FIRED, orientation="H", source_cluster_index=0,
        )
        with pytest.raises(InvalidTransition):
            transition(inst, BoosterState.CHAIN_TRIGGERED)


# ---------------------------------------------------------------------------
# TEST-P8-002: BoosterTracker adjacency detection
# ---------------------------------------------------------------------------

class TestBoosterTrackerAdjacency:
    """TEST-P8-002: Adjacency detection correct."""

    def test_adjacent_booster_detected(self, default_config: MasterConfig) -> None:
        tracker = BoosterTracker(default_config.board)
        # Place booster at (3, 3)
        tracker.add(Symbol.R, Position(3, 3), orientation="H")

        # Cluster includes (2, 3) — orthogonal neighbor of (3, 3)
        cluster_positions = frozenset({
            Position(0, 3), Position(1, 3), Position(2, 3),
            Position(0, 4), Position(0, 5),
        })
        adjacent = tracker.check_adjacency(cluster_positions)
        assert len(adjacent) == 1
        assert adjacent[0].position == Position(3, 3)

    def test_distant_booster_not_detected(self, default_config: MasterConfig) -> None:
        tracker = BoosterTracker(default_config.board)
        # Place booster at (6, 6) — far from cluster
        tracker.add(Symbol.B, Position(6, 6))

        cluster_positions = frozenset({
            Position(0, 0), Position(1, 0), Position(2, 0),
            Position(0, 1), Position(0, 2),
        })
        adjacent = tracker.check_adjacency(cluster_positions)
        assert len(adjacent) == 0

    def test_excluded_position_not_detected(self, default_config: MasterConfig) -> None:
        """Boosters at excluded positions are skipped by check_adjacency.

        Prevents freshly-spawned boosters from arming on their source cluster.
        """
        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.R, Position(3, 3), orientation="H")

        cluster_positions = frozenset({Position(2, 3), Position(1, 3)})

        # Without exclusion — found
        assert len(tracker.check_adjacency(cluster_positions)) == 1

        # With exclusion — not found
        assert len(tracker.check_adjacency(
            cluster_positions,
            exclude_positions=frozenset({Position(3, 3)}),
        )) == 0

    def test_only_dormant_detected(self, default_config: MasterConfig) -> None:
        """Already-armed boosters are not returned by check_adjacency."""
        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.R, Position(3, 3), orientation="H")
        # Arm it
        cluster = frozenset({Position(2, 3), Position(1, 3), Position(0, 3),
                             Position(0, 4), Position(0, 5)})
        tracker.arm_adjacent(cluster)

        # Now check adjacency again — should find nothing (already armed)
        adjacent = tracker.check_adjacency(cluster)
        assert len(adjacent) == 0


# ---------------------------------------------------------------------------
# TEST-P8-003: BoosterTracker row-major ordering
# ---------------------------------------------------------------------------

class TestBoosterTrackerRowMajor:
    """TEST-P8-003: get_armed returns boosters in row-major order."""

    def test_row_major_sort(self, default_config: MasterConfig) -> None:
        tracker = BoosterTracker(default_config.board)

        # Add 3 boosters at different positions
        tracker.add(Symbol.R, Position(5, 2), orientation="H")
        tracker.add(Symbol.B, Position(1, 0))
        tracker.add(Symbol.R, Position(3, 4), orientation="V")

        # Manually arm all three
        # (1, 0) — row 0
        tracker._boosters[Position(1, 0)] = transition(
            tracker._boosters[Position(1, 0)], BoosterState.ARMED
        )
        # (5, 2) — row 2
        tracker._boosters[Position(5, 2)] = transition(
            tracker._boosters[Position(5, 2)], BoosterState.ARMED
        )
        # (3, 4) — row 4
        tracker._boosters[Position(3, 4)] = transition(
            tracker._boosters[Position(3, 4)], BoosterState.ARMED
        )

        armed = tracker.get_armed()
        assert len(armed) == 3
        # Row-major: sorted by (row, reel)
        assert armed[0].position == Position(1, 0)   # row 0, reel 1
        assert armed[1].position == Position(5, 2)   # row 2, reel 5
        assert armed[2].position == Position(3, 4)   # row 4, reel 3


# ---------------------------------------------------------------------------
# TEST-P8-004: BoosterTracker position update after gravity
# ---------------------------------------------------------------------------

class TestBoosterTrackerGravity:
    """TEST-P8-004: Position update after gravity."""

    def test_position_remap(self, default_config: MasterConfig) -> None:
        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.R, Position(3, 2), orientation="H")

        # Gravity moves booster from (3, 2) to (3, 5)
        tracker.update_positions_after_gravity({
            Position(3, 2): Position(3, 5),
        })

        assert tracker.get_at(Position(3, 2)) is None
        moved = tracker.get_at(Position(3, 5))
        assert moved is not None
        assert moved.booster_type is Symbol.R
        assert moved.position == Position(3, 5)
        assert moved.orientation == "H"

    def test_stationary_booster_unchanged(self, default_config: MasterConfig) -> None:
        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.B, Position(6, 6))

        # Empty position map — nothing moves
        tracker.update_positions_after_gravity({})

        assert tracker.get_at(Position(6, 6)) is not None


# ---------------------------------------------------------------------------
# TEST-P8-005 through P8-007: ASP booster lifecycle
# (These tests require Clingo. They are integration tests that verify the
#  ASP rules produce correct booster lifecycle atoms.)
# ---------------------------------------------------------------------------

def _make_rocket_signature(config: MasterConfig) -> ArchetypeSignature:
    """Synthetic rocket archetype: cascade depth 2, must spawn rocket at step 0."""
    from ..archetypes.registry import CascadeStepConstraint
    return ArchetypeSignature(
        id="test_rocket_h_fire",
        family="rocket",
        criteria="basegame",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(9, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        required_cascade_depth=Range(1, 2),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.5, 50.0),
        triggers_freespin=False,
        reaches_wincap=False,
    )


def _make_chain_signature(config: MasterConfig) -> ArchetypeSignature:
    """Synthetic chain archetype: cascade depth 2, R spawns step 0, B spawns step 1."""
    from ..archetypes.registry import CascadeStepConstraint
    return ArchetypeSignature(
        id="test_chain_r_b",
        family="chain",
        criteria="basegame",
        required_cluster_count=Range(1, 1),
        required_cluster_sizes=(Range(9, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        required_cascade_depth=Range(2, 2),
        cascade_steps=(
            # Step 0: cluster of 9-10 → spawns rocket
            CascadeStepConstraint(
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
            ),
            # Step 1: cluster of 11-12 → spawns bomb
            CascadeStepConstraint(
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(11, 11),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
            ),
        ),
        required_booster_spawns={"R": Range(1, 1), "B": Range(1, 1)},
        required_booster_fires={},
        required_chain_depth=Range(0, 1),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 200.0),
        triggers_freespin=False,
        reaches_wincap=False,
    )


class TestBoosterChainConfig:
    """TEST-P8-007: Booster chain configuration rules."""

    def test_p8_007_lb_cannot_initiate_chain(self, default_config: MasterConfig) -> None:
        """TEST-P8-007: LB cannot initiate chains — config excludes LB/SLB from chain_initiators."""
        initiators = default_config.boosters.chain_initiators
        initiator_lower = [i.lower() for i in initiators]
        assert "r" in initiator_lower, "R must be a chain initiator"
        assert "b" in initiator_lower, "B must be a chain initiator"
        # LB and SLB should NOT be chain initiators
        assert "LB" not in initiators, "LB must not be a chain initiator"
        assert "SLB" not in initiators, "SLB must not be a chain initiator"


# ---------------------------------------------------------------------------
# TEST-P8-008: CSP centroid placement constraint
# ---------------------------------------------------------------------------

class TestCSPBoosterCentroidPlacement:
    """TEST-P8-008: Centroid matches shared compute_centroid."""

    def test_booster_within_cluster(self, default_config: MasterConfig) -> None:
        rules = BoosterRules(default_config.boosters, default_config.board, default_config.symbols)
        constraint = BoosterCentroidPlacement(rules)

        # Cluster positions — booster placed at centroid (which is a member)
        cluster_positions = frozenset({
            Position(2, 2), Position(3, 2), Position(4, 2),
            Position(2, 3), Position(3, 3),
        })
        centroid = rules.compute_centroid(cluster_positions)

        context = SolverContext(default_config.board)
        context.clusters.append(ClusterAssignment(
            symbol=Symbol.L1, positions=cluster_positions, size=5,
        ))
        context.booster_placements.append(BoosterPlacement(
            booster_type=Symbol.R, position=centroid,
        ))

        assert constraint.is_satisfied(context)

    def test_booster_outside_cluster_fails(self, default_config: MasterConfig) -> None:
        rules = BoosterRules(default_config.boosters, default_config.board, default_config.symbols)
        constraint = BoosterCentroidPlacement(rules)

        cluster_positions = frozenset({
            Position(0, 0), Position(1, 0), Position(2, 0),
            Position(0, 1), Position(0, 2),
        })

        context = SolverContext(default_config.board)
        context.clusters.append(ClusterAssignment(
            symbol=Symbol.L1, positions=cluster_positions, size=5,
        ))
        # Place booster outside the cluster
        context.booster_placements.append(BoosterPlacement(
            booster_type=Symbol.R, position=Position(6, 6),
        ))

        assert not constraint.is_satisfied(context)


# ---------------------------------------------------------------------------
# TEST-P8-009: CSP arm adjacency constraint
# ---------------------------------------------------------------------------

class TestCSPBoosterArmAdjacency:
    """TEST-P8-009: Arm adjacency ensures cluster touches booster."""

    def test_cluster_adjacent_to_booster(self, default_config: MasterConfig) -> None:
        # Booster at (3, 3), cluster has (2, 3) which neighbors it
        constraint = BoosterArmAdjacency(
            booster_positions=frozenset({Position(3, 3)}),
            board_config=default_config.board,
        )
        context = SolverContext(default_config.board)
        context.clusters.append(ClusterAssignment(
            symbol=Symbol.L1,
            positions=frozenset({
                Position(2, 3), Position(1, 3), Position(0, 3),
                Position(0, 4), Position(0, 5),
            }),
            size=5,
        ))
        assert constraint.is_satisfied(context)

    def test_cluster_not_adjacent_to_booster(self, default_config: MasterConfig) -> None:
        # Booster at (6, 6), cluster far away at top-left
        constraint = BoosterArmAdjacency(
            booster_positions=frozenset({Position(6, 6)}),
            board_config=default_config.board,
        )
        context = SolverContext(default_config.board)
        context.clusters.append(ClusterAssignment(
            symbol=Symbol.L1,
            positions=frozenset({
                Position(0, 0), Position(1, 0), Position(2, 0),
                Position(0, 1), Position(0, 2),
            }),
            size=5,
        ))
        assert not constraint.is_satisfied(context)

    def test_empty_booster_positions_trivially_satisfied(
        self, default_config: MasterConfig,
    ) -> None:
        constraint = BoosterArmAdjacency(
            booster_positions=frozenset(),
            board_config=default_config.board,
        )
        context = SolverContext(default_config.board)
        assert constraint.is_satisfied(context)


# ---------------------------------------------------------------------------
# TEST-P8-010: Spawn order matches config
# ---------------------------------------------------------------------------

class TestSpawnOrder:
    """TEST-P8-010: Spawns happen in config spawn order, no position conflicts."""

    def test_spawn_order_and_no_conflicts(self, default_config: MasterConfig) -> None:
        from ..pipeline.cascade_generator import CascadeInstanceGenerator
        from ..primitives.cluster_detection import Cluster

        tracker = BoosterTracker(default_config.board)
        rules = BoosterRules(default_config.boosters, default_config.board, default_config.symbols)

        # Create clusters of different sizes to trigger different booster types
        # Size 9 → Rocket, Size 11 → Bomb (W is size 7 but skipped for tracker)
        cluster_r = Cluster(
            symbol=Symbol.L1,
            positions=frozenset({
                Position(0, 0), Position(1, 0), Position(2, 0),
                Position(3, 0), Position(4, 0), Position(5, 0),
                Position(0, 1), Position(1, 1), Position(2, 1),
            }),
            wild_positions=frozenset(),
            size=9,
        )
        cluster_b = Cluster(
            symbol=Symbol.H1,
            positions=frozenset({
                Position(0, 3), Position(1, 3), Position(2, 3),
                Position(3, 3), Position(4, 3), Position(5, 3),
                Position(0, 4), Position(1, 4), Position(2, 4),
                Position(3, 4), Position(4, 4),
            }),
            wild_positions=frozenset(),
            size=11,
        )

        # Use a temporary generator just for its _spawn_boosters method
        # Build it minimally — we only need config and booster_rules
        from unittest.mock import MagicMock
        gen = MagicMock()
        gen._config = default_config
        gen._booster_rules = rules

        # Call the method directly — board arg needed for Wild writes (not hit here)
        board_mock = MagicMock()
        placements = CascadeInstanceGenerator._spawn_boosters(
            gen, [cluster_r, cluster_b], tracker, board_mock,
        )

        # Should spawn R first, then B (per config spawn order: W, R, B, LB, SLB)
        assert len(placements) == 2
        assert placements[0].booster_type is Symbol.R
        assert placements[1].booster_type is Symbol.B

        # No position conflicts — spawned at different positions
        positions = [p.position for p in placements]
        assert len(set(positions)) == 2, "Booster positions must not conflict"

        # Verify both are in the tracker as DORMANT
        for placement in placements:
            booster = tracker.get_at(placement.position)
            assert booster is not None
            assert booster.state is BoosterState.DORMANT


# ---------------------------------------------------------------------------
# TEST-P8-011: Booster phase row-major fire with stubs
# ---------------------------------------------------------------------------

class TestBoosterPhaseRowMajorFire:
    """TEST-P8-011: Row-major fire with stub functions."""

    def test_row_major_fire_order(self, default_config: MasterConfig) -> None:
        tracker = BoosterTracker(default_config.board)
        rules = BoosterRules(default_config.boosters, default_config.board, default_config.symbols)

        # Add 3 boosters and arm them
        positions = [Position(4, 5), Position(1, 1), Position(6, 3)]
        for pos in positions:
            tracker.add(Symbol.R, pos, orientation="H")

        # Arm all three by directly transitioning (bypass adjacency for unit test)
        for pos in positions:
            tracker._boosters[pos] = transition(
                tracker._boosters[pos], BoosterState.ARMED,
            )

        chain_initiators = frozenset(
            symbol_from_name(n) for n in default_config.boosters.chain_initiators
        )
        executor = BoosterPhaseExecutor(tracker, rules, chain_initiators)
        results = executor.execute_booster_phase(Board.empty(default_config.board))

        # All 3 should fire
        assert len(results) == 3

        # Fire order should be row-major: (1,1) row=1, (6,3) row=3, (4,5) row=5
        assert results[0].booster.position == Position(1, 1)
        assert results[1].booster.position == Position(6, 3)
        assert results[2].booster.position == Position(4, 5)

        # All should be in FIRED state in the tracker
        for pos in positions:
            booster = tracker.get_at(pos)
            assert booster is not None
            assert booster.state is BoosterState.FIRED

        # Stub fire returns empty affected_positions
        for result in results:
            assert result.affected_positions == frozenset()
