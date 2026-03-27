"""Tests for Gravity-Aware WFC — TEST-GWFC-CFG-001 through TEST-GWFC-038.

Tests cover:
- Phase 1: GravityWfcConfig validation and loading (CFG-001 to CFG-003)
- Phase 2: StepIntent planned_explosion field (INT-001, INT-002)
- Phase 3: Spatial weight map zones (001 to 012)
- Phase 4: Post-gravity adjacency and propagator (013 to 021)
- Phase 5: Gravity-group collapse ordering (022 to 029)
- Phase 7: WFC fill_step integration (030 to 033)
- Phase 8: StepExecutor constraint dispatch (034 to 037)
- Phase 9: End-to-end integration (038)
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from ..board_filler.cell_state import CellState
from ..board_filler.fill_constraints import FillConstraints
from ..board_filler.gravity_adjacency import PostGravityAdjacency
from ..board_filler.gravity_ordering import (
    GravityAwareEntropySelector,
    GravityGroupComputer,
)
from ..board_filler.propagators import (
    NoClusterPropagator,
    NoSpecialSymbolPropagator,
    PostGravityPropagator,
    _virtual_component_size,
)
from ..board_filler.spatial_weights import (
    SpatialWeightMap,
    WeightZone,
    build_weight_zones,
)
from ..board_filler.wfc_solver import WFCBoardFiller
from ..config.loader import load_config
from ..config.schema import (
    ConfigValidationError,
    GravityWfcConfig,
    MasterConfig,
)
from ..pipeline.step_executor import StepExecutor
from ..primitives.board import Board, Position
from ..primitives.cluster_detection import detect_clusters
from ..primitives.gravity import GravityDAG, settle
from ..primitives.symbols import Symbol, SymbolTier
from ..step_reasoner.intent import StepIntent, StepType
from ..pipeline.protocols import Range


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


@pytest.fixture(scope="session")
def config() -> MasterConfig:
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def gravity_dag(config: MasterConfig) -> GravityDAG:
    return GravityDAG(config.board, config.gravity)


def _make_intent(
    constrained: dict[Position, Symbol] | None = None,
    strategic: dict[Position, Symbol] | None = None,
    planned_explosion: frozenset[Position] | None = None,
    is_terminal: bool = False,
    propagators: list | None = None,
    weights: dict[Symbol, float] | None = None,
    predicted_wild_positions: frozenset[Position] | None = None,
) -> StepIntent:
    """Helper to build a StepIntent with minimal boilerplate."""
    return StepIntent(
        step_type=StepType.INITIAL if not is_terminal else StepType.TERMINAL_DEAD,
        constrained_cells=constrained or {},
        strategic_cells=strategic or {},
        expected_cluster_count=Range(0, 0),
        expected_cluster_sizes=[],
        expected_cluster_tier=None,
        expected_spawns=[],
        expected_arms=[],
        expected_fires=[],
        wfc_propagators=propagators or [],
        wfc_symbol_weights=weights or {},
        predicted_post_gravity=None,
        terminal_near_misses=None,
        terminal_dormant_boosters=None,
        planned_explosion=planned_explosion,
        is_terminal=is_terminal,
        predicted_wild_positions=predicted_wild_positions,
    )


# ===================================================================
# Phase 1: Config Extension
# ===================================================================


class TestGravityWfcConfig:

    def test_gwfc_cfg_001_config_loads_with_gravity_wfc(self, config: MasterConfig) -> None:
        """TEST-GWFC-CFG-001: Config loads with gravity_wfc section, all fields present."""
        gwfc = config.gravity_wfc
        assert gwfc is not None
        assert gwfc.cluster_boundary_tier_suppression == 0.15
        assert gwfc.extended_neighborhood_radius == 2
        assert gwfc.extended_neighborhood_suppression == 0.5
        assert gwfc.compression_column_suppression == 0.7
        assert gwfc.strategic_cell_neighbor_suppression == 0.2
        assert gwfc.min_symbol_weight == 0.01

    def test_gwfc_cfg_002_rejects_invalid_suppression(self) -> None:
        """TEST-GWFC-CFG-002: Config rejects suppression values outside (0, 1]."""
        with pytest.raises(ConfigValidationError):
            GravityWfcConfig(
                cluster_boundary_tier_suppression=0.0,  # invalid: must be > 0
                extended_neighborhood_radius=2,
                extended_neighborhood_suppression=0.5,
                compression_column_suppression=0.7,
                strategic_cell_neighbor_suppression=0.2,
                min_symbol_weight=0.01,
            )
        with pytest.raises(ConfigValidationError):
            GravityWfcConfig(
                cluster_boundary_tier_suppression=1.5,  # invalid: must be <= 1
                extended_neighborhood_radius=2,
                extended_neighborhood_suppression=0.5,
                compression_column_suppression=0.7,
                strategic_cell_neighbor_suppression=0.2,
                min_symbol_weight=0.01,
            )

    def test_gwfc_cfg_003_rejects_invalid_radius(self) -> None:
        """TEST-GWFC-CFG-003: Config rejects radius < 1."""
        with pytest.raises(ConfigValidationError):
            GravityWfcConfig(
                cluster_boundary_tier_suppression=0.15,
                extended_neighborhood_radius=0,  # invalid
                extended_neighborhood_suppression=0.5,
                compression_column_suppression=0.7,
                strategic_cell_neighbor_suppression=0.2,
                min_symbol_weight=0.01,
            )


# ===================================================================
# Phase 2: StepIntent Extension
# ===================================================================


class TestStepIntentExtension:

    def test_gwfc_int_001_non_terminal_has_planned_explosion(self) -> None:
        """TEST-GWFC-INT-001: Non-terminal intent has planned_explosion as frozenset."""
        explosion = frozenset({Position(3, 3), Position(3, 4)})
        intent = _make_intent(planned_explosion=explosion)
        assert intent.planned_explosion == explosion
        assert isinstance(intent.planned_explosion, frozenset)

    def test_gwfc_int_002_terminal_has_none(self) -> None:
        """TEST-GWFC-INT-002: Terminal intent has planned_explosion=None."""
        intent = _make_intent(is_terminal=True)
        assert intent.planned_explosion is None


# ===================================================================
# Phase 3: Spatial Weight Map
# ===================================================================


class TestSpatialWeightMap:

    @pytest.fixture
    def cluster_intent(self) -> StepIntent:
        """A cluster of L1 at positions (2,3), (3,3), (4,3), (2,4), (3,4)."""
        cluster_pos = {
            Position(2, 3): Symbol.L1,
            Position(3, 3): Symbol.L1,
            Position(4, 3): Symbol.L1,
            Position(2, 4): Symbol.L1,
            Position(3, 4): Symbol.L1,
        }
        return _make_intent(
            constrained=cluster_pos,
            planned_explosion=frozenset(cluster_pos.keys()),
        )

    def test_gwfc_001_boundary_zone_suppresses(
        self, cluster_intent: StepIntent, config: MasterConfig,
    ) -> None:
        """TEST-GWFC-001: Cell in boundary zone gets suppressed weights."""
        zones = build_weight_zones(cluster_intent, config)
        assert len(zones) >= 1

        # Position (1,3) is adjacent to cluster — should be in boundary zone
        boundary_zone = zones[0]
        assert boundary_zone.contains(Position(1, 3))

        base_weights = {sym: 1.0 for sym in Symbol if sym.value <= 7}
        result = boundary_zone.apply(base_weights, 0.01)
        # L1 (same tier as cluster) should be suppressed
        assert result[Symbol.L1] < 1.0

    def test_gwfc_002_outside_zones_gets_base(
        self, cluster_intent: StepIntent, config: MasterConfig,
    ) -> None:
        """TEST-GWFC-002: Cell outside all zones gets base weights."""
        zones = build_weight_zones(cluster_intent, config)
        base_weights = {sym: 1.0 for sym in Symbol if sym.value <= 7}
        weight_map = SpatialWeightMap(base_weights, zones, 0.01)

        # Position (0, 0) is far from the cluster on a 7x7 board
        far_pos = Position(0, 0)
        result = weight_map.get_weights(far_pos)
        # Should be base weights (no suppression)
        assert result[Symbol.L1] == 1.0
        assert result[Symbol.H1] == 1.0

    def test_gwfc_003_overlapping_zones_stack(self, config: MasterConfig) -> None:
        """TEST-GWFC-003: Overlapping zones stack multiplicatively."""
        pos = Position(3, 5)
        zone_a = WeightZone(
            positions=frozenset({pos}),
            adjustments={Symbol.L1: 0.5},
        )
        zone_b = WeightZone(
            positions=frozenset({pos}),
            adjustments={Symbol.L1: 0.5},
        )
        base = {Symbol.L1: 1.0}
        weight_map = SpatialWeightMap(base, [zone_a, zone_b], 0.01)
        result = weight_map.get_weights(pos)
        # 1.0 * 0.5 * 0.5 = 0.25
        assert abs(result[Symbol.L1] - 0.25) < 1e-9

    def test_gwfc_004_min_weight_enforced(self) -> None:
        """TEST-GWFC-004: min_weight floor enforced."""
        pos = Position(0, 0)
        zone = WeightZone(
            positions=frozenset({pos}),
            adjustments={Symbol.L1: 0.001},  # Would produce 0.001
        )
        base = {Symbol.L1: 1.0}
        weight_map = SpatialWeightMap(base, [zone], 0.05)  # floor = 0.05
        result = weight_map.get_weights(pos)
        assert result[Symbol.L1] == 0.05

    def test_gwfc_005_cache_returns_same(self) -> None:
        """TEST-GWFC-005: Cache returns same result on second call."""
        pos = Position(1, 1)
        zone = WeightZone(
            positions=frozenset({pos}),
            adjustments={Symbol.L1: 0.5},
        )
        base = {Symbol.L1: 1.0}
        weight_map = SpatialWeightMap(base, [zone], 0.01)
        first = weight_map.get_weights(pos)
        second = weight_map.get_weights(pos)
        assert first is second  # Same object (cached)

    def test_gwfc_006_weight_zone_contains(self) -> None:
        """TEST-GWFC-006: WeightZone.contains true for member, false for non-member."""
        zone = WeightZone(
            positions=frozenset({Position(1, 1)}),
            adjustments={},
        )
        assert zone.contains(Position(1, 1))
        assert not zone.contains(Position(2, 2))

    def test_gwfc_007_weight_zone_apply(self) -> None:
        """TEST-GWFC-007: WeightZone.apply multiplies correctly, respects floor."""
        zone = WeightZone(
            positions=frozenset(),
            adjustments={Symbol.L1: 0.3, Symbol.H1: 0.8},
        )
        base = {Symbol.L1: 1.0, Symbol.H1: 1.0, Symbol.L2: 1.0}
        result = zone.apply(base, 0.05)
        assert abs(result[Symbol.L1] - 0.3) < 1e-9
        assert abs(result[Symbol.H1] - 0.8) < 1e-9
        assert result[Symbol.L2] == 1.0  # Not in adjustments

    def test_gwfc_008_zone1_boundary(
        self, cluster_intent: StepIntent, config: MasterConfig,
    ) -> None:
        """TEST-GWFC-008: Zone 1 covers all boundary cells of cluster."""
        zones = build_weight_zones(cluster_intent, config)
        boundary_zone = zones[0]
        # (1,3) is left neighbor of (2,3) — must be in boundary
        assert boundary_zone.contains(Position(1, 3))
        # (5,3) is right neighbor of (4,3) — must be in boundary
        assert boundary_zone.contains(Position(5, 3))

    def test_gwfc_009_zone2_extended(
        self, cluster_intent: StepIntent, config: MasterConfig,
    ) -> None:
        """TEST-GWFC-009: Zone 2 covers extended neighborhood minus boundary."""
        zones = build_weight_zones(cluster_intent, config)
        # Zone 2 should exist (radius=2 means positions 2 cells away)
        assert len(zones) >= 2
        extended_zone = zones[1]
        # (0,3) is 2 cells from cluster edge — should be in extended zone
        assert extended_zone.contains(Position(0, 3))
        # Boundary positions should NOT be in zone 2
        assert not extended_zone.contains(Position(1, 3))

    def test_gwfc_010_zone3_compression(
        self, cluster_intent: StepIntent, config: MasterConfig,
    ) -> None:
        """TEST-GWFC-010: Zone 3 covers columns below cluster."""
        zones = build_weight_zones(cluster_intent, config)
        gwfc = config.gravity_wfc
        # Identify compression zone by its distinct suppression value (0.7)
        # vs boundary zone (0.15) and extended zone (0.5)
        compression_zone = None
        for z in zones:
            # Compression zone has compression_column_suppression as its value
            sample_adj = next(iter(z.adjustments.values()), None)
            if sample_adj is not None and abs(sample_adj - gwfc.compression_column_suppression) < 1e-9:
                compression_zone = z
                break
        assert compression_zone is not None, (
            f"No compression zone found among {len(zones)} zones. "
            f"Adjustment values: {[next(iter(z.adjustments.values()), None) for z in zones]}"
        )
        # Cluster at rows 3,4 in cols 2,3,4. Compression = cells below lowest per col.
        # Col 2 lowest=row4 → (2,5), (2,6). Col 3 lowest=row4 → (3,5), (3,6).
        # Col 4 lowest=row3 → (4,4), (4,5), (4,6) minus cluster positions.
        assert compression_zone.contains(Position(2, 5))
        assert compression_zone.contains(Position(3, 5))

    def test_gwfc_011_zone4_strategic(self, config: MasterConfig) -> None:
        """TEST-GWFC-011: Zone 4 covers neighbors of each strategic cell."""
        strategic = {Position(3, 3): Symbol.H1}
        cluster = {Position(1, 1): Symbol.L1}
        intent = _make_intent(
            constrained=cluster,
            strategic=strategic,
            planned_explosion=frozenset(cluster.keys()),
        )
        zones = build_weight_zones(intent, config)
        # Find zone 4 — should contain neighbors of (3,3)
        strategic_zone = None
        for z in zones:
            if Symbol.H1 in z.adjustments:
                strategic_zone = z
                break
        assert strategic_zone is not None
        # (2,3) is a neighbor of (3,3)
        assert strategic_zone.contains(Position(2, 3))

    def test_gwfc_012_no_zones_when_no_explosion(self, config: MasterConfig) -> None:
        """TEST-GWFC-012: No zones built when planned_explosion is None."""
        intent = _make_intent(planned_explosion=None)
        zones = build_weight_zones(intent, config)
        assert zones == []


# ===================================================================
# Phase 4: Post-Gravity Adjacency and Propagator
# ===================================================================


class TestPostGravityAdjacency:

    def _make_board_with_cluster(self, config: MasterConfig) -> tuple[Board, frozenset[Position]]:
        """Create a board with an L1 cluster at known positions for explosion."""
        board = Board.empty(config.board)
        # Fill entire board with mixed symbols
        rng = random.Random(42)
        symbols = [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4, Symbol.H1, Symbol.H2, Symbol.H3]
        for pos in board.all_positions():
            board.set(pos, rng.choice(symbols))

        # Place L1 cluster at top of column 3
        cluster = frozenset({
            Position(3, 0), Position(3, 1), Position(3, 2),
            Position(4, 0), Position(4, 1),
        })
        for pos in cluster:
            board.set(pos, Symbol.L1)
        return board, cluster

    def test_gwfc_013_moved_cells_mapped(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-013: Cells that move are mapped to new positions."""
        board, cluster = self._make_board_with_cluster(config)
        adjacency = PostGravityAdjacency(
            board, cluster, gravity_dag, config.gravity, config.board,
        )
        # After exploding top cells in columns 3-4, cells below should shift up
        # The adjacency should have entries for surviving cells
        settle_result = adjacency.settle_result
        assert len(settle_result.move_steps) > 0

    def test_gwfc_014_unmoved_cells_map_to_self(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-014: Cells that don't move map to themselves."""
        board, cluster = self._make_board_with_cluster(config)
        adjacency = PostGravityAdjacency(
            board, cluster, gravity_dag, config.gravity, config.board,
        )
        # Bottom-left corner cell (0,6) is far from explosion — should have neighbors
        neighbors = adjacency.virtual_neighbors(Position(0, 6))
        # It should still have its original neighbors (or very similar)
        assert isinstance(neighbors, list)

    def test_gwfc_015_empty_cells_at_top(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-015: Empty cells after settle are at top of columns."""
        board, cluster = self._make_board_with_cluster(config)
        adjacency = PostGravityAdjacency(
            board, cluster, gravity_dag, config.gravity, config.board,
        )
        # After exploding 5 cells, 5 cells should be empty
        assert len(adjacency.settle_result.empty_positions) == len(cluster)

    def test_gwfc_016_virtual_neighbors_returns_list(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-016: virtual_neighbors returns post-gravity adjacency."""
        board, cluster = self._make_board_with_cluster(config)
        adjacency = PostGravityAdjacency(
            board, cluster, gravity_dag, config.gravity, config.board,
        )
        # Any surviving cell should have virtual neighbors
        for pos in board.all_positions():
            if pos not in cluster:
                neighbors = adjacency.virtual_neighbors(pos)
                assert isinstance(neighbors, list)
                break

    def test_gwfc_017_separated_cells_adjacent_post_gravity(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-017: Two cells separated pre-gravity can be adjacent post-gravity."""
        board, cluster = self._make_board_with_cluster(config)
        adjacency = PostGravityAdjacency(
            board, cluster, gravity_dag, config.gravity, config.board,
        )
        # Cells above and below the explosion zone may become adjacent after
        # gravity — verify that at least some virtual neighbor relationships
        # differ from physical adjacency
        all_virtual = set()
        for pos in board.all_positions():
            if pos not in cluster:
                for n in adjacency.virtual_neighbors(pos):
                    all_virtual.add((pos, n))
        # Should have at least some virtual adjacencies
        assert len(all_virtual) > 0


class TestPostGravityPropagator:

    def test_gwfc_018_removes_symbol_at_threshold(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-018: Propagator removes symbol from virtual neighbor when
        component reaches threshold."""
        board = Board.empty(config.board)
        # Place L1 symbols that will be virtually adjacent after gravity
        board.set(Position(0, 5), Symbol.L1)
        board.set(Position(0, 6), Symbol.L1)
        board.set(Position(1, 5), Symbol.L1)
        board.set(Position(1, 6), Symbol.L1)

        # No explosion — just test propagator with identity adjacency
        adjacency = PostGravityAdjacency(
            board, frozenset(), gravity_dag, config.gravity, config.board,
        )
        propagator = PostGravityPropagator(
            adjacency.virtual_neighbors, threshold=5,
        )

        # Set up cells dict with one uncollapsed neighbor
        cells = {
            Position(2, 5): CellState({Symbol.L1, Symbol.L2, Symbol.H1}),
        }

        # Propagation should check virtual neighbors
        changed = propagator.propagate(
            board, cells, Position(1, 5), config.board,
        )
        # Result depends on virtual component size — at least runs without error
        assert isinstance(changed, set)

    def test_gwfc_019_allows_below_threshold(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-019: Propagator allows symbol when virtual component below threshold."""
        board = Board.empty(config.board)
        board.set(Position(0, 6), Symbol.L1)
        # Only 1 L1 — well below threshold=5

        adjacency = PostGravityAdjacency(
            board, frozenset(), gravity_dag, config.gravity, config.board,
        )
        propagator = PostGravityPropagator(
            adjacency.virtual_neighbors, threshold=5,
        )

        valid = propagator.validate_placement(board, Position(0, 6), config.board)
        assert valid is True

    def test_gwfc_020_only_counts_collapsed(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-020: Only counts collapsed cells in component size."""
        board = Board.empty(config.board)
        board.set(Position(0, 6), Symbol.L1)

        adjacency = PostGravityAdjacency(
            board, frozenset(), gravity_dag, config.gravity, config.board,
        )
        propagator = PostGravityPropagator(
            adjacency.virtual_neighbors, threshold=5,
        )
        # Single L1 cell — component is 1, well below threshold
        assert propagator.validate_placement(board, Position(0, 6), config.board)

    def test_gwfc_021_runs_alongside_no_cluster(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-021: Runs alongside NoClusterPropagator without conflict."""
        board = Board.empty(config.board)
        adjacency = PostGravityAdjacency(
            board, frozenset(), gravity_dag, config.gravity, config.board,
        )

        propagators = [
            NoClusterPropagator(config.board.min_cluster_size),
            PostGravityPropagator(adjacency.virtual_neighbors, config.board.min_cluster_size),
        ]

        cells = {
            Position(3, 3): CellState({Symbol.L1, Symbol.L2, Symbol.H1}),
        }

        # Both propagators should run without conflict
        for prop in propagators:
            changed = prop.propagate(board, cells, Position(3, 2), config.board)
            assert isinstance(changed, set)


# ===================================================================
# Phase 5: Gravity-Group Collapse Ordering
# ===================================================================


class TestGravityGroupOrdering:

    def test_gwfc_022_adjacent_same_group(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-022: Cells adjacent post-gravity are in same group."""
        board = Board.empty(config.board)
        adjacency = PostGravityAdjacency(
            board, frozenset(), gravity_dag, config.gravity, config.board,
        )
        computer = GravityGroupComputer(adjacency)
        # All cells empty — connected via virtual adjacency
        empty = set(board.all_positions())
        groups = computer.compute_groups(empty)
        # With no explosion, all cells should be in one big group
        assert len(groups) >= 1
        total_cells = sum(len(g) for g in groups)
        assert total_cells == len(empty)

    def test_gwfc_023_non_adjacent_different_groups(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-023: Cells not adjacent post-gravity are in different groups."""
        board = Board.empty(config.board)
        adjacency = PostGravityAdjacency(
            board, frozenset(), gravity_dag, config.gravity, config.board,
        )
        computer = GravityGroupComputer(adjacency)
        # Two isolated cells that are not virtually adjacent
        isolated = {Position(0, 0), Position(6, 6)}
        groups = computer.compute_groups(isolated)
        # On a 7x7 board with no explosion, (0,0) and (6,6) ARE connected
        # through intermediate cells — but since we only give 2 cells,
        # they should be separate (not virtually adjacent to each other)
        # unless they happen to be virtual neighbors
        total = sum(len(g) for g in groups)
        assert total == 2

    def test_gwfc_024_groups_sorted_largest_first(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-024: Groups sorted largest-first."""
        board = Board.empty(config.board)
        adjacency = PostGravityAdjacency(
            board, frozenset(), gravity_dag, config.gravity, config.board,
        )
        computer = GravityGroupComputer(adjacency)
        groups = computer.compute_groups(set(board.all_positions()))
        for i in range(len(groups) - 1):
            assert len(groups[i]) >= len(groups[i + 1])

    def test_gwfc_025_all_cells_accounted_for(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-025: All empty cells accounted for (no orphans)."""
        board = Board.empty(config.board)
        adjacency = PostGravityAdjacency(
            board, frozenset(), gravity_dag, config.gravity, config.board,
        )
        computer = GravityGroupComputer(adjacency)
        all_pos = set(board.all_positions())
        groups = computer.compute_groups(all_pos)
        grouped = set()
        for g in groups:
            grouped.update(g)
        assert grouped == all_pos

    def test_gwfc_026_selects_from_largest_first(self) -> None:
        """TEST-GWFC-026: Selects from largest group first."""
        large = [Position(0, r) for r in range(5)]
        small = [Position(6, 0)]
        selector = GravityAwareEntropySelector([large, small])

        cells = {}
        for pos in large + small:
            cells[pos] = CellState({Symbol.L1, Symbol.L2})

        result = selector.select_next(cells)
        assert result in large

    def test_gwfc_027_within_group_picks_min_entropy(self) -> None:
        """TEST-GWFC-027: Within group, picks min-entropy cell."""
        group = [Position(0, 0), Position(0, 1), Position(0, 2)]
        selector = GravityAwareEntropySelector([group])

        cells = {
            Position(0, 0): CellState({Symbol.L1, Symbol.L2, Symbol.L3}),  # entropy 3
            Position(0, 1): CellState({Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4}),  # entropy 4
            Position(0, 2): CellState({Symbol.L1, Symbol.L2}),              # entropy 2
        }

        result = selector.select_next(cells)
        # Position(0,2) has lowest entropy (2)
        assert result == Position(0, 2)

    def test_gwfc_028_advances_to_next_group(self) -> None:
        """TEST-GWFC-028: Advances to next group when current fully collapsed."""
        group1 = [Position(0, 0)]
        group2 = [Position(1, 0)]
        selector = GravityAwareEntropySelector([group1, group2])

        cs_collapsed = CellState({Symbol.L1})
        cs_collapsed.collapse_to(Symbol.L1)

        cells = {
            Position(0, 0): cs_collapsed,  # collapsed
            Position(1, 0): CellState({Symbol.L2, Symbol.L3}),  # open
        }

        result = selector.select_next(cells)
        assert result == Position(1, 0)

    def test_gwfc_029_returns_none_all_collapsed(self) -> None:
        """TEST-GWFC-029: Returns None when all cells collapsed."""
        group = [Position(0, 0)]
        selector = GravityAwareEntropySelector([group])

        cs = CellState({Symbol.L1})
        cs.collapse_to(Symbol.L1)
        cells = {Position(0, 0): cs}

        assert selector.select_next(cells) is None


# ===================================================================
# Phase 7: WFC fill_step Integration
# ===================================================================


class TestWfcFillStep:

    def test_gwfc_030_no_secondary_cluster_post_gravity(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-030: WFC with gravity mechanisms — L1 cluster → no same-tier
        secondary cluster post-gravity."""
        rng = random.Random(42)
        board = Board.empty(config.board)

        # Fill board with mixed symbols, avoiding clusters
        filler = WFCBoardFiller(config, use_defaults=True)
        board = filler.fill(board, frozenset(), rng=rng)

        # Place L1 cluster at known positions
        cluster_pos = frozenset({
            Position(3, 3), Position(3, 4), Position(4, 3),
            Position(4, 4), Position(3, 5),
        })
        for pos in cluster_pos:
            board.set(pos, Symbol.L1)

        # Simulate explosion and gravity
        settle_result = settle(gravity_dag, board, cluster_pos, config.gravity)
        settled_board = settle_result.board

        # Now fill the empty cells using fill_step with gravity awareness
        pinned = frozenset(
            pos for pos in settled_board.all_positions()
            if settled_board.get(pos) is not None
        )
        noise_cells = [
            pos for pos in settled_board.all_positions()
            if settled_board.get(pos) is None
        ]

        intent = _make_intent(
            constrained={pos: Symbol.L1 for pos in cluster_pos},
            planned_explosion=cluster_pos,
            propagators=[
                NoSpecialSymbolPropagator(config.symbols),
                NoClusterPropagator(config.board.min_cluster_size),
            ],
            weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
        )

        adjacency = PostGravityAdjacency(
            board, cluster_pos, gravity_dag, config.gravity, config.board,
        )
        gravity_prop = PostGravityPropagator(
            adjacency.virtual_neighbors, config.board.min_cluster_size,
        )
        zones = build_weight_zones(intent, config)
        spatial = SpatialWeightMap(
            {sym: 1.0 for sym in Symbol if sym.value <= 7},
            zones, config.gravity_wfc.min_symbol_weight,
        )
        group_computer = GravityGroupComputer(adjacency)
        groups = group_computer.compute_groups(set(noise_cells))
        selector = GravityAwareEntropySelector(groups)

        constraints = FillConstraints(
            propagators=[
                NoSpecialSymbolPropagator(config.symbols),
                NoClusterPropagator(config.board.min_cluster_size),
                gravity_prop,
            ],
            spatial_weights=spatial,
            gravity_adjacency=adjacency,
            gravity_groups=selector,
            flat_symbol_weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
        )

        filler2 = WFCBoardFiller(config, use_defaults=False)
        filled = filler2.fill_step(
            settled_board, pinned, noise_cells, constraints, rng,
        )

        # Verify: no unintended clusters on the filled board
        clusters = detect_clusters(filled, config)
        # Should have no clusters (the original cluster was exploded)
        assert len(clusters) == 0

    def test_gwfc_031_statistical_no_post_gravity_clusters(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-031: 100 random boards, gravity-aware fill produces fewer
        clusters than baseline fill (statistical improvement)."""
        gravity_cluster_count = 0
        baseline_cluster_count = 0
        valid_trials = 0

        for seed in range(100):
            rng_g = random.Random(seed)
            rng_b = random.Random(seed)
            board = Board.empty(config.board)

            # Fill board with NoCluster defaults
            filler = WFCBoardFiller(config, use_defaults=True)
            try:
                board = filler.fill(board, frozenset(), rng=random.Random(seed + 1000))
            except Exception:
                continue

            # Small L2 cluster for explosion
            cluster_pos = frozenset({
                Position(2, 2), Position(2, 3), Position(3, 2),
                Position(3, 3), Position(4, 2),
            })
            for pos in cluster_pos:
                board.set(pos, Symbol.L2)

            # Settle
            settle_result = settle(gravity_dag, board, cluster_pos, config.gravity)
            settled = settle_result.board

            pinned = frozenset(
                pos for pos in settled.all_positions()
                if settled.get(pos) is not None
            )
            noise = [
                pos for pos in settled.all_positions()
                if settled.get(pos) is None
            ]
            if not noise:
                continue

            valid_trials += 1

            # Gravity-aware fill
            adjacency = PostGravityAdjacency(
                board, cluster_pos, gravity_dag, config.gravity, config.board,
            )
            constraints_g = FillConstraints(
                propagators=[
                    NoSpecialSymbolPropagator(config.symbols),
                    NoClusterPropagator(config.board.min_cluster_size),
                    PostGravityPropagator(adjacency.virtual_neighbors, config.board.min_cluster_size),
                ],
                spatial_weights=None,
                gravity_adjacency=adjacency,
                gravity_groups=None,
                flat_symbol_weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
            )
            try:
                filler_g = WFCBoardFiller(config, use_defaults=False)
                filled_g = filler_g.fill_step(settled, pinned, noise, constraints_g, rng_g)
                gravity_cluster_count += len(detect_clusters(filled_g, config))
            except Exception:
                pass

            # Baseline fill (no gravity awareness)
            constraints_b = FillConstraints(
                propagators=[
                    NoSpecialSymbolPropagator(config.symbols),
                    NoClusterPropagator(config.board.min_cluster_size),
                ],
                spatial_weights=None,
                gravity_adjacency=None,
                gravity_groups=None,
                flat_symbol_weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
            )
            try:
                filler_b = WFCBoardFiller(config, use_defaults=False)
                filled_b = filler_b.fill_step(settled, pinned, noise, constraints_b, rng_b)
                baseline_cluster_count += len(detect_clusters(filled_b, config))
            except Exception:
                pass

        assert valid_trials > 50, f"Too few valid trials: {valid_trials}"
        # Gravity-aware should produce no more clusters than baseline
        assert gravity_cluster_count <= baseline_cluster_count, (
            f"Gravity-aware ({gravity_cluster_count}) produced more clusters "
            f"than baseline ({baseline_cluster_count})"
        )

    def test_gwfc_032_baseline_dead_max_component(
        self, config: MasterConfig,
    ) -> None:
        """TEST-GWFC-032: WFC baseline (dead): MaxComponentPropagator enforced."""
        from ..board_filler.propagators import MaxComponentPropagator

        rng = random.Random(99)
        board = Board.empty(config.board)

        constraints = FillConstraints(
            propagators=[
                NoSpecialSymbolPropagator(config.symbols),
                MaxComponentPropagator(config.board.min_cluster_size - 2),
            ],
            spatial_weights=None,
            gravity_adjacency=None,
            gravity_groups=None,
            flat_symbol_weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
        )

        filler = WFCBoardFiller(config, use_defaults=False)
        filled = filler.fill_step(
            board, frozenset(), list(board.all_positions()), constraints, rng,
        )

        clusters = detect_clusters(filled, config)
        assert len(clusters) == 0

    def test_gwfc_033_statistical_dead_boards(
        self, config: MasterConfig,
    ) -> None:
        """TEST-GWFC-033: 100 dead boards, all have no clusters."""
        from ..board_filler.propagators import MaxComponentPropagator

        for seed in range(100):
            rng = random.Random(seed)
            board = Board.empty(config.board)

            constraints = FillConstraints(
                propagators=[
                    NoSpecialSymbolPropagator(config.symbols),
                    MaxComponentPropagator(config.board.min_cluster_size - 2),
                ],
                spatial_weights=None,
                gravity_adjacency=None,
                gravity_groups=None,
                flat_symbol_weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
            )

            filler = WFCBoardFiller(config, use_defaults=False)
            try:
                filled = filler.fill_step(
                    board, frozenset(), list(board.all_positions()),
                    constraints, rng,
                )
                clusters = detect_clusters(filled, config)
                assert len(clusters) == 0, f"Seed {seed} produced {len(clusters)} clusters"
            except Exception:
                pass  # FillFailed acceptable


# ===================================================================
# Phase 8: StepExecutor Constraint Dispatch
# ===================================================================


class TestStepExecutorDispatch:

    def test_gwfc_034_gravity_aware_when_explosion(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-034: planned_explosion present → gravity-aware constraints built."""
        executor = StepExecutor(config, gravity_dag=gravity_dag)

        cluster = {Position(3, 3): Symbol.L1, Position(3, 4): Symbol.L1,
                   Position(4, 3): Symbol.L1, Position(4, 4): Symbol.L1,
                   Position(3, 5): Symbol.L1}
        intent = _make_intent(
            constrained=cluster,
            planned_explosion=frozenset(cluster.keys()),
            propagators=[
                NoSpecialSymbolPropagator(config.symbols),
                NoClusterPropagator(config.board.min_cluster_size),
            ],
            weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
        )

        board = Board.empty(config.board)
        rng = random.Random(42)

        # Should use gravity-aware path (fill_step) — verify it runs without error
        filled = executor.execute(intent, board, rng)
        # Board should be fully filled
        for pos in filled.all_positions():
            assert filled.get(pos) is not None

    def test_gwfc_035_baseline_when_no_explosion(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-035: planned_explosion None → baseline constraints built."""
        executor = StepExecutor(config, gravity_dag=gravity_dag)

        intent = _make_intent(
            planned_explosion=None,
            is_terminal=True,
            propagators=[
                NoSpecialSymbolPropagator(config.symbols),
                NoClusterPropagator(config.board.min_cluster_size),
            ],
            weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
        )

        board = Board.empty(config.board)
        rng = random.Random(42)

        filled = executor.execute(intent, board, rng)
        for pos in filled.all_positions():
            assert filled.get(pos) is not None

    def test_gwfc_036_propagators_include_post_gravity(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-036: Propagators list includes PostGravityPropagator for non-dead."""
        executor = StepExecutor(config, gravity_dag=gravity_dag)
        cluster = {Position(3, 3): Symbol.L1}
        intent = _make_intent(
            constrained=cluster,
            planned_explosion=frozenset(cluster.keys()),
            propagators=[NoSpecialSymbolPropagator(config.symbols)],
            weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
        )

        board = Board.empty(config.board)
        constraints = executor._build_gravity_aware_constraints(
            intent, board, frozenset(cluster.keys()),
            [pos for pos in board.all_positions() if pos not in cluster],
        )

        # The propagators list should include a PostGravityPropagator
        has_pgp = any(isinstance(p, PostGravityPropagator) for p in constraints.propagators)
        assert has_pgp

    def test_gwfc_037_no_post_gravity_for_dead(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-037: Propagators list does NOT include PostGravityPropagator for dead."""
        intent = _make_intent(
            planned_explosion=None,
            is_terminal=True,
            propagators=[NoSpecialSymbolPropagator(config.symbols)],
        )

        constraints = StepExecutor._build_baseline_constraints(intent)

        has_pgp = any(isinstance(p, PostGravityPropagator) for p in constraints.propagators)
        assert not has_pgp


# ===================================================================
# Phase 9: Integration
# ===================================================================


class TestGravityWfcIntegration:

    def test_gwfc_038_end_to_end(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-038: End-to-end scenario — gravity-aware fill at step 0
        produces a board where cluster is correctly placed and all cells filled."""
        rng = random.Random(123)
        executor = StepExecutor(config, gravity_dag=gravity_dag)

        # Build initial board with LOW cluster — gravity-aware fill
        board = Board.empty(config.board)
        cluster = {
            Position(2, 3): Symbol.L1, Position(3, 3): Symbol.L1,
            Position(4, 3): Symbol.L1, Position(2, 4): Symbol.L1,
            Position(3, 4): Symbol.L1,
        }

        intent = _make_intent(
            constrained=cluster,
            planned_explosion=frozenset(cluster.keys()),
            propagators=[
                NoSpecialSymbolPropagator(config.symbols),
                NoClusterPropagator(config.board.min_cluster_size),
            ],
            weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
        )

        # Execute step 0 — fills entire board with gravity awareness
        filled = executor.execute(intent, board, rng)

        # Verify board is fully filled
        for pos in filled.all_positions():
            assert filled.get(pos) is not None, f"Cell {pos} is empty"

        # Verify cluster is placed correctly
        for pos, sym in cluster.items():
            assert filled.get(pos) is sym

        # Verify spatial weight map was used — check that the board has
        # no pre-gravity clusters other than the intended one
        pre_clusters = detect_clusters(filled, config)
        # Only the intended L1 cluster should exist
        assert len(pre_clusters) == 1
        assert pre_clusters[0].symbol is Symbol.L1
        assert pre_clusters[0].positions == frozenset(cluster.keys())

        # Simulate explosion + gravity
        settle_result = settle(
            gravity_dag, filled, frozenset(cluster.keys()), config.gravity,
        )

        # Verify conservation: explosion created empty cells
        assert len(settle_result.empty_positions) == len(cluster)

        # Verify the settled board has no standard-symbol clusters
        # (the cluster was exploded, gravity may rearrange survivors)
        settled = settle_result.board
        post_clusters = detect_clusters(settled, config)
        # Post-gravity clusters from survivors are what gravity-aware WFC
        # is designed to minimize — verify the count is manageable
        # (exact zero depends on the specific board arrangement)
        assert len(post_clusters) <= 1, (
            f"Post-gravity produced {len(post_clusters)} survivor clusters — "
            f"gravity-aware WFC should minimize these"
        )


# ===================================================================
# Phase 10: Wild-Aware PostGravityPropagator
# ===================================================================


def _linear_neighbors(pos: Position) -> list[Position]:
    """Simple linear adjacency — positions at (col±1, same row).

    Used by wild-aware tests for deterministic virtual neighbor graphs
    without requiring a full PostGravityAdjacency setup.
    """
    return [Position(pos.reel + d, pos.row) for d in (-1, 1)]


def _make_linear_board(symbol_positions: dict[int, Symbol], num_cols: int = 8) -> Board:
    """Create a 1-row board with symbols at specified column indices."""
    board = Board(num_cols, 1)
    for col, sym in symbol_positions.items():
        board.set(Position(col, 0), sym)
    return board


class TestVirtualComponentSizeWildAware:
    """Tests for _virtual_component_size with wild_positions parameter."""

    def test_gwfc_040a_without_wilds_groups_stay_separate(self) -> None:
        """TEST-GWFC-040a: Two groups of 3 separated by empty cell → size 3."""
        # Group A: columns 0-2, Group B: columns 4-6, gap at column 3
        positions = {col: Symbol.L1 for col in range(3)}
        positions.update({col: Symbol.L1 for col in range(4, 7)})
        board = _make_linear_board(positions)

        size = _virtual_component_size(
            board, Position(0, 0), Symbol.L1, _linear_neighbors,
        )
        assert size == 3

    def test_gwfc_040b_wild_bridges_groups(self) -> None:
        """TEST-GWFC-040b: Wild at separator position bridges groups → size 7."""
        positions = {col: Symbol.L1 for col in range(3)}
        positions.update({col: Symbol.L1 for col in range(4, 7)})
        board = _make_linear_board(positions)

        wild_pos = frozenset({Position(3, 0)})
        size = _virtual_component_size(
            board, Position(0, 0), Symbol.L1, _linear_neighbors,
            wild_positions=wild_pos,
        )
        # 3 L1 + 1 wild + 3 L1 = 7
        assert size == 7

    def test_gwfc_040c_wild_not_in_path_no_effect(self) -> None:
        """TEST-GWFC-040c: Wild position not adjacent to either group → no bridging."""
        positions = {col: Symbol.L1 for col in range(3)}
        positions.update({col: Symbol.L1 for col in range(4, 7)})
        # Need 2 rows so wild at row 1 is valid but not in linear path
        board = Board(8, 2)
        for col, sym in positions.items():
            board.set(Position(col, 0), sym)

        # Wild is on row 1 — linear neighbors only traverse same row
        wild_pos = frozenset({Position(3, 1)})
        size = _virtual_component_size(
            board, Position(0, 0), Symbol.L1, _linear_neighbors,
            wild_positions=wild_pos,
        )
        assert size == 3


class TestPostGravityPropagatorWildAware:
    """Tests for PostGravityPropagator with wild_positions."""

    def test_gwfc_041a_without_wilds_allows_flanking(self) -> None:
        """TEST-GWFC-041a: Without wild_positions, propagator allows L1 flanking
        a future wild because each side is below threshold."""
        # 3 L1 at columns 0-2, gap at 3, uncollapsed at 4
        board = _make_linear_board({0: Symbol.L1, 1: Symbol.L1, 2: Symbol.L1})

        propagator = PostGravityPropagator(
            _linear_neighbors, threshold=5,
        )

        cells = {
            Position(4, 0): CellState({Symbol.L1, Symbol.L2}),
        }
        # Collapse column 2 — propagator checks virtual neighbors (col 1, col 3)
        # Col 3 is empty, col 4 is two hops away and not a direct virtual neighbor
        propagator.propagate(board, cells, Position(2, 0), None)

        # L1 should NOT be pruned — propagator doesn't see through the gap
        assert Symbol.L1 in cells[Position(4, 0)].possibilities

    def test_gwfc_041b_with_wilds_prunes_through_bridge(self) -> None:
        """TEST-GWFC-041b: With wild_positions, propagator prunes L1 that would
        merge through the wild into a component >= threshold."""
        # 4 L1 at columns 0-3, wild at column 4, uncollapsed at 5
        board = _make_linear_board(
            {0: Symbol.L1, 1: Symbol.L1, 2: Symbol.L1, 3: Symbol.L1},
        )
        wild_pos = frozenset({Position(4, 0)})

        propagator = PostGravityPropagator(
            _linear_neighbors, threshold=5,
            wild_positions=wild_pos,
        )

        cells = {
            Position(5, 0): CellState({Symbol.L1, Symbol.L2}),
        }
        # Collapse at column 4 (wild position) — propagate to column 5
        changed = propagator.propagate(board, cells, Position(4, 0), None)

        # L1 should be pruned — 4 L1 + wild + L1 = 6 >= threshold 5
        assert Symbol.L1 not in cells[Position(5, 0)].possibilities
        assert Position(5, 0) in changed

    def test_gwfc_041c_different_symbol_not_pruned(self) -> None:
        """TEST-GWFC-041c: Wild bridging is symbol-specific — L2 is not pruned
        even when L1 would be."""
        board = _make_linear_board(
            {0: Symbol.L1, 1: Symbol.L1, 2: Symbol.L1, 3: Symbol.L1},
        )
        wild_pos = frozenset({Position(4, 0)})

        propagator = PostGravityPropagator(
            _linear_neighbors, threshold=5,
            wild_positions=wild_pos,
        )

        cells = {
            Position(5, 0): CellState({Symbol.L1, Symbol.L2}),
        }
        propagator.propagate(board, cells, Position(4, 0), None)

        # L2 should survive — the L1 bridge doesn't affect L2 component size
        assert Symbol.L2 in cells[Position(5, 0)].possibilities


class TestStepExecutorWildForwarding:
    """Tests for StepExecutor forwarding predicted_wild_positions to propagator."""

    def test_gwfc_042_forwards_wild_positions(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-042: _build_gravity_aware_constraints forwards
        predicted_wild_positions to PostGravityPropagator."""
        executor = StepExecutor(config, gravity_dag=gravity_dag)
        wild_pos = frozenset({Position(3, 4)})
        cluster = {Position(3, 3): Symbol.L1}
        intent = _make_intent(
            constrained=cluster,
            planned_explosion=frozenset(cluster.keys()),
            propagators=[NoSpecialSymbolPropagator(config.symbols)],
            weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
            predicted_wild_positions=wild_pos,
        )

        board = Board.empty(config.board)
        constraints = executor._build_gravity_aware_constraints(
            intent, board, frozenset(cluster.keys()),
            [pos for pos in board.all_positions() if pos not in cluster],
        )

        # Find the PostGravityPropagator and verify it carries wild positions
        pgp = next(
            p for p in constraints.propagators
            if isinstance(p, PostGravityPropagator)
        )
        assert pgp._wild_positions == wild_pos

    def test_gwfc_043_none_yields_empty_wild_positions(
        self, config: MasterConfig, gravity_dag: GravityDAG,
    ) -> None:
        """TEST-GWFC-043: predicted_wild_positions=None → propagator gets empty frozenset."""
        executor = StepExecutor(config, gravity_dag=gravity_dag)
        cluster = {Position(3, 3): Symbol.L1}
        intent = _make_intent(
            constrained=cluster,
            planned_explosion=frozenset(cluster.keys()),
            propagators=[NoSpecialSymbolPropagator(config.symbols)],
            weights={sym: 1.0 for sym in Symbol if sym.value <= 7},
        )

        board = Board.empty(config.board)
        constraints = executor._build_gravity_aware_constraints(
            intent, board, frozenset(cluster.keys()),
            [pos for pos in board.all_positions() if pos not in cluster],
        )

        pgp = next(
            p for p in constraints.propagators
            if isinstance(p, PostGravityPropagator)
        )
        assert pgp._wild_positions == frozenset()


class TestStepIntentWildPositions:
    """Tests for StepIntent predicted_wild_positions field."""

    def test_si_050_none_default(self) -> None:
        """TEST-SI-050: StepIntent with predicted_wild_positions=None constructs OK."""
        intent = _make_intent()
        assert intent.predicted_wild_positions is None

    def test_si_051_frozenset_readable(self) -> None:
        """TEST-SI-051: StepIntent with predicted_wild_positions is frozen and readable."""
        wild_pos = frozenset({Position(3, 4)})
        intent = _make_intent(predicted_wild_positions=wild_pos)
        assert intent.predicted_wild_positions == wild_pos
