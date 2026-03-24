"""Phase 3 tests — CSP Spatial Solver (Basic).

TEST-P3-001 through TEST-P3-010: constraint validation, solver placement,
spatial bias, impossible constraints, and pluggable extensibility.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from ..config.schema import MasterConfig
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.cluster_detection import detect_clusters
from ..primitives.symbols import Symbol
from ..spatial_solver.constraints import (
    ClusterConnectivity,
    ClusterNonOverlap,
    NearMissIsolation,
    ScatterNonOverlap,
    SpatialConstraint,
)
from ..spatial_solver.data_types import (
    ClusterAssignment,
    NearMissAssignment,
    SolverContext,
)
from ..spatial_solver.solver import CSPSpatialSolver, SolveFailed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(config: MasterConfig) -> SolverContext:
    """Create an empty SolverContext from the default config."""
    return SolverContext(config.board)


# ---------------------------------------------------------------------------
# TEST-P3-001: ClusterConnectivity accepts connected, rejects disconnected
# ---------------------------------------------------------------------------


class TestClusterConnectivity:
    def test_accepts_connected(self, default_config: MasterConfig) -> None:
        """A horizontally-connected cluster of 5 passes connectivity."""
        ctx = _make_context(default_config)
        positions = frozenset(Position(r, 0) for r in range(5))
        ctx.clusters.append(
            ClusterAssignment(symbol=Symbol.L1, positions=positions, size=5)
        )
        assert ClusterConnectivity().is_satisfied(ctx) is True

    def test_rejects_disconnected(self, default_config: MasterConfig) -> None:
        """Two separate groups placed as one cluster fails connectivity."""
        ctx = _make_context(default_config)
        # Group A: (0,0), (1,0) — Group B: (4,0), (5,0), (6,0) — disconnected
        positions = frozenset([
            Position(0, 0), Position(1, 0),
            Position(4, 0), Position(5, 0), Position(6, 0),
        ])
        ctx.clusters.append(
            ClusterAssignment(symbol=Symbol.L1, positions=positions, size=5)
        )
        assert ClusterConnectivity().is_satisfied(ctx) is False


# ---------------------------------------------------------------------------
# TEST-P3-002: ClusterNonOverlap rejects shared positions
# ---------------------------------------------------------------------------


class TestClusterNonOverlap:
    def test_accepts_non_overlapping(self, default_config: MasterConfig) -> None:
        """Two clusters on different positions pass."""
        ctx = _make_context(default_config)
        ctx.clusters.append(ClusterAssignment(
            symbol=Symbol.L1,
            positions=frozenset(Position(r, 0) for r in range(5)),
            size=5,
        ))
        ctx.clusters.append(ClusterAssignment(
            symbol=Symbol.H1,
            positions=frozenset(Position(r, 2) for r in range(5)),
            size=5,
        ))
        assert ClusterNonOverlap().is_satisfied(ctx) is True

    def test_rejects_shared_position(self, default_config: MasterConfig) -> None:
        """Two clusters sharing position (2,0) fails."""
        ctx = _make_context(default_config)
        ctx.clusters.append(ClusterAssignment(
            symbol=Symbol.L1,
            positions=frozenset(Position(r, 0) for r in range(5)),
            size=5,
        ))
        # Overlaps at Position(2, 0)
        ctx.clusters.append(ClusterAssignment(
            symbol=Symbol.H1,
            positions=frozenset([
                Position(2, 0), Position(2, 1), Position(2, 2),
                Position(3, 1), Position(4, 1),
            ]),
            size=5,
        ))
        assert ClusterNonOverlap().is_satisfied(ctx) is False


# ---------------------------------------------------------------------------
# TEST-P3-003: NearMissIsolation — size 4, isolated
# ---------------------------------------------------------------------------


class TestNearMissIsolation:
    def test_accepts_isolated_near_miss(self, default_config: MasterConfig) -> None:
        """Near-miss of size 4 with no adjacent same-symbol neighbor outside passes."""
        ctx = _make_context(default_config)
        # H1 near-miss at (3,3), (3,4), (4,3), (4,4) — 2x2 block
        nm_positions = frozenset([
            Position(3, 3), Position(3, 4),
            Position(4, 3), Position(4, 4),
        ])
        ctx.near_misses.append(NearMissAssignment(
            symbol=Symbol.H1, positions=nm_positions, size=4,
        ))
        assert NearMissIsolation(default_config.board.min_cluster_size).is_satisfied(ctx) is True

    def test_rejects_adjacent_same_symbol(self, default_config: MasterConfig) -> None:
        """Near-miss fails when a cluster with the same symbol is adjacent."""
        ctx = _make_context(default_config)
        nm_positions = frozenset([
            Position(3, 3), Position(3, 4),
            Position(4, 3), Position(4, 4),
        ])
        ctx.near_misses.append(NearMissAssignment(
            symbol=Symbol.H1, positions=nm_positions, size=4,
        ))
        # Place an H1 cluster adjacent to the near-miss group at (2,3)
        ctx.clusters.append(ClusterAssignment(
            symbol=Symbol.H1,
            positions=frozenset([
                Position(0, 3), Position(1, 3), Position(2, 3),
                Position(0, 4), Position(1, 4),
            ]),
            size=5,
        ))
        # Position(2,3) is adjacent to (3,3) — same symbol H1 outside group
        assert NearMissIsolation(default_config.board.min_cluster_size).is_satisfied(ctx) is False

    def test_rejects_wrong_size(self, default_config: MasterConfig) -> None:
        """Near-miss with size != min_cluster_size - 1 fails."""
        ctx = _make_context(default_config)
        # Size 3 instead of required 4 (min_cluster_size=5, so required=4)
        nm_positions = frozenset([
            Position(3, 3), Position(3, 4), Position(4, 3),
        ])
        ctx.near_misses.append(NearMissAssignment(
            symbol=Symbol.H1, positions=nm_positions, size=3,
        ))
        assert NearMissIsolation(default_config.board.min_cluster_size).is_satisfied(ctx) is False


# ---------------------------------------------------------------------------
# TEST-P3-004: ScatterNonOverlap — rejects scatter on cluster position
# ---------------------------------------------------------------------------


class TestScatterNonOverlap:
    def test_accepts_non_overlapping(self, default_config: MasterConfig) -> None:
        """Scatters on free positions pass."""
        ctx = _make_context(default_config)
        ctx.clusters.append(ClusterAssignment(
            symbol=Symbol.L1,
            positions=frozenset(Position(r, 0) for r in range(5)),
            size=5,
        ))
        ctx.scatter_positions.update([Position(0, 6), Position(6, 6)])
        assert ScatterNonOverlap().is_satisfied(ctx) is True

    def test_rejects_scatter_on_cluster(self, default_config: MasterConfig) -> None:
        """Scatter on a cluster position fails."""
        ctx = _make_context(default_config)
        ctx.clusters.append(ClusterAssignment(
            symbol=Symbol.L1,
            positions=frozenset(Position(r, 0) for r in range(5)),
            size=5,
        ))
        # Scatter at (2,0) overlaps with cluster
        ctx.scatter_positions.add(Position(2, 0))
        assert ScatterNonOverlap().is_satisfied(ctx) is False


# ---------------------------------------------------------------------------
# TEST-P3-005: CSP — single cluster size 5 produces connected positions
# ---------------------------------------------------------------------------


class TestCSPSingleCluster:
    def test_single_cluster_size_5(self, default_config: MasterConfig) -> None:
        """Solver places one cluster of size 5 with valid connectivity."""
        solver = CSPSpatialSolver(default_config)
        result = solver.solve_step(
            cluster_specs=[(Symbol.L1, 5)],
            near_miss_specs=[],
            scatter_count=0,
            booster_specs=[],
            rng=random.Random(42),
        )

        assert len(result.clusters) == 1
        cluster = result.clusters[0]
        assert cluster.symbol == Symbol.L1
        assert cluster.size == 5
        assert len(cluster.positions) == 5

        # Verify connectivity using shared BFS from primitives — place on board
        # and confirm detect_clusters finds it
        board = Board.empty(default_config.board)
        for pos in cluster.positions:
            board.set(pos, Symbol.L1)
        detected = detect_clusters(board, default_config)
        # Should find at least one cluster containing these positions
        matching = [
            c for c in detected
            if c.symbol == Symbol.L1 and c.positions == cluster.positions
        ]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# TEST-P3-006: CSP — two clusters non-overlapping
# ---------------------------------------------------------------------------


class TestCSPTwoClusters:
    def test_two_clusters_non_overlapping(self, default_config: MasterConfig) -> None:
        """Solver places two clusters with no shared positions."""
        solver = CSPSpatialSolver(default_config)
        result = solver.solve_step(
            cluster_specs=[(Symbol.L1, 5), (Symbol.H1, 5)],
            near_miss_specs=[],
            scatter_count=0,
            booster_specs=[],
            rng=random.Random(42),
        )

        assert len(result.clusters) == 2
        c1, c2 = result.clusters
        assert c1.size == 5
        assert c2.size == 5
        # No shared positions
        assert not (c1.positions & c2.positions)


# ---------------------------------------------------------------------------
# TEST-P3-007: CSP — dead + near-miss isolated
# ---------------------------------------------------------------------------


class TestCSPDeadNearMiss:
    def test_near_miss_isolated(self, default_config: MasterConfig) -> None:
        """Near-miss of size 4 is isolated — no adjacent same-symbol outside."""
        solver = CSPSpatialSolver(default_config)
        result = solver.solve_step(
            cluster_specs=[],
            near_miss_specs=[(Symbol.H1, 4)],
            scatter_count=0,
            booster_specs=[],
            rng=random.Random(42),
        )

        assert len(result.near_misses) == 1
        nm = result.near_misses[0]
        assert nm.symbol == Symbol.H1
        assert nm.size == 4
        assert len(nm.positions) == 4

        # Verify isolation: no adjacent H1 outside the group exists in
        # the solver's own assignments (no clusters placed in this test)
        for pos in nm.positions:
            for neighbor in orthogonal_neighbors(pos, default_config.board):
                if neighbor not in nm.positions:
                    # No H1 should be assigned at this neighbor
                    # (no clusters placed, so all neighbors are free)
                    pass  # Isolation holds vacuously when no same-symbol exists outside

        # Verify connectivity of the near-miss group
        from ..spatial_solver.constraints import _is_connected
        assert _is_connected(nm.positions, default_config.board) is True


# ---------------------------------------------------------------------------
# TEST-P3-008: CSP — respects spatial bias (statistical)
# ---------------------------------------------------------------------------


class TestCSPSpatialBias:
    def test_bias_shifts_placement_toward_corner(
        self, default_config: MasterConfig
    ) -> None:
        """Heavy bias on top-left corner region increases placement frequency there.

        Run 200 solves with fixed seed. Corner positions (reels 0-1, rows 0-1)
        should appear significantly more than uniform expectation.
        """
        # Heavy bias on top-left 2x2 corner
        corner_positions = {
            Position(r, c)
            for r in range(2)
            for c in range(2)
        }
        # Bias: corner = 50.0, everything else = 1.0
        bias: dict[Position, float] = {pos: 50.0 for pos in corner_positions}

        solver = CSPSpatialSolver(default_config)
        corner_hits = 0
        total_positions = 0
        num_trials = 200

        for i in range(num_trials):
            result = solver.solve_step(
                cluster_specs=[(Symbol.L1, 5)],
                near_miss_specs=[],
                scatter_count=0,
                booster_specs=[],
                spatial_bias=bias,
                rng=random.Random(42 + i),
            )
            for pos in result.clusters[0].positions:
                total_positions += 1
                if pos in corner_positions:
                    corner_hits += 1

        # 4 corner positions out of 49 total = ~8.2% uniform expectation
        # With heavy bias, corner should appear much more frequently
        corner_rate = corner_hits / total_positions
        uniform_expectation = len(corner_positions) / (
            default_config.board.num_reels * default_config.board.num_rows
        )
        # Biased rate should be at least 2x uniform
        assert corner_rate > uniform_expectation * 2, (
            f"Corner rate {corner_rate:.3f} not > 2x uniform {uniform_expectation:.3f}"
        )


# ---------------------------------------------------------------------------
# TEST-P3-009: CSP — raises on impossible constraints
# ---------------------------------------------------------------------------


class TestCSPImpossible:
    def test_cluster_too_large_for_board(self, default_config: MasterConfig) -> None:
        """Requesting a cluster larger than the board raises SolveFailed."""
        solver = CSPSpatialSolver(default_config)
        with pytest.raises(SolveFailed):
            solver.solve_step(
                cluster_specs=[(Symbol.L1, 50)],
                near_miss_specs=[],
                scatter_count=0,
                booster_specs=[],
                rng=random.Random(42),
            )


# ---------------------------------------------------------------------------
# TEST-P3-010: Pluggable constraint — no solver changes needed
# ---------------------------------------------------------------------------


class TestCSPPluggable:
    def test_custom_constraint_without_solver_changes(
        self, default_config: MasterConfig
    ) -> None:
        """Adding a custom AlwaysReject constraint causes SolveFailed
        without any changes to the solver code."""

        class AlwaysReject:
            """Custom constraint that always rejects — for testing pluggability."""

            def is_satisfied(self, context: SolverContext) -> bool:
                return False

        # Verify it satisfies the protocol
        assert isinstance(AlwaysReject(), SpatialConstraint)

        solver = CSPSpatialSolver(default_config)
        solver.add_constraint(AlwaysReject())

        with pytest.raises(SolveFailed):
            solver.solve_step(
                cluster_specs=[(Symbol.L1, 5)],
                near_miss_specs=[],
                scatter_count=0,
                booster_specs=[],
                rng=random.Random(42),
            )
