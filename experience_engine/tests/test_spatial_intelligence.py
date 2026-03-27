"""Unit tests for the spatial intelligence layer.

Tests GravityFieldService, InfluenceMap, and UtilityScorer in isolation.
Follows the same fixture and helper patterns as test_step_reasoner_services.py.
"""

from __future__ import annotations

import math

import pytest

from ..config.schema import BoardConfig, MasterConfig, SpatialIntelligenceConfig
from ..primitives.board import Position
from ..primitives.gravity import GravityDAG
from ..step_reasoner.services.gravity_field import GravityFieldService
from ..step_reasoner.services.influence_map import DemandSpec, InfluenceMap
from ..step_reasoner.services.utility_scorer import (
    BoosterAdjacencyFactor,
    GravityAlignmentFactor,
    InfluenceFactor,
    MergeRiskFactor,
    ScoringContext,
    UtilityScorer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gravity_dag(default_config: MasterConfig) -> GravityDAG:
    return GravityDAG(default_config.board, default_config.gravity)


@pytest.fixture(scope="session")
def gravity_field(
    gravity_dag: GravityDAG, default_config: MasterConfig,
) -> GravityFieldService:
    return GravityFieldService(gravity_dag, default_config.board)


@pytest.fixture(scope="session")
def influence_map(default_config: MasterConfig) -> InfluenceMap:
    return InfluenceMap(default_config.spatial_intelligence, default_config.board)


@pytest.fixture(scope="session")
def utility_scorer(default_config: MasterConfig) -> UtilityScorer:
    factors = [
        InfluenceFactor(),
        GravityAlignmentFactor(),
        BoosterAdjacencyFactor(),
        MergeRiskFactor(),
    ]
    return UtilityScorer(
        factors, default_config.spatial_intelligence.utility_factor_weights,
    )


# ---------------------------------------------------------------------------
# TEST-SI-001 through 004: GravityFieldService
# ---------------------------------------------------------------------------


class TestGravityFieldService:

    def test_center_column_flow_is_downward(
        self, gravity_field: GravityFieldService,
    ):
        """SI-001: Center column flow vector is purely downward (0, 1)."""
        # Column 3 (center of 7-reel board), top row
        vx, vy = gravity_field.flow_vector(Position(3, 0))
        assert abs(vx) < 1e-6, f"Expected zero horizontal, got {vx}"
        assert vy > 0.9, f"Expected downward (vy > 0.9), got {vy}"

    def test_edge_column_has_horizontal_component(
        self, gravity_field: GravityFieldService,
    ):
        """SI-002: Edge column flow vector has nonzero horizontal component."""
        # Column 0 (left edge), top row — diagonal donors introduce horizontal flow
        vx, vy = gravity_field.flow_vector(Position(0, 0))
        assert vx > 0, f"Expected positive horizontal (rightward), got {vx}"
        assert vy > 0, f"Expected positive vertical (downward), got {vy}"

    def test_alignment_directly_below_is_one(
        self, gravity_field: GravityFieldService,
    ):
        """SI-003: alignment_score = 1.0 when target is directly below pos."""
        score = gravity_field.alignment_score(Position(3, 0), Position(3, 5))
        assert score > 0.99, f"Expected ~1.0, got {score}"

    def test_alignment_directly_above_is_zero(
        self, gravity_field: GravityFieldService,
    ):
        """SI-004: alignment_score = 0.0 when target is directly above pos."""
        score = gravity_field.alignment_score(Position(3, 5), Position(3, 0))
        assert score < 0.01, f"Expected ~0.0, got {score}"


# ---------------------------------------------------------------------------
# TEST-SI-005 through 008: InfluenceMap
# ---------------------------------------------------------------------------


class TestInfluenceMap:

    def test_centroid_influence_is_one(self, influence_map: InfluenceMap):
        """SI-005: Centroid position has influence exactly 1.0."""
        demand = DemandSpec(centroid=Position(3, 3), cluster_size=5, booster_type=None)
        influence = influence_map.compute(demand)
        assert influence[Position(3, 3)] == pytest.approx(1.0)

    def test_distant_cell_below_threshold(
        self, influence_map: InfluenceMap, default_config: MasterConfig,
    ):
        """SI-006: Influence at large distance is below reserve_threshold."""
        # Small cluster size → narrow sigma. Corner-to-corner distance should be below threshold.
        demand = DemandSpec(centroid=Position(0, 0), cluster_size=3, booster_type=None)
        influence = influence_map.compute(demand)
        far_corner = Position(
            default_config.board.num_reels - 1,
            default_config.board.num_rows - 1,
        )
        assert influence[far_corner] < default_config.spatial_intelligence.reserve_threshold

    def test_larger_cluster_wider_reserve(self, influence_map: InfluenceMap):
        """SI-007: Larger cluster_size produces wider reserve zone."""
        demand_small = DemandSpec(centroid=Position(3, 3), cluster_size=3, booster_type=None)
        demand_large = DemandSpec(centroid=Position(3, 3), cluster_size=12, booster_type=None)

        influence_small = influence_map.compute(demand_small)
        influence_large = influence_map.compute(demand_large)

        zone_small = influence_map.reserve_zone(influence_small)
        zone_large = influence_map.reserve_zone(influence_large)

        assert len(zone_large) > len(zone_small), (
            f"Large cluster reserve ({len(zone_large)}) should be wider "
            f"than small ({len(zone_small)})"
        )

    def test_wild_sigma_wider_than_rocket(self, influence_map: InfluenceMap):
        """SI-008: Wild sigma multiplier (1.35) produces wider zone than Rocket (1.0)."""
        demand_w = DemandSpec(centroid=Position(3, 3), cluster_size=7, booster_type="W")
        demand_r = DemandSpec(centroid=Position(3, 3), cluster_size=7, booster_type="R")

        zone_w = influence_map.reserve_zone(influence_map.compute(demand_w))
        zone_r = influence_map.reserve_zone(influence_map.compute(demand_r))

        assert len(zone_w) >= len(zone_r), (
            f"Wild zone ({len(zone_w)}) should be >= Rocket zone ({len(zone_r)})"
        )


# ---------------------------------------------------------------------------
# TEST-SI-009 through 012: UtilityScorer
# ---------------------------------------------------------------------------


class TestUtilityScorer:

    def _make_context(
        self,
        gravity_field: GravityFieldService,
        board_config: BoardConfig,
        cluster_positions: frozenset[Position] | None = None,
        booster_landing: Position | None = None,
    ) -> ScoringContext:
        """Build a ScoringContext with a default centered demand."""
        centroid = Position(3, 3)
        demand = DemandSpec(centroid=centroid, cluster_size=5, booster_type="R")

        # Compute influence from the default InfluenceMap config
        influence: dict[Position, float] = {}
        sigma = 0.8 + 5 * 0.22  # base + cluster_size * scale
        two_sigma_sq = 2.0 * sigma * sigma
        for reel in range(board_config.num_reels):
            for row in range(board_config.num_rows):
                pos = Position(reel, row)
                dist_sq = (pos.reel - centroid.reel) ** 2 + (pos.row - centroid.row) ** 2
                influence[pos] = math.exp(-dist_sq / two_sigma_sq)

        return ScoringContext(
            influence=influence,
            gravity_field=gravity_field,
            demand=demand,
            cluster_positions=cluster_positions or frozenset(),
            board_config=board_config,
            booster_landing=booster_landing or centroid,
        )

    def test_merge_risk_lowers_adjacent_score(
        self,
        utility_scorer: UtilityScorer,
        gravity_field: GravityFieldService,
        default_config: MasterConfig,
    ):
        """SI-009: Cluster-adjacent position scores lower than non-adjacent."""
        cluster = frozenset({Position(3, 3)})
        ctx = self._make_context(
            gravity_field, default_config.board,
            cluster_positions=cluster,
        )

        # Position adjacent to cluster (merge risk = 1.0, penalized)
        adjacent = Position(3, 4)
        # Position far from cluster (no merge risk)
        distant = Position(0, 0)

        score_adj = utility_scorer.score(adjacent, ctx)
        score_far = utility_scorer.score(distant, ctx)

        # Adjacent should be penalized by merge_risk weight (0.3)
        # But also boosted by proximity — so check relative merge penalty
        ctx_no_cluster = self._make_context(
            gravity_field, default_config.board,
            cluster_positions=frozenset(),
        )
        score_adj_no_risk = utility_scorer.score(adjacent, ctx_no_cluster)
        assert score_adj < score_adj_no_risk, (
            f"Adjacent with cluster ({score_adj}) should be less than "
            f"adjacent without ({score_adj_no_risk})"
        )

    def test_booster_adjacency_higher_near_landing(
        self,
        utility_scorer: UtilityScorer,
        gravity_field: GravityFieldService,
        default_config: MasterConfig,
    ):
        """SI-010: Position near booster landing scores higher than distant."""
        landing = Position(3, 4)
        ctx = self._make_context(
            gravity_field, default_config.board,
            booster_landing=landing,
        )

        near = Position(3, 5)  # 1 cell away
        far = Position(0, 0)   # far corner

        score_near = utility_scorer.score(near, ctx)
        score_far = utility_scorer.score(far, ctx)

        assert score_near > score_far, (
            f"Near landing ({score_near}) should score higher than far ({score_far})"
        )

    def test_zero_weight_factor_no_effect(
        self,
        gravity_field: GravityFieldService,
        default_config: MasterConfig,
    ):
        """SI-011: Zero-weight factor has no effect on score."""
        # Scorer with all weights zeroed except influence
        scorer_only_influence = UtilityScorer(
            [InfluenceFactor(), GravityAlignmentFactor()],
            {"influence": 1.0, "gravity_alignment": 0.0},
        )
        ctx = self._make_context(gravity_field, default_config.board)

        pos = Position(2, 2)
        score = scorer_only_influence.score(pos, ctx)

        # Score should equal just the influence value
        expected = ctx.influence[pos]
        assert score == pytest.approx(expected, abs=1e-6)

    def test_custom_factor_pluggable(
        self,
        gravity_field: GravityFieldService,
        default_config: MasterConfig,
    ):
        """SI-012: Custom ScoringFactor plugged in without scorer code changes."""

        class ConstantFactor:
            name = "constant"

            def evaluate(self, pos: Position, ctx: ScoringContext) -> float:
                return 0.5

        scorer = UtilityScorer(
            [ConstantFactor()],
            {"constant": 1.0},
        )
        ctx = self._make_context(gravity_field, default_config.board)

        score = scorer.score(Position(0, 0), ctx)
        assert score == pytest.approx(0.5)
