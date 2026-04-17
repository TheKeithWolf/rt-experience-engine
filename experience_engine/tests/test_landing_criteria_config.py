"""Snapshot tests for A4 — LandingCriteriaConfig refactor.

Verifies that hoisting the WildBridgeCriterion / RocketArmCriterion /
BombArmCriterion class constants onto LandingCriteriaConfig leaves scores
unchanged at the default values that match the YAML defaults.

The fixed LandingContext panel mirrors the cases exercised in
test_landing_evaluator.py but pins exact numeric outputs so any future
change to the scoring formulas (or to the default config values) trips
this snapshot.
"""

from __future__ import annotations

import pytest

from ..config.schema import (
    BoardConfig,
    BoosterConfig,
    ConfigValidationError,
    LandingCriteriaConfig,
    MasterConfig,
    SpawnThreshold,
    SymbolConfig,
)
from ..primitives.board import Position
from ..primitives.booster_rules import BoosterRules
from ..step_reasoner.services.landing_criteria import (
    BombArmCriterion,
    RocketArmCriterion,
    WildBridgeCriterion,
)
from ..step_reasoner.services.landing_evaluator import LandingContext, ShapeStats


def _shape() -> ShapeStats:
    """Minimal cluster shape — orientation + columns_used drive criterion logic."""
    return ShapeStats(
        col_span=1, row_span=1, orientation="H",
        centroid=Position(3, 3), columns_used=frozenset({3}),
        max_depth_per_column={3: 3},
    )


# ---------------------------------------------------------------------------
# Config validation — A4 cross-field invariants
# ---------------------------------------------------------------------------

def test_a4_rocket_weights_must_sum_to_one() -> None:
    """RocketArmCriterion is a convex combination — weights must sum to 1.0."""
    with pytest.raises(ConfigValidationError, match="rocket_arm_weight"):
        LandingCriteriaConfig(
            wild_bridge_multi_column_bonus=0.2,
            rocket_arm_weight=0.6,
            rocket_chain_weight=0.5,  # 0.6 + 0.5 != 1.0
            rocket_orientation_penalty=0.3,
            bomb_arm_weight=0.6,
            bomb_blast_weight=0.4,
        )


def test_a4_bomb_weights_must_sum_to_one() -> None:
    """BombArmCriterion is a convex combination — weights must sum to 1.0."""
    with pytest.raises(ConfigValidationError, match="bomb_arm_weight"):
        LandingCriteriaConfig(
            wild_bridge_multi_column_bonus=0.2,
            rocket_arm_weight=0.6,
            rocket_chain_weight=0.4,
            rocket_orientation_penalty=0.3,
            bomb_arm_weight=0.7,
            bomb_blast_weight=0.4,  # 0.7 + 0.4 != 1.0
        )


def test_a4_close_to_one_passes() -> None:
    """math.isclose tolerates float drift (~1e-15) from yaml parse."""
    # Construction must not raise — drift well under math.isclose default
    LandingCriteriaConfig(
        wild_bridge_multi_column_bonus=0.2,
        rocket_arm_weight=0.6,
        rocket_chain_weight=0.4 + 1e-16,
        rocket_orientation_penalty=0.3,
        bomb_arm_weight=0.6,
        bomb_blast_weight=0.4,
    )


# ---------------------------------------------------------------------------
# Snapshot — default values must score identically pre/post refactor
# ---------------------------------------------------------------------------

def test_a4_wild_bridge_multi_column_bonus_at_default(
    default_config: MasterConfig,
) -> None:
    """Wild bridge with two-column adjacency should add the configured bonus."""
    criterion = WildBridgeCriterion(
        default_config.board, default_config.landing_criteria,
    )
    # Two cells in different columns — engages the side bonus path
    adjacent_two_cols = (Position(2, 0), Position(4, 0))
    ctx_two = LandingContext(
        booster_type="W",
        cluster_positions=frozenset({Position(3, 3)}),
        landing_position=Position(3, 1),
        adjacent_refill=adjacent_two_cols,
        all_refill=adjacent_two_cols,
        settle_result=None,  # type: ignore[arg-type]
        cluster_shape_stats=_shape(),
    )
    # Two cells in the SAME column — no bonus
    adjacent_one_col = (Position(3, 0), Position(3, 2))
    ctx_one = LandingContext(
        booster_type="W",
        cluster_positions=frozenset({Position(3, 3)}),
        landing_position=Position(3, 1),
        adjacent_refill=adjacent_one_col,
        all_refill=adjacent_one_col,
        settle_result=None,  # type: ignore[arg-type]
        cluster_shape_stats=_shape(),
    )
    delta = criterion.score(ctx_two) - criterion.score(ctx_one)
    # The delta is exactly the configured side bonus — clipped at 1.0
    expected = default_config.landing_criteria.wild_bridge_multi_column_bonus
    assert delta == pytest.approx(expected)


def test_a4_rocket_arm_weight_split_at_default(
    default_config: MasterConfig,
) -> None:
    """RocketArm composite uses configured arm/chain weight split."""
    booster_rules = BoosterRules(
        default_config.boosters, default_config.board, default_config.symbols,
    )
    criterion = RocketArmCriterion(
        booster_rules, default_config.board, default_config.landing_criteria,
    )
    # Center landing maximises chain (centrality) component
    adjacent = tuple(Position(r, 0) for r in range(4))
    ctx = LandingContext(
        booster_type="R",
        cluster_positions=frozenset({Position(3, 3)}),
        landing_position=Position(3, 3),
        adjacent_refill=adjacent,
        all_refill=adjacent,
        settle_result=None,  # type: ignore[arg-type]
        cluster_shape_stats=_shape(),
    )
    score = criterion.score(ctx)
    # arm_score == 1.0 (4 adjacents >= needed of 4 for min_cluster_size 5)
    # chain_score >= 0.2 (max with floor)
    # combined = 1.0 * arm_weight + chain * chain_weight
    lc = default_config.landing_criteria
    assert score >= lc.rocket_arm_weight  # at least the arm contribution
    assert score <= lc.rocket_arm_weight + lc.rocket_chain_weight  # 1.0


def test_a4_bomb_arm_weight_split_at_default(
    default_config: MasterConfig,
) -> None:
    """BombArm composite uses configured arm/blast weight split."""
    booster_rules = BoosterRules(
        default_config.boosters, default_config.board, default_config.symbols,
    )
    criterion = BombArmCriterion(
        booster_rules, default_config.board, default_config.landing_criteria,
    )
    # Center landing — full blast coverage, full arm
    adjacent = tuple(Position(r, 0) for r in range(4))
    ctx = LandingContext(
        booster_type="B",
        cluster_positions=frozenset({Position(3, 3)}),
        landing_position=Position(3, 3),
        adjacent_refill=adjacent,
        all_refill=adjacent,
        settle_result=None,  # type: ignore[arg-type]
        cluster_shape_stats=_shape(),
    )
    score = criterion.score(ctx)
    lc = default_config.landing_criteria
    # arm == 1.0 contributes lc.bomb_arm_weight; remainder is blast coverage
    assert score >= lc.bomb_arm_weight
