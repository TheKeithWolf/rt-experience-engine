"""Tests for A5: BoosterArmStrategy tier classification.

Verifies that `cluster_tier` in the StepIntent is derived from the chosen
arm-cluster symbol via `tier_of(symbol, symbol_config)`, replacing the
previous hardcoded `SymbolTier.ANY` fallback (which masked all tier signal).

Test technique: substitute a `_FixedSymbolClusterBuilder` test-double that
returns a known low-tier or high-tier symbol from `select_symbol`, then
assert the StepIntent reports the corresponding tier. A subclass override
is required because `ClusterBuilder` uses `__slots__` (no monkeypatch).
"""

from __future__ import annotations

import random

import pytest

from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.gravity import GravityDAG
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.symbols import Symbol, SymbolTier, tier_of
from ..step_reasoner.context import BoardContext, DormantBooster
from ..step_reasoner.evaluators import ChainEvaluator, PayoutEstimator, SpawnEvaluator
from ..step_reasoner.progress import ProgressTracker
from ..step_reasoner.services.boundary_analyzer import BoundaryAnalyzer
from ..step_reasoner.services.cluster_builder import ClusterBuilder
from ..step_reasoner.services.forward_simulator import ForwardSimulator
from ..step_reasoner.services.landing_criteria import (
    BombArmCriterion,
    LightballArmCriterion,
    RocketArmCriterion,
    WildBridgeCriterion,
)
from ..step_reasoner.services.landing_evaluator import BoosterLandingEvaluator
from ..step_reasoner.services.seed_planner import SeedPlanner
from ..step_reasoner.strategies.booster_arm import BoosterArmStrategy
from ..archetypes.registry import ArchetypeSignature
from ..variance.hints import VarianceHints


# ---------------------------------------------------------------------------
# Helpers — minimal context builders mirroring test_booster_arm_affinity.py
# ---------------------------------------------------------------------------

def _make_signature() -> ArchetypeSignature:
    """A rocket-family signature that drives BoosterArmStrategy down the
    arm-cluster path. Values chosen so progress.current_step_size_ranges()
    returns a usable Range and required_chain_depth doesn't force chaining.
    """
    return ArchetypeSignature(
        id="test_sig",
        family="rocket",
        criteria="basegame",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 8),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.5, 10.0),
        required_booster_spawns={},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        max_component_size=None,
        triggers_freespin=False,
        reaches_wincap=False,
    )


def _make_variance(config: MasterConfig) -> VarianceHints:
    all_positions = [
        Position(r, c)
        for r in range(config.board.num_reels)
        for c in range(config.board.num_rows)
    ]
    return VarianceHints(
        spatial_bias={p: 1.0 for p in all_positions},
        symbol_weights={s: 1.0 for s in Symbol},
        near_miss_symbol_preference=(),
        cluster_size_preference=tuple(range(5, 16)),
    )


def _make_context(
    config: MasterConfig, booster_pos: Position,
) -> BoardContext:
    """Empty board with one dormant rocket — keeps plan_step deterministic."""
    board = Board.empty(config.board)
    grid_mults = GridMultiplierGrid(config.grid_multiplier, config.board)
    return BoardContext(
        board=board,
        grid_multipliers=grid_mults,
        dormant_boosters=[DormantBooster("R", booster_pos, "H", spawned_step=0)],
        active_wilds=[],
        _board_config=config.board,
    )


# ---------------------------------------------------------------------------
# Strategy + service fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


class _FixedSymbolClusterBuilder(ClusterBuilder):
    """Test-double: returns a pre-set symbol from select_symbol.

    Required because ClusterBuilder uses __slots__ (cannot monkeypatch).
    All other ClusterBuilder methods retain real behavior so the strategy
    exercises real boundary analysis, sizing, and adjacency logic.
    The forced symbol is set on the instance via `force_symbol` after
    construction (a separate __slot__ so it lives alongside the parent state).
    """

    __slots__ = ("force_symbol",)

    def select_symbol(self, *args, **kwargs):  # noqa: D401 - test override
        return self.force_symbol


@pytest.fixture
def services(default_config: MasterConfig):
    """Bundle of shared services needed to construct BoosterArmStrategy.

    Mirrors the production registry wiring at registry.py — using the
    real services keeps the test sensitive to wiring drift.
    """
    spawn_eval = SpawnEvaluator(default_config.boosters)
    payout_eval = PayoutEstimator(
        default_config.paytable, default_config.centipayout,
        default_config.win_levels, default_config.symbols,
        default_config.grid_multiplier,
    )
    boundary = BoundaryAnalyzer(default_config.board, default_config.symbols)
    cluster_builder = _FixedSymbolClusterBuilder(
        spawn_eval, payout_eval, default_config.board, default_config.symbols,
        boundary,
        multi_seed_threshold=default_config.solvers.multi_seed_threshold,
        reasoner_config=default_config.reasoner,
    )
    dag = GravityDAG(default_config.board, default_config.gravity)
    forward = ForwardSimulator(dag, default_config.board, default_config.gravity)
    seed_planner = SeedPlanner(forward, default_config.board, default_config.symbols)
    chain_eval = ChainEvaluator(default_config.boosters)
    booster_rules = BoosterRules(
        default_config.boosters, default_config.board, default_config.symbols,
    )
    lc = default_config.landing_criteria
    landing_criteria = {
        "W": WildBridgeCriterion(default_config.board, lc),
        "R": RocketArmCriterion(booster_rules, default_config.board, lc),
        "B": BombArmCriterion(booster_rules, default_config.board, lc),
        "LB": LightballArmCriterion(default_config.board),
        "SLB": LightballArmCriterion(default_config.board),
    }
    landing_eval = BoosterLandingEvaluator(
        forward, booster_rules, default_config.board, landing_criteria,
    )
    return {
        "cluster_builder": cluster_builder,
        "forward": forward,
        "seed_planner": seed_planner,
        "chain_eval": chain_eval,
        "landing_eval": landing_eval,
    }


# ---------------------------------------------------------------------------
# Tests — A5 acceptance criteria
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("forced_symbol", "expected_tier"),
    [
        # Pick first low_tier and first high_tier symbol so the test moves
        # with config (not a hardcoded ordinal). symbol_from_name keeps
        # this rule-driven, not magic-derived.
        ("low", SymbolTier.LOW),
        ("high", SymbolTier.HIGH),
    ],
)
def test_a5_cluster_tier_follows_selected_symbol(
    forced_symbol: str,
    expected_tier: SymbolTier,
    default_config: MasterConfig,
    services: dict,
    rng: random.Random,
) -> None:
    """A5: StepIntent.expected_cluster_tier == tier_of(selected_symbol).

    Forces a known symbol via _FixedSymbolClusterBuilder and asserts the
    resulting StepIntent reports the matching tier, not the previous
    SymbolTier.ANY masking fallback.
    """
    sym_config = default_config.symbols
    if forced_symbol == "low":
        symbol = Symbol[sym_config.low_tier[0]]
    else:
        symbol = Symbol[sym_config.high_tier[0]]
    # Sanity: the chosen symbol's actual tier must match the expectation —
    # otherwise the test asserts the wrong invariant.
    assert tier_of(symbol, sym_config) is expected_tier

    booster_pos = Position(3, 3)
    context = _make_context(default_config, booster_pos)
    sig = _make_signature()
    progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
    progress.steps_completed = 1
    variance = _make_variance(default_config)

    cluster_builder = services["cluster_builder"]
    cluster_builder.force_symbol = symbol

    strategy = BoosterArmStrategy(
        default_config, services["forward"], cluster_builder,
        services["seed_planner"], services["chain_eval"],
        services["landing_eval"], rng,
    )
    intent = strategy.plan_step(context, progress, sig, variance)

    assert intent.expected_cluster_tier is expected_tier
