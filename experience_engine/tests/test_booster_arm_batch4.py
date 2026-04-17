"""Tests for Batch 4 — A1 (early landing-score reject) and A6 (always plan
arm seeds) in BoosterArmStrategy.

A1 verifies the probe rejects geometrically impossible booster targets
BEFORE find_positions is called. We use a TrackingClusterBuilder subclass
to assert find_positions is not invoked on the early-reject path.

A6 verifies non-chain arms (rocket family) populate StepIntent.strategic_cells
when refill space is available — previously they returned an empty dict
because plan_arm_seeds was gated on the chain branch.
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
from ..primitives.symbols import Symbol, SymbolTier
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


class _TrackingClusterBuilder(ClusterBuilder):
    """Counts find_positions invocations so the test can assert the early
    reject path skipped the expensive search.
    """

    __slots__ = ("find_positions_calls",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.find_positions_calls = 0

    def find_positions(self, *args, **kwargs):  # noqa: D401
        self.find_positions_calls += 1
        return super().find_positions(*args, **kwargs)


def _make_signature() -> ArchetypeSignature:
    """Rocket signature — exercises the non-chain arm path."""
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


def _empty_context(
    config: MasterConfig, booster_pos: Position,
) -> BoardContext:
    board = Board.empty(config.board)
    return BoardContext(
        board=board,
        grid_multipliers=GridMultiplierGrid(config.grid_multiplier, config.board),
        dormant_boosters=[DormantBooster("R", booster_pos, "H", spawned_step=0)],
        active_wilds=[],
        _board_config=config.board,
    )


def _surrounded_context(
    config: MasterConfig, booster_pos: Position,
) -> BoardContext:
    """Booster fully surrounded by survivors — zero adjacent refill cells.

    The probe must reject this geometry without calling find_positions.
    """
    board = Board.empty(config.board)
    # Fill every cell with a non-booster, non-empty symbol so the booster
    # has no adjacent empties — the probe's first ring is empty.
    placeholder = Symbol[config.symbols.standard[0]]
    for r in range(config.board.num_reels):
        for c in range(config.board.num_rows):
            pos = Position(r, c)
            if pos != booster_pos:
                board.set(pos, placeholder)
    return BoardContext(
        board=board,
        grid_multipliers=GridMultiplierGrid(config.grid_multiplier, config.board),
        dormant_boosters=[DormantBooster("R", booster_pos, "H", spawned_step=0)],
        active_wilds=[],
        _board_config=config.board,
    )


@pytest.fixture
def services(default_config: MasterConfig):
    spawn_eval = SpawnEvaluator(default_config.boosters)
    payout_eval = PayoutEstimator(
        default_config.paytable, default_config.centipayout,
        default_config.win_levels, default_config.symbols,
        default_config.grid_multiplier,
    )
    boundary = BoundaryAnalyzer(default_config.board, default_config.symbols)
    cluster_builder = _TrackingClusterBuilder(
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
# A1 — early reject path
# ---------------------------------------------------------------------------

def test_a1_probe_rejects_surrounded_booster_before_find_positions(
    default_config: MasterConfig, services: dict,
) -> None:
    """A1: when the booster has no adjacent empties, the probe must raise
    BEFORE the cluster_builder's expensive find_positions runs.
    """
    booster_pos = Position(3, 3)
    context = _surrounded_context(default_config, booster_pos)
    sig = _make_signature()
    progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
    progress.steps_completed = 1
    variance = _make_variance(default_config)
    rng = random.Random(42)
    cluster_builder = services["cluster_builder"]

    strategy = BoosterArmStrategy(
        default_config, services["forward"], cluster_builder,
        services["seed_planner"], services["chain_eval"],
        services["landing_eval"], rng,
    )
    with pytest.raises(ValueError, match="probe"):
        strategy.plan_step(context, progress, sig, variance)

    # The critical assertion: find_positions was never invoked
    assert cluster_builder.find_positions_calls == 0


# ---------------------------------------------------------------------------
# A6 — always plan arm seeds
# ---------------------------------------------------------------------------

def test_a6_non_chain_rocket_arm_populates_strategic_cells(
    default_config: MasterConfig, services: dict,
) -> None:
    """A6: a non-chain rocket arm step must populate strategic_cells when
    refill space is available. Previously the seed planning was gated on
    the _needs_chain branch, so non-chain arcs returned strategic_cells={}
    and the WFC refill ran without any guidance.
    """
    booster_pos = Position(3, 3)
    context = _empty_context(default_config, booster_pos)
    sig = _make_signature()
    # required_chain_depth.min_val == 0 → _needs_chain returns False
    assert sig.required_chain_depth.min_val == 0
    progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
    progress.steps_completed = 1
    variance = _make_variance(default_config)
    rng = random.Random(42)

    strategy = BoosterArmStrategy(
        default_config, services["forward"], services["cluster_builder"],
        services["seed_planner"], services["chain_eval"],
        services["landing_eval"], rng,
    )
    intent = strategy.plan_step(context, progress, sig, variance)

    # The empty board has plenty of refill — strategic_cells should not
    # be empty on the non-chain path after A6.
    assert intent.strategic_cells, (
        "Non-chain rocket arm should populate strategic_cells when refill "
        "space is available — A6 lifted plan_arm_seeds out of the chain gate"
    )
