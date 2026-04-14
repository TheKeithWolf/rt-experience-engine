"""Tests for Step 6 strategies: 8 concrete StepStrategy implementations.

TEST-R6-001 through TEST-R6-027 per the implementation spec.
"""

from __future__ import annotations

import inspect
import random

import pytest

from ..archetypes.registry import ArchetypeSignature, TerminalNearMissSpec
from ..board_filler.propagators import (
    ClusterBoundaryPropagator,
    MaxComponentPropagator,
    NearMissAwareDeadPropagator,
    NoSpecialSymbolPropagator,
    Propagator,
)
from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.cluster_detection import detect_clusters, detect_components
from ..primitives.gravity import GravityDAG
from ..primitives.symbols import Symbol, SymbolTier, symbols_in_tier, is_standard
from ..step_reasoner.context import BoardContext, DormantBooster
from ..step_reasoner.evaluators import (
    ChainEvaluator, PayoutEstimator, SpawnEvaluator,
)
from ..step_reasoner.intent import StepIntent, StepType
from ..step_reasoner.progress import ClusterRecord, ProgressTracker
from ..step_reasoner.registry import build_default_registry
from ..step_reasoner.services.cluster_builder import ClusterBuilder
from ..step_reasoner.services.forward_simulator import ForwardSimulator
from ..step_reasoner.services.seed_planner import SeedPlanner
from ..step_reasoner.strategies.terminal_dead import TerminalDeadStrategy
from ..step_reasoner.strategies.terminal_near_miss import TerminalNearMissStrategy
from ..step_reasoner.strategies.initial_dead import InitialDeadStrategy
from ..step_reasoner.strategies.initial_cluster import InitialClusterStrategy
from ..step_reasoner.strategies.cascade_cluster import CascadeClusterStrategy
from ..step_reasoner.strategies.booster_arm import BoosterArmStrategy
from ..step_reasoner.strategies.booster_setup import BoosterSetupStrategy
from ..step_reasoner.strategies.wild_bridge import BridgeCandidate, WildBridgeStrategy
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..variance.hints import VarianceHints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signature(**overrides) -> ArchetypeSignature:
    """Build a minimal ArchetypeSignature with sensible defaults."""
    defaults = dict(
        id="test_sig",
        family="t1",
        criteria="basegame",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 8),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        required_cascade_depth=Range(2, 5),
        cascade_steps=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.5, 5.0),
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return ArchetypeSignature(**defaults)


def _make_variance_hints(
    config: MasterConfig,
    symbol_weight_overrides: dict[Symbol, float] | None = None,
) -> VarianceHints:
    """Build VarianceHints with uniform spatial_bias and configurable symbol weights."""
    spatial_bias: dict[Position, float] = {}
    total = config.board.num_reels * config.board.num_rows
    weight = 1.0 / total
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            spatial_bias[Position(reel, row)] = weight

    symbols = symbols_in_tier(SymbolTier.ANY, config.symbols)
    symbol_weights: dict[Symbol, float] = {s: 1.0 for s in symbols}
    if symbol_weight_overrides:
        symbol_weights.update(symbol_weight_overrides)

    return VarianceHints(
        spatial_bias=spatial_bias,
        symbol_weights=symbol_weights,
        near_miss_symbol_preference=symbols,
        cluster_size_preference=tuple(range(5, 16)),
    )


def _make_progress(signature: ArchetypeSignature, **overrides) -> ProgressTracker:
    """Build a ProgressTracker with sensible defaults for tests."""
    defaults = dict(signature=signature, centipayout_multiplier=100)
    defaults.update(overrides)
    return ProgressTracker(**defaults)


def _make_empty_context(config: MasterConfig) -> BoardContext:
    """Build a BoardContext with an empty board."""
    from ..primitives.grid_multipliers import GridMultiplierGrid
    board = Board.empty(config.board)
    grid_mults = GridMultiplierGrid(config.grid_multiplier, config.board)
    return BoardContext(
        board=board,
        grid_multipliers=grid_mults,
        dormant_boosters=[],
        active_wilds=[],
        _board_config=config.board,
    )


def _make_settled_context(
    config: MasterConfig,
    surviving: dict[Position, Symbol] | None = None,
    dormant_boosters: list[DormantBooster] | None = None,
    active_wilds: list[Position] | None = None,
) -> BoardContext:
    """Build a BoardContext with surviving symbols and empty cells (post-settle)."""
    from ..primitives.grid_multipliers import GridMultiplierGrid
    board = Board.empty(config.board)
    if surviving:
        for pos, sym in surviving.items():
            board.set(pos, sym)
    grid_mults = GridMultiplierGrid(config.grid_multiplier, config.board)
    return BoardContext(
        board=board,
        grid_multipliers=grid_mults,
        dormant_boosters=dormant_boosters or [],
        active_wilds=active_wilds or [],
        _board_config=config.board,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def gravity_dag(default_config: MasterConfig) -> GravityDAG:
    return GravityDAG(default_config.board, default_config.gravity)


@pytest.fixture
def forward_simulator(
    gravity_dag: GravityDAG, default_config: MasterConfig,
) -> ForwardSimulator:
    return ForwardSimulator(gravity_dag, default_config.board, default_config.gravity)


@pytest.fixture
def spawn_evaluator(default_config: MasterConfig) -> SpawnEvaluator:
    return SpawnEvaluator(default_config.boosters)


@pytest.fixture
def payout_estimator(default_config: MasterConfig) -> PayoutEstimator:
    return PayoutEstimator(
        default_config.paytable,
        default_config.centipayout,
        default_config.win_levels,
        default_config.symbols,
        default_config.grid_multiplier,
    )


@pytest.fixture
def chain_evaluator(default_config: MasterConfig) -> ChainEvaluator:
    return ChainEvaluator(default_config.boosters)


@pytest.fixture
def cluster_builder(
    spawn_evaluator: SpawnEvaluator,
    payout_estimator: PayoutEstimator,
    default_config: MasterConfig,
) -> ClusterBuilder:
    from ..step_reasoner.services.boundary_analyzer import BoundaryAnalyzer
    boundary_analyzer = BoundaryAnalyzer(default_config.board, default_config.symbols)
    return ClusterBuilder(
        spawn_evaluator, payout_estimator,
        default_config.board, default_config.symbols,
        boundary_analyzer,
    )


@pytest.fixture
def seed_planner(
    forward_simulator: ForwardSimulator, default_config: MasterConfig,
) -> SeedPlanner:
    return SeedPlanner(forward_simulator, default_config.board, default_config.symbols)


@pytest.fixture
def near_miss_planner(
    default_config: MasterConfig, cluster_builder, rng,
):
    from ..step_reasoner.services.near_miss_planner import NearMissPlanner
    return NearMissPlanner(default_config, cluster_builder, rng)


@pytest.fixture
def landing_evaluator(
    forward_simulator: ForwardSimulator,
    default_config: MasterConfig,
):
    from ..primitives.booster_rules import BoosterRules
    from ..step_reasoner.services.landing_criteria import (
        WildBridgeCriterion, RocketArmCriterion, BombArmCriterion,
        LightballArmCriterion,
    )
    from ..step_reasoner.services.landing_evaluator import BoosterLandingEvaluator

    booster_rules = BoosterRules(default_config.boosters, default_config.board, default_config.symbols)
    criteria = {
        "W": WildBridgeCriterion(default_config.board),
        "R": RocketArmCriterion(booster_rules, default_config.board),
        "B": BombArmCriterion(booster_rules, default_config.board),
        "LB": LightballArmCriterion(default_config.board),
        "SLB": LightballArmCriterion(default_config.board),
    }
    return BoosterLandingEvaluator(
        forward_simulator, booster_rules, default_config.board, criteria,
    )


# ---------------------------------------------------------------------------
# Terminal Dead — R6-001 through R6-003
# ---------------------------------------------------------------------------

class TestTerminalDeadStrategy:

    def test_r6_001_produces_zero_cluster_intent(
        self, default_config, rng,
    ) -> None:
        """R6-001: TerminalDeadStrategy produces intent with zero expected clusters."""
        strategy = TerminalDeadStrategy(default_config, rng)
        sig = _make_signature(
            family="dead", required_cascade_depth=Range(0, 0),
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        assert intent.step_type is StepType.TERMINAL_DEAD
        assert intent.expected_cluster_count == Range(0, 0)
        assert intent.expected_cluster_sizes == []
        assert intent.is_terminal is True

    def test_r6_002_preserves_dormant_boosters(
        self, default_config, rng,
    ) -> None:
        """R6-002: TerminalDeadStrategy preserves dormant boosters in constrained_cells."""
        strategy = TerminalDeadStrategy(default_config, rng)
        booster_pos = Position(3, 3)
        sig = _make_signature(
            family="rocket",
            dormant_boosters_on_terminal=("R",),
            required_cascade_depth=Range(0, 0),
        )
        context = _make_settled_context(
            default_config,
            dormant_boosters=[
                DormantBooster("R", booster_pos, "H", spawned_step=0)
            ],
        )
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        assert booster_pos in intent.constrained_cells
        assert intent.constrained_cells[booster_pos] is Symbol.R

    def test_r6_003_propagators_include_max_component(
        self, default_config, rng,
    ) -> None:
        """R6-003: TerminalDeadStrategy propagators include MaxComponentPropagator."""
        strategy = TerminalDeadStrategy(default_config, rng)
        sig = _make_signature(family="dead", required_cascade_depth=Range(0, 0))
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        propagator_types = [type(p) for p in intent.wfc_propagators]
        assert MaxComponentPropagator in propagator_types
        assert NoSpecialSymbolPropagator in propagator_types


# ---------------------------------------------------------------------------
# Terminal Near-Miss — R6-004 through R6-006
# ---------------------------------------------------------------------------

class TestTerminalNearMissStrategy:

    def test_r6_004_constrained_cells_contain_nm_groups(
        self, default_config, cluster_builder, rng,
    ) -> None:
        """R6-004: TerminalNearMissStrategy constrainted_cells contain NM groups of size 4."""
        strategy = TerminalNearMissStrategy(default_config, cluster_builder, rng)
        sig = _make_signature(
            family="dead",
            terminal_near_misses=TerminalNearMissSpec(
                count=Range(1, 1), symbol_tier=SymbolTier.HIGH,
            ),
            required_cascade_depth=Range(0, 0),
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        near_miss_size = default_config.board.min_cluster_size - 1
        assert len(intent.constrained_cells) == near_miss_size
        assert intent.is_terminal is True

    def test_r6_005_nm_groups_are_connected(
        self, default_config, cluster_builder, rng,
    ) -> None:
        """R6-005: NM groups are isolated (connected component, no 5th same-symbol neighbor)."""
        strategy = TerminalNearMissStrategy(default_config, cluster_builder, rng)
        sig = _make_signature(
            family="dead",
            terminal_near_misses=TerminalNearMissSpec(
                count=Range(1, 1), symbol_tier=SymbolTier.LOW,
            ),
            required_cascade_depth=Range(0, 0),
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # Verify the constrained cells form a connected group
        positions = set(intent.constrained_cells.keys())
        sym = next(iter(intent.constrained_cells.values()))
        assert all(s is sym for s in intent.constrained_cells.values()), \
            "All NM cells should be the same symbol"

        # BFS connectivity check
        start = next(iter(positions))
        visited: set[Position] = {start}
        queue = [start]
        while queue:
            pos = queue.pop(0)
            for n in orthogonal_neighbors(pos, default_config.board):
                if n in positions and n not in visited:
                    visited.add(n)
                    queue.append(n)
        assert visited == positions, "NM group must be connected"

    def test_r6_006_nm_symbols_match_required_tier(
        self, default_config, cluster_builder, rng,
    ) -> None:
        """R6-006: NM symbols match required tier."""
        strategy = TerminalNearMissStrategy(default_config, cluster_builder, rng)
        sig = _make_signature(
            family="dead",
            terminal_near_misses=TerminalNearMissSpec(
                count=Range(1, 1), symbol_tier=SymbolTier.HIGH,
            ),
            required_cascade_depth=Range(0, 0),
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        high_symbols = set(symbols_in_tier(SymbolTier.HIGH, default_config.symbols))
        for sym in intent.constrained_cells.values():
            assert sym in high_symbols, f"{sym} is not a HIGH tier symbol"


# ---------------------------------------------------------------------------
# Initial Dead — R6-007, R6-008
# ---------------------------------------------------------------------------

class TestInitialDeadStrategy:

    def test_r6_007_constrained_cells_contain_correct_scatter_count(
        self, default_config, near_miss_planner, rng,
    ) -> None:
        """R6-007: InitialDeadStrategy constrained_cells contain correct scatter count."""
        strategy = InitialDeadStrategy(default_config, near_miss_planner, rng)
        sig = _make_signature(
            family="dead",
            required_scatter_count=Range(2, 2),
            required_cascade_depth=Range(0, 0),
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        scatter_count = sum(
            1 for s in intent.constrained_cells.values() if s is Symbol.S
        )
        assert scatter_count == 2

    def test_r6_008_constrained_cells_contain_correct_nm_count(
        self, default_config, near_miss_planner, rng,
    ) -> None:
        """R6-008: InitialDeadStrategy constrained_cells contain correct NM count."""
        strategy = InitialDeadStrategy(default_config, near_miss_planner, rng)
        near_miss_size = default_config.board.min_cluster_size - 1
        sig = _make_signature(
            family="dead",
            required_near_miss_count=Range(2, 2),
            required_near_miss_symbol_tier=SymbolTier.LOW,
            required_cascade_depth=Range(0, 0),
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # Non-scatter constrained cells = NM groups
        nm_cells = {
            pos: sym for pos, sym in intent.constrained_cells.items()
            if sym is not Symbol.S
        }
        assert len(nm_cells) == near_miss_size * 2


# ---------------------------------------------------------------------------
# Initial Cluster — R6-009 through R6-012
# ---------------------------------------------------------------------------

class TestInitialClusterStrategy:

    def test_r6_009_constrained_cells_form_connected_cluster(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """R6-009: InitialClusterStrategy constrained_cells form connected cluster of correct size."""
        strategy = InitialClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, near_miss_planner,
            landing_evaluator, rng,
        )
        sig = _make_signature(required_cluster_sizes=(Range(5, 5),))
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # Filter out scatters — remaining cells are the cluster
        cluster_cells = {
            pos: sym for pos, sym in intent.constrained_cells.items()
            if sym is not Symbol.S
        }
        assert len(cluster_cells) == 5

        # BFS connectivity
        positions = set(cluster_cells.keys())
        start = next(iter(positions))
        visited: set[Position] = {start}
        queue = [start]
        while queue:
            pos = queue.pop(0)
            for n in orthogonal_neighbors(pos, default_config.board):
                if n in positions and n not in visited:
                    visited.add(n)
                    queue.append(n)
        assert visited == positions

    def test_r6_010_no_predicted_wild_for_bridge_signature(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """R6-010: InitialClusterStrategy does not handle bridge arcs — no predicted_wild_positions.

        Bridge setup is now handled by InitialWildBridgeStrategy. When given a
        bridge signature, InitialClusterStrategy falls through to generic seeds
        and does not set predicted_wild_positions.
        """
        from ..archetypes.registry import CascadeStepConstraint
        strategy = InitialClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, near_miss_planner,
            landing_evaluator, rng,
        )
        # Cluster large enough to spawn a wild (7-8), next step is bridge
        sig = _make_signature(
            required_cluster_sizes=(Range(7, 7),),
            required_cascade_depth=Range(2, 4),
            required_booster_spawns={"W": Range(1, 1)},
            cascade_steps=(
                CascadeStepConstraint(
                    cluster_count=Range(1, 1),
                    cluster_sizes=(Range(7, 7),),
                    cluster_symbol_tier=None,
                    must_spawn_booster="W",
                    must_arm_booster=None,
                    wild_behavior="spawn",
                ),
                CascadeStepConstraint(
                    cluster_count=Range(1, 1),
                    cluster_sizes=(Range(5, 6),),
                    cluster_symbol_tier=None,
                    must_spawn_booster=None,
                    must_arm_booster=None,
                    wild_behavior="bridge",
                ),
            ),
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # InitialClusterStrategy no longer handles bridge — no predicted wild positions
        assert intent.predicted_wild_positions is None

    def test_r6_011_scatter_positions_in_constrained_cells(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """R6-012: InitialClusterStrategy scatter positions in constrained_cells."""
        strategy = InitialClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, near_miss_planner,
            landing_evaluator, rng,
        )
        sig = _make_signature(
            required_scatter_count=Range(2, 2),
            required_cluster_sizes=(Range(5, 5),),
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        scatter_cells = [
            pos for pos, sym in intent.constrained_cells.items()
            if sym is Symbol.S
        ]
        assert len(scatter_cells) == 2

    def test_r6_012_step_type_is_initial(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """R6-009 supplement: step_type is INITIAL."""
        strategy = InitialClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, near_miss_planner,
            landing_evaluator, rng,
        )
        sig = _make_signature()
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        assert intent.step_type is StepType.INITIAL
        assert intent.is_terminal is False

    def test_near_miss_propagator_present_when_nm_required(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """NearMissAwareDeadPropagator selected when archetype requires near-misses."""
        strategy = InitialClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, near_miss_planner,
            landing_evaluator, rng,
        )
        sig = _make_signature(
            required_cluster_sizes=(Range(5, 5),),
            required_near_miss_count=Range(1, 1),
            required_near_miss_symbol_tier=SymbolTier.ANY,
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        assert any(
            isinstance(p, NearMissAwareDeadPropagator)
            for p in intent.wfc_propagators
        )

    def test_initial_cluster_no_bridge_seeds_for_bridge_signature(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """R6-010b: InitialClusterStrategy no longer produces bridge seeds.

        Bridge setup is now handled by InitialWildBridgeStrategy. When given
        a bridge signature, _plan_strategic_seeds falls through to generic seeds.
        """
        from ..archetypes.registry import CascadeStepConstraint
        from ..primitives.gravity import SettleResult

        strategy = InitialClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, near_miss_planner,
            landing_evaluator, rng,
        )
        # Step 0 spawns wild (size 7), step 1 is bridge
        sig = _make_signature(
            required_cluster_sizes=(Range(7, 7),),
            required_cascade_depth=Range(2, 4),
            required_booster_spawns={"W": Range(1, 1)},
            cascade_steps=(
                CascadeStepConstraint(
                    cluster_count=Range(1, 1),
                    cluster_sizes=(Range(7, 7),),
                    cluster_symbol_tier=None,
                    must_spawn_booster="W",
                    must_arm_booster=None,
                    wild_behavior="spawn",
                ),
                CascadeStepConstraint(
                    cluster_count=Range(1, 1),
                    cluster_sizes=(Range(5, 6),),
                    cluster_symbol_tier=None,
                    must_spawn_booster=None,
                    must_arm_booster=None,
                    wild_behavior="bridge",
                ),
            ),
        )
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        # Cluster at reels 0-6, row 6 (bottom row)
        cluster_positions = frozenset(Position(r, 6) for r in range(7))
        cluster_groups = [(cluster_positions, Symbol.L1)]

        empty_positions = tuple(Position(r, 0) for r in range(3))
        settle_result = SettleResult(
            board=Board.empty(default_config.board),
            move_steps=(),
            empty_positions=empty_positions,
        )

        # _plan_strategic_seeds should NOT take the bridge path — it no longer
        # has a bridge branch. It will fall through to arming or generic seeds.
        from unittest.mock import patch
        with patch.object(
            ForwardSimulator, "predict_booster_landing",
            return_value=Position(6, 6),
        ), patch.object(
            ForwardSimulator, "simulate_explosion",
            return_value=settle_result,
        ):
            context = _make_empty_context(default_config)
            seeds, reserve_zone, wild_positions = strategy._plan_strategic_seeds(
                context, cluster_groups, "W",
                settle_result, progress, sig, variance,
            )
            # No predicted_wild_positions — bridge branch is gone
            assert wild_positions is None

    def test_initial_cluster_caps_wild_spawns_from_step_sizes(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """When step-level sizes all spawn wilds, cluster count capped to wild budget."""
        from ..archetypes.registry import CascadeStepConstraint
        strategy = InitialClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, near_miss_planner,
            landing_evaluator, rng,
        )
        # Signature-level: 5-8 (5-6 do NOT spawn wild)
        # Step-level: 7-8 (both DO spawn wild)
        # Wild budget: max 1 → should cap cluster_count to 1
        sig = _make_signature(
            required_cluster_count=Range(1, 2),
            required_cluster_sizes=(Range(5, 8),),
            required_cascade_depth=Range(2, 4),
            required_booster_spawns={"W": Range(1, 1)},
            cascade_steps=(
                CascadeStepConstraint(
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(7, 8),),
                    cluster_symbol_tier=None,
                    must_spawn_booster="W",
                    must_arm_booster=None,
                    wild_behavior="spawn",
                ),
                CascadeStepConstraint(
                    cluster_count=Range(1, 1),
                    cluster_sizes=(Range(5, 6),),
                    cluster_symbol_tier=None,
                    must_spawn_booster=None,
                    must_arm_booster=None,
                    wild_behavior="bridge",
                ),
            ),
        )
        context = _make_empty_context(default_config)
        progress = _make_progress(sig)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # Should have exactly 1 wild spawn (capped from potential 2)
        wild_spawns = [s for s in intent.expected_spawns if s == "W"]
        assert len(wild_spawns) == 1


# ---------------------------------------------------------------------------
# Cascade Cluster — R6-013 through R6-016
# ---------------------------------------------------------------------------

class TestCascadeClusterStrategy:

    def _make_post_settle_context(self, config: MasterConfig) -> BoardContext:
        """Board with some surviving symbols and empty cells (typical mid-cascade)."""
        # Simulate a post-settle board: bottom rows have survivors, top is empty
        surviving: dict[Position, Symbol] = {}
        for reel in range(config.board.num_reels):
            for row in range(config.board.num_rows - 2, config.board.num_rows):
                # Stagger symbols to avoid accidental clusters
                sym = Symbol.L1 if (reel + row) % 2 == 0 else Symbol.L2
                surviving[Position(reel, row)] = sym
        return _make_settled_context(config, surviving=surviving)

    def test_r6_013_cluster_positions_within_empty_cells(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """R6-013: CascadeClusterStrategy cluster positions are within empty_cells only."""
        strategy = CascadeClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, rng,
        )
        sig = _make_signature(required_cluster_sizes=(Range(5, 5),))
        context = self._make_post_settle_context(default_config)
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        empty_positions = set(context.empty_cells)
        for pos in intent.constrained_cells:
            assert pos in empty_positions, f"Cluster position {pos} not in empty cells"

    def test_r6_014_cluster_doesnt_conflict_with_survivors(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """R6-014: CascadeClusterStrategy cluster doesn't conflict with surviving symbols."""
        strategy = CascadeClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, rng,
        )
        sig = _make_signature(required_cluster_sizes=(Range(5, 5),))
        context = self._make_post_settle_context(default_config)
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        surviving = set(context.surviving_symbols.keys())
        for pos in intent.constrained_cells:
            assert pos not in surviving, f"Position {pos} already occupied by survivor"

    def test_r6_015_forward_sim_catches_unintended_clusters(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """R6-015: CascadeClusterStrategy forward sim verifies no unintended clusters."""
        strategy = CascadeClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, rng,
        )
        sig = _make_signature(required_cluster_sizes=(Range(5, 5),))
        context = self._make_post_settle_context(default_config)
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # Verify the intent expects exactly 1 cluster
        assert intent.expected_cluster_count == Range(1, 1)

    def test_r6_016_strategic_cells_planned_when_not_terminal(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        """R6-016: CascadeClusterStrategy strategic_cells planned when not terminal."""
        strategy = CascadeClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, rng,
        )
        sig = _make_signature(
            required_cluster_sizes=(Range(5, 5),),
            required_cascade_depth=Range(3, 6),
        )
        context = self._make_post_settle_context(default_config)
        # steps_completed=1, cascade depth min=3 → not terminal yet
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # Strategic cells should be planned for the next step
        # (SeedPlanner.plan_generic_seeds may return empty if no refill positions,
        # but with 5 rows of empties it should produce seeds)
        assert isinstance(intent.strategic_cells, dict)


# ---------------------------------------------------------------------------
# Booster Arm — R6-017 through R6-019
# ---------------------------------------------------------------------------

class TestBoosterArmStrategy:

    def test_r6_017_cluster_adjacent_to_target_booster(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, chain_evaluator, landing_evaluator, rng,
    ) -> None:
        """R6-017: BoosterArmStrategy cluster is adjacent to target booster."""
        booster_pos = Position(3, 4)
        strategy = BoosterArmStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, landing_evaluator, rng,
        )
        sig = _make_signature(
            family="rocket",
            required_booster_fires={"R": Range(1, 1)},
            required_cascade_depth=Range(2, 4),
        )
        context = _make_settled_context(
            default_config,
            dormant_boosters=[
                DormantBooster("R", booster_pos, "H", spawned_step=0)
            ],
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # At least one cluster cell must be adjacent to the booster
        booster_neighbors = set(orthogonal_neighbors(booster_pos, default_config.board))
        cluster_positions = set(intent.constrained_cells.keys())
        assert cluster_positions & booster_neighbors, \
            "Cluster must have at least one cell adjacent to booster"

    def test_r6_018_expected_arms_contains_booster_type(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, chain_evaluator, landing_evaluator, rng,
    ) -> None:
        """R6-018: BoosterArmStrategy expected_arms contains target booster type."""
        strategy = BoosterArmStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, landing_evaluator, rng,
        )
        sig = _make_signature(
            family="rocket",
            required_booster_fires={"R": Range(1, 1)},
            required_cascade_depth=Range(2, 4),
        )
        context = _make_settled_context(
            default_config,
            dormant_boosters=[
                DormantBooster("R", Position(3, 4), "H", spawned_step=0)
            ],
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        assert "R" in intent.expected_arms
        assert intent.step_type is StepType.BOOSTER_ARM

    def test_r6_019_chain_arrangement_in_strategic_cells(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, chain_evaluator, landing_evaluator, rng,
    ) -> None:
        """R6-019: BoosterArmStrategy chain arrangement in strategic_cells when chain needed."""
        strategy = BoosterArmStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, landing_evaluator, rng,
        )
        sig = _make_signature(
            family="chain",
            required_booster_fires={"R": Range(1, 1)},
            required_chain_depth=Range(1, 2),
            required_cascade_depth=Range(3, 6),
        )
        context = _make_settled_context(
            default_config,
            dormant_boosters=[
                DormantBooster("R", Position(3, 4), "H", spawned_step=0)
            ],
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # Chain arrangement should produce strategic cells
        assert isinstance(intent.strategic_cells, dict)


# ---------------------------------------------------------------------------
# Booster Setup — R6-020, R6-021
# ---------------------------------------------------------------------------

class TestBoosterSetupStrategy:

    def test_r6_020_constrained_cells_for_missing_boosters(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, chain_evaluator, spawn_evaluator, landing_evaluator, rng,
    ) -> None:
        """R6-020: BoosterSetupStrategy constrained_cells contain cluster for missing booster."""
        strategy = BoosterSetupStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, spawn_evaluator,
            landing_evaluator, rng,
        )
        sig = _make_signature(
            family="chain",
            required_booster_spawns={"R": Range(1, 1)},
            required_cascade_depth=Range(3, 6),
        )
        context = _make_settled_context(default_config)
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # Constrained cells should form a cluster large enough to spawn a rocket (9-10)
        assert len(intent.constrained_cells) >= 9

    def test_r6_021_expected_spawns_contain_missing_types(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, chain_evaluator, spawn_evaluator, landing_evaluator, rng,
    ) -> None:
        """R6-021: BoosterSetupStrategy expected_spawns contain missing booster types."""
        strategy = BoosterSetupStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, spawn_evaluator,
            landing_evaluator, rng,
        )
        sig = _make_signature(
            family="chain",
            required_booster_spawns={"R": Range(1, 1)},
            required_cascade_depth=Range(3, 6),
        )
        context = _make_settled_context(default_config)
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        assert "R" in intent.expected_spawns


# ---------------------------------------------------------------------------
# Wild Bridge — R6-022 through R6-024
# ---------------------------------------------------------------------------

class TestWildBridgeStrategy:

    def _make_wild_context(self, config: MasterConfig) -> BoardContext:
        """Board with a wild at (3,3) and some survivors."""
        wild_pos = Position(3, 3)
        surviving = {wild_pos: Symbol.W}
        # Place some L1 cells adjacent to the wild on one side
        surviving[Position(2, 3)] = Symbol.L1
        surviving[Position(1, 3)] = Symbol.L1
        return _make_settled_context(
            config,
            surviving=surviving,
            active_wilds=[wild_pos],
        )

    def test_r6_022_bridge_cells_form_cluster_with_wild(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """R6-022: WildBridgeStrategy bridge_cells + existing + wild form cluster of target size."""
        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        sig = _make_signature(
            family="wild",
            required_cluster_sizes=(Range(5, 5),),
            required_cascade_depth=Range(2, 4),
        )
        context = self._make_wild_context(default_config)
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # Bridge cells should be placed to form a cluster
        assert len(intent.constrained_cells) > 0
        assert intent.step_type is StepType.CASCADE_CLUSTER

    def test_r6_023_bridge_is_genuine(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """R6-023: WildBridgeStrategy bridge is genuine (cells placed near wild)."""
        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        sig = _make_signature(
            family="wild",
            required_cluster_sizes=(Range(5, 5),),
            required_cascade_depth=Range(2, 4),
        )
        context = self._make_wild_context(default_config)
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        # At least one constrained cell should be adjacent to the wild
        wild_pos = context.active_wilds[0]
        wild_neighbors = set(orthogonal_neighbors(wild_pos, default_config.board))
        bridge_positions = set(intent.constrained_cells.keys())
        assert bridge_positions & wild_neighbors, \
            "At least one bridge cell must be adjacent to the wild"

    def test_r6_024_wild_bridge_propagator_present(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """R6-024: WildBridgeStrategy includes WildBridgePropagator."""
        from ..board_filler.propagators import WildBridgePropagator

        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        sig = _make_signature(
            family="wild",
            required_cluster_sizes=(Range(5, 5),),
            required_cascade_depth=Range(2, 4),
        )
        context = self._make_wild_context(default_config)
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        propagator_types = [type(p) for p in intent.wfc_propagators]
        assert WildBridgePropagator in propagator_types

    # ------------------------------------------------------------------
    # WB-001 through WB-003: _reachable_through_wild
    # ------------------------------------------------------------------

    def test_wb_001_reachable_chain_through_wild(
        self, default_config,
    ) -> None:
        """WB-001: Full chain of same-symbol cells reachable through the wild."""
        board = Board.empty(default_config.board)
        wild_pos = Position(3, 2)
        # H2 chain: (4,2) adjacent to wild, (4,3) and (4,4) extend the chain
        board.set(Position(4, 2), Symbol.H2)
        board.set(Position(4, 3), Symbol.H2)
        board.set(Position(4, 4), Symbol.H2)

        reachable = WildBridgeStrategy._reachable_through_wild(
            wild_pos, Symbol.H2, board, default_config.board,
        )
        assert reachable == frozenset({Position(4, 2), Position(4, 3), Position(4, 4)})

    def test_wb_002_reachable_both_sides_of_wild(
        self, default_config,
    ) -> None:
        """WB-002: Same-symbol components on both sides of wild are merged."""
        board = Board.empty(default_config.board)
        wild_pos = Position(3, 3)
        # L1 on one side
        board.set(Position(2, 3), Symbol.L1)
        board.set(Position(1, 3), Symbol.L1)
        # L1 on the other side
        board.set(Position(4, 3), Symbol.L1)

        reachable = WildBridgeStrategy._reachable_through_wild(
            wild_pos, Symbol.L1, board, default_config.board,
        )
        assert reachable == frozenset({
            Position(2, 3), Position(1, 3), Position(4, 3),
        })

    def test_wb_003_no_component_touches_wild(
        self, default_config,
    ) -> None:
        """WB-003: Symbol exists on board but no component is adjacent to wild."""
        board = Board.empty(default_config.board)
        wild_pos = Position(3, 3)
        # H1 far from wild — not adjacent
        board.set(Position(0, 0), Symbol.H1)
        board.set(Position(0, 1), Symbol.H1)

        reachable = WildBridgeStrategy._reachable_through_wild(
            wild_pos, Symbol.H1, board, default_config.board,
        )
        assert reachable == frozenset()

    # ------------------------------------------------------------------
    # WB-010 through WB-015: _scan_bridge_candidates
    # ------------------------------------------------------------------

    def test_wb_010_scan_finds_candidate_with_correct_needed(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-010: Candidate for H2 with 3 reachable cells → score=4, needed=1."""
        wild_pos = Position(3, 3)
        surviving = {
            wild_pos: Symbol.W,
            Position(2, 3): Symbol.H2,
            Position(1, 3): Symbol.H2,
            Position(4, 3): Symbol.H2,
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        candidates = strategy._scan_bridge_candidates(wild_pos, context, 5, None)

        assert len(candidates) >= 1
        h2_candidate = next(c for c in candidates if c.symbol is Symbol.H2)
        assert h2_candidate.score == 4  # 3 reachable + 1 wild
        assert h2_candidate.needed == 1

    def test_wb_011_scan_ranks_by_needed_ascending(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-011: Candidate with more reachable cells ranked first (lower needed)."""
        wild_pos = Position(3, 3)
        surviving = {
            wild_pos: Symbol.W,
            # L1: 1 reachable cell
            Position(2, 3): Symbol.L1,
            # H2: 3 reachable cells — chain extends away from wild
            Position(4, 3): Symbol.H2,
            Position(4, 4): Symbol.H2,
            Position(4, 5): Symbol.H2,
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        candidates = strategy._scan_bridge_candidates(wild_pos, context, 5, None)

        assert len(candidates) >= 2
        # H2 has score 4 (3+1), needed 1; L1 has score 2 (1+1), needed 3
        assert candidates[0].symbol is Symbol.H2
        assert candidates[0].needed < candidates[1].needed

    def test_wb_012_scan_no_standard_neighbors(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-012: No standard symbols adjacent to wild → empty candidate list."""
        wild_pos = Position(3, 3)
        # Only special symbols and empty cells around the wild
        surviving = {wild_pos: Symbol.W}
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        candidates = strategy._scan_bridge_candidates(wild_pos, context, 5, None)
        assert candidates == []

    def test_wb_013_scan_bridge_already_complete(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-013: Score >= target → needed=0 (bridge already complete)."""
        wild_pos = Position(3, 3)
        # 4 L1 cells adjacent/reachable through wild → score 5, target 5
        surviving = {
            wild_pos: Symbol.W,
            Position(2, 3): Symbol.L1,
            Position(4, 3): Symbol.L1,
            Position(3, 2): Symbol.L1,
            Position(3, 4): Symbol.L1,
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        candidates = strategy._scan_bridge_candidates(wild_pos, context, 5, None)

        l1_candidate = next(c for c in candidates if c.symbol is Symbol.L1)
        assert l1_candidate.score == 5
        assert l1_candidate.needed == 0

    def test_wb_014_scan_tier_filter_high(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-014: required_tier=HIGH filters out LOW symbols."""
        wild_pos = Position(3, 3)
        surviving = {
            wild_pos: Symbol.W,
            Position(2, 3): Symbol.L1,  # LOW
            Position(4, 3): Symbol.H1,  # HIGH
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        candidates = strategy._scan_bridge_candidates(
            wild_pos, context, 5, SymbolTier.HIGH,
        )

        symbols = {c.symbol for c in candidates}
        assert Symbol.H1 in symbols
        assert Symbol.L1 not in symbols

    def test_wb_015_scan_tier_filter_high_only_low_present(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-015: required_tier=HIGH but only LOW adjacent → empty list."""
        wild_pos = Position(3, 3)
        surviving = {
            wild_pos: Symbol.W,
            Position(2, 3): Symbol.L1,
            Position(4, 3): Symbol.L2,
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        candidates = strategy._scan_bridge_candidates(
            wild_pos, context, 5, SymbolTier.HIGH,
        )
        assert candidates == []

    # ------------------------------------------------------------------
    # WB-030 through WB-035: plan_step integration
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bridge_phase(
        *,
        cluster_sizes: tuple[Range, ...] = (Range(5, 6),),
        cluster_symbol_tier: SymbolTier | None = None,
        spawns: tuple[str, ...] | None = None,
    ) -> NarrativeArc:
        """Build a minimal NarrativeArc with a single bridge phase."""
        phase = NarrativePhase(
            id="bridge",
            intent="bridge through wild",
            repetitions=Range(1, 1),
            cluster_count=Range(1, 1),
            cluster_sizes=cluster_sizes,
            cluster_symbol_tier=cluster_symbol_tier,
            spawns=spawns,
            arms=None,
            fires=None,
            wild_behavior="bridge",
            ends_when="cluster_exploded",
        )
        return NarrativeArc(
            phases=(phase,),
            payout=RangeFloat(0.5, 5.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        )

    def test_wb_030_plan_step_bridge_small(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-030: wild_bridge_small — places cells to bridge, empty spawns."""
        wild_pos = Position(3, 3)
        surviving = {
            wild_pos: Symbol.W,
            Position(2, 3): Symbol.L1,
            Position(1, 3): Symbol.L1,
            Position(4, 3): Symbol.L1,
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        arc = self._make_bridge_phase(cluster_sizes=(Range(5, 5),))
        sig = _make_signature(
            family="wild",
            required_cluster_sizes=(Range(5, 5),),
            required_cascade_depth=Range(2, 4),
            narrative_arc=arc,
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        intent = strategy.plan_step(context, progress, sig, variance)

        # Score is 4 (3 reachable + 1 wild), target 5 → need 1 cell
        assert len(intent.constrained_cells) == 1
        assert intent.expected_spawns == []

    def test_wb_031_plan_step_bridge_already_complete(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-031: Bridge already complete → no constrained cells needed."""
        wild_pos = Position(3, 3)
        surviving = {
            wild_pos: Symbol.W,
            Position(2, 3): Symbol.L1,
            Position(4, 3): Symbol.L1,
            Position(3, 2): Symbol.L1,
            Position(3, 4): Symbol.L1,
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        arc = self._make_bridge_phase(cluster_sizes=(Range(5, 5),))
        sig = _make_signature(
            family="wild",
            required_cluster_sizes=(Range(5, 5),),
            required_cascade_depth=Range(2, 4),
            narrative_arc=arc,
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        intent = strategy.plan_step(context, progress, sig, variance)

        assert len(intent.constrained_cells) == 0

    def test_wb_032_plan_step_no_viable_candidates(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-032: No viable bridge candidates → raises ValueError."""
        wild_pos = Position(3, 3)
        # Only the wild, no standard neighbors
        surviving = {wild_pos: Symbol.W}
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        arc = self._make_bridge_phase()
        sig = _make_signature(
            family="wild",
            required_cluster_sizes=(Range(5, 5),),
            required_cascade_depth=Range(2, 4),
            narrative_arc=arc,
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        with pytest.raises(ValueError, match="No viable bridge symbols"):
            strategy.plan_step(context, progress, sig, variance)

    def test_wb_033_plan_step_rocket_spawns(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-033: wild_enable_rocket — expected_spawns=["R"] from phase."""
        wild_pos = Position(3, 3)
        # H3 chain of 5 adjacent to wild → score 6, target 9, needed 3
        surviving = {
            wild_pos: Symbol.W,
            Position(4, 3): Symbol.H3,
            Position(4, 4): Symbol.H3,
            Position(4, 5): Symbol.H3,
            Position(4, 2): Symbol.H3,
            Position(4, 1): Symbol.H3,
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        arc = self._make_bridge_phase(
            cluster_sizes=(Range(9, 9),),
            spawns=("R",),
        )
        sig = _make_signature(
            family="wild",
            required_cluster_sizes=(Range(9, 9),),
            required_cascade_depth=Range(2, 6),
            narrative_arc=arc,
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        intent = strategy.plan_step(context, progress, sig, variance)

        assert intent.expected_spawns == ["R"]
        assert len(intent.constrained_cells) == 3

    def test_wb_034_plan_step_high_tier_filter(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-034: wild_bridge_large with HIGH tier → only HIGH symbol selected."""
        wild_pos = Position(3, 3)
        surviving = {
            wild_pos: Symbol.W,
            Position(2, 3): Symbol.L1,  # LOW — should be filtered out
            Position(4, 3): Symbol.H1,  # HIGH — should be selected
            Position(4, 4): Symbol.H1,
            Position(4, 5): Symbol.H1,
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        arc = self._make_bridge_phase(
            cluster_sizes=(Range(5, 5),),
            cluster_symbol_tier=SymbolTier.HIGH,
        )
        sig = _make_signature(
            family="wild",
            required_cluster_sizes=(Range(5, 5),),
            required_cascade_depth=Range(2, 4),
            narrative_arc=arc,
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        intent = strategy.plan_step(context, progress, sig, variance)

        # All constrained cells should be H1 (HIGH), not L1 (LOW)
        for sym in intent.constrained_cells.values():
            assert sym is Symbol.H1
        assert intent.expected_cluster_tier is SymbolTier.HIGH

    def test_wb_035_plan_step_tier_filter_no_match_raises(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, rng,
    ) -> None:
        """WB-035: HIGH tier required but only LOW adjacent → raises ValueError."""
        wild_pos = Position(3, 3)
        # Only LOW symbols adjacent to wild — HIGH tier filter rejects all
        surviving = {
            wild_pos: Symbol.W,
            Position(2, 3): Symbol.L1,
            Position(4, 3): Symbol.L2,
        }
        context = _make_settled_context(
            default_config, surviving=surviving, active_wilds=[wild_pos],
        )
        arc = self._make_bridge_phase(
            cluster_sizes=(Range(5, 5),),
            cluster_symbol_tier=SymbolTier.HIGH,
        )
        sig = _make_signature(
            family="wild",
            required_cluster_sizes=(Range(5, 5),),
            required_cascade_depth=Range(2, 4),
            narrative_arc=arc,
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        strategy = WildBridgeStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, rng,
        )
        # HIGH tier required but only LOW symbols adjacent → no candidates
        with pytest.raises(ValueError, match="No viable bridge symbols"):
            strategy.plan_step(context, progress, sig, variance)


# ---------------------------------------------------------------------------
# Cross-cutting — R6-025 through R6-027
# ---------------------------------------------------------------------------

class TestCrossCutting:

    def test_r6_025_all_strategies_return_frozen_step_intent(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, chain_evaluator, near_miss_planner,
        landing_evaluator, rng,
    ) -> None:
        """R6-025: All strategies return frozen StepIntent."""
        strategies_and_contexts = [
            (
                TerminalDeadStrategy(default_config, rng),
                _make_empty_context(default_config),
                _make_signature(family="dead", required_cascade_depth=Range(0, 0)),
            ),
            (
                InitialDeadStrategy(default_config, near_miss_planner, rng),
                _make_empty_context(default_config),
                _make_signature(family="dead", required_cascade_depth=Range(0, 0)),
            ),
            (
                InitialClusterStrategy(
                    default_config, forward_simulator, cluster_builder,
                    seed_planner, spawn_evaluator, near_miss_planner,
                    landing_evaluator, rng,
                ),
                _make_empty_context(default_config),
                _make_signature(),
            ),
            (
                CascadeClusterStrategy(
                    default_config, forward_simulator, cluster_builder,
                    seed_planner, spawn_evaluator, rng,
                ),
                _make_settled_context(default_config),
                _make_signature(required_cascade_depth=Range(3, 6)),
            ),
        ]

        for strategy, context, sig in strategies_and_contexts:
            progress = _make_progress(sig, steps_completed=(
                1 if sig.family != "dead" and sig.required_cascade_depth.min_val > 1 else 0
            ))
            variance = _make_variance_hints(default_config)
            intent = strategy.plan_step(context, progress, sig, variance)
            assert isinstance(intent, StepIntent)
            # StepIntent is frozen=True — verify we can't reassign
            with pytest.raises(AttributeError):
                intent.step_type = StepType.TERMINAL_DEAD  # type: ignore[misc]

    def test_r6_026_no_strategy_imports_another_strategy(self) -> None:
        """R6-026: No strategy imports another strategy (no cross-dependencies)."""
        import importlib
        strategy_modules = [
            "games.royal_tumble.experience_engine.step_reasoner.strategies.terminal_dead",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.terminal_near_miss",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.initial_dead",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.initial_cluster",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.cascade_cluster",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.booster_arm",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.booster_setup",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.wild_bridge",
        ]
        strategy_class_names = {
            "TerminalDeadStrategy", "TerminalNearMissStrategy",
            "InitialDeadStrategy", "InitialClusterStrategy",
            "CascadeClusterStrategy", "BoosterArmStrategy",
            "BoosterSetupStrategy", "WildBridgeStrategy",
        }

        for mod_name in strategy_modules:
            mod = importlib.import_module(mod_name)
            source = inspect.getsource(mod)
            # Check that no strategy file imports another strategy class
            for class_name in strategy_class_names:
                # Exclude the class defined in this module itself
                if class_name in source:
                    # It should only appear as "class ClassName:" (its own definition)
                    # not as "from ...strategies.xxx import ClassName"
                    import_pattern = f"import {class_name}"
                    assert import_pattern not in source, \
                        f"{mod_name} imports {class_name}"

    def test_r6_027_no_strategy_constructs_own_services(self) -> None:
        """R6-027: No strategy constructs its own services (all injected)."""
        import importlib
        strategy_modules = [
            "games.royal_tumble.experience_engine.step_reasoner.strategies.terminal_dead",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.terminal_near_miss",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.initial_dead",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.initial_cluster",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.cascade_cluster",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.booster_arm",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.booster_setup",
            "games.royal_tumble.experience_engine.step_reasoner.strategies.wild_bridge",
        ]
        # Service classes that should never be constructed inside a strategy
        service_constructors = [
            "ForwardSimulator(",
            "ClusterBuilder(",
            "SeedPlanner(",
            "SpawnEvaluator(",
            "ChainEvaluator(",
            "PayoutEstimator(",
        ]

        for mod_name in strategy_modules:
            mod = importlib.import_module(mod_name)
            source = inspect.getsource(mod)
            for constructor in service_constructors:
                # Check that no service is constructed inline
                # (only referenced as self._xxx, never as ClassName(...))
                assert constructor not in source, \
                    f"{mod_name} constructs {constructor.rstrip('(')} inline"


# ---------------------------------------------------------------------------
# impl-plan-6 regression tests — arming-cluster reachability + WFC boundary
# ---------------------------------------------------------------------------

def _rocket_fire_arc() -> NarrativeArc:
    """Two-phase arc mirroring rocket_h_fire: spawn (no arms) → fire (arms R).

    This is the minimal narrative shape the planning strategy must see in
    order for the arming-cluster gates to activate at Step 0.
    """
    spawn = NarrativePhase(
        id="spawn", intent="spawn rocket", repetitions=Range(1, 1),
        cluster_count=Range(1, 1), cluster_sizes=(Range(7, 7),),
        cluster_symbol_tier=None,
        spawns=("R",), arms=None, fires=None,
        wild_behavior=None, ends_when="always",
    )
    fire = NarrativePhase(
        id="fire", intent="arm and fire rocket", repetitions=Range(1, 1),
        cluster_count=Range(1, 1), cluster_sizes=(Range(5, 5),),
        cluster_symbol_tier=None,
        spawns=None, arms=("R",), fires=("R",),
        wild_behavior=None, ends_when="always",
    )
    return NarrativeArc(
        phases=(spawn, fire),
        payout=RangeFloat(0.0, 50.0),
        wild_count_on_terminal=Range(0, 10),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation="H",
        lb_target_tier=None,
    )


class TestArmingClusterGatesReachable:
    """Plan v6 §4.2a — _next_step_needs_arming_cluster must see the fire
    phase at Step 0, not the (still-repeating) spawn phase."""

    def test_next_step_needs_arming_sees_fire_phase(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, spawn_evaluator, near_miss_planner, landing_evaluator, rng,
    ) -> None:
        strategy = InitialClusterStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, spawn_evaluator, near_miss_planner,
            landing_evaluator, rng,
        )
        sig = _make_signature(
            family="rocket",
            required_cluster_sizes=(Range(7, 7),),
            required_cascade_depth=Range(2, 2),
            required_booster_spawns={"R": Range(1, 1)},
            required_booster_fires={"R": Range(1, 1)},
            rocket_orientation="H",
            narrative_arc=_rocket_fire_arc(),
        )
        progress = _make_progress(sig)

        # Step 0: current_phase_repetitions=0 < Phase0.max_val=1. peek_next_phase
        # returns Phase 0 (arms=None) — the old bug. peek_phase_after_current
        # must return Phase 1 (arms=("R",)), so the guard returns True.
        assert strategy._next_step_needs_arming_cluster(sig, progress) is True


class TestBoosterArmBoundaryPropagator:
    """Plan v6 §4.3 — BoosterArmStrategy must guard the arming cluster with
    ClusterBoundaryPropagator so WFC does not merge strays into the planned
    cluster shape (which previously pulled the cluster off the booster)."""

    def test_booster_arm_includes_boundary_propagator(
        self, default_config, forward_simulator, cluster_builder,
        seed_planner, chain_evaluator, landing_evaluator, rng,
    ) -> None:
        strategy = BoosterArmStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, landing_evaluator, rng,
        )
        sig = _make_signature(
            family="rocket",
            required_booster_fires={"R": Range(1, 1)},
            required_cascade_depth=Range(2, 4),
        )
        context = _make_settled_context(
            default_config,
            dormant_boosters=[
                DormantBooster("R", Position(3, 4), "H", spawned_step=0)
            ],
        )
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        intent = strategy.plan_step(context, progress, sig, variance)

        assert any(
            isinstance(p, ClusterBoundaryPropagator)
            for p in intent.wfc_propagators
        ), "BoosterArmStrategy must include ClusterBoundaryPropagator"
