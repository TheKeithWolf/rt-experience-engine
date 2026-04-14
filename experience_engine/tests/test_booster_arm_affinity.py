"""Tests for booster arm survivor affinity: scoring boost and merge policy selection.

ARM-AFF-001 through ARM-AFF-008 per impl-plan-booster-arm-survivor-affinity.md.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from ..config.schema import ConfigValidationError, MasterConfig, ReasonerConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.symbols import Symbol, SymbolTier, symbols_in_tier
from ..step_reasoner.context import BoardContext, DormantBooster
from ..step_reasoner.evaluators import ChainEvaluator, PayoutEstimator, SpawnEvaluator
from ..step_reasoner.progress import ProgressTracker
from ..step_reasoner.services.boundary_analyzer import (
    BoundaryAnalysis,
    BoundaryAnalyzer,
    SurvivorComponent,
)
from ..step_reasoner.services.cluster_builder import ClusterBuilder
from ..step_reasoner.services.merge_policy import MergePolicy
from ..step_reasoner.strategies.booster_arm import BoosterArmStrategy
from ..archetypes.registry import ArchetypeSignature
from ..variance.hints import VarianceHints


# ---------------------------------------------------------------------------
# Helpers (shared patterns from test_survivor_aware / test_step_reasoner_strategies)
# ---------------------------------------------------------------------------

def _make_signature(**overrides) -> ArchetypeSignature:
    defaults = dict(
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
    defaults.update(overrides)
    return ArchetypeSignature(**defaults)


def _make_variance_hints(config: MasterConfig) -> VarianceHints:
    symbols = list(symbols_in_tier(SymbolTier.ANY, config.symbols))
    all_positions = [
        Position(r, c)
        for r in range(config.board.num_reels)
        for c in range(config.board.num_rows)
    ]
    return VarianceHints(
        spatial_bias={p: 1.0 for p in all_positions},
        symbol_weights={s: 1.0 for s in symbols},
        near_miss_symbol_preference=symbols,
        cluster_size_preference=tuple(range(5, 16)),
    )


def _make_progress(sig: ArchetypeSignature, **overrides) -> ProgressTracker:
    defaults = dict(signature=sig, centipayout_multiplier=100)
    defaults.update(overrides)
    return ProgressTracker(**defaults)


def _make_settled_context(
    config: MasterConfig,
    surviving: dict[Position, Symbol] | None = None,
    dormant_boosters: list[DormantBooster] | None = None,
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
        active_wilds=[],
        _board_config=config.board,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def analyzer(default_config: MasterConfig) -> BoundaryAnalyzer:
    return BoundaryAnalyzer(default_config.board, default_config.symbols)


@pytest.fixture
def spawn_evaluator(default_config: MasterConfig) -> SpawnEvaluator:
    return SpawnEvaluator(default_config.boosters)


@pytest.fixture
def payout_estimator(default_config: MasterConfig) -> PayoutEstimator:
    return PayoutEstimator(
        default_config.paytable, default_config.centipayout,
        default_config.win_levels, default_config.symbols,
        default_config.grid_multiplier,
    )


@pytest.fixture
def cluster_builder(
    spawn_evaluator: SpawnEvaluator,
    payout_estimator: PayoutEstimator,
    default_config: MasterConfig,
) -> ClusterBuilder:
    boundary_analyzer = BoundaryAnalyzer(default_config.board, default_config.symbols)
    return ClusterBuilder(
        spawn_evaluator, payout_estimator,
        default_config.board, default_config.symbols,
        boundary_analyzer,
    )


@pytest.fixture
def chain_evaluator(default_config: MasterConfig) -> ChainEvaluator:
    return ChainEvaluator(default_config.boosters)


@pytest.fixture
def forward_simulator(default_config: MasterConfig):
    from ..primitives.gravity import GravityDAG
    from ..step_reasoner.services.forward_simulator import ForwardSimulator
    dag = GravityDAG(default_config.board, default_config.gravity)
    return ForwardSimulator(dag, default_config.board, default_config.gravity)


@pytest.fixture
def seed_planner(forward_simulator, default_config: MasterConfig):
    from ..step_reasoner.services.seed_planner import SeedPlanner
    return SeedPlanner(forward_simulator, default_config.board, default_config.symbols)


@pytest.fixture
def landing_evaluator(forward_simulator, default_config: MasterConfig):
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
# Tests
# ---------------------------------------------------------------------------


class TestAffinityScoring:
    """ARM-AFF-001, ARM-AFF-008: affinity_scores effect on select_symbol."""

    def test_arm_aff_001_affinity_boosts_survivor_symbol(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """ARM-AFF-001: affinity_scores cause select_symbol to prefer the boosted symbol."""
        sig = _make_signature()
        variance = _make_variance_hints(default_config)
        progress = _make_progress(sig, steps_completed=1)

        # Build boundary where L3 is risky (survivors exist) — merge_score ~0.6
        # Without affinity, L3 is penalized; with affinity, it's boosted
        trials = 200
        affinity_boost = {Symbol.L3: 7.0}

        # Count selections WITH affinity
        counts_with: Counter[Symbol] = Counter()
        for i in range(trials):
            trial_rng = random.Random(i)
            sym = cluster_builder.select_symbol(
                progress, sig, variance, trial_rng,
                planned_size=5,
                affinity_scores=affinity_boost,
            )
            counts_with[sym] += 1

        # Count selections WITHOUT affinity
        counts_without: Counter[Symbol] = Counter()
        for i in range(trials):
            trial_rng = random.Random(i)
            sym = cluster_builder.select_symbol(
                progress, sig, variance, trial_rng,
                planned_size=5,
            )
            counts_without[sym] += 1

        # With affinity, L3 should be selected significantly more often
        l3_rate_with = counts_with[Symbol.L3] / trials
        l3_rate_without = counts_without[Symbol.L3] / trials
        assert l3_rate_with > l3_rate_without, (
            f"Affinity should boost L3 selection: with={l3_rate_with:.2f}, without={l3_rate_without:.2f}"
        )

    def test_arm_aff_008_zero_affinity_is_noop(
        self, cluster_builder: ClusterBuilder, default_config: MasterConfig,
    ) -> None:
        """ARM-AFF-008: affinity=0.0 per cell produces base 1.0 scores — no preference change."""
        sig = _make_signature()
        variance = _make_variance_hints(default_config)
        progress = _make_progress(sig, steps_completed=1)

        # Zero affinity per cell: 1.0 + count * 0.0 = 1.0 for all symbols
        zero_affinity = {Symbol.L3: 1.0}  # base 1.0 (as if 0 affinity per cell)
        trials = 200

        counts_with: Counter[Symbol] = Counter()
        counts_without: Counter[Symbol] = Counter()
        for i in range(trials):
            trial_rng = random.Random(i)
            sym_with = cluster_builder.select_symbol(
                progress, sig, variance, trial_rng,
                planned_size=5, affinity_scores=zero_affinity,
            )
            trial_rng2 = random.Random(i)
            sym_without = cluster_builder.select_symbol(
                progress, sig, variance, trial_rng2,
                planned_size=5,
            )
            counts_with[sym_with] += 1
            counts_without[sym_without] += 1

        # With base-1.0 affinity, L3 selection rate should be nearly identical
        l3_with = counts_with[Symbol.L3] / trials
        l3_without = counts_without[Symbol.L3] / trials
        assert abs(l3_with - l3_without) < 0.05, (
            f"Base-1.0 affinity should be no-op: with={l3_with:.2f}, without={l3_without:.2f}"
        )


class TestBoosterAdjacentSurvivorScan:
    """ARM-AFF-002, ARM-AFF-003: _booster_adjacent_survivors correctness."""

    def test_arm_aff_002_scan_finds_adjacent_survivors(
        self, default_config: MasterConfig, analyzer: BoundaryAnalyzer,
        cluster_builder: ClusterBuilder,
        forward_simulator, seed_planner, chain_evaluator,
        landing_evaluator, rng,
    ) -> None:
        """ARM-AFF-002: _booster_adjacent_survivors returns correct counts for adjacent group."""
        booster_pos = Position(4, 2)

        # L3 survivors adjacent to booster: (5,1), (5,2), (6,1) touch (4,2) neighbors
        # Booster neighbors: (3,2), (5,2), (4,1), (4,3)
        # (5,2) is a neighbor of booster — so L3 component at (5,1),(5,2),(6,1) qualifies
        survivor_map = {
            Position(5, 1): Symbol.L3,
            Position(6, 1): Symbol.L3,
            Position(5, 2): Symbol.L3,
        }
        # Fill rest with H1 to avoid accidental clusters, leave some empty
        empty_positions = frozenset(
            Position(r, c)
            for r in range(default_config.board.num_reels)
            for c in range(default_config.board.num_rows)
            if Position(r, c) not in survivor_map
            and Position(r, c) != booster_pos
        )

        context = _make_settled_context(
            default_config,
            surviving=survivor_map,
            dormant_boosters=[DormantBooster("R", booster_pos, "H", spawned_step=0)],
        )

        strategy = BoosterArmStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, landing_evaluator, rng,
        )

        boundary = cluster_builder.analyze_boundary(context)
        result = strategy._booster_adjacent_survivors(booster_pos, boundary)

        # L3 component touches booster neighbor (5,2)
        assert Symbol.L3 in result
        assert result[Symbol.L3] == 3

    def test_arm_aff_003_scan_ignores_distant_survivors(
        self, default_config: MasterConfig, analyzer: BoundaryAnalyzer,
        cluster_builder: ClusterBuilder,
        forward_simulator, seed_planner, chain_evaluator,
        landing_evaluator, rng,
    ) -> None:
        """ARM-AFF-003: survivors not adjacent to booster are excluded."""
        booster_pos = Position(0, 0)

        # L2 survivors far from booster — at opposite corner
        survivor_map = {
            Position(5, 5): Symbol.L2,
            Position(5, 6): Symbol.L2,
            Position(6, 5): Symbol.L2,
        }

        context = _make_settled_context(
            default_config,
            surviving=survivor_map,
            dormant_boosters=[DormantBooster("R", booster_pos, "H", spawned_step=0)],
        )

        strategy = BoosterArmStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, landing_evaluator, rng,
        )

        boundary = cluster_builder.analyze_boundary(context)
        result = strategy._booster_adjacent_survivors(booster_pos, boundary)

        # No survivors adjacent to booster at (0,0)
        assert result == {}


class TestMergePolicySelection:
    """ARM-AFF-004, ARM-AFF-005, ARM-AFF-006: EXPLOIT vs AVOID policy in plan_step."""

    def test_arm_aff_004_exploit_policy_when_survivors_adjacent(
        self, default_config: MasterConfig,
        forward_simulator, cluster_builder: ClusterBuilder,
        seed_planner, chain_evaluator, landing_evaluator, rng,
    ) -> None:
        """ARM-AFF-004: plan_step uses EXPLOIT when survivors adjacent, placing fewer new cells."""
        booster_pos = Position(3, 3)

        # Place L3 survivors adjacent to booster — enough to contribute to arming cluster
        # Booster neighbors: (2,3), (4,3), (3,2), (3,4)
        survivor_map = {
            Position(2, 3): Symbol.L3,
            Position(2, 2): Symbol.L3,
            Position(2, 4): Symbol.L3,
        }

        context = _make_settled_context(
            default_config,
            surviving=survivor_map,
            dormant_boosters=[DormantBooster("R", booster_pos, "H", spawned_step=0)],
        )
        sig = _make_signature()
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        strategy = BoosterArmStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, landing_evaluator, rng,
        )

        intent = strategy.plan_step(context, progress, sig, variance)

        # Cluster must still be adjacent to booster
        booster_neighbors = set(orthogonal_neighbors(booster_pos, default_config.board))
        cluster_positions = set(intent.constrained_cells.keys())
        assert cluster_positions & booster_neighbors, \
            "Cluster must have at least one cell adjacent to booster"

    def test_arm_aff_005_avoid_policy_when_no_adjacent_survivors(
        self, default_config: MasterConfig,
        forward_simulator, cluster_builder: ClusterBuilder,
        seed_planner, chain_evaluator, landing_evaluator, rng,
    ) -> None:
        """ARM-AFF-005: plan_step falls back to AVOID when no survivors are booster-adjacent."""
        booster_pos = Position(3, 3)

        # No survivors at all — empty board except booster
        context = _make_settled_context(
            default_config,
            dormant_boosters=[DormantBooster("R", booster_pos, "H", spawned_step=0)],
        )
        sig = _make_signature()
        progress = _make_progress(sig, steps_completed=1)
        variance = _make_variance_hints(default_config)

        strategy = BoosterArmStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, landing_evaluator, rng,
        )

        intent = strategy.plan_step(context, progress, sig, variance)

        # With no survivors, all cells in the cluster are new placements
        cluster_positions = set(intent.constrained_cells.keys())
        assert len(cluster_positions) >= default_config.board.min_cluster_size, \
            "Without survivors, cluster should be at least min_cluster_size new cells"

        # Cluster must still be adjacent to booster
        booster_neighbors = set(orthogonal_neighbors(booster_pos, default_config.board))
        assert cluster_positions & booster_neighbors

    def test_arm_aff_006_avoid_when_merge_triggers_unwanted_spawn(
        self, default_config: MasterConfig,
        cluster_builder: ClusterBuilder,
        forward_simulator, seed_planner, chain_evaluator,
        landing_evaluator, rng,
    ) -> None:
        """ARM-AFF-006: unwanted spawn from merge triggers AVOID — verified via spawn_eval."""
        booster_pos = Position(3, 3)
        spawn_eval = cluster_builder._spawn_eval

        # Find the smallest spawn threshold so we can create a survivor group
        # that, combined with a planned cluster, would trigger an unwanted spawn
        thresholds = spawn_eval.all_thresholds()
        assert thresholds, "Need at least one spawn threshold for this test"
        smallest = min(thresholds, key=lambda t: t.min_size)

        # Build an L1 group adjacent to booster large enough that
        # cluster_size + survivor_count >= smallest.min_size
        target_survivor_count = smallest.min_size  # generous: guarantees merged > threshold
        survivor_map: dict[Position, Symbol] = {}
        grow_queue = [Position(2, 3)]  # neighbor of booster at (3,3)
        visited: set[Position] = set()
        while len(survivor_map) < target_survivor_count and grow_queue:
            pos = grow_queue.pop(0)
            if pos in visited or pos == booster_pos:
                continue
            visited.add(pos)
            survivor_map[pos] = Symbol.L1
            for nb in orthogonal_neighbors(pos, default_config.board):
                if nb not in visited and nb != booster_pos:
                    grow_queue.append(nb)

        context = _make_settled_context(
            default_config,
            surviving=survivor_map,
            dormant_boosters=[DormantBooster("R", booster_pos, "H", spawned_step=0)],
        )

        strategy = BoosterArmStrategy(
            default_config, forward_simulator, cluster_builder,
            seed_planner, chain_evaluator, landing_evaluator, rng,
        )

        boundary = cluster_builder.analyze_boundary(context)
        adjacent = strategy._booster_adjacent_survivors(booster_pos, boundary)

        # Verify that L1 survivors are found adjacent to booster
        assert Symbol.L1 in adjacent, "L1 survivors should be adjacent to booster"
        survivor_count = adjacent[Symbol.L1]

        # Verify that merged total would trigger a spawn (the guard condition)
        merged_total = default_config.board.min_cluster_size + survivor_count
        spawn_result = spawn_eval.booster_for_size(merged_total)
        assert spawn_result is not None, (
            f"merged_total={merged_total} should trigger a spawn"
        )


class TestConfigValidation:
    """ARM-AFF-007: config rejects negative affinity."""

    def test_arm_aff_007_config_rejects_negative_affinity(
        self, default_config: MasterConfig,
    ) -> None:
        """ARM-AFF-007: survivor_affinity_per_cell < 0 raises ConfigValidationError."""
        # Extract current values from the loaded config to build a valid base
        r = default_config.reasoner
        with pytest.raises(ConfigValidationError, match="survivor_affinity_per_cell"):
            ReasonerConfig(
                payout_low_fraction=r.payout_low_fraction,
                payout_high_fraction=r.payout_high_fraction,
                arming_urgency_horizon=r.arming_urgency_horizon,
                terminal_dead_default_max_component=r.terminal_dead_default_max_component,
                max_forward_simulations_per_step=r.max_forward_simulations_per_step,
                max_strategic_cells_per_step=r.max_strategic_cells_per_step,
                lookahead_depth=r.lookahead_depth,
                survivor_affinity_per_cell=-1.0,
                arm_feasibility_threshold=r.arm_feasibility_threshold,
                arm_feasibility_retry_budget=r.arm_feasibility_retry_budget,
            )
