"""Strategy registry — name-based lookup for step strategies.

StrategyRegistry provides register/get/available operations with duplicate
detection. build_default_registry is the factory that constructs shared
services once and registers all 9 strategies.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .strategies.protocol import StepStrategy

if TYPE_CHECKING:
    from ..config.schema import MasterConfig
    from ..pipeline.protocols import SpatialSolver
    from ..primitives.gravity import GravityDAG
    from .evaluators import ChainEvaluator, PayoutEstimator, SpawnEvaluator


class StrategyRegistry:
    """Name-based lookup for registered step strategies.

    Duplicate names raise ValueError on registration.
    Unknown names raise KeyError on lookup.
    """

    __slots__ = ("_strategies",)

    def __init__(self) -> None:
        self._strategies: dict[str, StepStrategy] = {}

    def register(self, name: str, strategy: StepStrategy) -> None:
        """Register a strategy under a unique name."""
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' already registered")
        self._strategies[name] = strategy

    def get(self, name: str) -> StepStrategy:
        """Look up a strategy by name. Raises KeyError if not registered."""
        if name not in self._strategies:
            raise KeyError(
                f"Unknown strategy '{name}' — registered: {sorted(self._strategies)}"
            )
        return self._strategies[name]

    def available(self) -> list[str]:
        """All registered strategy names, sorted alphabetically."""
        return sorted(self._strategies.keys())


def build_default_registry(
    config: MasterConfig,
    gravity_dag: GravityDAG,
    spatial_solver: SpatialSolver,
    spawn_evaluator: SpawnEvaluator,
    chain_evaluator: ChainEvaluator,
    payout_evaluator: PayoutEstimator,
    rng: random.Random | None = None,
) -> StrategyRegistry:
    """Construct the default registry with all 9 strategies.

    Shared services (ForwardSimulator, ClusterBuilder, SeedPlanner) are
    constructed once here and injected into all strategies that need them.
    A single rng instance is shared across all strategies for reproducibility.
    """
    from .services.boundary_analyzer import BoundaryAnalyzer
    from .services.forward_simulator import ForwardSimulator
    from .services.cluster_builder import ClusterBuilder
    from .services.gravity_field import GravityFieldService
    from .services.influence_map import InfluenceMap
    from .services.landing_criteria import (
        WildBridgeCriterion, RocketArmCriterion, BombArmCriterion,
        LightballArmCriterion, ArmFeasibilityCriterion, CompositeCriterion,
    )
    from .services.landing_evaluator import BoosterLandingEvaluator
    from .services.near_miss_planner import NearMissPlanner
    from .services.seed_planner import SeedPlanner
    from .services.spatial_context import StepSpatialContext
    from .services.utility_scorer import (
        UtilityScorer, InfluenceFactor, GravityAlignmentFactor,
        BoosterAdjacencyFactor, MergeRiskFactor,
    )
    from ..primitives.booster_rules import BoosterRules
    from .strategies.booster_arm import BoosterArmStrategy
    from .strategies.booster_setup import BoosterSetupStrategy
    from .strategies.cascade_cluster import CascadeClusterStrategy
    from .strategies.initial_cluster import InitialClusterStrategy
    from .strategies.initial_dead import InitialDeadStrategy
    from .strategies.initial_wild_bridge import InitialWildBridgeStrategy
    from .strategies.terminal_dead import TerminalDeadStrategy
    from .strategies.terminal_near_miss import TerminalNearMissStrategy
    from .strategies.wild_bridge import WildBridgeStrategy

    if rng is None:
        rng = random.Random()

    # Shared services — constructed once, injected into strategies that need them
    forward_sim = ForwardSimulator(gravity_dag, config.board, config.gravity)
    boundary_analyzer = BoundaryAnalyzer(config.board, config.symbols)
    cluster_builder = ClusterBuilder(
        spawn_evaluator, payout_evaluator, config.board, config.symbols,
        boundary_analyzer, centipayout_multiplier=config.centipayout.multiplier,
        max_seed_retries=config.solvers.max_seed_retries,
        multi_seed_threshold=config.solvers.multi_seed_threshold,
        multi_seed_count=config.solvers.multi_seed_count,
    )
    seed_planner = SeedPlanner(forward_sim, config.board, config.symbols)
    near_miss_planner = NearMissPlanner(config, cluster_builder, rng)

    # Booster landing evaluator — shared service for post-gravity viability scoring.
    # Criteria dict dispatch replaces per-strategy if/else on booster type.
    booster_rules = BoosterRules(config.boosters, config.board, config.symbols)
    # Rocket needs the composite gate — RocketArmCriterion alone only checks
    # immediate adjacency and centrality, which scores well at Step 0 but
    # doesn't guarantee the post-gravity refill zone has enough connected
    # empty space for an arming cluster. ArmFeasibilityCriterion BFSs the
    # refill pool; aggregating with `min` makes the worse score dominate so
    # both constraints must hold.
    landing_criteria = {
        "W": WildBridgeCriterion(config.board),
        "R": CompositeCriterion(
            (
                RocketArmCriterion(booster_rules, config.board),
                ArmFeasibilityCriterion(config.board),
            ),
            aggregate=min,
        ),
        "B": BombArmCriterion(booster_rules, config.board),
        "LB": LightballArmCriterion(config.board),
        "SLB": LightballArmCriterion(config.board),
    }
    landing_eval = BoosterLandingEvaluator(
        forward_sim, booster_rules, config.board, landing_criteria,
    )

    # Bridge path tracer — deterministic path from wild landing to refill zone.
    # Used exclusively by InitialWildBridgeStrategy.
    from .services.bridge_path_tracer import BridgePathTracer
    bridge_path_tracer = BridgePathTracer(config.board)

    # Spatial intelligence — constructed once when config section is present.
    # Gives strategies foresight about where future steps need space.
    spatial: StepSpatialContext | None = None
    if config.spatial_intelligence is not None:
        gravity_field_svc = GravityFieldService(gravity_dag, config.board)
        influence_map_svc = InfluenceMap(
            config.spatial_intelligence, config.board,
        )
        factors = [
            InfluenceFactor(),
            GravityAlignmentFactor(),
            BoosterAdjacencyFactor(),
            MergeRiskFactor(),
        ]
        utility_scorer = UtilityScorer(
            factors, config.spatial_intelligence.utility_factor_weights,
        )
        spatial = StepSpatialContext(
            gravity_field_svc, influence_map_svc, utility_scorer,
        )

    registry = StrategyRegistry()

    # Terminal strategies — simplest, minimal dependencies.
    # No spatial intelligence — terminal steps have no future-step demand.
    registry.register("terminal_dead", TerminalDeadStrategy(config, rng))
    registry.register("terminal_near_miss", TerminalNearMissStrategy(
        config, cluster_builder, rng,
    ))

    # Initial strategies — step 0 on empty board
    registry.register("initial_dead", InitialDeadStrategy(
        config, near_miss_planner, rng,
    ))
    registry.register("initial_cluster", InitialClusterStrategy(
        config, forward_sim, cluster_builder, seed_planner,
        spawn_evaluator, near_miss_planner, landing_eval, rng,
        spatial=spatial,
    ))
    # Initial wild bridge — step 0 for arcs where the next phase is a wild bridge.
    # Uses BridgePathTracer for deterministic path tracing instead of probabilistic seeding.
    registry.register("initial_wild_bridge", InitialWildBridgeStrategy(
        config, forward_sim, cluster_builder, seed_planner,
        spawn_evaluator, near_miss_planner, landing_eval,
        bridge_path_tracer, rng, spatial=spatial,
    ))

    # Cascade strategy — general mid-cascade
    registry.register("cascade_cluster", CascadeClusterStrategy(
        config, forward_sim, cluster_builder, seed_planner,
        spawn_evaluator, rng, spatial=spatial,
    ))

    # Wild strategy — bridge through wild symbol
    registry.register("wild_bridge", WildBridgeStrategy(
        config, forward_sim, cluster_builder, seed_planner, rng,
        spatial=spatial,
    ))

    # Booster strategies — arming and chain arrangement
    registry.register("booster_arm", BoosterArmStrategy(
        config, forward_sim, cluster_builder, seed_planner,
        chain_evaluator, landing_eval, rng, spatial=spatial,
    ))
    registry.register("booster_setup", BoosterSetupStrategy(
        config, forward_sim, cluster_builder, seed_planner,
        chain_evaluator, spawn_evaluator, landing_eval, rng,
        spatial=spatial,
    ))

    return registry
