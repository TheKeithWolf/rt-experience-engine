"""Strategy registry — name-based lookup for step strategies.

StrategyRegistry provides register/get/available operations with duplicate
detection. build_default_registry is the factory that constructs shared
services once and registers all 8 strategies.
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
    """Construct the default registry with all 8 strategies.

    Shared services (ForwardSimulator, ClusterBuilder, SeedPlanner) are
    constructed once here and injected into all strategies that need them.
    A single rng instance is shared across all strategies for reproducibility.
    """
    from .services.boundary_analyzer import BoundaryAnalyzer
    from .services.forward_simulator import ForwardSimulator
    from .services.cluster_builder import ClusterBuilder
    from .services.near_miss_planner import NearMissPlanner
    from .services.seed_planner import SeedPlanner
    from .strategies.booster_arm import BoosterArmStrategy
    from .strategies.booster_setup import BoosterSetupStrategy
    from .strategies.cascade_cluster import CascadeClusterStrategy
    from .strategies.initial_cluster import InitialClusterStrategy
    from .strategies.initial_dead import InitialDeadStrategy
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
    )
    seed_planner = SeedPlanner(forward_sim, config.board, config.symbols)
    near_miss_planner = NearMissPlanner(config, cluster_builder, rng)

    registry = StrategyRegistry()

    # Terminal strategies — simplest, minimal dependencies
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
        spawn_evaluator, near_miss_planner, rng,
    ))

    # Cascade strategy — general mid-cascade
    registry.register("cascade_cluster", CascadeClusterStrategy(
        config, forward_sim, cluster_builder, seed_planner, spawn_evaluator, rng,
    ))

    # Wild strategy — bridge through wild symbol
    registry.register("wild_bridge", WildBridgeStrategy(
        config, forward_sim, cluster_builder, seed_planner, rng,
    ))

    # Booster strategies — arming and chain arrangement
    registry.register("booster_arm", BoosterArmStrategy(
        config, forward_sim, cluster_builder, seed_planner, chain_evaluator, rng,
    ))
    registry.register("booster_setup", BoosterSetupStrategy(
        config, forward_sim, cluster_builder, seed_planner,
        chain_evaluator, spawn_evaluator, rng,
    ))

    return registry
