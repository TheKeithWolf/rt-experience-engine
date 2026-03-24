"""Parallel generation — distributes archetype generation across worker processes.

Partitions by family to preserve intra-family variance feedback coherence.
Each worker constructs its own full pipeline (generators, validator, accumulators,
clingo Control) to avoid pickling non-serializable objects.

GravityDAG and Paytable are reconstructed per-worker from config (~1ms, deterministic).
Clingo ASP rules are loaded per-worker from the file path.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, replace

from .archetypes.bomb import register_bomb_archetypes
from .archetypes.chain import register_chain_archetypes
from .archetypes.dead import register_dead_archetypes
from .archetypes.lb import register_lb_archetypes
from .archetypes.registry import ArchetypeRegistry
from .archetypes.rocket import register_rocket_archetypes
from .archetypes.slb import register_slb_archetypes
from .archetypes.tier1 import register_cascade_t1_archetypes, register_static_t1_archetypes
from .archetypes.trigger import register_trigger_archetypes
from .archetypes.wincap import register_wincap_archetypes
from .archetypes.wild import register_wild_archetypes
from .config.schema import MasterConfig
from .pipeline.data_types import GeneratedInstance
from .population.allocator import BudgetAllocation, allocate_budget
from .population.controller import PopulationController, PopulationResult
from .validation.metrics import InstanceMetrics

# All family registration functions — same order as run.py
_FAMILY_REGISTRARS = (
    register_dead_archetypes,
    register_static_t1_archetypes,
    register_cascade_t1_archetypes,
    register_wild_archetypes,
    register_rocket_archetypes,
    register_bomb_archetypes,
    register_lb_archetypes,
    register_slb_archetypes,
    register_chain_archetypes,
    register_trigger_archetypes,
    register_wincap_archetypes,
)


@dataclass(frozen=True, slots=True)
class WorkerTask:
    """Specification for one parallel generation worker — fully picklable."""

    # Archetype IDs this worker is responsible for
    archetype_ids: tuple[str, ...]
    # Seed offset so each worker produces unique RNG sequences
    seed: int
    # Full config — frozen dataclasses are picklable
    config: MasterConfig
    # Starting sim_id for this worker's instances (contiguous assignment)
    sim_id_offset: int


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Result from one parallel worker — picklable for IPC."""

    instances: tuple[GeneratedInstance, ...]
    metrics: tuple[InstanceMetrics, ...]
    failure_log: tuple[tuple[str, int, str], ...]
    total_generated: int
    total_failed: int


def _build_worker_registry(config: MasterConfig, archetype_ids: frozenset[str]) -> ArchetypeRegistry:
    """Build a full registry then extract only the requested archetypes.

    Each worker rebuilds the registry from scratch — the full registry is needed
    for signature validation, then we filter to the assigned subset.
    """
    full = ArchetypeRegistry(config)
    for registrar in _FAMILY_REGISTRARS:
        registrar(full)

    # Build a filtered registry with only assigned archetypes
    filtered = ArchetypeRegistry(config)
    for arch_id in archetype_ids:
        sig = full.get(arch_id)
        filtered.register(sig)
    return filtered


def _worker_fn(task: WorkerTask) -> WorkerResult:
    """Worker process entry point — builds its own pipeline and generates instances.

    Each worker creates its own:
    - ArchetypeRegistry (filtered to assigned archetypes)
    - StaticInstanceGenerator + CascadeInstanceGenerator
    - InstanceValidator + PopulationAccumulators
    - Sequence planner (backend from config.planner.backend)
    - GravityDAG (reconstructed from config, deterministic)

    Non-serializable objects are constructed fresh per worker.
    """
    # Lazy imports — heavy dependencies constructed per worker
    from .pipeline.cascade_generator import CascadeInstanceGenerator
    from .pipeline.instance_generator import StaticInstanceGenerator
    from .primitives.gravity import GravityDAG
    from .validation.validator import InstanceValidator

    config = task.config
    archetype_ids = frozenset(task.archetype_ids)

    # Build filtered registry for this worker's archetypes
    registry = _build_worker_registry(config, archetype_ids)

    # Build pipeline components — each worker owns its own instances
    gravity_dag = GravityDAG(config.board, config.gravity)
    static_gen = StaticInstanceGenerator(config, registry, gravity_dag)

    # Build StepReasoner components for cascade generation
    from .step_reasoner.evaluators import ChainEvaluator, PayoutEstimator, SpawnEvaluator
    from .step_reasoner.assessor import StepAssessor
    from .step_reasoner.selector import StrategySelector, DEFAULT_SELECTION_RULES
    from .step_reasoner.reasoner import StepReasoner
    from .step_reasoner.registry import build_default_registry
    from .pipeline.step_executor import StepExecutor
    from .pipeline.step_validator import StepValidator
    from .pipeline.simulator import StepTransitionSimulator
    from .spatial_solver.solver import CSPSpatialSolver

    spawn_eval = SpawnEvaluator(config.boosters)
    chain_eval = ChainEvaluator(config.boosters)
    payout_eval = PayoutEstimator(
        config.paytable, config.centipayout, config.win_levels,
        config.symbols, config.grid_multiplier,
    )
    csp_solver = CSPSpatialSolver(config)
    strategy_registry = build_default_registry(
        config, gravity_dag, csp_solver,
        spawn_eval, chain_eval, payout_eval,
    )
    assessor = StepAssessor(spawn_eval, chain_eval, payout_eval, config.reasoner)
    selector = StrategySelector(DEFAULT_SELECTION_RULES)
    reasoner = StepReasoner(strategy_registry, selector, assessor)
    executor = StepExecutor(config, gravity_dag=gravity_dag)
    step_validator = StepValidator(config)
    simulator = StepTransitionSimulator(gravity_dag, config)

    cascade_gen = CascadeInstanceGenerator(
        config, registry, gravity_dag,
        reasoner, executor, step_validator, simulator,
    )
    validator = InstanceValidator(config, registry)

    controller = PopulationController(config, registry, static_gen, cascade_gen, validator)
    result = controller.run(task.seed)

    # Remap sim_ids to the assigned contiguous range
    remapped_instances: list[GeneratedInstance] = []
    for i, inst in enumerate(result.instances):
        remapped = replace(inst, sim_id=task.sim_id_offset + i)
        remapped_instances.append(remapped)

    return WorkerResult(
        instances=tuple(remapped_instances),
        metrics=result.metrics,
        failure_log=result.failure_log,
        total_generated=result.total_generated,
        total_failed=result.total_failed,
    )


def _partition_by_family(
    registry: ArchetypeRegistry,
    num_workers: int,
) -> list[tuple[str, ...]]:
    """Partition archetypes into groups by family, then round-robin across workers.

    Groups by family first — keeps variance feedback coherent within families.
    If more families than workers, some workers handle multiple families.
    """
    families = sorted(registry.registered_families())
    partitions: list[list[str]] = [[] for _ in range(min(num_workers, len(families)))]

    for i, family in enumerate(families):
        bucket = i % len(partitions)
        for sig in registry.get_family(family):
            partitions[bucket].append(sig.id)

    # Filter out empty partitions (shouldn't happen but safety)
    return [tuple(p) for p in partitions if p]


def run_parallel_generation(
    config: MasterConfig,
    registry: ArchetypeRegistry,
    num_workers: int,
    seed: int = 42,
) -> PopulationResult:
    """Distribute archetype generation across worker processes.

    Partitions allocations by family so intra-family variance steering is preserved.
    Each worker gets a contiguous sim_id range and independent seed.
    Results are merged into a single PopulationResult.
    """
    partitions = _partition_by_family(registry, num_workers)

    # Pre-compute sim_id offsets by estimating allocation counts per partition
    # Each worker runs allocate_budget internally, but we need sim_id offsets beforehand
    # We allocate sim_ids proportionally based on partition archetype counts
    all_allocations = allocate_budget(config, registry)
    alloc_map = {a.archetype_id: a.count for a in all_allocations}

    sim_id_offset = 0
    tasks: list[WorkerTask] = []
    for i, partition in enumerate(partitions):
        # Count how many books this partition will generate
        partition_count = sum(alloc_map.get(aid, 0) for aid in partition)

        tasks.append(WorkerTask(
            archetype_ids=partition,
            seed=seed + i,
            config=config,
            sim_id_offset=sim_id_offset,
        ))
        sim_id_offset += partition_count

    # Run workers in a process pool
    with multiprocessing.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_fn, tasks)

    # Merge results — concatenate instances, metrics, failure_logs
    all_instances: list[GeneratedInstance] = []
    all_metrics: list[InstanceMetrics] = []
    all_failures: list[tuple[str, int, str]] = []
    total_gen = 0
    total_fail = 0

    for worker_result in results:
        all_instances.extend(worker_result.instances)
        all_metrics.extend(worker_result.metrics)
        all_failures.extend(worker_result.failure_log)
        total_gen += worker_result.total_generated
        total_fail += worker_result.total_failed

    return PopulationResult(
        instances=tuple(all_instances),
        metrics=tuple(all_metrics),
        total_generated=total_gen,
        total_failed=total_fail,
        failure_log=tuple(all_failures),
    )
