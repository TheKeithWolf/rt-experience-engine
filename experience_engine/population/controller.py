"""Population controller — orchestrates the full generation → validate → accumulate loop.

Owns the variance engine accumulators. Coordinates generator, validator, and
accumulators into the complete feedback loop. Produces PopulationResult with
all generated instances, validation metrics, and failure logs.

Routes each archetype to the correct generator: static (cascade_depth=0) or
cascade (cascade_depth > 0). All retry limits and budget parameters from
MasterConfig — zero hardcoded values.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..archetypes.registry import ArchetypeRegistry
from ..config.schema import MasterConfig
from ..pipeline.cascade_generator import CascadeInstanceGenerator
from ..pipeline.data_types import GeneratedInstance, GenerationResult
from ..pipeline.instance_generator import StaticInstanceGenerator
from ..pipeline.reel_generator import ReelStripGenerator
from ..validation.metrics import InstanceMetrics
from ..validation.validator import InstanceValidator
from ..variance.accumulators import PopulationAccumulators
from ..variance.bias_computation import compute_hints
from ..variance.hints import VarianceHints
from .allocator import BudgetAllocation, allocate_budget

if TYPE_CHECKING:
    from ..rl_archive.generator import RLArchiveGenerator


@dataclass(frozen=True, slots=True)
class PopulationResult:
    """Result of a full population generation run."""

    instances: tuple[GeneratedInstance, ...]
    metrics: tuple[InstanceMetrics, ...]
    total_generated: int
    total_failed: int
    # (archetype_id, attempt_count, failure_reason) for each failed slot
    failure_log: tuple[tuple[str, int, str], ...]


class PopulationController:
    """Orchestrates the full generation loop for a population.

    Manages the feedback cycle: compute variance hints → generate → validate
    → update accumulators → repeat. Routes each archetype to the static or
    cascade generator based on required_cascade_depth. The accumulators steer
    subsequent generations toward uniform coverage (CONSTRAINT-VE-1 through VE-6).
    """

    __slots__ = (
        "_config", "_registry", "_static_generator", "_cascade_generator",
        "_validator", "_accumulators", "_rl_archive_generator",
        "_reel_generator",
    )

    def __init__(
        self,
        config: MasterConfig,
        registry: ArchetypeRegistry,
        static_generator: StaticInstanceGenerator,
        cascade_generator: CascadeInstanceGenerator | None,
        validator: InstanceValidator,
        rl_archive_generator: RLArchiveGenerator | None = None,
        reel_generator: ReelStripGenerator | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._static_generator = static_generator
        self._cascade_generator = cascade_generator
        self._validator = validator
        self._accumulators = PopulationAccumulators.create(config)
        self._rl_archive_generator = rl_archive_generator
        self._reel_generator = reel_generator

    def _select_generator(
        self, archetype_id: str,
    ) -> StaticInstanceGenerator | CascadeInstanceGenerator | RLArchiveGenerator | ReelStripGenerator:
        """Route archetypes to generators along two independent dimensions.

        Dimension 1 — family-based: the reel family is strip-driven, not
        solver-driven, and always routes to the ReelStripGenerator regardless
        of cascade depth. This is a distinct dispatch axis, not an extension
        of the depth chain below.

        Dimension 2 — cascade depth (unchanged):
          depth == 0 → static pipeline
          depth 1    → cascade pipeline (rule-based StepReasoner)
          depth >= 2 → RL archive if available, else cascade fallback
        """
        sig = self._registry.get(archetype_id)

        if sig.family == "reel":
            if self._reel_generator is None:
                raise RuntimeError(
                    f"Archetype '{archetype_id}' is in the reel family "
                    f"but no ReelStripGenerator was provided to "
                    f"PopulationController.",
                )
            return self._reel_generator

        depth = sig.required_cascade_depth.max_val

        if depth == 0:
            return self._static_generator

        # Deep cascades: prefer RL archive if available
        #if depth >= 2 and self._rl_archive_generator is not None:
        #    return self._rl_archive_generator

        # Shallow cascades or fallback when no archive
        
        #if self._cascade_generator is None:
        #    raise RuntimeError(
        #        f"Archetype '{archetype_id}' requires cascade_depth "
        #        f"{depth} but no cascade generator was provided. "
        #        f"Initialize PopulationController with a CascadeInstanceGenerator."
        #    )
        
        return self._cascade_generator

    def run(self, seed: int = 42) -> PopulationResult:
        """Execute the full population generation loop.

        1. Allocate budget across registered archetypes
        2. Reset accumulators (CONSTRAINT-VE-5)
        3. For each allocation slot:
           a. Select correct generator (static vs cascade) for the archetype
           b. Compute variance hints from current accumulators
           c. Generate instance
           d. Validate instance
           e. Valid → append + update accumulators (CONSTRAINT-VE-4)
           f. Invalid → log failure
        4. Return PopulationResult with all instances and metrics
        """
        allocations = allocate_budget(self._config, self._registry)
        self._accumulators.reset()

        instances: list[GeneratedInstance] = []
        all_metrics: list[InstanceMetrics] = []
        failure_log: list[tuple[str, int, str]] = []

        # Assign sequential sim_ids across all allocations
        sim_id_counter = 0

        for allocation in allocations:
            # Select generator once per archetype — all instances share it
            generator = self._select_generator(allocation.archetype_id)

            for _ in range(allocation.count):
                hints = compute_hints(self._accumulators, self._config)
                max_val_retries = self._config.solvers.max_validation_retries
                accepted = False

                for val_attempt in range(max_val_retries):
                    # Reseed RNG each validation attempt — different board state
                    instance_rng = random.Random(
                        seed + sim_id_counter + val_attempt * 10000
                    )

                    result = generator.generate(
                        archetype_id=allocation.archetype_id,
                        sim_id=sim_id_counter,
                        hints=hints,
                        rng=instance_rng,
                    )

                    if not result.success or result.instance is None:
                        # Generation failed — log and stop retrying
                        failure_log.append((
                            allocation.archetype_id,
                            result.attempts,
                            result.failure_reason or "unknown",
                        ))
                        break

                    metrics = self._validator.validate(result.instance)
                    all_metrics.append(metrics)

                    if metrics.is_valid:
                        instances.append(result.instance)
                        self._accumulators.update(result.instance)
                        accepted = True
                        break

                    # Validation failed — retry with fresh RNG unless last attempt
                    if val_attempt == max_val_retries - 1:
                        failure_log.append((
                            allocation.archetype_id,
                            result.attempts,
                            f"validation failed: {'; '.join(metrics.validation_errors)}",
                        ))

                sim_id_counter += 1

        return PopulationResult(
            instances=tuple(instances),
            metrics=tuple(all_metrics),
            total_generated=len(instances),
            total_failed=len(failure_log),
            failure_log=tuple(failure_log),
        )
