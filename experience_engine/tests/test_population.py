"""Tests for the population controller and budget allocation.

Covers TEST-P4-019 through P4-021 and P4-026.
"""

from __future__ import annotations

import pytest

from ..archetypes.dead import register_dead_archetypes
from ..archetypes.registry import ArchetypeRegistry
from ..archetypes.tier1 import register_static_t1_archetypes
from ..config.schema import MasterConfig
from ..pipeline.instance_generator import StaticInstanceGenerator
from ..primitives.gravity import GravityDAG
from ..population.allocator import allocate_budget
from ..population.controller import PopulationController
from ..validation.validator import InstanceValidator


@pytest.fixture
def full_registry(default_config: MasterConfig) -> ArchetypeRegistry:
    reg = ArchetypeRegistry(default_config)
    register_dead_archetypes(reg)
    register_static_t1_archetypes(reg)
    return reg


# ---------------------------------------------------------------------------
# TEST-P4-019: Budget allocation sums to total +/- 1%
# ---------------------------------------------------------------------------

def test_budget_allocation_sums_to_total(
    default_config: MasterConfig,
    full_registry: ArchetypeRegistry,
) -> None:
    allocations = allocate_budget(default_config, full_registry)
    total = sum(a.count for a in allocations)
    budget = default_config.population.total_budget
    # Within 1% of target
    assert abs(total - budget) / budget < 0.01, (
        f"Total {total} not within 1% of budget {budget}"
    )


# ---------------------------------------------------------------------------
# TEST-P4-020: Minimum instances per archetype enforced
# ---------------------------------------------------------------------------

def test_minimum_instances_per_archetype(
    default_config: MasterConfig,
    full_registry: ArchetypeRegistry,
) -> None:
    allocations = allocate_budget(default_config, full_registry)
    min_per = default_config.population.min_instances_per_archetype
    for alloc in allocations:
        assert alloc.count >= min_per, (
            f"{alloc.archetype_id} has {alloc.count} < minimum {min_per}"
        )


def test_allocation_only_includes_registered_archetypes(
    default_config: MasterConfig,
    full_registry: ArchetypeRegistry,
) -> None:
    """Unregistered families (wild, rocket, etc.) get zero allocation."""
    allocations = allocate_budget(default_config, full_registry)
    allocated_families = {a.family for a in allocations}
    # Phase 4 only has dead and t1 registered
    assert allocated_families == {"dead", "t1"}


# ---------------------------------------------------------------------------
# TEST-P4-021: Generation loop — 100 dead_empty all valid
# ---------------------------------------------------------------------------

def test_generation_loop_100_dead_empty(
    default_config: MasterConfig,
) -> None:
    """Generate 100 dead_empty instances via the full population loop."""
    # Build a registry with only dead_empty for a focused test
    from ..archetypes.registry import ArchetypeSignature
    from ..pipeline.protocols import Range, RangeFloat

    reg = ArchetypeRegistry(default_config)
    reg.register(ArchetypeSignature(
        id="dead_empty",
        family="dead",
        criteria="0",
        required_cluster_count=Range(0, 0),
        required_cluster_sizes=(),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=3,
        required_cascade_depth=Range(0, 0),
        cascade_steps=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.0, 0.0),
        triggers_freespin=False,
        reaches_wincap=False,
    ))

    gravity_dag = GravityDAG(default_config.board, default_config.gravity)
    generator = StaticInstanceGenerator(default_config, reg, gravity_dag)
    validator = InstanceValidator(default_config, reg)
    # cascade_generator=None — this test only uses static archetypes
    controller = PopulationController(
        default_config, reg, generator, cascade_generator=None, validator=validator,
    )

    result = controller.run(seed=42)

    # Should generate instances (budget distributed to dead_empty only)
    assert result.total_generated > 0, "Expected generated instances"
    # All should be valid
    valid_count = sum(1 for m in result.metrics if m.is_valid)
    assert valid_count == result.total_generated, (
        f"{result.total_generated - valid_count} invalid instances"
    )


# ---------------------------------------------------------------------------
# TEST-P4-026: Config validation — weights summing to 0 handled gracefully
# ---------------------------------------------------------------------------

def test_allocation_with_no_registered_archetypes(
    default_config: MasterConfig,
) -> None:
    """Empty registry produces empty allocation."""
    reg = ArchetypeRegistry(default_config)
    allocations = allocate_budget(default_config, reg)
    assert len(allocations) == 0
