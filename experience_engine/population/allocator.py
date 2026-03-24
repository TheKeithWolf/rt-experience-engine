"""Budget allocation — distributes total_budget across registered archetypes.

Weights from config.population are renormalized to only include families
with registered archetypes. Per-family budgets distributed by archetype
weights. Minimum instances per archetype enforced from config.

All thresholds from MasterConfig — zero hardcoded values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..archetypes.registry import ArchetypeRegistry
from ..config.schema import MasterConfig


@dataclass(frozen=True, slots=True)
class BudgetAllocation:
    """Budget for a single archetype — how many instances to generate."""

    archetype_id: str
    family: str
    count: int


def allocate_budget(
    config: MasterConfig,
    registry: ArchetypeRegistry,
) -> tuple[BudgetAllocation, ...]:
    """Distribute total_budget across registered archetypes.

    Algorithm:
    1. Filter family_weights to families with registered archetypes only
    2. Renormalize remaining family weights to sum to 1.0
    3. Per family: distribute by archetype weights (renormalized)
    4. Enforce min_instances_per_archetype floor
    5. Adjust rounding so sum approximates total_budget
    """
    total_budget = config.population.total_budget
    min_per = config.population.min_instances_per_archetype

    # Build family weight lookup from config (stored as tuple of tuples)
    family_weight_map: dict[str, float] = dict(config.population.family_weights)

    # Build per-family archetype weight lookup
    archetype_weight_map: dict[str, dict[str, float]] = {}
    for fam_name, arch_weights in config.population.archetype_weights:
        archetype_weight_map[fam_name] = dict(arch_weights)

    # Only include families with registered archetypes
    active_families = registry.registered_families()
    active_weights: dict[str, float] = {
        f: family_weight_map.get(f, 0.0)
        for f in active_families
        if family_weight_map.get(f, 0.0) > 0.0
    }

    if not active_weights:
        return ()

    # Renormalize family weights to sum to 1.0
    weight_sum = sum(active_weights.values())
    normalized_family: dict[str, float] = {
        f: w / weight_sum for f, w in active_weights.items()
    }

    allocations: list[BudgetAllocation] = []

    for family, family_frac in sorted(normalized_family.items()):
        # Raw family budget
        family_budget_raw = total_budget * family_frac

        # Get archetype weights for this family — only registered ones
        registered_ids = {s.id for s in registry.get_family(family)}
        all_arch_weights = archetype_weight_map.get(family, {})
        arch_weights: dict[str, float] = {
            aid: w for aid, w in all_arch_weights.items()
            if aid in registered_ids and w > 0.0
        }

        if not arch_weights:
            continue

        # Renormalize archetype weights within family
        arch_weight_sum = sum(arch_weights.values())
        normalized_arch: dict[str, float] = {
            aid: w / arch_weight_sum for aid, w in arch_weights.items()
        }

        # Distribute family budget to archetypes
        for arch_id in sorted(normalized_arch.keys()):
            arch_frac = normalized_arch[arch_id]
            raw_count = family_budget_raw * arch_frac
            # Floor to integer, enforce minimum
            count = max(math.floor(raw_count), min_per)
            allocations.append(BudgetAllocation(
                archetype_id=arch_id,
                family=family,
                count=count,
            ))

    # Adjust to match total_budget — distribute remainder to largest allocations
    current_total = sum(a.count for a in allocations)
    diff = total_budget - current_total

    if diff > 0 and allocations:
        # Add remainder one-by-one to archetypes in round-robin by descending count
        sorted_indices = sorted(
            range(len(allocations)),
            key=lambda i: allocations[i].count,
            reverse=True,
        )
        for i in range(diff):
            idx = sorted_indices[i % len(sorted_indices)]
            old = allocations[idx]
            allocations[idx] = BudgetAllocation(
                archetype_id=old.archetype_id,
                family=old.family,
                count=old.count + 1,
            )
    elif diff < 0 and allocations:
        # Trim excess from largest allocations (preserve minimums)
        sorted_indices = sorted(
            range(len(allocations)),
            key=lambda i: allocations[i].count,
            reverse=True,
        )
        remaining = -diff
        for idx in sorted_indices:
            if remaining <= 0:
                break
            old = allocations[idx]
            can_trim = old.count - min_per
            trim = min(can_trim, remaining)
            if trim > 0:
                allocations[idx] = BudgetAllocation(
                    archetype_id=old.archetype_id,
                    family=old.family,
                    count=old.count - trim,
                )
                remaining -= trim

    return tuple(allocations)
