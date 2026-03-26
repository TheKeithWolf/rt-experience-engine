"""Dead family archetype definitions — 11 archetypes paying 0.0x.

Dead spins form the majority of the population. Ordered by ascending
player tension: from empty boards to multi-scatter + multi-near-miss boards.
All distinguishing values come from the spec — zero hardcoded beyond what
the archetype signature literally defines (ranges, tiers).
"""

from __future__ import annotations

from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier
from .registry import ArchetypeRegistry, ArchetypeSignature


def _dead_base() -> dict:
    """Shared fields for all dead archetypes — avoids repeating 20 identical lines."""
    return dict(
        family="dead",
        criteria="0",
        required_cluster_count=Range(0, 0),
        required_cluster_sizes=(),
        required_cluster_symbols=None,
        required_cascade_depth=Range(0, 0),
        cascade_steps=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        narrative_arc=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.0, 0.0),
        triggers_freespin=False,
        reaches_wincap=False,
    )


def register_dead_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 11 dead family archetypes."""

    base = _dead_base

    # Baseline loss — minimal board activity
    registry.register(ArchetypeSignature(
        id="dead_empty",
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 1),
        required_near_miss_symbol_tier=None,
        max_component_size=3,
        **base(),
    ))

    # Low-stakes near-miss — one almost-cluster with a low-tier symbol
    registry.register(ArchetypeSignature(
        id="dead_near_miss_low",
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 1),
        required_near_miss_symbol_tier=SymbolTier.LOW,
        max_component_size=4,
        **base(),
    ))

    # High-stakes near-miss — one almost-cluster with a high-tier symbol
    registry.register(ArchetypeSignature(
        id="dead_near_miss_high",
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 1),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        max_component_size=4,
        **base(),
    ))

    # Multiple near-misses — 2-3 almost-clusters, any tier
    registry.register(ArchetypeSignature(
        id="dead_near_miss_multi",
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(2, 3),
        required_near_miss_symbol_tier=SymbolTier.ANY,
        max_component_size=4,
        **base(),
    ))

    # Feature awareness — 2 scatters visible, no near-miss
    registry.register(ArchetypeSignature(
        id="dead_scatter_2",
        required_scatter_count=Range(2, 2),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=4,
        **base(),
    ))

    # Dual tension — 2 scatters + near-misses
    registry.register(ArchetypeSignature(
        id="dead_scatter_2_near_miss",
        required_scatter_count=Range(2, 2),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.ANY,
        max_component_size=4,
        **base(),
    ))

    # Premium pressure — multiple high-tier near-misses
    registry.register(ArchetypeSignature(
        id="dead_near_miss_high_multi",
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(2, 3),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        max_component_size=4,
        **base(),
    ))

    # Maximum near-miss density — at least 3 near-misses, no scatters
    registry.register(ArchetypeSignature(
        id="dead_saturated",
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(3, 10),
        required_near_miss_symbol_tier=SymbolTier.ANY,
        max_component_size=4,
        **base(),
    ))

    # Near-trigger — 3 scatters (one short of freespin)
    registry.register(ArchetypeSignature(
        id="dead_scatter_3",
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=4,
        **base(),
    ))

    # Multi-vector tension — 3 scatters + near-misses
    registry.register(ArchetypeSignature(
        id="dead_scatter_3_near_miss",
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.ANY,
        max_component_size=4,
        **base(),
    ))

    # Apex tension dead — 3 scatters + high-tier near-misses
    registry.register(ArchetypeSignature(
        id="dead_scatter_3_near_miss_high",
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        max_component_size=4,
        **base(),
    ))
