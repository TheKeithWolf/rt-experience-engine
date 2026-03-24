"""Tier 1 family archetype definitions — static and cascade variants.

Small wins from clusters of 5-6. These are bread-and-butter basegame wins.
Phase 4 registers 4 static archetypes (cascade_depth=0). Phase 5 adds 8
cascade archetypes (cascade_depth >= 1) that require the ASP sequence planner.
"""

from __future__ import annotations

from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier
from .registry import ArchetypeRegistry, ArchetypeSignature, CascadeStepConstraint


def _t1_base() -> dict:
    """Shared fields for all static t1 archetypes."""
    return dict(
        family="t1",
        criteria="basegame",
        # Phase 4 static archetypes have no cascade, booster, or narrative constraints
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
        # No dead-board component cap — clusters are expected
        max_component_size=None,
        triggers_freespin=False,
        reaches_wincap=False,
    )


def register_static_t1_archetypes(registry: ArchetypeRegistry) -> None:
    """Register 4 static (cascade_depth=0) t1 archetypes."""

    base = _t1_base

    # Simplest win — single small cluster
    registry.register(ArchetypeSignature(
        id="t1_single",
        required_cluster_count=Range(1, 1),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        payout_range=RangeFloat(0.1, 3.0),
        **base(),
    ))

    # Win + almost-more — cluster plus near-misses
    registry.register(ArchetypeSignature(
        id="t1_near_miss",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.ANY,
        payout_range=RangeFloat(0.1, 3.0),
        **base(),
    ))

    # Multiple clusters — 2-3 small clusters on the same board
    registry.register(ArchetypeSignature(
        id="t1_multi",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        payout_range=RangeFloat(0.2, 9.0),
        **base(),
    ))

    # Win + scatter tease — cluster with 2 scatters (feature awareness)
    # Spec cascade range is 0-2 but static generator uses depth=0 only
    registry.register(ArchetypeSignature(
        id="t1_scatter_2",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(2, 2),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        payout_range=RangeFloat(0.1, 12.0),
        **base(),
    ))


# ---------------------------------------------------------------------------
# Cascade t1 archetypes (Phase 5) — require ASP sequence planner
# ---------------------------------------------------------------------------

def _t1_cascade_base(**overrides: object) -> dict:
    """Shared fields for cascade t1 archetypes.

    Same family/criteria as static t1, but with cascade_depth > 0.
    No booster constraints at t1 level — clusters stay in 5-6 size range,
    well below any booster spawn threshold.
    """
    defaults = dict(
        family="t1",
        criteria="basegame",
        required_booster_spawns={},
        required_booster_fires={},
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
    return defaults


def register_cascade_t1_archetypes(registry: ArchetypeRegistry) -> None:
    """Register 8 cascade (cascade_depth >= 1) t1 archetypes.

    These require the step reasoner to produce per-step intents
    before the CSP spatial solver can place clusters per cascade step.
    """
    base = _t1_cascade_base

    # Single cascade — one win, refill produces another, then dead
    registry.register(ArchetypeSignature(
        id="t1_cascade_1",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(1, 1),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.2, 6.0),
        **base(),
    ))

    # Double cascade — two refill cycles before dead
    registry.register(ArchetypeSignature(
        id="t1_cascade_2",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 2),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.4, 12.0),
        **base(),
    ))

    # Cascade with near-misses — win + almost-more + cascade continues
    registry.register(ArchetypeSignature(
        id="t1_near_miss_cascade",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.ANY,
        required_cascade_depth=Range(1, 2),
        cascade_steps=None,
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.2, 12.0),
        **base(),
    ))

    # Multiple clusters per step with cascade — broader payout potential
    registry.register(ArchetypeSignature(
        id="t1_multi_cascade",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(1, 2),
        cascade_steps=None,
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.4, 15.0),
        **base(),
    ))

    # Deep cascade — 3 to 6 cascade steps for extended tumble experience
    registry.register(ArchetypeSignature(
        id="t1_cascade_3plus",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 6),
        cascade_steps=None,
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.6, 30.0),
        **base(),
    ))

    # Long cascade forced to LOW tier symbols — extended tumble, modest payout
    # symbol_tier_per_step covers all possible steps up to max depth
    registry.register(ArchetypeSignature(
        id="t1_low_cascade",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 6),
        cascade_steps=None,
        # Force LOW tier on every possible non-terminal step (0 through max_depth)
        symbol_tier_per_step={0: SymbolTier.LOW, 1: SymbolTier.LOW,
                              2: SymbolTier.LOW, 3: SymbolTier.LOW,
                              4: SymbolTier.LOW, 5: SymbolTier.LOW,
                              6: SymbolTier.LOW},
        payout_range=RangeFloat(0.3, 8.0),
        **base(),
    ))

    # Win + 3 scatters (near-trigger) — can be static or shallow cascade
    registry.register(ArchetypeSignature(
        id="t1_scatter_3",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(0, 2),
        cascade_steps=None,
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.1, 12.0),
        **base(),
    ))

    # Deep cascade + 3 scatters — extended tumble near feature trigger tension
    registry.register(ArchetypeSignature(
        id="t1_cascade_scatter_3",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=None,
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.6, 30.0),
        **base(),
    ))
