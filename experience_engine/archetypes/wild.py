"""Wild family archetype definitions — 17 archetypes covering idle, bridge,
narrative arcs (late saves, storms, fakeouts), scatter tension, and near-miss tension.

Wilds spawn from clusters of 7-8 and can bridge disconnected standard-symbol
groups. Each archetype defines a specific player experience pattern using wilds.
"""

from __future__ import annotations

from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier
from .registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    CascadeStepConstraint,
    TerminalNearMissSpec,
)


def _wild_base() -> dict:
    """Shared fields for all wild archetypes.

    Wild family is basegame criteria. No booster fires, chains, rocket orientation,
    or feature triggers — those come in later phases (9+).
    """
    return dict(
        family="wild",
        criteria="basegame",
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        triggers_freespin=False,
        reaches_wincap=False,
        # Wilds produce clusters — no dead-board component cap
        max_component_size=None,
    )


def register_wild_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 17 wild-family archetypes.

    Organized by experience category: core (6), near-miss tension (3),
    late saves (2), storms (2), scatter tension (2), fakeouts (2).
    """
    base = _wild_base

    # -----------------------------------------------------------------------
    # Core wild archetypes (6)
    # -----------------------------------------------------------------------

    # Wild spawns from 7-8 cluster but sits idle — doesn't bridge
    registry.register(ArchetypeSignature(
        id="wild_idle",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(1, 2),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                must_spawn_booster="W",
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior="spawn",
            ),
        ),
        required_booster_spawns={"W": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.2, 8.0),
        **base(),
    ))

    # Wild bridges two small groups into a larger effective cluster
    registry.register(ArchetypeSignature(
        id="wild_bridge_small",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(1, 3),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                must_spawn_booster="W",
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior="spawn",
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior="bridge",
            ),
        ),
        required_booster_spawns={"W": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.4, 15.0),
        **base(),
    ))

    # Wild bridges groups into 9-10 size — escalation toward rocket territory
    registry.register(ArchetypeSignature(
        id="wild_bridge_large",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 2)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.6, 20.0),
        **base(),
    ))

    # Wild bridges cluster to 9-10 size, triggering a rocket spawn
    registry.register(ArchetypeSignature(
        id="wild_enable_rocket",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1), "R": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 30.0),
        **base(),
    ))

    # Wild bridges cluster to 11-12 size, triggering a bomb spawn
    registry.register(ArchetypeSignature(
        id="wild_enable_bomb",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1), "B": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(2.0, 50.0),
        **base(),
    ))

    # Multiple wilds on board — 2-3 wild spawns across cascade steps
    registry.register(ArchetypeSignature(
        id="wild_multi",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        required_booster_spawns={"W": Range(2, 3)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 30.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Near-miss tension (3)
    # -----------------------------------------------------------------------

    # Idle wild near HIGH near-miss groups — tantalizing "what if" moment
    registry.register(ArchetypeSignature(
        id="wild_near_miss_single",
        required_cluster_count=Range(1, 1),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        required_cascade_depth=Range(1, 2),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                must_spawn_booster="W",
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior="spawn",
            ),
        ),
        required_booster_spawns={"W": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.2, 8.0),
        **base(),
    ))

    # Multiple wilds idle near multiple near-miss groups
    registry.register(ArchetypeSignature(
        id="wild_near_miss_multi",
        required_cluster_count=Range(2, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(2, 3),
        required_near_miss_symbol_tier=SymbolTier.ANY,
        required_cascade_depth=Range(1, 2),
        cascade_steps=None,
        required_booster_spawns={"W": Range(2, 2)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.4, 10.0),
        **base(),
    ))

    # LOW cluster + idle wild + HIGH near-misses — mismatch tension
    registry.register(ArchetypeSignature(
        id="wild_near_miss_high_idle",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        required_cascade_depth=Range(1, 2),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.1, 6.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Late-stage saves (2)
    # -----------------------------------------------------------------------

    # LOW cascade fades → late wild spawn saves the run
    # Narrative: early steps forced LOW tier, late step unconstrained
    registry.register(ArchetypeSignature(
        id="wild_late_save",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1)},
        # Early steps forced LOW — late step unconstrained for the wild save
        symbol_tier_per_step={0: SymbolTier.LOW, 1: SymbolTier.LOW},
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.4, 12.0),
        **base(),
    ))

    # Same as wild_late_save but HIGH wild cluster on the save step
    registry.register(ArchetypeSignature(
        id="wild_late_save_high",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1)},
        # Early steps forced LOW — late HIGH wild cluster is the payoff
        symbol_tier_per_step={0: SymbolTier.LOW, 1: SymbolTier.LOW},
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.6, 15.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Cascade storms (2)
    # -----------------------------------------------------------------------

    # Long LOW cascade with many wilds — steady low-value cascade pacing
    registry.register(ArchetypeSignature(
        id="wild_storm_low",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(4, 6),
        cascade_steps=None,
        required_booster_spawns={"W": Range(2, 4)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.6, 8.0),
        **base(),
    ))

    # Wild-to-wild domino — each wild bridges to the next in a chain
    registry.register(ArchetypeSignature(
        id="wild_storm_bridge_chain",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=None,
        required_booster_spawns={"W": Range(2, 3)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 20.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Scatter tension (2)
    # -----------------------------------------------------------------------

    # Wild + 3 scatters — one scatter short of trigger, plus wild action
    registry.register(ArchetypeSignature(
        id="wild_scatter_tease",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(1, 3),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.4, 15.0),
        **base(),
    ))

    # Deep wild cascade + 3 scatters — extended near-trigger experience
    registry.register(ArchetypeSignature(
        id="wild_scatter_cascade_tease",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 2)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.6, 20.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Escalation fakeouts (2)
    # -----------------------------------------------------------------------

    # Wild→wild→fizzle — builds up momentum then cascade dies with near-misses
    registry.register(ArchetypeSignature(
        id="wild_escalation_fizzle",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 2)},
        symbol_tier_per_step=None,
        # Terminal board shows HIGH near-misses — "it almost kept going"
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2),
            symbol_tier=SymbolTier.HIGH,
        ),
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.6, 12.0),
        **base(),
    ))

    # Wild enables rocket spawn → rocket stays dormant on terminal board
    # Terminal shows dormant rocket + near-misses — maximum unfulfilled potential
    registry.register(ArchetypeSignature(
        id="wild_rocket_tease",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1), "R": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2),
            symbol_tier=SymbolTier.ANY,
        ),
        # Rocket must survive on the terminal dead board for visual impact
        dormant_boosters_on_terminal=("R",),
        payout_range=RangeFloat(1.0, 20.0),
        **base(),
    ))
