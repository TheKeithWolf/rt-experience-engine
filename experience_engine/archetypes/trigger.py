"""Trigger family archetype definitions — 4 archetypes covering freespin
trigger scenarios with 4 or 5 scatters, optionally accompanied by wins
or booster spawns.

Trigger archetypes have criteria="freegame" and triggers_freespin=True.
All scatter counts satisfy CONTRACT-SIG-4: scatter_count.min >=
config.freespin.min_scatters_to_trigger.
"""

from __future__ import annotations

from ..pipeline.protocols import Range, RangeFloat
from .registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    CascadeStepConstraint,
)


def _trigger_base() -> dict:
    """Shared fields for all trigger archetypes.

    Trigger family is freegame criteria. No rocket orientation, no LB tier
    targeting, no wincap. All trigger archetypes award freespins.
    """
    return dict(
        family="trigger",
        criteria="freegame",
        triggers_freespin=True,
        reaches_wincap=False,
        rocket_orientation=None,
        lb_target_tier=None,
        # Narrative constraints not used — trigger boards are about scatter placement
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
    )


def register_trigger_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 4 trigger-family archetypes.

    Organized by scatter count and complexity: pure trigger (2),
    trigger with win (1), trigger with booster (1).
    """
    base = _trigger_base

    # -----------------------------------------------------------------------
    # trigger_4s — 4 scatters, no wins, static dead board
    # -----------------------------------------------------------------------
    # Minimum scatter trigger — the player sees 4 scatters land and enters
    # freespins with no base game payout. Clean trigger experience.
    registry.register(ArchetypeSignature(
        id="trigger_4s",
        required_cluster_count=Range(0, 0),
        required_cluster_sizes=(),
        required_cluster_symbols=None,
        # 4 scatters = minimum to trigger freespin (config.freespin.min_scatters_to_trigger)
        required_scatter_count=Range(4, 4),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        # Dead board — no clusters, max component capped below cluster threshold
        max_component_size=4,
        # Static — no cascade
        required_cascade_depth=Range(0, 0),
        cascade_steps=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        # No payout — trigger itself is the reward
        payout_range=RangeFloat(0.0, 0.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # trigger_4s_with_win — 4 scatters + small cluster wins
    # -----------------------------------------------------------------------
    # Trigger accompanied by modest cluster payout. The player gets both
    # a base game win and freespins — feels doubly rewarding.
    registry.register(ArchetypeSignature(
        id="trigger_4s_with_win",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        required_scatter_count=Range(4, 4),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        # Clusters present — no dead-board component cap
        max_component_size=None,
        # Optional short cascade alongside the trigger (0 = static, 1-2 = cascade)
        required_cascade_depth=Range(0, 2),
        cascade_steps=(
            # Step 0: 1-2 small clusters alongside the scatters
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
            ),
        ),
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        # Modest cluster payout — t1-level alongside the trigger
        payout_range=RangeFloat(0.1, 3.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # trigger_5s — 5 scatters, no wins, static dead board
    # -----------------------------------------------------------------------
    # Maximum scatter trigger — all available scatter positions filled.
    # Rare premium trigger with no base game payout.
    registry.register(ArchetypeSignature(
        id="trigger_5s",
        required_cluster_count=Range(0, 0),
        required_cluster_sizes=(),
        required_cluster_symbols=None,
        # 5 scatters = config.freespin.max_scatters_on_board
        required_scatter_count=Range(5, 5),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=4,
        required_cascade_depth=Range(0, 0),
        cascade_steps=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        payout_range=RangeFloat(0.0, 0.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # trigger_5s_with_booster — 5 scatters + booster-spawning clusters
    # -----------------------------------------------------------------------
    # Premium trigger with booster presence — 5 scatters plus clusters large
    # enough to spawn rockets or bombs. Boosters are dormant (no fires
    # required) — they carry visual excitement into freespin.
    registry.register(ArchetypeSignature(
        id="trigger_5s_with_booster",
        required_cluster_count=Range(1, 2),
        # Clusters large enough to spawn R (9-10) or B (11-12) or larger
        required_cluster_sizes=(Range(9, 14),),
        required_cluster_symbols=None,
        required_scatter_count=Range(5, 5),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        # Cascade needed — booster-spawning clusters require at least 1 step
        required_cascade_depth=Range(1, 3),
        cascade_steps=(
            # Step 0: booster-spawning cluster(s) alongside scatters
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                # Spawn R or B — dormant presence for visual excitement
                must_spawn_booster=("R", "B"),
                must_arm_booster=None,
            ),
        ),
        # Boosters spawn but are not required to fire — dormant visual presence
        required_booster_spawns={"R": Range(0, 1), "B": Range(0, 1)},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        # Moderate payout from the booster-spawning clusters
        payout_range=RangeFloat(0.5, 10.0),
        **base(),
    ))
