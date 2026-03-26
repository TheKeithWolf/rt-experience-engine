"""Trigger family archetype definitions — 4 archetypes covering freespin
trigger scenarios with 4 or 5 scatters, optionally accompanied by wins
or booster spawns.

Trigger archetypes have criteria="freegame" and triggers_freespin=True.
All scatter counts satisfy CONTRACT-SIG-4: scatter_count.min >=
config.freespin.min_scatters_to_trigger.
"""

from __future__ import annotations

from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from .registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    build_arc_signature,
)


def _trigger_base_static() -> dict:
    """Shared fields for depth-0 trigger archetypes (no cascade, no arc)."""
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
        narrative_arc=None,
    )


def _trigger_base_arc() -> dict:
    """Shared fields for arc-based trigger archetypes (used with build_arc_signature).

    Does NOT include fields that build_arc_signature derives from the arc
    (terminal_near_misses, dormant_boosters_on_terminal, rocket_orientation, lb_target_tier).
    """
    return dict(
        family="trigger",
        criteria="freegame",
        triggers_freespin=True,
        reaches_wincap=False,
    )


def register_trigger_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 4 trigger-family archetypes.

    Organized by scatter count and complexity: pure trigger (2),
    trigger with win (1), trigger with booster (1).
    """
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
        **_trigger_base_static(),
    ))

    # -----------------------------------------------------------------------
    # trigger_4s_with_win — 4 scatters + small cluster wins
    # -----------------------------------------------------------------------
    # Trigger accompanied by modest cluster payout. The player gets both
    # a base game win and freespins — feels doubly rewarding.
    _trigger_4s_win_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="scatter_win",
                intent="Small clusters land alongside the scatters.",
                # 0 = static (no cascade), 1-2 = short cascade alongside trigger
                repetitions=Range(0, 2),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None, arms=None, fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.1, 3.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _trigger_4s_win_arc,
        id="trigger_4s_with_win",
        required_cluster_symbols=None,
        required_scatter_count=Range(4, 4),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_trigger_base_arc(),
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
        **_trigger_base_static(),
    ))

    # -----------------------------------------------------------------------
    # trigger_5s_with_booster — 5 scatters + booster-spawning clusters
    # -----------------------------------------------------------------------
    # Premium trigger with booster presence — 5 scatters plus clusters large
    # enough to spawn rockets or bombs. Boosters are dormant (no fires
    # required) — they carry visual excitement into freespin.
    _trigger_5s_booster_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="booster_spawn",
                intent="Large clusters spawn boosters alongside scatters.",
                # 1-3 steps of booster-spawning cascade alongside trigger
                repetitions=Range(1, 3),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                # R or B may spawn — dormant visual presence
                spawns=("R", "B"),
                arms=None, fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.5, 10.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _trigger_5s_booster_arc,
        id="trigger_5s_with_booster",
        required_cluster_symbols=None,
        required_scatter_count=Range(5, 5),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_trigger_base_arc(),
    ))
