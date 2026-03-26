"""Lightball family archetype definitions — 4 archetypes covering direct fire,
cascade integration, and chain reception patterns.

Lightballs spawn from clusters of 13-14 and clear ALL instances of the single
most abundant standard symbol on the board. They cannot initiate chains but
can be chain-triggered by rockets or bombs (R/B from config.chain_initiators).

Target symbol selection uses payout_rank (from config) as tiebreaker when
multiple symbols share the highest count.
"""

from __future__ import annotations

from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier
from .registry import (
    ArchetypeRegistry,
    build_arc_signature,
)


def _lb_arc_base(**overrides: object) -> dict:
    """Shared fields for arc-based lightball archetypes (used with build_arc_signature).

    Only includes fields that build_arc_signature does NOT derive —
    identity, feature flags, and component constraints. All cascade
    structure comes from the NarrativeArc via build_arc_signature.
    """
    defaults = dict(
        family="lb",
        criteria="basegame",
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return defaults


def register_lb_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 4 lightball-family archetypes.

    Organized by experience: direct fire (2, tier-targeted), cascade (1),
    chain-triggered (1).
    """

    # -----------------------------------------------------------------------
    # Direct fire archetypes — LB spawns from 13-14 cluster and fires
    # -----------------------------------------------------------------------

    # LB fires targeting the most abundant LOW-tier symbol — moderate payout
    _lb_fire_low_arc = NarrativeArc(
        phases=(
            # Initial cluster spawns LB
            NarrativePhase(
                id="spawn_lb",
                intent="Large cluster spawns lightball booster.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(13, 14),),
                cluster_symbol_tier=None,
                spawns=("LB",),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # New cluster arms and fires the LB
            NarrativePhase(
                id="arm_fire_lb",
                intent="Cluster arms the lightball which fires immediately.",
                repetitions=Range(1, 2),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("LB",),
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(2.0, 20.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        # Board composition biased so the most abundant symbol is LOW-tier
        lb_target_tier=SymbolTier.LOW,
    )
    registry.register(build_arc_signature(
        _lb_fire_low_arc,
        id="lb_fire_low",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_lb_arc_base(),
    ))

    # LB fires targeting the most abundant HIGH-tier symbol — higher payout
    # because HIGH symbols have larger paytable values
    _lb_fire_high_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_lb",
                intent="Large cluster spawns lightball booster.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(13, 14),),
                cluster_symbol_tier=None,
                spawns=("LB",),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            NarrativePhase(
                id="arm_fire_lb",
                intent="Cluster arms the lightball which fires immediately.",
                repetitions=Range(1, 2),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("LB",),
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(4.0, 35.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        # Board composition biased so the most abundant symbol is HIGH-tier
        lb_target_tier=SymbolTier.HIGH,
    )
    registry.register(build_arc_signature(
        _lb_fire_high_arc,
        id="lb_fire_high",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_lb_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Cascade integration — LB fires during extended cascade
    # -----------------------------------------------------------------------

    # LB fires mid-cascade — any tier target, longer sequence
    _lb_cascade_arc = NarrativeArc(
        phases=(
            # Initial clusters, including one large enough to spawn LB
            NarrativePhase(
                id="spawn_lb",
                intent="Large cluster spawns lightball during cascade opening.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(13, 14),),
                cluster_symbol_tier=None,
                spawns=("LB",),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Cascade continues, building toward LB fire
            NarrativePhase(
                id="cascade_build",
                intent="Mid-cascade clusters sustain momentum before LB fires.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Cluster arms and fires the LB
            NarrativePhase(
                id="arm_fire_lb",
                intent="Cluster arms the lightball which fires immediately.",
                repetitions=Range(1, 2),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("LB",),
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(4.0, 40.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        # No tier bias — LB targets whichever symbol is most abundant
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _lb_cascade_arc,
        id="lb_cascade",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_lb_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Chain-triggered — R or B chain-triggers a dormant LB
    # -----------------------------------------------------------------------

    # Rocket or bomb fires, hits a dormant LB, LB chain-fires and clears a symbol.
    # CONTRACT-SIG-7: chain_depth=1 because two distinct booster types fire.
    _lb_chain_triggered_arc = NarrativeArc(
        phases=(
            # Large cluster spawns chain initiator (R or B) + LB spawns too
            NarrativePhase(
                id="spawn_initiator_and_lb",
                intent="Large cluster spawns both the chain initiator and the lightball.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                spawns=("LB", "R"),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Cluster arms the initiator, which fires and chains into LB
            NarrativePhase(
                id="arm_initiator_chain_lb",
                intent="Cluster arms initiator which fires and chain-triggers dormant LB.",
                repetitions=Range(1, 3),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 10),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("R",),
                # Both initiator and chain target fire during this phase
                fires=("R", "LB"),
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(6.0, 90.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        # Chain depth 1: initiator (R/B) fires then chain-triggers LB
        required_chain_depth=Range(1, 1),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _lb_chain_triggered_arc,
        id="lb_chain_triggered",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_lb_arc_base(),
    ))
