"""SuperLightball family archetype definitions — 4 archetypes covering direct fire,
cascade with multiplier stacking, and chain reception patterns.

SuperLightballs spawn from clusters of 15-49 and clear ALL instances of the two
most abundant standard symbols. Unlike LB, SLB also increments grid multipliers
at every cleared position (values from config.grid_multiplier). They cannot
initiate chains but can be chain-triggered by R/B.

Target symbol selection uses payout_rank (from config) as tiebreaker.
"""

from __future__ import annotations

from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from .registry import (
    ArchetypeRegistry,
    build_arc_signature,
)


def _slb_arc_base(**overrides: object) -> dict:
    """Shared fields for arc-based superlightball archetypes (used with build_arc_signature).

    Only includes fields that build_arc_signature does NOT derive —
    identity, feature flags, and component constraints. All cascade
    structure comes from the NarrativeArc via build_arc_signature.
    """
    defaults = dict(
        family="slb",
        criteria="basegame",
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return defaults


def register_slb_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 4 superlightball-family archetypes.

    Organized by experience: direct fire (1), cascade with elevated mults (1),
    chain-triggered (1), multiplier stacking (1).
    """

    # -----------------------------------------------------------------------
    # Direct fire — SLB spawns from 15+ cluster, arms, fires
    # -----------------------------------------------------------------------

    # SLB fires targeting two most abundant symbols — massive board clear
    _slb_fire_arc = NarrativeArc(
        phases=(
            # Massive cluster spawns SLB
            NarrativePhase(
                id="spawn_slb",
                intent="Massive cluster spawns superlightball booster.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(15, 49),),
                cluster_symbol_tier=None,
                spawns=("SLB",),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # New cluster arms and fires the SLB
            NarrativePhase(
                id="arm_fire_slb",
                intent="Cluster arms the superlightball which fires immediately.",
                repetitions=Range(1, 2),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("SLB",),
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(5.0, 40.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        # SLB doesn't use LB target tier — it always targets the top 2
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _slb_fire_arc,
        id="slb_fire",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_slb_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Cascade with elevated multipliers — SLB fires after grid mults built up
    # -----------------------------------------------------------------------

    # Extended cascade builds grid multipliers before SLB fires — cleared
    # positions already have elevated mults, amplifying the double-symbol wipe
    _slb_cascade_arc = NarrativeArc(
        phases=(
            # Large cluster spawns SLB
            NarrativePhase(
                id="spawn_slb",
                intent="Large cluster spawns superlightball during cascade opening.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(15, 49),),
                cluster_symbol_tier=None,
                spawns=("SLB",),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Cascade builds grid multipliers
            NarrativePhase(
                id="cascade_build_mults",
                intent="Mid-cascade clusters build grid multipliers at cleared positions.",
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
            # Cluster arms and fires SLB on elevated-mult board
            NarrativePhase(
                id="arm_fire_slb",
                intent="Cluster arms SLB which fires on board with elevated grid multipliers.",
                repetitions=Range(1, 3),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("SLB",),
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(10.0, 80.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _slb_cascade_arc,
        id="slb_cascade",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_slb_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Chain-triggered — R or B chain-triggers a dormant SLB
    # -----------------------------------------------------------------------

    # Initiator (R or B) fires and chain-triggers a dormant SLB.
    # CONTRACT-SIG-7: chain_depth=1, two distinct booster types fire.
    _slb_chain_triggered_arc = NarrativeArc(
        phases=(
            # Clusters spawn both the initiator and SLB
            NarrativePhase(
                id="spawn_initiator_and_slb",
                intent="Clusters spawn both the chain initiator and the superlightball.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(9, 49),),
                cluster_symbol_tier=None,
                spawns=("SLB", "R"),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Cluster arms initiator, which fires and chains into SLB
            NarrativePhase(
                id="arm_initiator_chain_slb",
                intent="Cluster arms initiator which fires and chain-triggers dormant SLB.",
                repetitions=Range(1, 3),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 10),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("R",),
                # Both initiator and chain target fire during this phase
                fires=("R", "SLB"),
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(10.0, 130.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        # Chain depth 1: initiator (R/B) fires then chain-triggers SLB
        required_chain_depth=Range(1, 1),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _slb_chain_triggered_arc,
        id="slb_chain_triggered",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_slb_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Multiplier stacking — SLB fires on board with pre-elevated grid mults
    # -----------------------------------------------------------------------

    # Deep cascade builds significant grid multipliers across many positions
    # before SLB fires — the double-symbol wipe inherits the elevated mults,
    # producing the highest payouts in the SLB family
    _slb_multiplier_stack_arc = NarrativeArc(
        phases=(
            # Large cluster spawns SLB
            NarrativePhase(
                id="spawn_slb",
                intent="Large cluster spawns superlightball at cascade start.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(15, 49),),
                cluster_symbol_tier=None,
                spawns=("SLB",),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Cascade with overlapping positions builds grid mults
            NarrativePhase(
                id="cascade_stack_mults",
                intent="Extended cascade overlaps positions to stack grid multipliers.",
                repetitions=Range(2, 2),
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(5, 10),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Cluster arms and fires SLB on heavily-multiplied board
            NarrativePhase(
                id="arm_fire_slb",
                intent="Cluster arms SLB which fires on heavily-multiplied board.",
                repetitions=Range(1, 2),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("SLB",),
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(15.0, 130.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _slb_multiplier_stack_arc,
        id="slb_multiplier_stack",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_slb_arc_base(),
    ))
