"""Lightball family archetype definitions — 4 archetypes covering direct fire,
cascade integration, and chain reception patterns.

Lightballs spawn from clusters of 13-14 and clear ALL instances of the single
most abundant standard symbol on the board. They cannot initiate chains but
can be chain-triggered by rockets or bombs (R/B from config.chain_initiators).

Target symbol selection uses payout_rank (from config) as tiebreaker when
multiple symbols share the highest count.
"""

from __future__ import annotations

from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier
from .registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    CascadeStepConstraint,
)


def _lb_base() -> dict:
    """Shared fields for all lightball archetypes.

    LB family is basegame criteria. No rocket orientation, feature triggers,
    or component caps. lb_target_tier varies per archetype.
    """
    return dict(
        family="lb",
        criteria="basegame",
        rocket_orientation=None,
        triggers_freespin=False,
        reaches_wincap=False,
        # LB produces clusters — no dead-board component cap
        max_component_size=None,
        # No narrative constraints for LB family
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
    )


def register_lb_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 4 lightball-family archetypes.

    Organized by experience: direct fire (2, tier-targeted), cascade (1),
    chain-triggered (1).
    """
    base = _lb_base

    # -----------------------------------------------------------------------
    # Direct fire archetypes — LB spawns from 13-14 cluster and fires
    # -----------------------------------------------------------------------

    # LB fires targeting the most abundant LOW-tier symbol — moderate payout
    registry.register(ArchetypeSignature(
        id="lb_fire_low",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(13, 14),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            # Step 0: initial cluster spawns LB
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(13, 14),),
                cluster_symbol_tier=None,
                must_spawn_booster="LB",
                must_arm_booster=None,
                wild_behavior=None,
            ),
            # Step 1: new cluster arms and fires the LB
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="LB",
                wild_behavior=None,
            ),
        ),
        required_booster_spawns={"LB": Range(1, 1)},
        required_booster_fires={"LB": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        # Board composition biased so the most abundant symbol is LOW-tier
        lb_target_tier=SymbolTier.LOW,
        payout_range=RangeFloat(2.0, 20.0),
        **base(),
    ))

    # LB fires targeting the most abundant HIGH-tier symbol — higher payout
    # because HIGH symbols have larger paytable values
    registry.register(ArchetypeSignature(
        id="lb_fire_high",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(13, 14),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(13, 14),),
                cluster_symbol_tier=None,
                must_spawn_booster="LB",
                must_arm_booster=None,
                wild_behavior=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="LB",
                wild_behavior=None,
            ),
        ),
        required_booster_spawns={"LB": Range(1, 1)},
        required_booster_fires={"LB": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        # Board composition biased so the most abundant symbol is HIGH-tier
        lb_target_tier=SymbolTier.HIGH,
        payout_range=RangeFloat(4.0, 35.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Cascade integration — LB fires during extended cascade
    # -----------------------------------------------------------------------

    # LB fires mid-cascade — any tier target, longer sequence
    registry.register(ArchetypeSignature(
        id="lb_cascade",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 14),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 4),
        cascade_steps=(
            # Step 0: initial clusters, including one large enough to spawn LB
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(13, 14),),
                cluster_symbol_tier=None,
                must_spawn_booster="LB",
                must_arm_booster=None,
                wild_behavior=None,
            ),
            # Step 1: cascade continues, building toward LB fire
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                wild_behavior=None,
            ),
            # Step 2: cluster arms and fires the LB
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="LB",
                wild_behavior=None,
            ),
        ),
        required_booster_spawns={"LB": Range(1, 1)},
        required_booster_fires={"LB": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        # No tier bias — LB targets whichever symbol is most abundant
        lb_target_tier=None,
        payout_range=RangeFloat(4.0, 40.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Chain-triggered — R or B chain-triggers a dormant LB
    # -----------------------------------------------------------------------

    # Rocket or bomb fires, hits a dormant LB, LB chain-fires and clears a symbol.
    # CONTRACT-SIG-7: chain_depth=1 because two distinct booster types fire.
    registry.register(ArchetypeSignature(
        id="lb_chain_triggered",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 14),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            # Step 0: large cluster spawns chain initiator (R or B) + LB spawns too
            CascadeStepConstraint(
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                must_spawn_booster="LB",
                must_arm_booster=None,
                wild_behavior=None,
            ),
            # Step 1: cluster arms the initiator, which fires and chains into LB
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                wild_behavior=None,
            ),
        ),
        required_booster_spawns={"LB": Range(1, 1), "R": Range(1, 1)},
        required_booster_fires={"LB": Range(1, 1), "R": Range(1, 1)},
        # Chain depth 1: initiator (R/B) fires → chain-triggers LB
        required_chain_depth=Range(1, 1),
        lb_target_tier=None,
        payout_range=RangeFloat(6.0, 90.0),
        **base(),
    ))
