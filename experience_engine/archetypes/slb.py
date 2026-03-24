"""SuperLightball family archetype definitions — 4 archetypes covering direct fire,
cascade with multiplier stacking, and chain reception patterns.

SuperLightballs spawn from clusters of 15-49 and clear ALL instances of the two
most abundant standard symbols. Unlike LB, SLB also increments grid multipliers
at every cleared position (values from config.grid_multiplier). They cannot
initiate chains but can be chain-triggered by R/B.

Target symbol selection uses payout_rank (from config) as tiebreaker.
"""

from __future__ import annotations

from ..pipeline.protocols import Range, RangeFloat
from .registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    CascadeStepConstraint,
)


def _slb_base() -> dict:
    """Shared fields for all superlightball archetypes.

    SLB family is basegame criteria. No rocket orientation, LB target tier,
    feature triggers, or component caps.
    """
    return dict(
        family="slb",
        criteria="basegame",
        rocket_orientation=None,
        # SLB doesn't use LB target tier — it always targets the top 2
        lb_target_tier=None,
        triggers_freespin=False,
        reaches_wincap=False,
        # SLB produces clusters — no dead-board component cap
        max_component_size=None,
        # No narrative constraints for SLB family
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
    )


def register_slb_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 4 superlightball-family archetypes.

    Organized by experience: direct fire (1), cascade with elevated mults (1),
    chain-triggered (1), multiplier stacking (1).
    """
    base = _slb_base

    # -----------------------------------------------------------------------
    # Direct fire — SLB spawns from 15+ cluster, arms, fires
    # -----------------------------------------------------------------------

    # SLB fires targeting two most abundant symbols — massive board clear
    registry.register(ArchetypeSignature(
        id="slb_fire",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(15, 49),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            # Step 0: massive cluster spawns SLB
            CascadeStepConstraint(
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(15, 49),),
                cluster_symbol_tier=None,
                must_spawn_booster="SLB",
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior=None,
            ),
            # Step 1: new cluster arms and fires the SLB
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="SLB",
                must_fire_booster="SLB",
                wild_behavior=None,
            ),
        ),
        required_booster_spawns={"SLB": Range(1, 1)},
        required_booster_fires={"SLB": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        payout_range=RangeFloat(5.0, 40.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Cascade with elevated multipliers — SLB fires after grid mults built up
    # -----------------------------------------------------------------------

    # Extended cascade builds grid multipliers before SLB fires — cleared
    # positions already have elevated mults, amplifying the double-symbol wipe
    registry.register(ArchetypeSignature(
        id="slb_cascade",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 49),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=(
            # Step 0: large cluster spawns SLB
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(15, 49),),
                cluster_symbol_tier=None,
                must_spawn_booster="SLB",
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior=None,
            ),
            # Step 1: cascade builds grid multipliers
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior=None,
            ),
            # Step 2: cluster arms and fires SLB on elevated-mult board
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="SLB",
                must_fire_booster="SLB",
                wild_behavior=None,
            ),
        ),
        required_booster_spawns={"SLB": Range(1, 1)},
        required_booster_fires={"SLB": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        payout_range=RangeFloat(10.0, 80.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Chain-triggered — R or B chain-triggers a dormant SLB
    # -----------------------------------------------------------------------

    # Initiator (R or B) fires and chain-triggers a dormant SLB.
    # CONTRACT-SIG-7: chain_depth=1, two distinct booster types fire.
    registry.register(ArchetypeSignature(
        id="slb_chain_triggered",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 49),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            # Step 0: clusters spawn both the initiator and SLB
            CascadeStepConstraint(
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(9, 49),),
                cluster_symbol_tier=None,
                must_spawn_booster="SLB",
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior=None,
            ),
            # Step 1: cluster arms initiator, which fires and chains into SLB
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
                wild_behavior=None,
            ),
        ),
        required_booster_spawns={"SLB": Range(1, 1), "R": Range(1, 1)},
        required_booster_fires={"SLB": Range(1, 1), "R": Range(1, 1)},
        required_chain_depth=Range(1, 1),
        payout_range=RangeFloat(10.0, 130.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Multiplier stacking — SLB fires on board with pre-elevated grid mults
    # -----------------------------------------------------------------------

    # Deep cascade builds significant grid multipliers across many positions
    # before SLB fires — the double-symbol wipe inherits the elevated mults,
    # producing the highest payouts in the SLB family
    registry.register(ArchetypeSignature(
        id="slb_multiplier_stack",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 49),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=(
            # Step 0: large cluster spawns SLB
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(15, 49),),
                cluster_symbol_tier=None,
                must_spawn_booster="SLB",
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior=None,
            ),
            # Step 1: cascade with overlapping positions builds grid mults
            CascadeStepConstraint(
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(5, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior=None,
            ),
            # Step 2: more cascade for mult accumulation
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
                wild_behavior=None,
            ),
            # Step 3: cluster arms and fires SLB on heavily-multiplied board
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="SLB",
                must_fire_booster="SLB",
                wild_behavior=None,
            ),
        ),
        required_booster_spawns={"SLB": Range(1, 1)},
        required_booster_fires={"SLB": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        payout_range=RangeFloat(15.0, 130.0),
        **base(),
    ))
