"""Chain family archetype definitions — 5 archetypes covering multi-booster
compositions, deep chains, parallel independent fires, organic cascade-to-booster
growth, and multi-phase booster sequences.

Chain archetypes are defined by the *interaction pattern* between boosters rather
than a specific booster type. They compose rockets, bombs, lightballs, and
superlightballs into complex multi-step sequences.

Chain depth semantics: chain_depth counts chain-triggered targets (ASP convention).
B→R = chain_depth 1 (one target). R→B→R = chain_depth 2 (two targets).
"""

from __future__ import annotations

from ..pipeline.protocols import Range, RangeFloat
from .registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    CascadeStepConstraint,
)


def _chain_base() -> dict:
    """Shared fields for all chain archetypes.

    Chain family is basegame criteria. No rocket orientation constraint (chains
    may contain rockets with either orientation), no lightball tier targeting,
    no feature triggers or wincap.
    """
    return dict(
        family="chain",
        criteria="basegame",
        rocket_orientation=None,
        lb_target_tier=None,
        triggers_freespin=False,
        reaches_wincap=False,
        # Chain archetypes produce clusters — no dead-board component cap
        max_component_size=None,
    )


def register_chain_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 5 chain-family archetypes.

    Organized by interaction pattern: cross-type chain (1), deep chain (1),
    parallel independent (1), organic growth (1), multi-phase (1).
    """
    base = _chain_base

    # -----------------------------------------------------------------------
    # chain_2_mixed — Two different booster types in a depth-2 chain
    # -----------------------------------------------------------------------
    # Simple multi-booster chain: e.g., R spawns, B spawns on a later step,
    # both arm and fire in sequence triggering each other (R→B→R or B→R→B).
    # The player sees two different boosters interact — introduces the concept
    # that boosters can trigger each other.
    registry.register(ArchetypeSignature(
        id="chain_2_mixed",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(9, 14),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 8),
        cascade_steps=(
            # Step 0: spawn first booster via 9-10 (rocket) or 11-12 (bomb) cluster
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                must_spawn_booster=("R", "B"),
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        # Both rocket and bomb must spawn and fire — ASP chooses which initiates
        required_booster_spawns={"R": Range(1, 2), "B": Range(1, 2)},
        required_booster_fires={"R": Range(1, 2), "B": Range(1, 2)},
        # Depth 2 = two chain-triggered targets (e.g., R→B→R)
        required_chain_depth=Range(2, 2),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(5.0, 200.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # chain_3_plus — Deep chain of 3-6 links, at least 1 rocket + 1 bomb
    # -----------------------------------------------------------------------
    # Extended multi-booster cascade: a sustained sequence of boosters triggering
    # each other. High drama, high payout. At least one rocket and one bomb
    # participate — ASP plans the full chain depth.
    registry.register(ArchetypeSignature(
        id="chain_3_plus",
        required_cluster_count=Range(1, 4),
        required_cluster_sizes=(Range(9, 14),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 12),
        # Let ASP plan freely — only constrain step 0 to spawn the first booster
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                must_spawn_booster=("R", "B"),
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        # Flexible spawn/fire counts — ASP fills to meet chain depth requirement
        required_booster_spawns={"R": Range(1, 4), "B": Range(1, 4)},
        required_booster_fires={"R": Range(1, 4), "B": Range(1, 4)},
        # Deep chain: 3-6 chain-triggered targets (4-7 boosters total)
        required_chain_depth=Range(3, 6),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(10.0, 500.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # multi_booster_parallel — 2+ boosters fire independently, no chain
    # -----------------------------------------------------------------------
    # Simultaneous action: multiple independent booster events on the same spin.
    # Boosters don't trigger each other — they coexist. chain_depth=0 triggers
    # the forbid_chain ASP constraint to prevent accidental chain links.
    registry.register(ArchetypeSignature(
        id="multi_booster_parallel",
        required_cluster_count=Range(2, 4),
        required_cluster_sizes=(Range(9, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 8),
        cascade_steps=(
            # Step 0: spawn both boosters from separate clusters
            CascadeStepConstraint(
                cluster_count=Range(2, 4),
                cluster_sizes=(Range(9, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster=("R", "B"),
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        required_booster_spawns={"R": Range(1, 2), "B": Range(1, 2)},
        required_booster_fires={"R": Range(1, 2), "B": Range(1, 2)},
        # Zero chains — boosters fire independently (triggers forbid_chain in ASP)
        required_chain_depth=Range(0, 0),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(5.0, 200.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # cascade_to_booster_to_cascade — Organic growth from t1 to booster
    # -----------------------------------------------------------------------
    # Small clusters (5-6) cascade and grow until they reach booster spawn
    # threshold (9+). The booster fires, refill cascades again. The booster
    # emerges naturally from cascading — feels earned rather than given.
    registry.register(ArchetypeSignature(
        id="cascade_to_booster_to_cascade",
        required_cluster_count=Range(2, 5),
        # Initial clusters are small t1-level — size constraint is on step 0
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=None,
        # No scatters — focus is on the organic cascade experience
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(4, 10),
        cascade_steps=(
            # Step 0: t1-level small clusters — cascade beginning
            CascadeStepConstraint(
                cluster_count=Range(2, 3),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Step 1: continued small cascade
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Step 2: transition — clusters grow toward booster threshold
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Step 3: growth hits booster threshold — organic spawn
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                must_spawn_booster=("R", "B"),
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        # Single booster spawns from the organic growth step — R or B
        required_booster_spawns={"R": Range(0, 1), "B": Range(0, 1)},
        required_booster_fires={"R": Range(0, 1), "B": Range(0, 1)},
        # No chain — single booster fires
        required_chain_depth=Range(0, 0),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(3.0, 150.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # booster_phase_multi — Multiple booster phases in a single spin
    # -----------------------------------------------------------------------
    # Episodic multi-phase spin: cascade → booster phase 1 → cascade → booster
    # phase 2 → terminal dead. The spin feels like it has chapters — build,
    # explode, rebuild, explode again. Each phase gets its own booster type.
    registry.register(ArchetypeSignature(
        id="booster_phase_multi",
        required_cluster_count=Range(2, 5),
        required_cluster_sizes=(Range(9, 14),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(4, 15),
        cascade_steps=(
            # Act 1: spawn rocket from 9-10 cluster
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Act 1 climax: arm + fire rocket (booster phase 1)
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
            # Act 2: spawn bomb from 11-12 cluster
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Act 2 climax: arm + fire bomb (booster phase 2)
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="B",
                must_fire_booster="B",
            ),
        ),
        # Both boosters must spawn and fire — in separate phases, not chained
        required_booster_spawns={"R": Range(1, 2), "B": Range(1, 2)},
        required_booster_fires={"R": Range(1, 2), "B": Range(1, 2)},
        # Zero chain depth — phases are independent, not chain-linked
        required_chain_depth=Range(0, 0),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(10.0, 500.0),
        **base(),
    ))
