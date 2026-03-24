"""Rocket family archetype definitions — 22 archetypes covering core fire patterns,
chain compositions, narrative arcs (near-miss tension, late saves, storms, fakeouts),
and scatter tension.

Rockets spawn from clusters of 9-10 and clear entire rows (H) or columns (V).
They are chain initiators — a fired rocket can trigger other boosters in its path.
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


def _rocket_base() -> dict:
    """Shared fields for all rocket archetypes.

    Rocket family is basegame criteria. No lightball targeting or feature triggers.
    """
    return dict(
        family="rocket",
        criteria="basegame",
        lb_target_tier=None,
        triggers_freespin=False,
        reaches_wincap=False,
        # Rockets produce clusters — no dead-board component cap
        max_component_size=None,
    )


def register_rocket_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 22 rocket-family archetypes.

    Organized by experience category: core (10), near-miss tension (3),
    late saves (2), storms (2), scatter tension (2), fakeouts (3).
    """
    base = _rocket_base

    # -----------------------------------------------------------------------
    # Core rocket archetypes (10)
    # -----------------------------------------------------------------------

    # Rocket spawns from 9-10 cluster but stays dormant — visual presence only
    registry.register(ArchetypeSignature(
        id="rocket_idle",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(9, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(1, 2),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.5, 12.0),
        **base(),
    ))

    # Rocket fires horizontally — clears entire row
    registry.register(ArchetypeSignature(
        id="rocket_h_fire",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            # Step 0: spawn the rocket from a 9-10 cluster
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Step 1: new cluster arms the rocket, it fires H
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation="H",
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 20.0),
        **base(),
    ))

    # Rocket fires vertically — clears entire column
    registry.register(ArchetypeSignature(
        id="rocket_v_fire",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation="V",
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 20.0),
        **base(),
    ))

    # Rocket fires but cascade immediately dies — short burst then nothing
    registry.register(ArchetypeSignature(
        id="rocket_fire_dead",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2), symbol_tier=SymbolTier.LOW,
        ),
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 15.0),
        **base(),
    ))

    # Rocket fires and clears feed a deeper cascade continuation
    registry.register(ArchetypeSignature(
        id="rocket_fire_cascade",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(2.0, 40.0),
        **base(),
    ))

    # Two rockets fire in sequence — first rocket's clear enables second to arm
    # CONTRACT-SIG-7: chain_depth=0 because both are same type (R→R is not a
    # cross-type chain, it's sequential same-type fire across cascade steps)
    registry.register(ArchetypeSignature(
        id="rocket_chain_rocket",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Second rocket spawns from a different cluster
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(2, 2)},
        required_booster_fires={"R": Range(2, 2)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(3.0, 50.0),
        **base(),
    ))

    # Rocket chain-triggers a bomb — R→B combined clear
    registry.register(ArchetypeSignature(
        id="rocket_chain_bomb",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            # Spawn both rocket (9-10) and bomb (11-12) across steps
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1), "B": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1), "B": Range(1, 1)},
        required_chain_depth=Range(1, 1),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(4.0, 60.0),
        **base(),
    ))

    # Rocket chain-triggers a lightball — R→LB symbol wipe
    registry.register(ArchetypeSignature(
        id="rocket_chain_lb",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 14),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 14),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1), "LB": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1), "LB": Range(1, 1)},
        required_chain_depth=Range(1, 1),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(5.0, 80.0),
        **base(),
    ))

    # Rocket chain-triggers a superlightball — R→SLB double wipe + multipliers
    registry.register(ArchetypeSignature(
        id="rocket_chain_slb",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 49),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 49),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1), "SLB": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1), "SLB": Range(1, 1)},
        required_chain_depth=Range(1, 1),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(8.0, 120.0),
        **base(),
    ))

    # Two rockets both fire independently — parallel row/column clears
    registry.register(ArchetypeSignature(
        id="rocket_dual",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(2, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(2, 2)},
        required_booster_fires={"R": Range(2, 2)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(3.0, 50.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Near-miss tension (3) — rocket presence amplifies near-miss frustration
    # -----------------------------------------------------------------------

    # Rocket spawns dormant near HIGH near-misses — "so close" feeling
    registry.register(ArchetypeSignature(
        id="rocket_near_miss_spawn",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(9, 10),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        required_cascade_depth=Range(1, 2),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=SymbolTier.LOW,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.5, 8.0),
        **base(),
    ))

    # Rocket fires H but bomb sits on adjacent row — orientation near-miss
    registry.register(ArchetypeSignature(
        id="rocket_near_miss_orientation",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 1),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1), "B": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation="H",
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        # Bomb remains dormant — rocket's H path missed it
        dormant_boosters_on_terminal=("B",),
        payout_range=RangeFloat(1.0, 15.0),
        **base(),
    ))

    # Two rockets idle surrounded by HIGH near-misses — maximum visual tension
    registry.register(ArchetypeSignature(
        id="rocket_near_miss_dual_idle",
        required_cluster_count=Range(2, 2),
        required_cluster_sizes=(Range(9, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(2, 3),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        required_cascade_depth=Range(1, 2),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(2, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        required_booster_spawns={"R": Range(2, 2)},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 15.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Late-stage saves (2) — LOW cascade fades, late rocket rescues the run
    # -----------------------------------------------------------------------

    # LOW cascade fading out → late rocket fire saves the payout
    registry.register(ArchetypeSignature(
        id="rocket_late_save",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=(
            # Early steps: LOW clusters only
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=SymbolTier.LOW,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Late step: rocket fires to rescue
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        # Narrative arc: early LOW, late unconstrained
        symbol_tier_per_step={0: SymbolTier.LOW, 1: SymbolTier.LOW},
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(2.0, 30.0),
        **base(),
    ))

    # Bomb spawns early, dormant — late rocket chain-triggers it for combo
    registry.register(ArchetypeSignature(
        id="rocket_late_save_chain",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=(
            # Step 0: LOW cluster
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=SymbolTier.LOW,
                must_spawn_booster=None,
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Step 1: bomb spawns early from 11-12 cluster
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            # Step 2+: rocket spawns, fires, and chains through the bomb
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1), "B": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1), "B": Range(1, 1)},
        required_chain_depth=Range(1, 1),
        rocket_orientation=None,
        # Narrative: early LOW → late unconstrained (bomb/rocket fireworks)
        symbol_tier_per_step={0: SymbolTier.LOW},
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(4.0, 60.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Cascade storms (2) — multiple rockets fire in sequence
    # -----------------------------------------------------------------------

    # 2-3 rockets fire during a LOW cascade — sustained low-value action
    registry.register(ArchetypeSignature(
        id="rocket_storm_low",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=SymbolTier.LOW,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=SymbolTier.LOW,
                must_spawn_booster="R",
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(2, 3)},
        required_booster_fires={"R": Range(2, 3)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(2.0, 30.0),
        **base(),
    ))

    # One H + one V rocket = cross pattern — dramatic visual
    registry.register(ArchetypeSignature(
        id="rocket_storm_cross",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(2, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(2, 2)},
        required_booster_fires={"R": Range(2, 2)},
        required_chain_depth=Range(0, 0),
        # No fixed orientation — one H and one V emerge from cluster shapes
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(3.0, 50.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Scatter tension (2) — rockets + scatters create dual excitement
    # -----------------------------------------------------------------------

    # Rocket + 3 scatters — near-trigger with booster action
    registry.register(ArchetypeSignature(
        id="rocket_scatter_tease",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 20.0),
        **base(),
    ))

    # Multiple rockets + 3 scatters + LOW clusters — chaotic tease
    registry.register(ArchetypeSignature(
        id="rocket_scatter_storm_tease",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=SymbolTier.LOW,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=SymbolTier.LOW,
                must_spawn_booster="R",
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(2, 2)},
        required_booster_fires={"R": Range(2, 2)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(2.0, 30.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Escalation fakeouts (3) — buildup that fizzles at the end
    # -----------------------------------------------------------------------

    # Rocket fires → cascade dies → HIGH near-misses on terminal board
    registry.register(ArchetypeSignature(
        id="rocket_escalation_fizzle",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2), symbol_tier=SymbolTier.HIGH,
        ),
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 15.0),
        **base(),
    ))

    # R→B chain starts but cascade dies afterwards + near-misses
    registry.register(ArchetypeSignature(
        id="rocket_chain_fizzle",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1), "B": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1), "B": Range(1, 1)},
        required_chain_depth=Range(1, 1),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2), symbol_tier=None,
        ),
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(3.0, 40.0),
        **base(),
    ))

    # Rocket fires → refill creates a dormant bomb → cascade dies with bomb visible
    registry.register(ArchetypeSignature(
        id="rocket_bomb_tease",
        required_cluster_count=Range(1, 3),
        required_cluster_sizes=(Range(5, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster="R",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="R",
                must_fire_booster="R",
            ),
        ),
        required_booster_spawns={"R": Range(1, 1), "B": Range(1, 1)},
        required_booster_fires={"R": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        # Bomb stays dormant and visible on the final dead board — tease
        dormant_boosters_on_terminal=("B",),
        payout_range=RangeFloat(1.0, 20.0),
        **base(),
    ))
