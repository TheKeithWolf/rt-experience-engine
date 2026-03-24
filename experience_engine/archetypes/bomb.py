"""Bomb family archetype definitions — 9 archetypes covering idle, fire, chain
compositions, multi-bomb, and near-miss tension patterns.

Bombs spawn from clusters of 11-12 and clear a 3×3 area (Manhattan radius from
config). They are chain initiators — a fired bomb can trigger other boosters
in its blast zone.
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


def _bomb_base() -> dict:
    """Shared fields for all bomb archetypes.

    Bomb family is basegame criteria. No rocket orientation, lightball targeting,
    or feature triggers.
    """
    return dict(
        family="bomb",
        criteria="basegame",
        rocket_orientation=None,
        lb_target_tier=None,
        triggers_freespin=False,
        reaches_wincap=False,
        # Bombs produce clusters — no dead-board component cap
        max_component_size=None,
    )


def register_bomb_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 9 bomb-family archetypes.

    Organized by experience: idle (1), fire (2), chains (4), multi (1),
    near-miss tension (1).
    """
    base = _bomb_base

    # -----------------------------------------------------------------------
    # Core bomb archetypes
    # -----------------------------------------------------------------------

    # Bomb spawns from 11-12 cluster but stays dormant — visual intimidation
    registry.register(ArchetypeSignature(
        id="bomb_idle",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(11, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(1, 2),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        required_booster_spawns={"B": Range(1, 1)},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 15.0),
        **base(),
    ))

    # Bomb fires — 3×3 area clear, cascade continues
    registry.register(ArchetypeSignature(
        id="bomb_fire",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="B",
                must_fire_booster="B",
            ),
        ),
        required_booster_spawns={"B": Range(1, 1)},
        required_booster_fires={"B": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(2.0, 25.0),
        **base(),
    ))

    # Bomb fires but cascade dies immediately — terminal near-misses
    registry.register(ArchetypeSignature(
        id="bomb_fire_dead",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 3),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="B",
                must_fire_booster="B",
            ),
        ),
        required_booster_spawns={"B": Range(1, 1)},
        required_booster_fires={"B": Range(1, 1)},
        required_chain_depth=Range(0, 0),
        symbol_tier_per_step=None,
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2), symbol_tier=None,
        ),
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.5, 20.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Chain archetypes (4) — bomb initiates chain into other boosters
    # -----------------------------------------------------------------------

    # Bomb chain-triggers a rocket — B→R combined clear
    registry.register(ArchetypeSignature(
        id="bomb_chain_rocket",
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
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 10),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="B",
                must_fire_booster="B",
            ),
        ),
        required_booster_spawns={"B": Range(1, 1), "R": Range(1, 1)},
        required_booster_fires={"B": Range(1, 1), "R": Range(1, 1)},
        required_chain_depth=Range(1, 1),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(4.0, 60.0),
        **base(),
    ))

    # Two bombs fire in sequence — first bomb's clear enables second to arm
    # CONTRACT-SIG-7: chain_depth=0 because both are same type (B→B is not a
    # cross-type chain, it's sequential same-type fire across cascade steps)
    registry.register(ArchetypeSignature(
        id="bomb_chain_bomb",
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
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster="B",
                must_fire_booster="B",
            ),
        ),
        required_booster_spawns={"B": Range(2, 2)},
        required_booster_fires={"B": Range(2, 2)},
        required_chain_depth=Range(0, 0),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(5.0, 70.0),
        **base(),
    ))

    # Bomb chain-triggers a lightball — B→LB symbol wipe
    registry.register(ArchetypeSignature(
        id="bomb_chain_lb",
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
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 14),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="B",
                must_fire_booster="B",
            ),
        ),
        required_booster_spawns={"B": Range(1, 1), "LB": Range(1, 1)},
        required_booster_fires={"B": Range(1, 1), "LB": Range(1, 1)},
        required_chain_depth=Range(1, 1),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(6.0, 90.0),
        **base(),
    ))

    # Bomb chain-triggers a superlightball — B→SLB double wipe + multipliers
    registry.register(ArchetypeSignature(
        id="bomb_chain_slb",
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
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 49),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="B",
                must_fire_booster="B",
            ),
        ),
        required_booster_spawns={"B": Range(1, 1), "SLB": Range(1, 1)},
        required_booster_fires={"B": Range(1, 1), "SLB": Range(1, 1)},
        required_chain_depth=Range(1, 1),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(10.0, 130.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Multi-bomb and near-miss tension
    # -----------------------------------------------------------------------

    # Two bombs both fire — double 3×3 clear
    registry.register(ArchetypeSignature(
        id="bomb_multi",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(5, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(2, 2),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                must_spawn_booster=None,
                must_arm_booster="B",
                must_fire_booster="B",
            ),
        ),
        required_booster_spawns={"B": Range(2, 2)},
        required_booster_fires={"B": Range(2, 2)},
        required_chain_depth=Range(0, 0),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(4.0, 50.0),
        **base(),
    ))

    # Bomb sits dormant surrounded by HIGH near-misses — explosive tension
    registry.register(ArchetypeSignature(
        id="bomb_near_miss",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(11, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        required_cascade_depth=Range(1, 2),
        cascade_steps=(
            CascadeStepConstraint(
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                must_spawn_booster="B",
                must_arm_booster=None,
                must_fire_booster=None,
            ),
        ),
        required_booster_spawns={"B": Range(1, 1)},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 12.0),
        **base(),
    ))
