"""Bomb family archetype definitions — 9 archetypes covering idle, fire, chain
compositions, multi-bomb, and near-miss tension patterns.

Bombs spawn from clusters of 11-12 and clear a 3x3 area (Manhattan radius from
config). They are chain initiators — a fired bomb can trigger other boosters
in its blast zone.
"""

from __future__ import annotations

from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier
from .registry import (
    ArchetypeRegistry,
    TerminalNearMissSpec,
    build_arc_signature,
)


def _bomb_arc_base(**overrides: object) -> dict:
    """Shared fields for arc-based bomb archetypes.

    Only includes fields that build_arc_signature does NOT derive —
    identity, feature flags, and component constraints. All cascade
    structure comes from the NarrativeArc via build_arc_signature.
    """
    defaults = dict(
        family="bomb",
        criteria="basegame",
        max_component_size=None,
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Reusable phase factories — avoid repeating identical NarrativePhase defs
# across the 9 archetypes. Each factory encodes one canonical beat in the
# bomb experience arc.
# ---------------------------------------------------------------------------

def _bomb_spawn_phase(
    cluster_count: Range = Range(1, 2),
    *,
    phase_id: str = "spawn_bomb",
    intent: str = "Cluster spawns a bomb",
) -> NarrativePhase:
    """Phase where an 11-12 cluster spawns a bomb."""
    return NarrativePhase(
        id=phase_id,
        intent=intent,
        repetitions=Range(1, 1),
        cluster_count=cluster_count,
        cluster_sizes=(Range(11, 12),),
        cluster_symbol_tier=None,
        spawns=("B",),
        arms=None,
        fires=None,
        wild_behavior=None,
        ends_when="always",
    )


def _bomb_fire_phase(
    cluster_sizes: tuple[Range, ...] = (Range(5, 6),),
    cluster_count: Range = Range(1, 2),
    *,
    phase_id: str = "fire_bomb",
    intent: str = "Cluster arms the bomb, which fires",
) -> NarrativePhase:
    """Phase where a new cluster arms and fires the bomb."""
    return NarrativePhase(
        id=phase_id,
        intent=intent,
        repetitions=Range(1, 1),
        cluster_count=cluster_count,
        cluster_sizes=cluster_sizes,
        cluster_symbol_tier=None,
        spawns=None,
        arms=("B",),
        fires=("B",),
        wild_behavior=None,
        ends_when="always",
    )


def register_bomb_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 9 bomb-family archetypes.

    Organized by experience: idle (1), fire (2), chains (4), multi (1),
    near-miss tension (1).
    """
    arc_base = _bomb_arc_base

    # -----------------------------------------------------------------------
    # Core bomb archetypes
    # -----------------------------------------------------------------------

    # Bomb spawns from 11-12 cluster but stays dormant — visual intimidation
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _bomb_spawn_phase(),
            ),
            payout=RangeFloat(1.0, 15.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="bomb_idle",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Bomb fires — 3x3 area clear, cascade continues
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _bomb_spawn_phase(),
                _bomb_fire_phase(),
            ),
            payout=RangeFloat(2.0, 25.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="bomb_fire",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Bomb fires but cascade dies immediately — terminal near-misses
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _bomb_spawn_phase(),
                _bomb_fire_phase(),
            ),
            payout=RangeFloat(1.5, 20.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=TerminalNearMissSpec(
                count=Range(1, 2), symbol_tier=None,
            ),
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="bomb_fire_dead",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Chain archetypes (4) — bomb initiates chain into other boosters
    # -----------------------------------------------------------------------

    # Bomb chain-triggers a rocket — B→R combined clear
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _bomb_spawn_phase(),
                # Bomb fires and its blast zone hits a dormant rocket, causing both to fire
                NarrativePhase(
                    id="fire_bomb_chain_rocket",
                    intent="Cluster arms bomb, blast zone chain-triggers dormant rocket",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(5, 10),),
                    cluster_symbol_tier=None,
                    spawns=None,
                    arms=("B",),
                    fires=("B", "R"),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(4.0, 60.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(1, 1),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="bomb_chain_rocket",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Two bombs fire in sequence — first bomb's clear enables second to arm
    # CONTRACT-SIG-7: chain_depth=0 because both are same type (B→B is not a
    # cross-type chain, it's sequential same-type fire across cascade steps)
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _bomb_spawn_phase(),
                # Second bomb spawns from a different cluster and arms the first
                NarrativePhase(
                    id="spawn_fire_bomb",
                    intent="Second bomb spawns while first fires",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(11, 12),),
                    cluster_symbol_tier=None,
                    spawns=("B",),
                    arms=("B",),
                    fires=("B",),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(5.0, 70.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="bomb_chain_bomb",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Bomb chain-triggers a lightball — B→LB symbol wipe
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _bomb_spawn_phase(),
                NarrativePhase(
                    id="fire_bomb_chain_lb",
                    intent="Cluster arms bomb, blast zone chain-triggers lightball",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(5, 14),),
                    cluster_symbol_tier=None,
                    spawns=None,
                    arms=("B",),
                    fires=("B", "LB"),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(6.0, 90.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(1, 1),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="bomb_chain_lb",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Bomb chain-triggers a superlightball — B→SLB double wipe + multipliers
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _bomb_spawn_phase(),
                NarrativePhase(
                    id="fire_bomb_chain_slb",
                    intent="Cluster arms bomb, blast zone chain-triggers superlightball",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(5, 49),),
                    cluster_symbol_tier=None,
                    spawns=None,
                    arms=("B",),
                    fires=("B", "SLB"),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(10.0, 130.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(1, 1),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="bomb_chain_slb",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Multi-bomb and near-miss tension
    # -----------------------------------------------------------------------

    # Two bombs both fire — double 3x3 clear
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _bomb_spawn_phase(
                    cluster_count=Range(2, 2),
                    phase_id="dual_spawn_bomb",
                    intent="Two clusters each spawn a bomb",
                ),
                _bomb_fire_phase(
                    phase_id="dual_fire_bomb",
                    intent="New cluster arms both bombs, they fire",
                ),
            ),
            payout=RangeFloat(4.0, 50.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="bomb_multi",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Bomb sits dormant surrounded by HIGH near-misses — explosive tension
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _bomb_spawn_phase(),
            ),
            payout=RangeFloat(1.0, 12.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="bomb_near_miss",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        **arc_base(),
    ))
