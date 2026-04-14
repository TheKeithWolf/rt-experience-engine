"""Rocket family archetype definitions — 22 archetypes covering core fire patterns,
chain compositions, narrative arcs (near-miss tension, late saves, storms, fakeouts),
and scatter tension.

Rockets spawn from clusters of 9-10 and clear entire rows (H) or columns (V).
They are chain initiators — a fired rocket can trigger other boosters in its path.
"""

from __future__ import annotations

from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier
from .registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    TerminalNearMissSpec,
    build_arc_signature,
)


def _rocket_base() -> dict:
    """Shared fields for all rocket archetypes that use direct ArchetypeSignature.

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


def _rocket_arc_base(**overrides: object) -> dict:
    """Shared fields for arc-based rocket archetypes.

    Only includes fields that build_arc_signature does NOT derive —
    identity, feature flags, and component constraints. All cascade
    structure comes from the NarrativeArc via build_arc_signature.
    """
    defaults = dict(
        family="rocket",
        criteria="basegame",
        max_component_size=None,
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Reusable phase factories — avoid repeating identical NarrativePhase defs
# across the 22 archetypes. Each factory encodes one canonical beat in the
# rocket experience arc.
# ---------------------------------------------------------------------------

def _rocket_spawn_phase(
    cluster_count: Range = Range(1, 2),
    cluster_symbol_tier: SymbolTier | None = None,
    *,
    phase_id: str = "spawn_rocket",
    intent: str = "Cluster spawns a rocket",
) -> NarrativePhase:
    """Phase where a 9-10 cluster spawns a rocket."""
    return NarrativePhase(
        id=phase_id,
        intent=intent,
        repetitions=Range(1, 1),
        cluster_count=cluster_count,
        cluster_sizes=(Range(9, 10),),
        cluster_symbol_tier=cluster_symbol_tier,
        spawns=("R",),
        arms=None,
        fires=None,
        wild_behavior=None,
        ends_when="always",
    )


def _rocket_fire_phase(
    cluster_sizes: tuple[Range, ...] = (Range(5, 6),),
    cluster_count: Range = Range(1, 2),
    *,
    phase_id: str = "fire_rocket",
    intent: str = "Cluster arms the rocket, which fires",
) -> NarrativePhase:
    """Phase where a new cluster arms and fires the rocket."""
    return NarrativePhase(
        id=phase_id,
        intent=intent,
        repetitions=Range(1, 1),
        cluster_count=cluster_count,
        cluster_sizes=cluster_sizes,
        cluster_symbol_tier=None,
        spawns=None,
        arms=("R",),
        fires=("R",),
        wild_behavior=None,
        ends_when="always",
    )


# Size ranges for chain-target spawn phases, derived from the same game rule
# as boosters.spawn_thresholds in default.yaml (B fires at 11-12, LB at 13-14,
# SLB at 15+). SLB capped at 16 so the phase fits within board space alongside
# the rocket spawn and arming cluster. SpawnEvaluator resolves the runtime
# mapping; this mirror exists only because NarrativePhase construction happens
# at import time, before the evaluator is available.
_CHAIN_TARGET_SIZES: dict[str, Range] = {
    "B": Range(11, 12),
    "LB": Range(13, 14),
    "SLB": Range(15, 16),
}


def _chain_target_spawn_phase(
    target_booster: str,
    *,
    phase_id: str = "spawn_chain_target",
    intent: str = "Cluster spawns the chain-target booster",
) -> NarrativePhase:
    """Phase where a cluster spawns the booster that a rocket will chain-trigger.

    Used by chain archetypes (rocket_chain_{bomb,fizzle,lb,slb}) to guarantee
    the target booster exists before the rocket fires, and by tease archetypes
    (rocket_bomb_tease, rocket_near_miss_orientation) to guarantee a dormant
    bomb appears on the terminal board.
    """
    return NarrativePhase(
        id=phase_id,
        intent=intent,
        repetitions=Range(1, 1),
        cluster_count=Range(1, 1),
        cluster_sizes=(_CHAIN_TARGET_SIZES[target_booster],),
        cluster_symbol_tier=None,
        spawns=(target_booster,),
        arms=None,
        fires=None,
        wild_behavior=None,
        ends_when="always",
    )


def _low_cluster_phase(
    *,
    phase_id: str = "low_cluster",
    intent: str = "LOW-tier cluster filler — modest value before the payoff",
    cluster_sizes: tuple[Range, ...] = (Range(5, 6),),
) -> NarrativePhase:
    """Phase of LOW-tier filler clusters with no booster activity.

    cluster_sizes override exists so space-tight arcs (late_save_chain) can
    tighten to the minimum (5-5) without forking the factory.
    """
    return NarrativePhase(
        id=phase_id,
        intent=intent,
        repetitions=Range(1, 1),
        cluster_count=Range(1, 2),
        cluster_sizes=cluster_sizes,
        cluster_symbol_tier=SymbolTier.LOW,
        spawns=None,
        arms=None,
        fires=None,
        wild_behavior=None,
        ends_when="always",
    )


def register_rocket_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 22 rocket-family archetypes.

    Organized by experience category: core (10), near-miss tension (3),
    late saves (2), storms (2), scatter tension (2), fakeouts (3).
    """
    arc_base = _rocket_arc_base

    # -----------------------------------------------------------------------
    # Core rocket archetypes (10)
    # -----------------------------------------------------------------------

    # Rocket spawns from 9-10 cluster but stays dormant — visual presence only
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
            ),
            payout=RangeFloat(0.5, 12.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_idle",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Rocket fires horizontally — clears entire row
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _rocket_fire_phase(),
            ),
            payout=RangeFloat(1.0, 20.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation="H",
            lb_target_tier=None,
        ),
        id="rocket_h_fire",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Rocket fires vertically — clears entire column
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _rocket_fire_phase(),
            ),
            payout=RangeFloat(1.0, 20.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation="V",
            lb_target_tier=None,
        ),
        id="rocket_v_fire",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Rocket fires but cascade immediately dies — short burst then nothing
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _rocket_fire_phase(),
            ),
            payout=RangeFloat(1.0, 15.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=TerminalNearMissSpec(
                count=Range(1, 2), symbol_tier=SymbolTier.LOW,
            ),
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_fire_dead",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Rocket fires and clears feed a deeper cascade continuation
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _rocket_fire_phase(),
            ),
            payout=RangeFloat(2.0, 40.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_fire_cascade",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Two rockets fire in sequence — first rocket's clear enables second to arm
    # CONTRACT-SIG-7: chain_depth=0 because both are same type (R→R is not a
    # cross-type chain, it's sequential same-type fire across cascade steps)
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                # Second rocket spawns from a different cluster and arms the first
                NarrativePhase(
                    id="spawn_fire_rocket",
                    intent="Second rocket spawns while first fires",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(9, 10),),
                    cluster_symbol_tier=None,
                    spawns=("R",),
                    arms=("R",),
                    fires=("R",),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(3.0, 50.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_chain_rocket",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Rocket chain-triggers a bomb — R→B combined clear
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                # Bomb spawns from a dedicated cluster so the rocket's fire
                # path has a real target — previously relied on accidental
                # post-refill spawns that almost never materialised.
                _chain_target_spawn_phase(
                    "B",
                    phase_id="spawn_chain_bomb",
                    intent="Cluster spawns the bomb that the rocket will chain-trigger",
                ),
                # Rocket fires and its path hits the dormant bomb, causing both to fire
                NarrativePhase(
                    id="fire_rocket_chain_bomb",
                    intent="Cluster arms rocket, fire path chain-triggers dormant bomb",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    # Arming cluster only — bomb already spawned in prior phase
                    cluster_sizes=(Range(5, 10),),
                    cluster_symbol_tier=None,
                    spawns=None,
                    arms=("R",),
                    fires=("R", "B"),
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
        id="rocket_chain_bomb",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Rocket chain-triggers a lightball — R→LB symbol wipe
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _chain_target_spawn_phase(
                    "LB",
                    phase_id="spawn_chain_lb",
                    intent="Cluster spawns the lightball that the rocket will chain-trigger",
                ),
                NarrativePhase(
                    id="fire_rocket_chain_lb",
                    intent="Cluster arms rocket, fire path chain-triggers lightball",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(5, 10),),
                    cluster_symbol_tier=None,
                    spawns=None,
                    arms=("R",),
                    fires=("R", "LB"),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(5.0, 80.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(1, 1),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_chain_lb",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Rocket chain-triggers a superlightball — R→SLB double wipe + multipliers
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _chain_target_spawn_phase(
                    "SLB",
                    phase_id="spawn_chain_slb",
                    intent="Cluster spawns the superlightball that the rocket will chain-trigger",
                ),
                NarrativePhase(
                    id="fire_rocket_chain_slb",
                    intent="Cluster arms rocket, fire path chain-triggers superlightball",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(5, 10),),
                    cluster_symbol_tier=None,
                    spawns=None,
                    arms=("R",),
                    fires=("R", "SLB"),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(8.0, 120.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(1, 1),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_chain_slb",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Two rockets both fire independently — parallel row/column clears
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                # Both rockets spawn in the same step from a 2-cluster reveal
                _rocket_spawn_phase(
                    cluster_count=Range(2, 2),
                    phase_id="dual_spawn_rocket",
                    intent="Two clusters each spawn a rocket",
                ),
                _rocket_fire_phase(
                    phase_id="dual_fire_rocket",
                    intent="New cluster arms both rockets, they fire",
                ),
            ),
            payout=RangeFloat(3.0, 50.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_dual",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Near-miss tension (3) — rocket presence amplifies near-miss frustration
    # -----------------------------------------------------------------------

    # Rocket spawns dormant near HIGH near-misses — "so close" feeling
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(
                    cluster_symbol_tier=SymbolTier.LOW,
                    phase_id="spawn_rocket_low",
                    intent="LOW-tier cluster spawns rocket near HIGH near-misses",
                ),
            ),
            payout=RangeFloat(0.5, 8.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_near_miss_spawn",
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        **arc_base(),
    ))

    # Rocket fires H but bomb sits on adjacent row — orientation near-miss
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                # Bomb spawned deliberately before the H-fire so the near-miss
                # geometry (bomb on the adjacent row) can actually materialise.
                _chain_target_spawn_phase(
                    "B",
                    phase_id="spawn_near_miss_bomb",
                    intent="Cluster spawns the bomb that the H-rocket will narrowly miss",
                ),
                _rocket_fire_phase(),
            ),
            payout=RangeFloat(1.0, 15.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            # Bomb remains dormant — rocket's H path missed it
            dormant_boosters_on_terminal=("B",),
            required_chain_depth=Range(0, 0),
            rocket_orientation="H",
            lb_target_tier=None,
        ),
        id="rocket_near_miss_orientation",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 1),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Two rockets idle surrounded by HIGH near-misses — maximum visual tension
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                # Sequential spawns — fitting two 9-10 clusters on one 49-cell
                # board is geometrically tight; splitting into two phases lets
                # gravity + refill free space between spawns.
                _rocket_spawn_phase(
                    cluster_count=Range(1, 1),
                    phase_id="dual_idle_spawn_1",
                    intent="First cluster spawns a dormant rocket",
                ),
                _rocket_spawn_phase(
                    cluster_count=Range(1, 1),
                    phase_id="dual_idle_spawn_2",
                    intent="Second cluster spawns another dormant rocket",
                ),
            ),
            payout=RangeFloat(1.0, 15.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_near_miss_dual_idle",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(2, 3),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        **arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Late-stage saves (2) — LOW cascade fades, late rocket rescues the run
    # -----------------------------------------------------------------------

    # LOW cascade fading out → late rocket fire saves the payout
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _low_cluster_phase(
                    phase_id="low_filler",
                    intent="Early LOW clusters — value fading before rocket rescue",
                ),
                _rocket_spawn_phase(),
                _rocket_fire_phase(),
            ),
            payout=RangeFloat(2.0, 30.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_late_save",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Bomb spawns early, dormant — late rocket chain-triggers it for combo
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _low_cluster_phase(
                    phase_id="low_filler",
                    intent="Early LOW clusters before bomb spawns",
                    # Tightened to the minimum-size cluster so the three
                    # remaining phases (B spawn, R spawn, R fire) have cells
                    # left on the 49-cell board.
                    cluster_sizes=(Range(5, 5),),
                ),
                # Bomb spawns from the minimum-size B cluster — tightened to
                # 11-11 to conserve cells for the rocket spawn + fire that
                # must follow on the same 49-cell board.
                NarrativePhase(
                    id="spawn_bomb_early",
                    intent="Bomb spawns early and waits dormant",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(11, 11),),
                    cluster_symbol_tier=None,
                    spawns=("B",),
                    arms=None,
                    fires=None,
                    wild_behavior=None,
                    ends_when="always",
                ),
                # Rocket spawns, fires, and chain-triggers the dormant bomb.
                # Cluster size tightened to 9-9 for the same space-conservation
                # reason as the bomb spawn above.
                NarrativePhase(
                    id="spawn_fire_rocket_chain",
                    intent="Rocket spawns, fires, and chain-triggers dormant bomb",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(9, 9),),
                    cluster_symbol_tier=None,
                    spawns=("R",),
                    arms=("R",),
                    fires=("R", "B"),
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
        id="rocket_late_save_chain",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Cascade storms (2) — multiple rockets fire in sequence
    # -----------------------------------------------------------------------

    # 2-3 rockets fire during a LOW cascade — sustained low-value action
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(
                    cluster_symbol_tier=SymbolTier.LOW,
                    phase_id="storm_spawn",
                    intent="LOW-tier cluster spawns first rocket",
                ),
                # Second rocket spawns while first fires — both LOW tier
                NarrativePhase(
                    id="storm_spawn_fire",
                    intent="LOW-tier cluster spawns another rocket and fires previous",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(9, 10),),
                    cluster_symbol_tier=SymbolTier.LOW,
                    spawns=("R",),
                    arms=("R",),
                    fires=("R",),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(2.0, 30.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_storm_low",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # One H + one V rocket = cross pattern — dramatic visual
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                # Sequential rocket spawns — two 9-10 clusters on one board
                # rarely fit; splitting gives gravity room to clear between.
                _rocket_spawn_phase(
                    cluster_count=Range(1, 1),
                    phase_id="cross_spawn_1",
                    intent="First cluster spawns a rocket (H or V)",
                ),
                _rocket_spawn_phase(
                    cluster_count=Range(1, 1),
                    phase_id="cross_spawn_2",
                    intent="Second cluster spawns the other-orientation rocket",
                ),
                _rocket_fire_phase(
                    phase_id="cross_fire",
                    intent="New cluster arms both rockets — cross pattern clear",
                ),
            ),
            payout=RangeFloat(3.0, 50.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            # No fixed orientation — one H and one V emerge from cluster shapes
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_storm_cross",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Scatter tension (2) — rockets + scatters create dual excitement
    # -----------------------------------------------------------------------

    # Rocket + 3 scatters — near-trigger with booster action
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _rocket_fire_phase(),
            ),
            payout=RangeFloat(1.0, 20.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_scatter_tease",
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Multiple rockets + 3 scatters + LOW clusters — chaotic tease
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(
                    cluster_symbol_tier=SymbolTier.LOW,
                    phase_id="scatter_storm_spawn",
                    intent="LOW-tier cluster spawns first rocket amid scatters",
                ),
                NarrativePhase(
                    id="scatter_storm_spawn_fire",
                    intent="LOW-tier cluster spawns second rocket and fires first",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(9, 10),),
                    cluster_symbol_tier=SymbolTier.LOW,
                    spawns=("R",),
                    arms=("R",),
                    fires=("R",),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(2.0, 30.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_scatter_storm_tease",
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Escalation fakeouts (3) — buildup that fizzles at the end
    # -----------------------------------------------------------------------

    # Rocket fires → cascade dies → HIGH near-misses on terminal board
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _rocket_fire_phase(),
            ),
            payout=RangeFloat(1.0, 15.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=TerminalNearMissSpec(
                count=Range(1, 2), symbol_tier=SymbolTier.HIGH,
            ),
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_escalation_fizzle",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # R→B chain starts but cascade dies afterwards + near-misses
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _chain_target_spawn_phase(
                    "B",
                    phase_id="spawn_chain_bomb_fizzle",
                    intent="Cluster spawns the bomb the rocket chain will trigger",
                ),
                NarrativePhase(
                    id="fire_rocket_chain_fizzle",
                    intent="Cluster arms rocket, chain triggers bomb, then cascade dies",
                    repetitions=Range(1, 1),
                    cluster_count=Range(1, 2),
                    cluster_sizes=(Range(5, 10),),
                    cluster_symbol_tier=None,
                    spawns=None,
                    arms=("R",),
                    fires=("R", "B"),
                    wild_behavior=None,
                    ends_when="always",
                ),
            ),
            payout=RangeFloat(3.0, 40.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=TerminalNearMissSpec(
                count=Range(1, 2), symbol_tier=None,
            ),
            dormant_boosters_on_terminal=None,
            required_chain_depth=Range(1, 1),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_chain_fizzle",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))

    # Rocket fires → refill creates a dormant bomb → cascade dies with bomb visible
    registry.register(build_arc_signature(
        arc=NarrativeArc(
            phases=(
                _rocket_spawn_phase(),
                _rocket_fire_phase(),
                # Dedicated bomb spawn — post-fire refill alone never reliably
                # produced the 11-12 cluster needed, so the tease archetype
                # now explicitly plants the dormant bomb.
                _chain_target_spawn_phase(
                    "B",
                    phase_id="spawn_dormant_bomb",
                    intent="Post-fire cluster spawns the dormant bomb that stays on terminal",
                ),
            ),
            payout=RangeFloat(1.0, 20.0),
            wild_count_on_terminal=Range(0, 0),
            terminal_near_misses=None,
            # Bomb stays dormant and visible on the final dead board — tease
            dormant_boosters_on_terminal=("B",),
            required_chain_depth=Range(0, 0),
            rocket_orientation=None,
            lb_target_tier=None,
        ),
        id="rocket_bomb_tease",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **arc_base(),
    ))
