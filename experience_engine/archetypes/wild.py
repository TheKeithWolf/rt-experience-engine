"""Wild family archetype definitions — 17 archetypes covering idle, bridge,
narrative arcs (late saves, storms, fakeouts), scatter tension, and near-miss tension.

Wilds spawn from clusters of 7-8 and can bridge disconnected standard-symbol
groups. Each archetype defines a specific player experience pattern using wilds.
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


def _wild_base() -> dict:
    """Shared fields for non-arc wild archetypes.

    Wild family is basegame criteria. No booster fires, chains, rocket orientation,
    or feature triggers — those come in later phases (9+).
    """
    return dict(
        family="wild",
        criteria="basegame",
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        triggers_freespin=False,
        reaches_wincap=False,
        # Wilds produce clusters — no dead-board component cap
        max_component_size=None,
    )


def _wild_arc_base(**overrides: object) -> dict:
    """Shared fields for arc-based wild archetypes (used with build_arc_signature).

    Only includes fields that build_arc_signature does NOT derive —
    identity, feature flags, and component constraints. All cascade
    structure comes from the NarrativeArc via build_arc_signature.
    """
    defaults = dict(
        family="wild",
        criteria="basegame",
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return defaults


def register_wild_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 17 wild-family archetypes.

    Organized by experience category: core (6), near-miss tension (3),
    late saves (2), storms (2), scatter tension (2), fakeouts (2).
    """
    base = _wild_base

    # -----------------------------------------------------------------------
    # Core wild archetypes (6)
    # -----------------------------------------------------------------------

    # Narrative Arc
    # Step 0: 1 Cluster of 7-8 + W spawn
    # Cascade to Settle (Grativy + Refill)
    # Step 1: Terminal board with W unused
    _wild_idle_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wild",
                intent="Cluster spawns a wild that sits idle on terminal.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.2, 8.0),
        # W spawns and remains idle on terminal
        wild_count_on_terminal=Range(1, 1),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _wild_idle_arc,
        id="wild_idle",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # Narrative Arc
    # Step 0: 1 Cluster of 7-8 + W spawn
    # Cascade to Settle (Grativy + Refill)
    # Step 1: W bridges two disconnected LOW symbol groups into one large cluster
    # Cascade to Settle (Grativy + Refill)
    # Step 2: Terminal board with W used in the bridge, so no wilds remain on terminal
    _wild_bridge_small_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wild",
                intent="Initial cluster spawns a wild.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
            NarrativePhase(
                id="bridge_small",
                intent="Wild bridges two disconnected LOW groups into a small cluster.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior="bridge",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.4, 15.0),
        # W consumed in bridge — no wilds on terminal
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _wild_bridge_small_arc,
        id="wild_bridge_small",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # Narrative Arc
    # Step 0: 1 Cluster of 7-8 + W spawn
    # Cascade to Settle (Grativy + Refill)
    # Step 1: W bridges two disconnected HIGH symbol groups into one large cluster
    # Cascade to Settle (Grativy + Refill)
    # Step 2: Terminal board with W used in the bridge, so no wilds remain on terminal
    registry.register(ArchetypeSignature(
        id="wild_bridge_large",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 2)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.6, 20.0),
        **base(),
    ))

    # Narrative Arc
    # Step 0: 2 Clusters of 7-8 + W spawn
    # Cascade to Settle (Grativy + Refill)
    # Step 1: W bridges a symbol group of size 9-10 spawning a Rocket booster
    # Cascade to Settle (Grativy + Refill)
    # Step 2: 1 Cluster of 5-6 ARMS the Rocket
    # Cascade to Settle (Grativy + Refill)
    # Step 3: Rocket fires, clearing a row/col
    # Cascade to Settle (Grativy + Refill)
    # Step 4: Terminal board with minimum of 1 or no wilds remain on terminal. No Rockets remain on terminal since it FIRED.
    _wild_enable_rocket_arc = NarrativeArc(
        phases=(
            # Step 0: initial cluster spawns W at centroid
            NarrativePhase(
                id="spawn_wild",
                intent="Initial cluster spawns wild at centroid.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
            # Step 1: W bridges two groups into 9-10 cluster → R spawns
            NarrativePhase(
                id="bridge_rocket",
                intent="Wild bridges groups into large cluster, spawning rocket.",
                # 1-2 steps: bridge may settle before the rocket-sized cluster forms
                repetitions=Range(1, 2),
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                spawns=("R",),
                arms=None,
                fires=None,
                wild_behavior="bridge",
                ends_when="always",
            ),
            # Step 2: cluster arms R
            NarrativePhase(
                id="arm_rocket",
                intent="Cluster arms the rocket booster.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("R",),
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Step 3: R fires → clears row/col (terminal — fire is not a cluster match)
            NarrativePhase(
                id="fire_rocket",
                intent="Rocket fires and clears a row or column.",
                repetitions=Range(1, 1),
                cluster_count=Range(0, 0),
                cluster_sizes=(),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=("R",),
                wild_behavior=None,
                ends_when="booster_fired",
            ),
        ),
        payout=RangeFloat(1.0, 30.0),
        # W may be consumed in bridge — 0 or 1 wilds on terminal
        wild_count_on_terminal=Range(0, 1),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        # Rocket must fire in this archetype
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _wild_enable_rocket_arc,
        id="wild_enable_rocket",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # Narrative Arc
    # Step 0: 2 Clusters of 7-8 + W spawn
    # Cascade to Settle (Grativy + Refill)
    # Step 1: W bridges a symbol group of size 11-12 spawning a Bomb booster
    # Cascade to Settle (Grativy + Refill)
    # Step 2: 1 Cluster of 5-6 ARMS the Bomb
    # Cascade to Settle (Grativy + Refill)
    # Step 3: Bomb fires, clearing a 3x3 area around it
    # Cascade to Settle (Grativy + Refill)
    # Step 4: Terminal board with minimum of 1 or no wilds remain on terminal. No Bombs remain on terminal since it FIRED.
    registry.register(ArchetypeSignature(
        id="wild_enable_bomb",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 12),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1), "B": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(2.0, 50.0),
        **base(),
    ))

    # Narrative Arc
    # Step 0: 2-4 Clusters of 7-8 spawning Wilds (1 per cluster)
    # Cascade to Settle (Grativy + Refill)
    # Step 1: W bridges symbol groups forming clusters
    # Cascade to Settle (Grativy + Refill). Until wilds can no longer bridge
    # Step 3: Terminal board. Wilds may be used in bridges, but any number of wilds can remain on terminal since they are not required to be used in a bridge.
    registry.register(ArchetypeSignature(
        id="wild_multi",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        required_booster_spawns={"W": Range(2, 3)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 30.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Near-miss tension (3)
    # -----------------------------------------------------------------------

    # Idle wild near HIGH near-miss groups — tantalizing "what if" moment
    _wild_near_miss_single_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wild",
                intent="Cluster spawns wild that idles near HIGH near-miss groups.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.2, 8.0),
        # W spawns and remains idle on terminal
        wild_count_on_terminal=Range(1, 1),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _wild_near_miss_single_arc,
        id="wild_near_miss_single",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # Multiple wilds idle near multiple near-miss groups
    registry.register(ArchetypeSignature(
        id="wild_near_miss_multi",
        required_cluster_count=Range(2, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(2, 3),
        required_near_miss_symbol_tier=SymbolTier.ANY,
        required_cascade_depth=Range(1, 2),
        cascade_steps=None,
        required_booster_spawns={"W": Range(2, 2)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.4, 10.0),
        **base(),
    ))

    # LOW cluster + idle wild + HIGH near-misses — mismatch tension
    registry.register(ArchetypeSignature(
        id="wild_near_miss_high_idle",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        required_cascade_depth=Range(1, 2),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.1, 6.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Late-stage saves (2)
    # -----------------------------------------------------------------------

    # LOW cascade fades → late wild spawn saves the run
    # Narrative: early phases forced LOW tier, late phase unconstrained
    _wild_late_save_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="low_cascade",
                intent="Early steps forced LOW tier — building the 'fading' feel.",
                # Steps 0 and 1 are both LOW-constrained
                repetitions=Range(2, 2),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=SymbolTier.LOW,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            NarrativePhase(
                id="wild_save",
                intent="Unconstrained late step where wild spawn rescues the cascade.",
                # Remaining steps (1-3) are unconstrained for the wild save
                repetitions=Range(1, 3),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.4, 12.0),
        # W spawns late — may or may not survive to terminal
        wild_count_on_terminal=Range(0, 1),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _wild_late_save_arc,
        id="wild_late_save",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # Same as wild_late_save but HIGH wild cluster on the save step
    _wild_late_save_high_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="low_cascade",
                intent="Early steps forced LOW tier — building the 'fading' feel.",
                repetitions=Range(2, 2),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=SymbolTier.LOW,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            NarrativePhase(
                id="high_wild_save",
                intent="Late HIGH wild cluster is the payoff — dramatic tier escalation.",
                repetitions=Range(1, 3),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.6, 15.0),
        wild_count_on_terminal=Range(0, 1),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _wild_late_save_high_arc,
        id="wild_late_save_high",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Cascade storms (2)
    # -----------------------------------------------------------------------

    # Long LOW cascade with many wilds — steady low-value cascade pacing
    registry.register(ArchetypeSignature(
        id="wild_storm_low",
        required_cluster_count=Range(2, 3),
        required_cluster_sizes=(Range(5, 6),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(4, 6),
        cascade_steps=None,
        required_booster_spawns={"W": Range(2, 4)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.6, 8.0),
        **base(),
    ))

    # Wild-to-wild domino — each wild bridges to the next in a chain
    registry.register(ArchetypeSignature(
        id="wild_storm_bridge_chain",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=None,
        required_booster_spawns={"W": Range(2, 3)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(1.0, 20.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Scatter tension (2)
    # -----------------------------------------------------------------------

    # Wild + 3 scatters — one scatter short of trigger, plus wild action
    registry.register(ArchetypeSignature(
        id="wild_scatter_tease",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(1, 3),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.4, 15.0),
        **base(),
    ))

    # Deep wild cascade + 3 scatters — extended near-trigger experience
    registry.register(ArchetypeSignature(
        id="wild_scatter_cascade_tease",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(3, 5),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 2)},
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.6, 20.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # Escalation fakeouts (2)
    # -----------------------------------------------------------------------

    # Wild→wild→fizzle — builds up momentum then cascade dies with near-misses
    registry.register(ArchetypeSignature(
        id="wild_escalation_fizzle",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 8),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 2)},
        symbol_tier_per_step=None,
        # Terminal board shows HIGH near-misses — "it almost kept going"
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2),
            symbol_tier=SymbolTier.HIGH,
        ),
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.6, 12.0),
        **base(),
    ))

    # Wild enables rocket spawn → rocket stays dormant on terminal board
    # Terminal shows dormant rocket + near-misses — maximum unfulfilled potential
    registry.register(ArchetypeSignature(
        id="wild_rocket_tease",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(7, 10),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(2, 4),
        cascade_steps=None,
        required_booster_spawns={"W": Range(1, 1), "R": Range(1, 1)},
        symbol_tier_per_step=None,
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2),
            symbol_tier=SymbolTier.ANY,
        ),
        # Rocket must survive on the terminal dead board for visual impact
        dormant_boosters_on_terminal=("R",),
        payout_range=RangeFloat(1.0, 20.0),
        **base(),
    ))
