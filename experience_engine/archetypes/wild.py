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
    # Cascade to Settle (Gravity + Refill)
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
    # Step 1: W bridges two disconnected LOW groups into 5-6 cluster
    # Step 2: Terminal board with W consumed in bridge
    _wild_bridge_small_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wild",
                intent="Initial cluster spawns a wild.",
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

    # FIX: Converted from direct ArchetypeSignature to arc-based.
    # Step 0: 1 Cluster of 7-8 + W spawn
    # Step 1: W bridges two disconnected HIGH groups into 9-10 cluster
    # Step 2: Terminal board with W consumed in bridge
    _wild_bridge_large_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wild",
                intent="Initial cluster spawns wild.",
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
                id="bridge_large",
                intent="Wild bridges two disconnected HIGH groups into a large cluster.",
                repetitions=Range(1, 2),
                cluster_count=Range(1, 2),
                # FIX: Bridge cluster size 5-8 — stays below rocket threshold (9+)
                # to avoid spawning an unintended rocket. The "large" refers to
                # the bridge being a HIGH-tier cluster, not necessarily a booster-spawning size.
                cluster_sizes=(Range(5, 8),),
                cluster_symbol_tier=SymbolTier.HIGH,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior="bridge",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.6, 20.0),
        # W consumed in bridge — no wilds on terminal
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _wild_bridge_large_arc,
        id="wild_bridge_large",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # Narrative Arc — W spawns, bridges to 9-10 (R spawns), cluster arms R, R fires
    # Step 0: 1 Cluster of 7-8 + W spawn
    # Step 1: W bridges groups into 9-10 cluster → R spawns
    # Step 2: 1 Cluster of 5-6 ARMS the Rocket
    # Step 3: Rocket fires (terminal — 0 clusters)
    _wild_enable_rocket_arc = NarrativeArc(
        phases=(
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
            NarrativePhase(
                id="bridge_rocket",
                intent="Wild bridges groups into large cluster, spawning rocket.",
                # FIX: repetitions Range(1,1) — R should only spawn once.
                # Was Range(1,2) which derived R budget as Range(1,2), allowing
                # 2 rockets when we only want 1.
                repetitions=Range(1, 1),
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                spawns=("R",),
                arms=None,
                fires=None,
                wild_behavior="bridge",
                ends_when="always",
            ),
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

    # FIX: Converted from direct ArchetypeSignature to arc-based.
    # Added required_booster_fires for Bomb (was missing — bomb must fire).
    # Step 0: 1 Cluster of 7-8 + W spawn
    # Step 1: W bridges groups into 11-12 cluster → B spawns
    # Step 2: 1 Cluster of 5-6 ARMS the Bomb
    # Step 3: Bomb fires (terminal — 0 clusters)
    _wild_enable_bomb_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wild",
                intent="Initial cluster spawns wild.",
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
            NarrativePhase(
                id="bridge_bomb",
                intent="Wild bridges groups into large cluster, spawning bomb.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                spawns=("B",),
                arms=None,
                fires=None,
                wild_behavior="bridge",
                ends_when="always",
            ),
            NarrativePhase(
                id="arm_bomb",
                intent="Cluster arms the bomb booster.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("B",),
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            NarrativePhase(
                id="fire_bomb",
                intent="Bomb fires, clearing a 3x3 area.",
                repetitions=Range(1, 1),
                cluster_count=Range(0, 0),
                cluster_sizes=(),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=("B",),
                wild_behavior=None,
                ends_when="booster_fired",
            ),
        ),
        payout=RangeFloat(2.0, 50.0),
        wild_count_on_terminal=Range(0, 1),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    # Derives: depth Range(3,3), spawns {"W": Range(1,1), "B": Range(1,1)},
    #          fires {"B": Range(1,1)}
    registry.register(build_arc_signature(
        _wild_enable_bomb_arc,
        id="wild_enable_bomb",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # FIX: Converted from direct ArchetypeSignature to arc-based.
    # Step 0: 2-3 Clusters of 7-8 spawning Wilds
    # Steps 1+: W bridges symbol groups forming clusters (optional)
    # Terminal: wilds may remain idle or be consumed in bridges
    _wild_multi_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wilds",
                intent="Multiple clusters each spawn a wild.",
                repetitions=Range(1, 1),
                cluster_count=Range(2, 3),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
            NarrativePhase(
                id="bridge_cascade",
                intent="Wilds bridge symbol groups, extending the cascade.",
                repetitions=Range(1, 3),
                cluster_count=Range(1, 2),
                # FIX: 5-6 for bridge steps — below wild threshold to prevent
                # unplanned W spawns from bridge clusters.
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior="bridge",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(1.0, 30.0),
        # Some wilds may survive bridging
        wild_count_on_terminal=Range(0, 3),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    # Derives: depth Range(2,4), spawns {"W": Range(2,3)} (from spawn phase cluster_count 2-3)
    # Note: spawns aggregation uses repetitions, not cluster_count. With spawns=("W",)
    # and repetitions=Range(1,1), derived W budget = Range(1,1). But we need 2-3 W spawns.
    # The cluster_count=Range(2,3) produces 2-3 clusters each spawning W, but the derivation
    # counts by repetitions not cluster_count. So we need repetitions=Range(2,3) with
    # cluster_count=Range(1,1) per repetition, OR we accept the derivation gives W=Range(1,1)
    # and override. Let's restructure:
    _wild_multi_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wilds",
                intent="Multiple clusters each spawn a wild.",
                # Each repetition produces 1 cluster of 7-8 → 1 W spawn.
                # 2-3 repetitions → 2-3 W spawns.
                repetitions=Range(2, 3),
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
            NarrativePhase(
                id="bridge_cascade",
                intent="Wilds bridge symbol groups, extending the cascade.",
                repetitions=Range(0, 2),
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
        payout=RangeFloat(1.0, 30.0),
        wild_count_on_terminal=Range(0, 3),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    # Derives: depth Range(2,5), spawns {"W": Range(2,3)}
    registry.register(build_arc_signature(
        _wild_multi_arc,
        id="wild_multi",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
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

    # FIX: Converted to arc-based for better step control.
    # Multiple wilds idle near multiple near-miss groups
    _wild_near_miss_multi_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_dual_wilds",
                intent="Two clusters each spawn a wild near near-miss groups.",
                repetitions=Range(1, 2),
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
        payout=RangeFloat(0.4, 10.0),
        wild_count_on_terminal=Range(1, 2),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    # Derives: depth Range(1,2), spawns {"W": Range(1,2)}
    registry.register(build_arc_signature(
        _wild_near_miss_multi_arc,
        id="wild_near_miss_multi",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(2, 3),
        required_near_miss_symbol_tier=SymbolTier.ANY,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # FIX: cluster_sizes changed from Range(5, 6) to Range(7, 8).
    # A cluster of 5-6 CANNOT spawn a wild (threshold is 7).
    # LOW cluster + idle wild + HIGH near-misses — mismatch tension
    _wild_near_miss_high_idle_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wild_idle",
                intent="Cluster spawns wild that idles near HIGH near-miss groups.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                # FIX: Was Range(5, 6) — impossible to spawn W from clusters < 7.
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=SymbolTier.LOW,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.1, 6.0),
        wild_count_on_terminal=Range(1, 1),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _wild_near_miss_high_idle_arc,
        id="wild_near_miss_high_idle",
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(1, 2),
        required_near_miss_symbol_tier=SymbolTier.HIGH,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Late-stage saves (2)
    # -----------------------------------------------------------------------

    # LOW cascade fades → late wild spawn saves the run
    _wild_late_save_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="low_cascade",
                intent="Early steps forced LOW tier — building the 'fading' feel.",
                repetitions=Range(2, 2),
                cluster_count=Range(1, 2),
                # FIX: Was Range(5, 8). Sizes 7-8 would spawn unplanned wilds
                # in the "fading" phase. Capped to 5-6 to stay below threshold.
                cluster_sizes=(Range(5, 6),),
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
                repetitions=Range(1, 3),
                cluster_count=Range(1, 2),
                # FIX: Was Range(5, 8). Must be 7-8 to guarantee W spawn.
                # Sizes 5-6 cannot spawn a wild, causing booster_spawn(W)=0 failures.
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.4, 12.0),
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
                # FIX: Was Range(5, 8). Same fix as wild_late_save.
                cluster_sizes=(Range(5, 6),),
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
                # FIX: Was Range(5, 8). Must be 7-8 to guarantee W spawn.
                cluster_sizes=(Range(7, 8),),
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

    # FIX: Converted to arc-based. Was direct signature with cluster_sizes
    # Range(5, 6) — impossible to spawn wilds (need 7+). All 5 attempts
    # failed with booster_spawn(W)=0.
    # Long LOW cascade with many wilds — steady low-value cascade pacing
    _wild_storm_low_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="wild_storm",
                intent="Repeated wild spawns from LOW clusters create a sustained cascade.",
                # Each repetition spawns 1 W from a 7-8 cluster.
                # 2-4 repetitions → 2-4 W spawns.
                repetitions=Range(2, 4),
                cluster_count=Range(1, 2),
                # FIX: Was Range(5, 6) — cannot spawn wilds. Changed to Range(7, 8).
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=SymbolTier.LOW,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
            NarrativePhase(
                id="storm_fade",
                intent="Storm winds down — smaller clusters without wild spawns.",
                repetitions=Range(1, 2),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=SymbolTier.LOW,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.6, 8.0),
        # Some wilds survive to terminal
        wild_count_on_terminal=Range(0, 4),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    # Derives: depth Range(3,6), spawns {"W": Range(2,4)}
    registry.register(build_arc_signature(
        _wild_storm_low_arc,
        id="wild_storm_low",
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # FIX: Converted to arc-based. Wild-to-wild domino — each wild bridges
    # to the next in a chain. Needs explicit spawn and bridge phases.
    _wild_storm_bridge_chain_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wilds",
                intent="Initial clusters spawn wilds for the bridge chain.",
                repetitions=Range(2, 3),
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(7, 8),),
                cluster_symbol_tier=None,
                spawns=("W",),
                arms=None,
                fires=None,
                wild_behavior="spawn",
                ends_when="always",
            ),
            NarrativePhase(
                id="bridge_chain",
                intent="Wilds bridge in sequence, forming a domino chain.",
                repetitions=Range(1, 2),
                cluster_count=Range(1, 2),
                # Bridge clusters stay below wild threshold
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior="bridge",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(1.0, 20.0),
        wild_count_on_terminal=Range(0, 3),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    # Derives: depth Range(3,5), spawns {"W": Range(2,3)}
    registry.register(build_arc_signature(
        _wild_storm_bridge_chain_arc,
        id="wild_storm_bridge_chain",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Scatter tension (2)
    # -----------------------------------------------------------------------

    # Wild + 3 scatters — one scatter short of trigger, plus wild action
    # Simple archetype — single step, works well as direct signature
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

    # FIX: Converted to arc-based. Deep wild cascade + 3 scatters.
    # Separates spawn steps from non-spawn steps to prevent W over-spawning.
    _wild_scatter_cascade_tease_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wild",
                intent="Initial cluster(s) spawn wild(s) with scatter tension.",
                repetitions=Range(1, 2),
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
                id="cascade_continuation",
                intent="Cascade continues with smaller clusters — no new spawns.",
                repetitions=Range(1, 3),
                cluster_count=Range(1, 2),
                # Below wild threshold — prevents accidental W spawns
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.6, 20.0),
        wild_count_on_terminal=Range(0, 2),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    # Derives: depth Range(2,5), spawns {"W": Range(1,2)}
    registry.register(build_arc_signature(
        _wild_scatter_cascade_tease_arc,
        id="wild_scatter_cascade_tease",
        required_cluster_symbols=None,
        required_scatter_count=Range(3, 3),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # Escalation fakeouts (2)
    # -----------------------------------------------------------------------

    # FIX: Converted to arc-based. Wild→wild→fizzle — builds up momentum
    # then cascade dies with near-misses on terminal.
    _wild_escalation_fizzle_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="escalation",
                intent="Wild spawns build momentum — player expects continuation.",
                repetitions=Range(1, 2),
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
                id="fizzle",
                intent="Cascade winds down — smaller clusters, board dies with near-misses.",
                repetitions=Range(1, 2),
                cluster_count=Range(1, 2),
                # Below wild threshold — no new spawns
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(0.6, 12.0),
        wild_count_on_terminal=Range(0, 2),
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2),
            symbol_tier=SymbolTier.HIGH,
        ),
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    # Derives: depth Range(2,4), spawns {"W": Range(1,2)}
    registry.register(build_arc_signature(
        _wild_escalation_fizzle_arc,
        id="wild_escalation_fizzle",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))

    # FIX: Converted to arc-based. Wild enables rocket spawn → rocket stays
    # dormant on terminal board. Separates W-spawn and R-spawn phases.
    _wild_rocket_tease_arc = NarrativeArc(
        phases=(
            NarrativePhase(
                id="spawn_wild",
                intent="Initial cluster spawns wild.",
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
                id="spawn_rocket_via_bridge",
                intent="Wild bridges groups into 9-10 cluster, spawning a rocket.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 1),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                spawns=("R",),
                arms=None,
                fires=None,
                wild_behavior="bridge",
                ends_when="always",
            ),
        ),
        payout=RangeFloat(1.0, 20.0),
        # W may survive; R stays dormant
        wild_count_on_terminal=Range(0, 1),
        terminal_near_misses=TerminalNearMissSpec(
            count=Range(1, 2),
            symbol_tier=SymbolTier.ANY,
        ),
        dormant_boosters_on_terminal=("R",),
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    # Derives: depth Range(2,2), spawns {"W": Range(1,1), "R": Range(1,1)}
    registry.register(build_arc_signature(
        _wild_rocket_tease_arc,
        id="wild_rocket_tease",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        **_wild_arc_base(),
    ))
