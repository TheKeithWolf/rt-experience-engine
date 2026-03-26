"""Chain family archetype definitions — 5 archetypes covering multi-booster
compositions, deep chains, parallel independent fires, organic cascade-to-booster
growth, and multi-phase booster sequences.

Chain archetypes are defined by the *interaction pattern* between boosters rather
than a specific booster type. They compose rockets, bombs, lightballs, and
superlightballs into complex multi-step sequences.

Chain depth semantics: chain_depth counts chain-triggered targets (ASP convention).
B->R = chain_depth 1 (one target). R->B->R = chain_depth 2 (two targets).
"""

from __future__ import annotations

from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from .registry import (
    ArchetypeRegistry,
    build_arc_signature,
)


def _chain_arc_base(**overrides: object) -> dict:
    """Shared fields for arc-based chain archetypes (used with build_arc_signature).

    Only includes fields that build_arc_signature does NOT derive —
    identity, feature flags, and component constraints. All cascade
    structure comes from the NarrativeArc via build_arc_signature.
    """
    defaults = dict(
        family="chain",
        criteria="basegame",
        triggers_freespin=False,
        reaches_wincap=False,
        # Chain archetypes produce clusters — no dead-board component cap
        max_component_size=None,
    )
    defaults.update(overrides)
    return defaults


def register_chain_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 5 chain-family archetypes.

    Organized by interaction pattern: cross-type chain (1), deep chain (1),
    parallel independent (1), organic growth (1), multi-phase (1).
    """

    # -----------------------------------------------------------------------
    # chain_2_mixed — Two different booster types in a depth-2 chain
    # -----------------------------------------------------------------------
    # Simple multi-booster chain: e.g., R spawns, B spawns on a later step,
    # both arm and fire in sequence triggering each other (R->B->R or B->R->B).
    # The player sees two different boosters interact — introduces the concept
    # that boosters can trigger each other.
    _chain_2_mixed_arc = NarrativeArc(
        phases=(
            # Spawn first booster via 9-10 (rocket) or 11-12 (bomb) cluster
            NarrativePhase(
                id="spawn_initiators",
                intent="Spawn first booster from cluster reaching booster threshold.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                spawns=("R", "B"),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # ASP plans the chain execution across remaining steps
            NarrativePhase(
                id="chain_execution",
                intent="Chain fires — boosters trigger each other in R/B sequence.",
                repetitions=Range(1, 7),
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(5, 14),),
                cluster_symbol_tier=None,
                spawns=("R", "B"),
                arms=("R", "B"),
                # Both R and B fire during the chain sequence
                fires=("R", "B"),
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(5.0, 200.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        # Depth 2 = two chain-triggered targets (e.g., R->B->R)
        required_chain_depth=Range(2, 2),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _chain_2_mixed_arc,
        id="chain_2_mixed",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **_chain_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # chain_3_plus — Deep chain of 3-6 links, at least 1 rocket + 1 bomb
    # -----------------------------------------------------------------------
    # Extended multi-booster cascade: a sustained sequence of boosters triggering
    # each other. High drama, high payout. At least one rocket and one bomb
    # participate — ASP plans the full chain depth.
    _chain_3_plus_arc = NarrativeArc(
        phases=(
            # Spawn the first booster to kick off the chain
            NarrativePhase(
                id="spawn_initiator",
                intent="Spawn first booster from large cluster to start deep chain.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                spawns=("R", "B"),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # ASP fills remaining chain — flexible depth and spawn/fire counts
            NarrativePhase(
                id="deep_chain_execution",
                intent="Sustained booster chain — each link triggers the next.",
                repetitions=Range(2, 11),
                cluster_count=Range(1, 4),
                cluster_sizes=(Range(5, 14),),
                cluster_symbol_tier=None,
                spawns=("R", "B"),
                arms=("R", "B"),
                # Both R and B fire during the deep chain
                fires=("R", "B"),
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(10.0, 500.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        # Deep chain: 3-6 chain-triggered targets (4-7 boosters total)
        required_chain_depth=Range(3, 6),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _chain_3_plus_arc,
        id="chain_3_plus",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **_chain_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # multi_booster_parallel — 2+ boosters fire independently, no chain
    # -----------------------------------------------------------------------
    # Simultaneous action: multiple independent booster events on the same spin.
    # Boosters don't trigger each other — they coexist. chain_depth=0 triggers
    # the forbid_chain ASP constraint to prevent accidental chain links.
    _multi_booster_parallel_arc = NarrativeArc(
        phases=(
            # Spawn both boosters from separate clusters
            NarrativePhase(
                id="spawn_parallel_boosters",
                intent="Spawn multiple independent boosters from separate clusters.",
                repetitions=Range(1, 1),
                cluster_count=Range(2, 4),
                cluster_sizes=(Range(9, 12),),
                cluster_symbol_tier=None,
                spawns=("R", "B"),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Boosters arm and fire independently — no chain interaction
            NarrativePhase(
                id="parallel_fire",
                intent="Boosters arm and fire independently with no chain interaction.",
                repetitions=Range(1, 7),
                cluster_count=Range(1, 4),
                cluster_sizes=(Range(5, 12),),
                cluster_symbol_tier=None,
                spawns=("R", "B"),
                arms=("R", "B"),
                fires=("R", "B"),
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(5.0, 200.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        # Zero chains — boosters fire independently (triggers forbid_chain in ASP)
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _multi_booster_parallel_arc,
        id="multi_booster_parallel",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **_chain_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # cascade_to_booster_to_cascade — Organic growth from t1 to booster
    # -----------------------------------------------------------------------
    # Small clusters (5-6) cascade and grow until they reach booster spawn
    # threshold (9+). The booster fires, refill cascades again. The booster
    # emerges naturally from cascading — feels earned rather than given.
    _cascade_to_booster_arc = NarrativeArc(
        phases=(
            # t1-level small clusters — cascade beginning
            NarrativePhase(
                id="small_cascade_start",
                intent="Small t1-level clusters start the organic cascade.",
                repetitions=Range(2, 2),
                cluster_count=Range(1, 3),
                cluster_sizes=(Range(5, 6),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Transition — clusters grow toward booster threshold
            NarrativePhase(
                id="growth_transition",
                intent="Clusters grow in size, approaching booster spawn threshold.",
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
            # Growth hits booster threshold — organic spawn
            NarrativePhase(
                id="organic_booster_spawn",
                intent="Cascade growth naturally reaches booster size threshold.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 14),),
                cluster_symbol_tier=None,
                spawns=("R", "B"),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Booster arms and fires, cascade continues from refill
            NarrativePhase(
                id="booster_fire_cascade",
                intent="Booster fires and refill triggers further cascade steps.",
                repetitions=Range(1, 6),
                cluster_count=Range(1, 5),
                cluster_sizes=(Range(5, 14),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("R", "B"),
                fires=("R", "B"),
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(3.0, 150.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        # No chain — single booster fires
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _cascade_to_booster_arc,
        id="cascade_to_booster_to_cascade",
        required_cluster_symbols=None,
        # No scatters — focus is on the organic cascade experience
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **_chain_arc_base(),
    ))

    # -----------------------------------------------------------------------
    # booster_phase_multi — Multiple booster phases in a single spin
    # -----------------------------------------------------------------------
    # Episodic multi-phase spin: cascade -> booster phase 1 -> cascade -> booster
    # phase 2 -> terminal dead. The spin feels like it has chapters — build,
    # explode, rebuild, explode again. Each phase gets its own booster type.
    _booster_phase_multi_arc = NarrativeArc(
        phases=(
            # Act 1: spawn rocket from 9-10 cluster
            NarrativePhase(
                id="act1_spawn_rocket",
                intent="First act opens — rocket spawns from medium cluster.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(9, 10),),
                cluster_symbol_tier=None,
                spawns=("R",),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Act 1 climax: arm + fire rocket (booster phase 1)
            NarrativePhase(
                id="act1_fire_rocket",
                intent="Rocket arms and fires — first booster phase climax.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 10),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("R",),
                fires=("R",),
                wild_behavior=None,
                ends_when="always",
            ),
            # Act 2: spawn bomb from 11-12 cluster
            NarrativePhase(
                id="act2_spawn_bomb",
                intent="Second act opens — bomb spawns from larger cluster.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(11, 12),),
                cluster_symbol_tier=None,
                spawns=("B",),
                arms=None,
                fires=None,
                wild_behavior=None,
                ends_when="always",
            ),
            # Act 2 climax: arm + fire bomb (booster phase 2)
            NarrativePhase(
                id="act2_fire_bomb",
                intent="Bomb arms and fires — second booster phase climax.",
                repetitions=Range(1, 1),
                cluster_count=Range(1, 2),
                cluster_sizes=(Range(5, 12),),
                cluster_symbol_tier=None,
                spawns=None,
                arms=("B",),
                fires=("B",),
                wild_behavior=None,
                ends_when="always",
            ),
        ),
        payout=RangeFloat(10.0, 500.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        # Zero chain depth — phases are independent, not chain-linked
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    registry.register(build_arc_signature(
        _booster_phase_multi_arc,
        id="booster_phase_multi",
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        **_chain_arc_base(),
    ))
