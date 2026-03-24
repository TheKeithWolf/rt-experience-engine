"""Wincap family archetype definitions — 2 archetypes covering near-wincap
(high payout approaching the cap) and wincap (exact cap hit with cascade halt).

Wincap archetypes require deep cascades with grid multiplier accumulation to
reach payout levels of 2000-5000x. ASP uses wincap_mode to relax payout
constraints because grid multiplier scaling (not modeled in ASP) provides
the majority of payout. Actual payout targeting happens in the cascade
generator and is enforced by the validator.
"""

from __future__ import annotations

from ..pipeline.protocols import Range, RangeFloat
from .registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
)


def _wincap_base() -> dict:
    """Shared fields for all wincap archetypes.

    Wincap family is wincap criteria. No rocket orientation constraint
    (deep cascades may contain multiple rockets), no LB tier targeting,
    no freespin triggers.
    """
    return dict(
        family="wincap",
        criteria="wincap",
        triggers_freespin=False,
        rocket_orientation=None,
        lb_target_tier=None,
        # No dead-board component cap — deep cascades with clusters
        max_component_size=None,
        # Wincap boards don't constrain narrative arcs or terminal aesthetics —
        # the focus is on reaching payout targets through deep cascade + multipliers
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
    )


def register_wincap_archetypes(registry: ArchetypeRegistry) -> None:
    """Register all 2 wincap-family archetypes.

    near_wincap approaches but does not hit the cap. wincap hits the cap
    exactly and halts the cascade.
    """
    base = _wincap_base

    # -----------------------------------------------------------------------
    # near_wincap — payout 2000-4999x, deep cascades + multiplier stacking
    # -----------------------------------------------------------------------
    # High-payout outcome that approaches but does not hit the win cap.
    # Requires deep cascade depth (8-25 steps) where grid multipliers
    # accumulate to scale cluster payouts into the thousands. Boosters
    # are flexible — the solver picks whatever combination reaches the
    # payout target. reaches_wincap=False means no cascade halt.
    registry.register(ArchetypeSignature(
        id="near_wincap",
        reaches_wincap=False,
        required_cluster_count=Range(1, 4),
        # Wide size range — solver needs flexibility to hit payout targets
        required_cluster_sizes=(Range(5, 49),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        # Deep cascades needed — grid multipliers accumulate over many steps
        required_cascade_depth=Range(8, 25),
        # No per-step constraints — ASP/CSP plan freely to hit the payout
        cascade_steps=None,
        # Flexible booster ranges — solver picks viable combinations
        required_booster_spawns={
            "R": Range(0, 3), "B": Range(0, 3),
            "LB": Range(0, 2), "SLB": Range(0, 2),
        },
        required_booster_fires={
            "R": Range(0, 3), "B": Range(0, 3),
        },
        # Chains allowed but not required — may emerge from deep cascades
        required_chain_depth=Range(0, 4),
        # Near-cap payout — below config.wincap.max_payout
        payout_range=RangeFloat(2000.0, 4999.0),
        **base(),
    ))

    # -----------------------------------------------------------------------
    # wincap — payout exactly 5000x (config.wincap.max_payout), cascade halts
    # -----------------------------------------------------------------------
    # The cascade hits the win cap and halts mid-step. No further game
    # mechanics execute after the cap — remaining freespins emit events
    # but add no wins. Even deeper cascade depth (8-30) than near_wincap
    # to ensure the multiplier stack reaches exactly the cap value.
    # The cascade generator clamps payout to config.wincap.max_payout
    # and breaks the cascade loop when reaches_wincap=True.
    registry.register(ArchetypeSignature(
        id="wincap",
        reaches_wincap=True,
        required_cluster_count=Range(1, 4),
        required_cluster_sizes=(Range(5, 49),),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 1),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        # Even deeper — needs to accumulate enough for the exact cap
        required_cascade_depth=Range(8, 30),
        cascade_steps=None,
        required_booster_spawns={
            "R": Range(0, 3), "B": Range(0, 3),
            "LB": Range(0, 2), "SLB": Range(0, 2),
        },
        required_booster_fires={
            "R": Range(0, 3), "B": Range(0, 3),
        },
        required_chain_depth=Range(0, 4),
        # Exact cap — payout clamped to config.wincap.max_payout by cascade generator
        payout_range=RangeFloat(5000.0, 5000.0),
        **base(),
    ))
