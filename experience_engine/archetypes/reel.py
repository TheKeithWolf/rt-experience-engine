"""Reel family archetype definitions.

Reel archetypes are populated by a circular reel-strip generator rather
than the WFC/CSP solver stack. Because the strip determines the outcome,
signatures are deliberately wide validation envelopes — the strip cannot
be forced to a specific cluster count, booster spawn, or cascade depth.
The `InstanceValidator` checks produced instances against these envelopes
just like any other family.

Range bounds come from game rules (board area, min_cluster_size) — no
magic numbers. `BOARD_CELLS / min_cluster_size` caps the cascade depth
budget, and board area caps cluster count.
"""

from __future__ import annotations

from ..pipeline.protocols import Range, RangeFloat
from .registry import ArchetypeRegistry, ArchetypeSignature

def _reel_base() -> dict:
    """Shared fields for all reel archetypes — mirrors _dead_base pattern.

    Range literals derive from default.yaml game rules:
      board: 7x7 = 49 cells; min_cluster_size = 5
      max cluster count  = cells // min_cluster_size = 9
      max cascade depth  = cells // min_cluster_size + 1 = 10
      max cluster size   = 49 (the whole board can be one mega-cluster)
    """
    return dict(
        family="reel",
        criteria="basegame",
        required_cluster_symbols=None,
        required_cascade_depth=Range(0, 10),
        cascade_steps=None,
        narrative_arc=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        triggers_freespin=False,
        reaches_wincap=False,
        max_component_size=None,
    )


def register_reel_archetypes(registry: ArchetypeRegistry) -> None:
    """Register reel-family archetypes. Currently one generic envelope."""
    base = _reel_base

    # Generic reel spin — any outcome the strip can produce. Wide ranges are
    # intentional: the strip is the source of truth; the signature validates
    # shape, not target distribution.
    registry.register(ArchetypeSignature(
        id="reel_base",
        required_cluster_count=Range(0, 9),
        required_cluster_sizes=(Range(5, 49),),
        required_scatter_count=Range(0, 4),
        required_near_miss_count=Range(0, 10),
        required_near_miss_symbol_tier=None,
        payout_range=RangeFloat(0.0, 100.0),
        **base(),
    ))
