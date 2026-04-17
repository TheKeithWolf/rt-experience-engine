"""Reusable NarrativePhase factories — DRY for the booster-family archetypes.

B4: lifts the per-family `_<family>_spawn_phase` and `_<family>_fire_phase`
factories that are structurally identical across rocket and bomb (and
near-identical across other families) into shared builders.

Scope: covers the "spawn the booster" and "arm-and-fire the booster"
shapes that match exactly across rocket and bomb. Families with
divergent semantics (lightball arms but does not fire the same step;
chain combines two boosters; wild has the bridge wild_behavior) keep
their own inline phase definitions because forcing them through the
shared builder would over-generalize beyond the observed shapes — the
opposite of the doc's guidance.

Adding a sixth family that matches the spawn/arm-fire shape requires
only a call site change; no edits here.
"""

from __future__ import annotations

from ..pipeline.protocols import Range
from ..primitives.symbols import SymbolTier
from .arc import NarrativePhase


# Default arming-cluster size range — derived from the smallest game-rule
# cluster size that an arm step typically targets (5-6 in royal_tumble's
# paytable bracket). Centralized here so the value is sourced from one
# place when re-tuned, rather than duplicated across families.
_DEFAULT_ARM_CLUSTER_SIZES: tuple[Range, ...] = (Range(5, 6),)
_DEFAULT_CLUSTER_COUNT: Range = Range(1, 2)


def spawn_phase(
    booster_type: str,
    cluster_sizes: tuple[Range, ...],
    *,
    cluster_count: Range = _DEFAULT_CLUSTER_COUNT,
    cluster_symbol_tier: SymbolTier | None = None,
    phase_id: str | None = None,
    intent: str | None = None,
) -> NarrativePhase:
    """Phase where a cluster spawns `booster_type`.

    cluster_sizes is required because it varies per family (rocket 9-10,
    bomb 11-12, lb 13-14, …) — there is no sensible default.
    """
    return NarrativePhase(
        id=phase_id or f"spawn_{booster_type.lower()}",
        intent=intent or f"Cluster spawns a {booster_type}",
        repetitions=Range(1, 1),
        cluster_count=cluster_count,
        cluster_sizes=cluster_sizes,
        cluster_symbol_tier=cluster_symbol_tier,
        spawns=(booster_type,),
        arms=None,
        fires=None,
        wild_behavior=None,
        ends_when="always",
    )


def arm_and_fire_phase(
    booster_type: str,
    *,
    cluster_sizes: tuple[Range, ...] = _DEFAULT_ARM_CLUSTER_SIZES,
    cluster_count: Range = _DEFAULT_CLUSTER_COUNT,
    phase_id: str | None = None,
    intent: str | None = None,
) -> NarrativePhase:
    """Phase where a new cluster arms `booster_type` and the booster fires.

    Used by booster families whose arming cluster also triggers the booster
    on the same step (rocket, bomb). For boosters that arm without firing
    in the same step (lightball), construct the NarrativePhase inline.
    """
    return NarrativePhase(
        id=phase_id or f"fire_{booster_type.lower()}",
        intent=intent or f"Cluster arms the {booster_type}, which fires",
        repetitions=Range(1, 1),
        cluster_count=cluster_count,
        cluster_sizes=cluster_sizes,
        cluster_symbol_tier=None,
        spawns=None,
        arms=(booster_type,),
        fires=(booster_type,),
        wild_behavior=None,
        ends_when="always",
    )
