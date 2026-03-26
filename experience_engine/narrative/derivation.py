"""Constraint derivation — compute structural constraints from a NarrativeArc.

derive_constraints() is a pure function that computes cascade_depth,
booster_spawns, booster_fires, cluster_count, and cluster_sizes from the
arc's phase definitions. This eliminates hand-specified derived fields and
ensures the arc is the single source of truth for cascade structure.

Used by build_arc_signature() to populate ArchetypeSignature fields
automatically — no manual specification of derived values.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pipeline.protocols import Range
from .arc import NarrativeArc, NarrativePhase


@dataclass(frozen=True, slots=True)
class DerivedConstraints:
    """Structural constraints computed from a NarrativeArc's phase definitions."""

    cascade_depth: Range
    booster_spawns: dict[str, Range]
    booster_fires: dict[str, Range]
    cluster_count: Range
    cluster_sizes: tuple[Range, ...]


def _is_terminal_phase(phase: NarrativePhase) -> bool:
    """Terminal phases produce zero clusters — they don't count toward cascade depth."""
    return phase.cluster_count.min_val == 0 and phase.cluster_count.max_val == 0


def _aggregate_booster_ranges(
    phases: tuple[NarrativePhase, ...],
    field: str,
) -> dict[str, Range]:
    """Sum repetition ranges per booster type across phases that use the given field.

    For each booster type appearing in any phase's spawns/fires tuple, the
    total range is the sum of repetition ranges across phases that include it.
    """
    totals: dict[str, list[int]] = {}  # type -> [sum_min, sum_max]

    for phase in phases:
        booster_types: tuple[str, ...] | None = getattr(phase, field)
        if booster_types is None:
            continue
        # Deduplicate within a single phase — each type counted once per phase
        for btype in set(booster_types):
            if btype not in totals:
                totals[btype] = [0, 0]
            totals[btype][0] += phase.repetitions.min_val
            totals[btype][1] += phase.repetitions.max_val

    return {btype: Range(vals[0], vals[1]) for btype, vals in totals.items()}


def derive_constraints(arc: NarrativeArc) -> DerivedConstraints:
    """Derive structural constraints from the arc's phase definitions.

    All derivation is purely from phase structure — no config needed.
    """
    # Cascade depth: sum of repetition ranges across non-terminal phases
    depth_min = 0
    depth_max = 0
    for phase in arc.phases:
        if not _is_terminal_phase(phase):
            depth_min += phase.repetitions.min_val
            depth_max += phase.repetitions.max_val

    # Cluster count: union of non-terminal phase ranges (min of mins, max of maxes)
    # Terminal phases (cluster_count=0) are excluded — they don't produce clusters
    # and would pull min_val to 0, letting select_cluster_count return 0
    count_mins: list[int] = []
    count_maxes: list[int] = []
    for phase in arc.phases:
        if not _is_terminal_phase(phase):
            count_mins.append(phase.cluster_count.min_val)
            count_maxes.append(phase.cluster_count.max_val)

    cluster_count = Range(
        min(count_mins) if count_mins else 0,
        max(count_maxes) if count_maxes else 0,
    )

    # Cluster sizes: deduplicated union of all distinct Range tuples across phases
    seen_sizes: list[Range] = []
    for phase in arc.phases:
        for size_range in phase.cluster_sizes:
            if size_range not in seen_sizes:
                seen_sizes.append(size_range)

    return DerivedConstraints(
        cascade_depth=Range(depth_min, depth_max),
        booster_spawns=_aggregate_booster_ranges(arc.phases, "spawns"),
        booster_fires=_aggregate_booster_ranges(arc.phases, "fires"),
        cluster_count=cluster_count,
        cluster_sizes=tuple(seen_sizes),
    )
