"""RegionConstraint — unified placement hint consumed by ClusterBuilder.

Both AtlasConfiguration (Tier 1) and TrajectorySketch (Tier 2) resolve to a
RegionConstraint per cascade step via the GuidanceSource protocol. The
ClusterBuilder only ever sees RegionConstraint — it does not know which tier
produced the guidance. This keeps the builder decoupled from both planners
and makes the tier swap a drop-in substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class BridgeHint:
    """Bridge-phase placement hint — where the wild should connect two sub-groups.

    Strategies use this to prefer candidates whose reachable cells align with
    the atlas-predicted group structure. gap_column is where the wild sits;
    left/right_columns are where each sub-group should live.
    """

    gap_column: int
    left_columns: frozenset[int]
    right_columns: frozenset[int]


@dataclass(frozen=True, slots=True)
class RegionConstraint:
    """Soft placement preference for a single cascade step.

    viable_columns — columns the planner judged reachable/safe for this step's
      cluster. Cells outside these columns receive a falloff penalty through
      RegionPreferenceFactor rather than a hard exclusion; the builder may
      still place outside if no valid shape fits inside.
    preferred_row_range — inclusive (min_row, max_row) window. None means
      any row is equally preferred (column-only constraint).
    profile_hint — optional ColumnProfile handle (typed as object to avoid a
      circular import with atlas.data_types; consumers that care introspect
      via the atlas package). Used only by debug/telemetry paths.
    """

    viable_columns: frozenset[int]
    preferred_row_range: tuple[int, int] | None
    profile_hint: object | None = None


@runtime_checkable
class GuidanceSource(Protocol):
    """Either an AtlasConfiguration or a TrajectorySketch.

    Each tier implements region_at() against its own internal step/phase
    structure. The accessor contract is the single DRY extraction point
    shared by every strategy that consumes planning guidance.
    """

    def region_at(self, step_index: int) -> RegionConstraint | None:
        ...


def region_for_step(
    guidance: GuidanceSource | None,
    step_index: int,
) -> RegionConstraint | None:
    """Return the RegionConstraint for the given step, or None if unguided.

    Strategies call this in preference to branching on guidance type — the
    None path preserves the historical unconstrained behavior, so adding the
    call is a no-op for games without atlas or trajectory planning.
    """
    if guidance is None:
        return None
    return guidance.region_at(step_index)


def bridge_hint_for_step(
    guidance: GuidanceSource | None,
    step_index: int,
) -> BridgeHint | None:
    """Return the BridgeHint for the given step, or None if not a bridge phase.

    Duck-typed: only AtlasConfiguration exposes bridge_hint_at(). Trajectory
    planners (Tier 2) return None implicitly — no bridge support needed there.
    """
    if guidance is None:
        return None
    accessor = getattr(guidance, "bridge_hint_at", None)
    if accessor is None:
        return None
    return accessor(step_index)


def landing_for_step(
    guidance: GuidanceSource | None,
    step_index: int,
):
    """Return the predicted booster landing for the given step, or None.

    Duck-typed accessor like bridge_hint_for_step: only AtlasConfiguration
    exposes landing_at() (routed to PhaseGuidance.booster_landing). Trajectory
    planners return None — their Tier-2 sketches do not carry pre-validated
    armability data.

    Returns a Position (imported lazily at call site to avoid circular imports).
    """
    if guidance is None:
        return None
    accessor = getattr(guidance, "landing_at", None)
    if accessor is None:
        return None
    return accessor(step_index)
