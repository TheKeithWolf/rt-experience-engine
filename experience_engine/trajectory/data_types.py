"""Trajectory planner data types — immutable sketches of a full arc rollout.

A TrajectoryWaypoint captures everything a cascade step's strategies need to
know about a sketch-fidelity prediction: where the cluster lives, what
booster spawns, where it lands, where its fire zone reaches, and which cells
must be reserved for the next step's arming demand.

Design reference: games/royal_tumble/docs/trajectory-planner-impl-plan.md §4.1.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..narrative.arc import NarrativeArc
from ..planning.region_constraint import RegionConstraint
from ..primitives.board import Position
from ..primitives.gravity import SettleResult
from ..primitives.symbols import Symbol
from ..step_reasoner.services.landing_evaluator import LandingContext


@dataclass(frozen=True, slots=True)
class TrajectoryWaypoint:
    """One waypoint per NarrativePhase that produces a cluster or booster interaction.

    Every field is sourced from an existing service — the planner composes,
    it does not invent. See §4.1 of the design doc for the per-field source
    table.

    cluster_region — positions chosen for this step's cluster by
      ClusterBuilder; projected to a RegionConstraint via columns/rows
    booster_landing_pos / booster_type — None for phases that don't spawn
    chain_target_pos — downstream booster position the fire zone must reach;
      only populated for chain-initiator phases
    reserve_zone — cells InfluenceMap.reserve_zone locked down for the next
      step's arming cluster; unioned with any prior reserves
    seed_hints — SeedPlanner arm/bridge recommendations keyed by position
    """

    phase_index: int
    cluster_region: frozenset[Position]
    cluster_symbol: Symbol
    booster_type: str | None
    booster_spawn_pos: Position | None
    booster_landing_pos: Position | None
    settle_result: SettleResult
    landing_context: LandingContext | None
    landing_score: float
    reserve_zone: frozenset[Position]
    chain_target_pos: Position | None
    seed_hints: dict[Position, Symbol]

    def as_region(self) -> RegionConstraint:
        """Project onto RegionConstraint so strategies can consume it uniformly.

        viable_columns are the distinct columns touched by cluster_region;
        preferred_row_range spans the observed row band. When the strategy
        runs during generation it re-uses the same builder path as the atlas
        tier, so there is no dual code path.
        """
        columns = frozenset(p.reel for p in self.cluster_region)
        if self.cluster_region:
            rows = tuple(p.row for p in self.cluster_region)
            preferred = (min(rows), max(rows))
        else:
            preferred = None
        return RegionConstraint(
            viable_columns=columns,
            preferred_row_range=preferred,
        )


@dataclass(frozen=True, slots=True)
class TrajectorySketch:
    """The planner's full-arc prediction.

    is_feasible — False if any waypoint violated its feasibility threshold or
      a dormant booster was destroyed mid-arc; strategies read this before
      trusting the sketch
    composite_score — product of per-waypoint landing scores, gated by the
      trajectory.sketch_feasibility_threshold
    """

    waypoints: tuple[TrajectoryWaypoint, ...]
    composite_score: float
    is_feasible: bool
    arc: NarrativeArc

    def region_at(self, step_index: int) -> RegionConstraint | None:
        """GuidanceSource implementation — delegates to waypoint.as_region()."""
        if not 0 <= step_index < len(self.waypoints):
            return None
        return self.waypoints[step_index].as_region()
