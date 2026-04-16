"""Trajectory scorer — composite feasibility scoring over waypoints.

The planner calls this once per sketch attempt. Per-waypoint scores are
produced by BoosterLandingEvaluator.score — no duplicate scoring criteria.
The composite is a product of per-waypoint scores, interpreted as a joint
feasibility probability: a single poor waypoint drags the whole sketch down.

Gating:
- Any waypoint below waypoint_feasibility_threshold marks the sketch infeasible.
- The composite below sketch_feasibility_threshold also marks infeasible.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..config.schema import TrajectoryConfig
from .data_types import TrajectoryWaypoint


@dataclass(frozen=True, slots=True)
class ScoredTrajectory:
    """Scorer output — orchestrator-facing result."""

    composite_score: float
    is_feasible: bool


class TrajectoryScorer:
    """Aggregates per-waypoint scores into a composite with threshold gating.

    __slots__ keeps the scorer cheap to instantiate; it holds only the
    trajectory thresholds from config. The planner passes waypoints as an
    iterable so the scorer has no dependency on TrajectorySketch's shape.
    """

    __slots__ = ("_config",)

    def __init__(self, config: TrajectoryConfig) -> None:
        self._config = config

    def score(self, waypoints: Iterable[TrajectoryWaypoint]) -> ScoredTrajectory:
        """Combine waypoint landing scores into the sketch's composite.

        An empty iterable scores 0.0 infeasible — a sketch with no waypoints
        cannot express any arc, so it must never pass feasibility.
        """
        composite = 1.0
        any_waypoint = False
        for waypoint in waypoints:
            any_waypoint = True
            if waypoint.landing_score < self._config.waypoint_feasibility_threshold:
                return ScoredTrajectory(composite_score=0.0, is_feasible=False)
            composite *= waypoint.landing_score
        if not any_waypoint:
            return ScoredTrajectory(composite_score=0.0, is_feasible=False)
        feasible = composite >= self._config.sketch_feasibility_threshold
        return ScoredTrajectory(composite_score=composite, is_feasible=feasible)
