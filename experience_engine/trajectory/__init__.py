"""Trajectory Planner — runtime sketch-fidelity forward simulation (Tier 2).

When the offline atlas has no match, the planner simulates the full arc
step-by-step using the engine's existing services, emitting per-waypoint
region constraints for ClusterBuilder.
"""

from .data_types import TrajectorySketch, TrajectoryWaypoint
from .phase_simulators import SketchDependencies, SketchState
from .planner import TrajectoryPlanner
from .scorer import ScoredTrajectory, TrajectoryScorer
from .sketch_fill import neutral_fill

__all__ = (
    "ScoredTrajectory",
    "SketchDependencies",
    "SketchState",
    "TrajectoryPlanner",
    "TrajectoryScorer",
    "TrajectorySketch",
    "TrajectoryWaypoint",
    "neutral_fill",
)
