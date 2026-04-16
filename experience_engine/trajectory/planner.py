"""Trajectory planner — runtime sketch-fidelity arc rollout.

Wraps the phase simulator registry and the trajectory scorer into a single
entry point the CascadeInstanceGenerator calls when the atlas misses. The
planner owns no physics — it orchestrates the simulators and packages the
resulting waypoints into a TrajectorySketch.

Retry logic lives in the generator, not here: TrajectoryPlanner.sketch()
produces exactly one sketch attempt per call. The orchestrator decides how
many attempts to spend.
"""

from __future__ import annotations

import random

from ..narrative.arc import NarrativeArc
from ..primitives.board import Board
from .data_types import TrajectorySketch, TrajectoryWaypoint
from .phase_simulators import (
    PhaseSimulator,
    SketchDependencies,
    SketchState,
    build_default_phase_registry,
    resolve_simulator,
)
from .scorer import TrajectoryScorer


class TrajectoryPlanner:
    """Produces one TrajectorySketch per call.

    Registry defaults to build_default_phase_registry() so callers can swap
    in a custom map for tests or future phase shapes without subclassing.
    """

    __slots__ = ("_deps", "_scorer", "_registry")

    def __init__(
        self,
        deps: SketchDependencies,
        scorer: TrajectoryScorer,
        registry: dict[tuple[bool, bool, bool], PhaseSimulator] | None = None,
    ) -> None:
        self._deps = deps
        self._scorer = scorer
        self._registry = registry or build_default_phase_registry()

    def sketch(
        self,
        arc: NarrativeArc,
        board: Board,
        rng: random.Random,
    ) -> TrajectorySketch:
        """Simulate every phase on a copy of the board, score, package.

        Failures (a simulator returning None) short-circuit into an infeasible
        sketch with whatever waypoints were produced so far. The orchestrator
        may still read those waypoints for telemetry — infeasibility stops
        the generator from trusting the guidance, not from inspecting it.
        """
        state = SketchState(board=board.copy())
        waypoints: list[TrajectoryWaypoint] = []
        for index, phase in enumerate(arc.phases):
            state.phase_index = index
            simulator = resolve_simulator(phase, self._registry)
            waypoint = simulator.simulate(phase, state, self._deps, rng)
            if waypoint is None:
                return TrajectorySketch(
                    waypoints=tuple(waypoints),
                    composite_score=0.0,
                    is_feasible=False,
                    arc=arc,
                )
            waypoints.append(waypoint)

        scored = self._scorer.score(waypoints)
        return TrajectorySketch(
            waypoints=tuple(waypoints),
            composite_score=scored.composite_score,
            is_feasible=scored.is_feasible,
            arc=arc,
        )
