"""Gym-style cascade environment for RL training.

Wraps the existing pipeline (StepExecutor, StepValidator, StepTransitionSimulator)
as the environment dynamics. The RL agent chooses cluster parameters via
CascadeAction; the environment executes them through the real pipeline and
returns observations and rewards.

The environment tracks NarrativePhase progression using greedy linear matching
(same logic as NarrativeArcValidator, but applied step-by-step during the
episode). The NarrativeArcValidator itself is used at episode end as ground
truth for archive insertion.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..board_filler.wfc_solver import FillFailed
from ..pipeline.step_validator import StepValidationFailed
from ..boosters.tracker import BoosterTracker
from ..config.schema import EnvironmentConfig, MasterConfig
from ..narrative.transitions import build_transition_rules
from ..pipeline.data_types import CascadeStepRecord, GeneratedInstance
from ..primitives.board import Board, Position
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..step_reasoner.context import BoardContext
from ..step_reasoner.progress import ProgressTracker
from ..variance.hints import VarianceHints
from .action_space import ActionInterpreter, CascadeAction
from .observation import CascadeObservation, ObservationBuilder

if TYPE_CHECKING:
    from ..archetypes.registry import ArchetypeSignature
    from ..narrative.arc import NarrativeArc
    from ..pipeline.simulator import StepTransitionSimulator
    from ..pipeline.step_executor import StepExecutor
    from ..pipeline.step_validator import StepValidator
    from ..primitives.gravity import GravityDAG
    from ..step_reasoner.results import StepResult
    from ..step_reasoner.services.cluster_builder import ClusterBuilder


@dataclass(frozen=True, slots=True)
class StepInfo:
    """Metadata returned by each environment step."""

    step_result: StepResult | None
    reason: str  # "ok", "fill_failed", "max_steps", "terminal"


class CascadeEnvironment:
    """Gym-style cascade environment for RL training episodes.

    Each episode generates one cascade sequence for a specific archetype.
    The environment reuses the real pipeline for step execution —
    no game rules are reimplemented. Phase progression is tracked online
    for reward shaping; the full NarrativeArcValidator runs at episode end.
    """

    __slots__ = (
        "_config", "_arc", "_signature", "_executor", "_validator",
        "_simulator", "_gravity_dag", "_obs_builder", "_action_interpreter",
        "_env_config",
        # Per-episode mutable state
        "_board", "_grid_mults", "_booster_tracker", "_progress",
        "_step_count", "_done", "_rng", "_transition_rules",
        "_step_records", "_variance_hints",
    )

    def __init__(
        self,
        config: MasterConfig,
        arc: NarrativeArc,
        signature: ArchetypeSignature,
        executor: StepExecutor,
        validator: StepValidator,
        simulator: StepTransitionSimulator,
        gravity_dag: GravityDAG,
        obs_builder: ObservationBuilder,
        action_interpreter: ActionInterpreter,
        env_config: EnvironmentConfig,
    ) -> None:
        self._config = config
        self._arc = arc
        self._signature = signature
        self._executor = executor
        self._validator = validator
        self._simulator = simulator
        self._gravity_dag = gravity_dag
        self._obs_builder = obs_builder
        self._action_interpreter = action_interpreter
        self._env_config = env_config

        # Mutable state — set in reset()
        self._board: Board | None = None
        self._grid_mults: GridMultiplierGrid | None = None
        self._booster_tracker: BoosterTracker | None = None
        self._progress: ProgressTracker | None = None
        self._step_count: int = 0
        self._done: bool = True
        self._rng: random.Random | None = None
        self._transition_rules: dict | None = None
        self._step_records: list[StepResult] = []
        self._variance_hints: VarianceHints | None = None

    def reset(
        self,
        rng: random.Random,
        variance_hints: VarianceHints | None = None,
    ) -> CascadeObservation:
        """Start a new episode with fresh board/progress state.

        Follows the same initialization pattern as
        CascadeInstanceGenerator._attempt_generation (cascade_generator.py:138-143).
        """
        self._board = Board.empty(self._config.board)
        self._grid_mults = GridMultiplierGrid(
            self._config.grid_multiplier, self._config.board,
        )
        self._booster_tracker = BoosterTracker(self._config.board)
        self._progress = ProgressTracker(
            self._signature, self._config.centipayout.multiplier,
        )
        self._transition_rules = build_transition_rules(
            self._config.board, self._config.symbols,
        )
        self._step_count = 0
        self._done = False
        self._rng = rng
        self._step_records = []
        self._variance_hints = variance_hints

        return self._build_observation()

    def step(
        self, action: CascadeAction,
    ) -> tuple[CascadeObservation, float, bool, StepInfo]:
        """Execute one cascade step and return (observation, reward, done, info).

        The environment never rejects a step — the board state is real.
        Reward guides the agent; the final NarrativeArcValidator confirms
        trajectory validity for archive insertion.
        """
        if self._done:
            raise RuntimeError("Episode is done — call reset() first")

        context = self._build_context()
        hints = self._variance_hints or _default_hints()

        # Convert action to pipeline StepIntent
        intent = self._action_interpreter.interpret(
            action, context, self._progress, hints, self._rng,
        )

        # Execute through the real pipeline — catch failures gracefully.
        # FillFailed: WFC couldn't fill the board (too many constraints).
        # StepValidationFailed: filled board violates step constraints
        # (e.g., terminal intent but clusters detected). Both end the episode
        # with a penalty — the agent learns to avoid these states.
        try:
            filled = self._executor.execute(intent, self._board, self._rng)
            step_result = self._validator.validate_step(
                filled, intent, self._progress, self._grid_mults,
            )
        except (FillFailed, StepValidationFailed):
            self._done = True
            obs = self._build_observation()
            return (
                obs,
                self._env_config.invalid_step_penalty,
                True,
                StepInfo(step_result=None, reason="step_failed"),
            )

        # Update progress tracker with this step's results
        self._progress.update(step_result)
        self._step_records.append(step_result)

        # Phase progression — greedy linear matching applied online
        self._advance_phase_if_ready(step_result)

        # Transition (gravity, booster spawning) unless terminal
        if not intent.is_terminal:
            transition_result = self._simulator.transition(
                self._board, step_result,
                self._booster_tracker, self._grid_mults,
            )
            self._board = transition_result.board
            self._progress.sync_active_wilds(
                self._board, self._config.board,
            )

        self._step_count += 1

        # Check termination
        done = (
            intent.is_terminal
            or self._step_count >= self._env_config.max_episode_steps
        )
        self._done = done

        # Build observation and compute reward
        obs = self._build_observation()
        reason = "terminal" if intent.is_terminal else (
            "max_steps" if self._step_count >= self._env_config.max_episode_steps
            else "ok"
        )
        info = StepInfo(step_result=step_result, reason=reason)

        # Reward is computed by the caller (PhaseRewardComputer) — environment
        # returns 0.0 as placeholder. The training loop computes the actual
        # reward using the RewardComputer protocol with full context.
        reward = 0.0

        return obs, reward, done, info

    def current_instance(self) -> GeneratedInstance | None:
        """Build a GeneratedInstance from accumulated step records.

        Returns None if the episode hasn't completed or had no steps.
        """
        if not self._step_records or self._board is None:
            return None

        # Build CascadeStepRecords from the accumulated StepResults
        cascade_steps: list[CascadeStepRecord] = []
        for step_result in self._step_records:
            num_rows = self._config.board.num_rows
            # Snapshot current grid multipliers in reel-major order
            snapshot = tuple(
                self._grid_mults.get(Position(r, row))
                for r in range(self._config.board.num_reels)
                for row in range(num_rows)
            )
            record = CascadeStepRecord(
                step_index=step_result.step_index,
                board_before=self._board,  # simplified — real impl tracks per-step
                board_after=self._board,
                clusters=tuple(
                    # Convert ClusterRecord to ClusterAssignment would be needed
                    # for full fidelity, but for archive quality scoring the
                    # step_payout and cluster sizes are what matter
                ),
                step_payout=step_result.step_payout / self._config.centipayout.multiplier,
                grid_multipliers_snapshot=snapshot,
                booster_spawn_types=tuple(
                    s.booster_type for s in step_result.spawns
                ),
                booster_spawn_positions=tuple(
                    (s.booster_type, s.position.reel, s.position.row)
                    for s in step_result.spawns
                ),
                booster_fire_records=(),
                gravity_record=None,
                booster_gravity_record=None,
            )
            cascade_steps.append(record)

        payout = self._progress.cumulative_payout / self._config.centipayout.multiplier
        return GeneratedInstance(
            sim_id=0,  # Stamped by the generator when sampled from archive
            archetype_id=self._signature.id,
            family=self._signature.family,
            criteria=self._signature.criteria,
            board=self._board,
            spatial_step=None,  # type: ignore[arg-type]
            payout=payout,
            centipayout=self._progress.cumulative_payout,
            win_level=0,  # Computed during full validation
            cascade_steps=tuple(cascade_steps),
            gravity_record=None,
        )

    @property
    def done(self) -> bool:
        """Whether the current episode has ended."""
        return self._done

    @property
    def progress(self) -> ProgressTracker | None:
        """Access the current ProgressTracker (for reward computation)."""
        return self._progress

    def _build_context(self) -> BoardContext:
        """Build BoardContext from current state."""
        return BoardContext.from_board(
            self._board,
            self._grid_mults,
            self._progress.dormant_boosters,
            self._progress.active_wilds,
            self._config.board,
        )

    def _build_observation(self) -> CascadeObservation:
        """Build observation from current state."""
        context = self._build_context()
        return self._obs_builder.build(context, self._progress, self._arc)

    def _advance_phase_if_ready(self, step_result: StepResult) -> None:
        """Advance narrative phase using greedy linear matching.

        Same logic as NarrativeArcValidator but applied online, step-by-step.
        Checks if the current phase's transition predicate fires or if max
        repetitions are reached.
        """
        phase = self._progress.current_phase()
        if phase is None:
            return

        # Check if max repetitions reached → advance
        if self._progress.current_phase_repetitions >= phase.repetitions.max_val:
            self._progress.advance_phase()
            return

        # Check transition predicate
        if phase.ends_when in self._transition_rules:
            predicate = self._transition_rules[phase.ends_when]
            context = self._build_context()
            if predicate(step_result, context):
                # Only advance if minimum repetitions are met
                if self._progress.current_phase_repetitions >= phase.repetitions.min_val:
                    self._progress.advance_phase()


def _default_hints() -> VarianceHints:
    """Create uniform variance hints when none are provided."""
    return VarianceHints(
        spatial_bias={},
        symbol_weights={},
        near_miss_symbol_preference=(),
        cluster_size_preference=(),
    )
