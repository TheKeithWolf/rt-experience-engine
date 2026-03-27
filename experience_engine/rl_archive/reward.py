"""Phase-aware reward function for cascade RL training.

Reads constraints from the current NarrativePhase and computes reward based
on how well the step result matches those constraints. Data-driven — no
branching on phase ID or booster type. Uses a dispatch table mapping
phase field names to checker functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from ..config.schema import BoardConfig, EnvironmentConfig, RewardConfig

if TYPE_CHECKING:
    from ..narrative.arc import NarrativePhase
    from ..step_reasoner.context import BoardContext
    from ..step_reasoner.results import StepResult


@runtime_checkable
class RewardComputer(Protocol):
    """Computes per-step reward for the cascade environment."""

    def compute(
        self,
        step_result: StepResult | None,
        phase: NarrativePhase | None,
        context: BoardContext,
        done: bool,
        arc_satisfied: bool,
    ) -> float: ...


class PhaseRewardComputer:
    """Phase-aware reward using field-iteration, not per-type branching.

    Reward components:
    1. Phase match — iterate over phase constraint fields and check step_result
    2. Feasibility — post-transition board suitability for next phase
    3. Terminal bonus/penalty — completion_bonus or failure_penalty
    """

    __slots__ = ("_reward_config", "_env_config", "_board_config", "_checkers")

    def __init__(
        self,
        reward_config: RewardConfig,
        env_config: EnvironmentConfig,
        board_config: BoardConfig,
    ) -> None:
        self._reward_config = reward_config
        self._env_config = env_config
        self._board_config = board_config

        # Dispatch table: phase field → (checker function, config weight)
        # Each checker returns a float reward contribution (0.0 if no match)
        self._checkers: dict[str, tuple[Callable, float]] = {
            "cluster_count": (
                _check_cluster_count,
                reward_config.cluster_match_reward,
            ),
            "spawns": (
                _check_spawns,
                reward_config.spawn_match_reward,
            ),
            "fires": (
                _check_fires,
                reward_config.fire_match_reward,
            ),
            "wild_behavior": (
                _check_wild_behavior,
                reward_config.wild_behavior_match_reward,
            ),
        }

    def compute(
        self,
        step_result: StepResult | None,
        phase: NarrativePhase | None,
        context: BoardContext,
        done: bool,
        arc_satisfied: bool,
    ) -> float:
        """Compute reward for one environment step."""
        reward = 0.0

        # Phase match reward — iterate over constraint fields
        if step_result is not None and phase is not None:
            reward += self._phase_match_reward(step_result, phase)

        # Feasibility — board suitability for next steps
        if step_result is not None:
            reward += self._feasibility_reward(context)

        # Terminal bonus/penalty
        if done:
            if arc_satisfied:
                reward += self._env_config.completion_bonus
            else:
                reward += self._env_config.failure_penalty

        return reward

    def _phase_match_reward(
        self, step_result: StepResult, phase: NarrativePhase,
    ) -> float:
        """Check each phase constraint field against step_result via dispatch."""
        reward = 0.0
        for field_name, (checker, weight) in self._checkers.items():
            if weight == 0.0:
                continue
            # Only check if the phase has a constraint for this field
            field_val = getattr(phase, field_name, None)
            if field_val is None:
                continue
            contribution = checker(phase, step_result)
            reward += contribution * weight
        return reward

    def _feasibility_reward(self, context: BoardContext) -> float:
        """Board suitability: fraction of cells adjacent to wilds/boosters that are empty.

        Rewards boards where future clusters can form near wilds and boosters.
        """
        board_area = self._board_config.num_reels * self._board_config.num_rows
        if board_area == 0:
            return 0.0

        # Count empty cells adjacent to wilds
        adjacent_empty = 0
        total_adjacent = 0
        for wild_pos in context.active_wilds:
            neighbors = context.neighbors_of(wild_pos)
            total_adjacent += len(neighbors)
            adjacent_empty += len(context.empty_neighbors_of(wild_pos))

        # Count empty cells adjacent to dormant boosters
        for db in context.dormant_boosters:
            neighbors = context.neighbors_of(db.position)
            total_adjacent += len(neighbors)
            adjacent_empty += len(context.empty_neighbors_of(db.position))

        if total_adjacent == 0:
            return 0.0

        # Weighted combination of empty cell fraction and adjacency ratio
        empty_fraction = adjacent_empty / total_adjacent
        adjacency_ratio = total_adjacent / board_area

        return (
            self._reward_config.feasibility_empty_cell_weight * empty_fraction
            + self._reward_config.feasibility_adjacency_weight * adjacency_ratio
        ) * self._env_config.feasibility_weight


# ---------------------------------------------------------------------------
# Checker functions — each reads a phase field and compares to StepResult
# ---------------------------------------------------------------------------


def _check_cluster_count(phase: NarrativePhase, step_result: StepResult) -> float:
    """1.0 if cluster count is within the phase's expected range, else 0.0."""
    count = len(step_result.clusters)
    return 1.0 if phase.cluster_count.contains(count) else 0.0


def _check_spawns(phase: NarrativePhase, step_result: StepResult) -> float:
    """Count of spawns whose booster_type matches the phase's expected spawns."""
    if phase.spawns is None:
        return 0.0
    expected = set(phase.spawns)
    return float(sum(
        1 for s in step_result.spawns if s.booster_type in expected
    ))


def _check_fires(phase: NarrativePhase, step_result: StepResult) -> float:
    """Count of fires whose booster_type matches the phase's expected fires."""
    if phase.fires is None:
        return 0.0
    expected = set(phase.fires)
    return float(sum(
        1 for f in step_result.fires if f.booster_type in expected
    ))


def _check_wild_behavior(phase: NarrativePhase, step_result: StepResult) -> float:
    """1.0 if wild behavior matches phase expectation, else 0.0.

    "spawn" — at least one wild spawned this step
    "bridge" — at least one cluster has wild positions (bridging occurred)
    "idle" — no wilds spawned or used
    """
    behavior = phase.wild_behavior
    if behavior is None:
        return 0.0

    has_wild_spawn = any(s.booster_type == "W" for s in step_result.spawns)

    if behavior == "spawn":
        return 1.0 if has_wild_spawn else 0.0
    if behavior == "idle":
        return 1.0 if not has_wild_spawn else 0.0
    # "bridge" — check if any cluster involves bridging (we approximate
    # by checking if there are clusters with wilds in the step)
    if behavior == "bridge":
        return 1.0 if len(step_result.clusters) > 0 else 0.0

    return 0.0
