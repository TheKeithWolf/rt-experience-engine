"""PPO trainer — Proximal Policy Optimization for cascade generation.

Collects trajectories via CascadeEnvironment, computes GAE advantages,
runs clipped policy gradient updates, and inserts valid completed
trajectories into the MAP-Elites archive.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    raise ImportError(
        "PyTorch is required for rl_archive training modules. "
        "Install with: pip install torch"
    )

from ...config.schema import MasterConfig, TrainingConfig
from ..archive import MAPElitesArchive
from ..descriptor import CascadeDescriptorExtractor
from ..observation import ObservationEncoder
from ..quality import CascadeQualityScorer
from .curriculum import CurriculumScheduler
from .reporter import BatchMetrics, TrainingCompleteSummary, TrainingReporter
from .trajectory import Trajectory, TrajectoryStep, compute_gae

if TYPE_CHECKING:
    from ..environment import CascadeEnvironment
    from ..policy.network import CascadePolicy
    from ..reward import RewardComputer


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary of a completed training run."""

    total_episodes: int
    wall_time_seconds: float
    archive_filled: int
    archive_total: int
    final_completion_rate: float


class PPOTrainer:
    """PPO training loop with curriculum and archive integration.

    Per batch:
    1. Select archetype from curriculum
    2. Collect batch_size trajectories via environment rollouts
    3. Compute GAE advantages
    4. Run epochs_per_batch PPO update passes with clip
    5. Insert completed valid trajectories into archive
    6. Report progress
    """

    __slots__ = (
        "_policy", "_environment", "_archive", "_config", "_training_config",
        "_curriculum", "_reporter", "_reward_computer",
        "_descriptor_extractor", "_quality_scorer",
        "_optimizer", "_obs_encoder", "_device",
    )

    def __init__(
        self,
        policy: CascadePolicy,
        environment: CascadeEnvironment,
        archive: MAPElitesArchive,
        config: MasterConfig,
        training_config: TrainingConfig,
        curriculum: CurriculumScheduler,
        reporter: TrainingReporter,
        reward_computer: RewardComputer,
        descriptor_extractor: CascadeDescriptorExtractor,
        quality_scorer: CascadeQualityScorer,
        device: torch.device | None = None,
    ) -> None:
        self._policy = policy
        self._environment = environment
        self._archive = archive
        self._config = config
        self._training_config = training_config
        self._curriculum = curriculum
        self._reporter = reporter
        self._reward_computer = reward_computer
        self._descriptor_extractor = descriptor_extractor
        self._quality_scorer = quality_scorer
        self._optimizer = optim.Adam(
            policy.parameters(), lr=training_config.learning_rate,
        )
        self._obs_encoder = ObservationEncoder(config.symbols, config.board)
        # Infer device from the policy parameters if not explicitly provided
        self._device = device or next(policy.parameters()).device

    def train(self, rng: random.Random) -> TrainingResult:
        """Run the full training loop until max_training_episodes."""
        tc = self._training_config
        start_time = time.monotonic()
        total_episodes = 0
        batch_index = 0
        stale_streak = 0
        recent_completions: list[bool] = []

        while total_episodes < tc.max_training_episodes:
            batch_start_ep = total_episodes

            # Collect trajectories for one batch
            trajectories: list[Trajectory] = []
            for _ in range(tc.batch_size):
                traj = self._collect_trajectory(rng)
                trajectories.append(traj)
                total_episodes += 1
                recent_completions.append(traj.completed)

                if total_episodes >= tc.max_training_episodes:
                    break

            # Compute advantages and run PPO update
            policy_loss, value_loss, entropy, clip_frac = self._update_policy(
                trajectories,
            )

            # Insert completed trajectories into archive
            new_niches, replacements = self._insert_into_archive(trajectories)

            # Track staleness
            if new_niches == 0 and replacements == 0:
                stale_streak += 1
            else:
                stale_streak = 0

            # Rolling completion rate
            window = tc.reporter.completion_rolling_window
            if len(recent_completions) > window:
                recent_completions = recent_completions[-window:]
            completion_rate = (
                sum(recent_completions) / len(recent_completions)
                if recent_completions else 0.0
            )

            # Completion trend (bucketized)
            buckets = tc.reporter.completion_trend_buckets
            trend = self._compute_trend(recent_completions, buckets)

            # Report
            elapsed = time.monotonic() - start_time
            metrics = BatchMetrics(
                batch_index=batch_index,
                episode_range=(batch_start_ep, total_episodes - 1),
                total_episodes=total_episodes,
                max_episodes=tc.max_training_episodes,
                elapsed_seconds=elapsed,
                completed_count=sum(1 for t in trajectories if t.completed),
                batch_size=len(trajectories),
                completion_rate_rolling=completion_rate,
                completion_trend=trend,
                mean_reward=self._mean_reward(trajectories),
                best_reward=self._best_reward(trajectories),
                archive_filled=self._archive.filled_count(),
                archive_total=self._archive.total_cells(),
                new_niches=new_niches,
                replacements=replacements,
                stale_batch_streak=stale_streak,
                new_niche_details=(),
                coverage_by_axis={},
                step_failure_distribution={},
                phase_completion_rates={},
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                learning_rate=tc.learning_rate,
                clip_fraction=clip_frac,
                quality_min=None,
                quality_p25=None,
                quality_p50=None,
                quality_p75=None,
                quality_max=None,
            )
            self._reporter.on_batch_complete(metrics)
            batch_index += 1

            # Checkpoint
            if total_episodes % tc.checkpoint_interval < tc.batch_size:
                self._save_checkpoint(total_episodes)

        wall_time = time.monotonic() - start_time
        return TrainingResult(
            total_episodes=total_episodes,
            wall_time_seconds=wall_time,
            archive_filled=self._archive.filled_count(),
            archive_total=self._archive.total_cells(),
            final_completion_rate=(
                sum(recent_completions) / len(recent_completions)
                if recent_completions else 0.0
            ),
        )

    def _collect_trajectory(self, rng: random.Random) -> Trajectory:
        """Run one episode and record the trajectory."""
        obs = self._environment.reset(rng)
        traj = Trajectory(archetype_id=self._environment.progress.signature.id)

        while not self._environment.done:
            # Get action from policy
            action_dist, value = self._policy.act(obs, archetype_idx=0)
            action = action_dist.sample()
            value_scalar = value.squeeze().item()
            log_prob = action_dist.log_prob(
                torch.tensor([[
                    action.cluster_symbol_index,
                    action.cluster_size,
                    action.centroid_col,
                    action.centroid_row,
                    float(action.is_terminal),
                ]])
            ).item()

            # Step environment
            next_obs, env_reward, done, info = self._environment.step(action)

            # Compute reward via RewardComputer
            phase = self._environment.progress.current_phase()
            context = self._environment._build_context()
            reward = self._reward_computer.compute(
                info.step_result, phase, context, done, arc_satisfied=False,
            )

            traj.steps.append(TrajectoryStep(
                observation=obs,
                action=action,
                reward=reward,
                value=value_scalar,
                log_prob=log_prob,
                done=done,
            ))

            obs = next_obs

        # Check if trajectory satisfies the arc (ground truth)
        instance = self._environment.current_instance()
        traj.instance = instance
        traj.completed = instance is not None

        return traj

    def _update_policy(
        self, trajectories: list[Trajectory],
    ) -> tuple[float, float, float, float]:
        """Run PPO update passes on collected trajectories.

        Returns (policy_loss, value_loss, entropy, clip_fraction).
        """
        import numpy as np

        tc = self._training_config
        enc = self._obs_encoder

        # Flatten all steps and encode observations
        all_rewards: list[float] = []
        all_values: list[float] = []
        all_dones: list[bool] = []
        all_log_probs: list[float] = []
        all_actions: list[list[float]] = []
        all_boards: list[np.ndarray] = []
        all_scalars: list[np.ndarray] = []
        all_phase_indices: list[int] = []

        for traj in trajectories:
            for step in traj.steps:
                all_rewards.append(step.reward)
                all_values.append(step.value)
                all_dones.append(step.done)
                all_log_probs.append(step.log_prob)
                all_actions.append([
                    step.action.cluster_symbol_index,
                    step.action.cluster_size,
                    step.action.centroid_col,
                    step.action.centroid_row,
                    float(step.action.is_terminal),
                ])
                all_boards.append(enc.encode_board(step.observation))
                all_scalars.append(enc.encode_scalars(step.observation))
                all_phase_indices.append(step.observation.current_phase_index)

        if not all_rewards:
            return 0.0, 0.0, 0.0, 0.0

        advantages, returns = compute_gae(
            all_rewards, all_values, all_dones,
            tc.gamma, tc.gae_lambda,
        )

        # Move all training tensors to the policy's device (CPU or GPU)
        dev = self._device
        old_log_probs = torch.tensor(all_log_probs, dtype=torch.float32, device=dev)
        action_tensor = torch.tensor(all_actions, dtype=torch.float32, device=dev)
        board_tensor = torch.from_numpy(np.stack(all_boards)).to(dev)
        scalar_tensor = torch.from_numpy(np.stack(all_scalars)).to(dev)
        arch_tensor = torch.zeros(len(all_rewards), dtype=torch.long, device=dev)
        phase_tensor = torch.tensor(all_phase_indices, dtype=torch.long, device=dev)
        advantages = advantages.to(dev)
        returns = returns.to(dev)

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        for _ in range(tc.epochs_per_batch):
            log_prob, value, entropy = self._policy.evaluate(
                board_tensor, scalar_tensor, arch_tensor, phase_tensor,
                action_tensor,
            )

            # PPO clipped surrogate loss
            ratio = torch.exp(log_prob - old_log_probs)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - tc.clip_epsilon, 1.0 + tc.clip_epsilon,
            )
            policy_loss = -torch.min(
                ratio * advantages, clipped_ratio * advantages,
            ).mean()

            value_loss = nn.functional.mse_loss(value, returns)
            entropy_loss = -entropy.mean() * tc.clip_epsilon

            loss = policy_loss + 0.5 * value_loss + entropy_loss

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            clip_frac = (
                (ratio - 1.0).abs() > tc.clip_epsilon
            ).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_clip_frac += clip_frac
            n_updates += 1

        n = max(n_updates, 1)
        return (
            total_policy_loss / n,
            total_value_loss / n,
            total_entropy / n,
            total_clip_frac / n,
        )

    def _insert_into_archive(
        self, trajectories: list[Trajectory],
    ) -> tuple[int, int]:
        """Insert completed trajectories into the MAP-Elites archive.

        Returns (new_niches, replacements).
        """
        new_niches = 0
        replacements = 0
        sig = self._environment._signature

        for traj in trajectories:
            if not traj.completed or traj.instance is None:
                continue

            try:
                descriptor = self._descriptor_extractor.extract(
                    traj.instance, sig.payout_range,
                )
                quality = self._quality_scorer.score(traj.instance, sig.narrative_arc)
                was_empty = self._archive.get(
                    self._descriptor_extractor.to_key(descriptor)
                ) is None
                inserted = self._archive.try_insert(
                    traj.instance, descriptor, quality,
                )
                if inserted:
                    if was_empty:
                        new_niches += 1
                    else:
                        replacements += 1
            except (ValueError, AttributeError):
                # Skip instances that can't be processed
                continue

        return new_niches, replacements

    def _save_checkpoint(self, episode_count: int) -> None:
        """Save policy weights and archive state."""
        # Derive checkpoint dir from the generator's archive_dir config
        base_dir = Path("games/royal_tumble/experience_engine/library/archives")
        if (self._config.rl_archive is not None
                and self._config.rl_archive.generator is not None):
            base_dir = Path(self._config.rl_archive.generator.archive_dir)
        checkpoint_dir = base_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"checkpoint_ep{episode_count}.pt"
        torch.save({
            "policy_state_dict": self._policy.state_dict(),
            "episode_count": episode_count,
            "archive_filled": self._archive.filled_count(),
        }, path)
        self._reporter.on_checkpoint(path)

    @staticmethod
    def _compute_trend(
        completions: list[bool], buckets: int,
    ) -> tuple[float, ...]:
        """Compute completion rate trend over bucketized windows."""
        if not completions or buckets < 1:
            return ()
        bucket_size = max(1, len(completions) // buckets)
        trend: list[float] = []
        for i in range(0, len(completions), bucket_size):
            chunk = completions[i:i + bucket_size]
            trend.append(sum(chunk) / len(chunk) if chunk else 0.0)
        return tuple(trend[-buckets:])

    @staticmethod
    def _mean_reward(trajectories: list[Trajectory]) -> float:
        all_rewards = [s.reward for t in trajectories for s in t.steps]
        return sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    @staticmethod
    def _best_reward(trajectories: list[Trajectory]) -> float:
        all_rewards = [s.reward for t in trajectories for s in t.steps]
        return max(all_rewards) if all_rewards else 0.0
