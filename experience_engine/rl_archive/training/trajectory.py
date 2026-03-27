"""Trajectory data structures and GAE advantage computation.

TrajectoryStep and Trajectory record one episode's worth of
(observation, action, reward, value, log_prob) tuples for PPO training.
compute_gae implements Generalized Advantage Estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for rl_archive training modules. "
        "Install with: pip install torch"
    )

from ..action_space import CascadeAction
from ..observation import CascadeObservation
from ...pipeline.data_types import GeneratedInstance


@dataclass(slots=True)
class TrajectoryStep:
    """One step in a training trajectory."""

    observation: CascadeObservation
    action: CascadeAction
    reward: float
    value: float
    log_prob: float
    done: bool


@dataclass(slots=True)
class Trajectory:
    """Complete episode trajectory for PPO training."""

    steps: list[TrajectoryStep] = field(default_factory=list)
    archetype_id: str = ""
    completed: bool = False
    instance: GeneratedInstance | None = None


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation.

    Returns (advantages, returns) as 1D tensors of length len(rewards).
    Uses the standard GAE(gamma, lambda) recursion from Schulman et al. 2016.
    """
    n = len(rewards)
    advantages = torch.zeros(n, dtype=torch.float32)
    last_advantage = 0.0

    for t in reversed(range(n)):
        # Next value is 0 if episode ended at this step
        next_value = values[t + 1] if t + 1 < n and not dones[t] else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        # GAE recursion: A_t = delta_t + gamma * lambda * A_{t+1}
        mask = 0.0 if dones[t] else 1.0
        last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        advantages[t] = last_advantage

    returns = advantages + torch.tensor(values[:n], dtype=torch.float32)
    return advantages, returns
