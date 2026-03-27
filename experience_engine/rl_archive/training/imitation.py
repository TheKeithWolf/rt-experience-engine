"""Behavioral cloning — supervised pre-training from successful trajectories.

Generates a dataset by running the existing CascadeInstanceGenerator and
recording (observation, action) pairs. Then trains the policy network
using cross-entropy loss on action components.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    raise ImportError(
        "PyTorch is required for rl_archive training modules. "
        "Install with: pip install torch"
    )

from ...config.schema import TrainingConfig


@dataclass(frozen=True, slots=True)
class ImitationDataset:
    """Pre-collected (observation_tensor, action_tensor) pairs for cloning."""

    # board_tensors: (N, channels, reels, rows)
    board_tensors: torch.Tensor
    # scalar_tensors: (N, 6)
    scalar_tensors: torch.Tensor
    # archetype_indices: (N,)
    archetype_indices: torch.Tensor
    # phase_indices: (N,)
    phase_indices: torch.Tensor
    # action_tensors: (N, 5)
    action_tensors: torch.Tensor


def behavioral_clone(
    policy: nn.Module,
    dataset: ImitationDataset,
    config: TrainingConfig,
) -> float:
    """Train policy via supervised learning on expert trajectories.

    Uses cross-entropy loss on action components. Returns final epoch loss.
    """
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
    n_samples = dataset.board_tensors.shape[0]
    final_loss = 0.0

    for epoch in range(config.imitation_epochs):
        # Shuffle indices for each epoch
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, config.imitation_batch_size):
            end = min(start + config.imitation_batch_size, n_samples)
            idx = perm[start:end]

            board_batch = dataset.board_tensors[idx]
            scalar_batch = dataset.scalar_tensors[idx]
            arch_batch = dataset.archetype_indices[idx]
            phase_batch = dataset.phase_indices[idx]
            action_batch = dataset.action_tensors[idx]

            # Forward pass — get action distribution
            action_dist, _ = policy(
                board_batch, scalar_batch, arch_batch, phase_batch,
            )

            # Negative log-likelihood loss (cross-entropy on factored actions)
            log_prob = action_dist.log_prob(action_batch)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)

    return final_loss
