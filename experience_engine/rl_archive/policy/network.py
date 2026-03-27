"""Cascade policy network — goal-conditioned actor-critic for cascade generation.

Combines BoardEncoder (CNN), ScalarEncoder (MLP), archetype/phase embeddings,
and PolicyHead/ValueHead into a single nn.Module. All dimensions from config.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is required for rl_archive policy modules. "
        "Install with: pip install torch"
    )

from ...config.schema import BoardConfig, PolicyConfig
from ..observation import CascadeObservation, ObservationEncoder
from .encoder import (
    ArchetypeEmbedding,
    BoardEncoder,
    PhaseEmbedding,
    ScalarEncoder,
)
from .heads import ActionDistribution, PolicyHead, ValueHead


class CascadePolicy(nn.Module):
    """Goal-conditioned actor-critic for cascade instance generation.

    Architecture:
    1. BoardEncoder (CNN) processes board channels → flat features
    2. ScalarEncoder (MLP) processes progress scalars
    3. Archetype + Phase embeddings provide goal conditioning
    4. Trunk MLP combines all features
    5. PolicyHead → factored action distribution
    6. ValueHead → scalar V(s)
    """

    def __init__(
        self,
        board_config: BoardConfig,
        num_symbols: int,
        policy_config: PolicyConfig,
        num_archetypes: int,
        max_phases: int,
        obs_encoder: ObservationEncoder,
    ) -> None:
        super().__init__()
        self._board_config = board_config
        self._num_symbols = num_symbols
        self._policy_config = policy_config
        self._obs_encoder = obs_encoder

        # Board CNN — input channels = symbol one-hot + extra board channels
        in_channels = num_symbols + policy_config.board_channels
        self._board_encoder = BoardEncoder(
            in_channels=in_channels,
            num_reels=board_config.num_reels,
            num_rows=board_config.num_rows,
            policy_config=policy_config,
        )

        # Scalar MLP
        self._scalar_encoder = ScalarEncoder(policy_config.trunk_hidden)

        # Goal conditioning embeddings
        self._archetype_emb = ArchetypeEmbedding(
            num_archetypes, policy_config.archetype_embedding_dim,
        )
        self._phase_emb = PhaseEmbedding(
            max_phases, policy_config.phase_embedding_dim,
        )

        # Trunk MLP — combines all feature streams
        trunk_input_dim = (
            self._board_encoder.out_features
            + policy_config.trunk_hidden  # scalar encoder output
            + policy_config.archetype_embedding_dim
            + policy_config.phase_embedding_dim
        )
        self._trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, policy_config.trunk_hidden),
            nn.ReLU(),
            nn.Linear(policy_config.trunk_hidden, policy_config.trunk_hidden),
            nn.ReLU(),
        )

        # Action size range: min_cluster_size to board_area
        board_area = board_config.num_reels * board_config.num_rows
        size_range = board_area - board_config.min_cluster_size + 1

        # Output heads
        self._policy_head = PolicyHead(
            trunk_dim=policy_config.trunk_hidden,
            num_symbols=num_symbols,
            size_range=size_range,
            num_reels=board_config.num_reels,
            num_rows=board_config.num_rows,
        )
        self._value_head = ValueHead(policy_config.trunk_hidden)

    def forward(
        self,
        board_tensor: torch.Tensor,
        scalar_tensor: torch.Tensor,
        archetype_idx: torch.Tensor,
        phase_idx: torch.Tensor,
    ) -> tuple[ActionDistribution, torch.Tensor]:
        """Full forward pass.

        Args:
            board_tensor: (batch, channels, reels, rows) float
            scalar_tensor: (batch, 6) float
            archetype_idx: (batch,) long
            phase_idx: (batch,) long

        Returns:
            (ActionDistribution, value: (batch, 1))
        """
        board_features = self._board_encoder(board_tensor)
        scalar_features = self._scalar_encoder(scalar_tensor)
        archetype_features = self._archetype_emb(archetype_idx)
        phase_features = self._phase_emb(phase_idx)

        trunk_input = torch.cat([
            board_features, scalar_features,
            archetype_features, phase_features,
        ], dim=-1)

        trunk = self._trunk(trunk_input)
        action_dist = self._policy_head(trunk)
        value = self._value_head(trunk)

        return action_dist, value

    def act(
        self,
        obs: CascadeObservation,
        archetype_idx: int,
        device: torch.device | None = None,
    ) -> tuple[ActionDistribution, torch.Tensor]:
        """Single observation → action distribution + value (no grad).

        Convenience method for inference during trajectory collection.
        """
        dev = device or next(self.parameters()).device

        board = torch.from_numpy(
            self._obs_encoder.encode_board(obs)
        ).unsqueeze(0).to(dev)

        scalars = torch.from_numpy(
            self._obs_encoder.encode_scalars(obs)
        ).unsqueeze(0).to(dev)

        arch_idx = torch.tensor([archetype_idx], dtype=torch.long, device=dev)
        phase_idx = torch.tensor(
            [obs.current_phase_index], dtype=torch.long, device=dev,
        )

        with torch.no_grad():
            return self.forward(board, scalars, arch_idx, phase_idx)

    def evaluate(
        self,
        board_tensor: torch.Tensor,
        scalar_tensor: torch.Tensor,
        archetype_idx: torch.Tensor,
        phase_idx: torch.Tensor,
        action_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns: (log_prob, value, entropy) — all (batch,) or (batch, 1)
        """
        action_dist, value = self.forward(
            board_tensor, scalar_tensor, archetype_idx, phase_idx,
        )
        log_prob = action_dist.log_prob(action_tensor)
        entropy = action_dist.entropy()
        return log_prob, value.squeeze(-1), entropy
