"""Board and scalar encoders for the cascade policy network.

BoardEncoder processes the multi-channel board tensor through a CNN.
ScalarEncoder processes progress scalars through an MLP.
ArchetypeEmbedding and PhaseEmbedding provide goal conditioning.

All spatial dimensions derived from config — no hardcoded board sizes.
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

from ...config.schema import PolicyConfig


class BoardEncoder(nn.Module):
    """CNN encoder for the multi-channel board tensor.

    Input shape: (batch, in_channels, num_reels, num_rows)
    Output shape: (batch, cnn_filters * num_reels * num_rows)

    Uses 3x3 convolutions with padding=1 to preserve spatial dimensions.
    No pooling — spatial resolution matters for centroid prediction.
    """

    def __init__(
        self,
        in_channels: int,
        num_reels: int,
        num_rows: int,
        policy_config: PolicyConfig,
    ) -> None:
        super().__init__()
        self._num_reels = num_reels
        self._num_rows = num_rows
        self._out_features = policy_config.cnn_filters * num_reels * num_rows

        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(policy_config.cnn_layers):
            layers.append(
                nn.Conv2d(current_channels, policy_config.cnn_filters,
                          kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU())
            current_channels = policy_config.cnn_filters
        self._cnn = nn.Sequential(*layers)

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, reels, rows) -> (batch, flat_features)"""
        features = self._cnn(x)
        return features.flatten(start_dim=1)


class ScalarEncoder(nn.Module):
    """MLP encoder for scalar progress features.

    Input: (batch, num_scalars)
    Output: (batch, trunk_hidden)
    """

    _NUM_SCALARS = 6  # phase_index, phase_rep, phases_remaining, cum_payout, pay_min, pay_max

    def __init__(self, trunk_hidden: int) -> None:
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(self._NUM_SCALARS, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 6) -> (batch, trunk_hidden)"""
        return self._mlp(x)


class ArchetypeEmbedding(nn.Module):
    """Learned embedding for archetype conditioning.

    Vocabulary size = number of registered archetypes (passed at init).
    """

    def __init__(self, num_archetypes: int, embedding_dim: int) -> None:
        super().__init__()
        self._embedding = nn.Embedding(num_archetypes, embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self._embedding.embedding_dim

    def forward(self, archetype_idx: torch.Tensor) -> torch.Tensor:
        """(batch,) int -> (batch, embedding_dim)"""
        return self._embedding(archetype_idx)


class PhaseEmbedding(nn.Module):
    """Learned embedding for narrative phase conditioning.

    Vocabulary size = max phase count across all registered arcs.
    """

    def __init__(self, max_phases: int, embedding_dim: int) -> None:
        super().__init__()
        self._embedding = nn.Embedding(max_phases, embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self._embedding.embedding_dim

    def forward(self, phase_idx: torch.Tensor) -> torch.Tensor:
        """(batch,) int -> (batch, embedding_dim)"""
        return self._embedding(phase_idx)
