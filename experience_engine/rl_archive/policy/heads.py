"""Policy and value heads for the cascade policy network.

PolicyHead outputs a factored action distribution over CascadeAction fields.
ValueHead outputs a scalar state value V(s).

The factored distribution treats each action component as independent given
the trunk features. log_prob and entropy are sums of individual components.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Bernoulli, Categorical
except ImportError:
    raise ImportError(
        "PyTorch is required for rl_archive policy modules. "
        "Install with: pip install torch"
    )

from ..action_space import CascadeAction


@dataclass
class ActionDistribution:
    """Factored distribution over CascadeAction components.

    Each component is an independent distribution conditioned on trunk features.
    """

    symbol_dist: Categorical
    size_dist: Categorical
    col_dist: Categorical
    row_dist: Categorical
    terminal_dist: Bernoulli

    def sample(self) -> CascadeAction:
        """Sample one action from the joint distribution.

        For batched distributions, takes the first sample (index 0).
        Use sample_batch() for full-batch sampling.
        """
        sym = self.symbol_dist.sample()
        size = self.size_dist.sample()
        col = self.col_dist.sample()
        row = self.row_dist.sample()
        term = self.terminal_dist.sample()

        # Handle both batched and unbatched distributions
        idx = 0 if sym.dim() > 0 else ...
        return CascadeAction(
            cluster_symbol_index=int(sym[idx].item() if sym.dim() > 0 else sym.item()),
            cluster_size=int(size[idx].item() if size.dim() > 0 else size.item()),
            centroid_col=int(col[idx].item() if col.dim() > 0 else col.item()),
            centroid_row=int(row[idx].item() if row.dim() > 0 else row.item()),
            is_terminal=bool(
                (term[idx].item() if term.dim() > 0 else term.item()) > 0.5
            ),
        )

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Log probability of an action tensor.

        action shape: (batch, 5) — [symbol_idx, size, col, row, terminal]
        Returns: (batch,) total log prob
        """
        return (
            self.symbol_dist.log_prob(action[:, 0])
            + self.size_dist.log_prob(action[:, 1])
            + self.col_dist.log_prob(action[:, 2])
            + self.row_dist.log_prob(action[:, 3])
            + self.terminal_dist.log_prob(action[:, 4])
        )

    def entropy(self) -> torch.Tensor:
        """Sum of component entropies. Returns: (batch,) or scalar."""
        return (
            self.symbol_dist.entropy()
            + self.size_dist.entropy()
            + self.col_dist.entropy()
            + self.row_dist.entropy()
            + self.terminal_dist.entropy()
        )


class PolicyHead(nn.Module):
    """Outputs a factored ActionDistribution from trunk features.

    Action components:
    - cluster_symbol_index: Categorical over num_symbols
    - cluster_size: Categorical over size_range (min_cluster_size..board_area)
    - centroid_col: Categorical over num_reels
    - centroid_row: Categorical over num_rows
    - is_terminal: Bernoulli
    """

    def __init__(
        self,
        trunk_dim: int,
        num_symbols: int,
        size_range: int,
        num_reels: int,
        num_rows: int,
    ) -> None:
        super().__init__()
        self._symbol_head = nn.Linear(trunk_dim, num_symbols)
        self._size_head = nn.Linear(trunk_dim, size_range)
        self._col_head = nn.Linear(trunk_dim, num_reels)
        self._row_head = nn.Linear(trunk_dim, num_rows)
        self._terminal_head = nn.Linear(trunk_dim, 1)

    def forward(self, trunk: torch.Tensor) -> ActionDistribution:
        """(batch, trunk_dim) -> ActionDistribution"""
        return ActionDistribution(
            symbol_dist=Categorical(logits=self._symbol_head(trunk)),
            size_dist=Categorical(logits=self._size_head(trunk)),
            col_dist=Categorical(logits=self._col_head(trunk)),
            row_dist=Categorical(logits=self._row_head(trunk)),
            terminal_dist=Bernoulli(logits=self._terminal_head(trunk).squeeze(-1)),
        )


class ValueHead(nn.Module):
    """Scalar state value V(s) from trunk features."""

    def __init__(self, trunk_dim: int) -> None:
        super().__init__()
        self._linear = nn.Linear(trunk_dim, 1)

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        """(batch, trunk_dim) -> (batch, 1)"""
        return self._linear(trunk)
