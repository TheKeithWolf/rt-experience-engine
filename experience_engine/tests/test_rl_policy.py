"""Tests for rl_archive/policy/ — encoder, heads, and network.

RLA-060 through RLA-069: Output shapes, config-driven dimensions,
finite log_probs, positive entropy, determinism.

Uses tiny config (3x3 board, 3 symbols) for speed.
"""

from __future__ import annotations

import pytest
import torch

from ..config.schema import BoardConfig, PolicyConfig, SymbolConfig
from ..rl_archive.observation import CascadeObservation, ObservationEncoder
from ..rl_archive.policy.encoder import (
    ArchetypeEmbedding,
    BoardEncoder,
    PhaseEmbedding,
    ScalarEncoder,
)
from ..rl_archive.policy.heads import PolicyHead, ValueHead
from ..rl_archive.policy.network import CascadePolicy


# ---------------------------------------------------------------------------
# Tiny config for fast tests
# ---------------------------------------------------------------------------

_TINY_BOARD = BoardConfig(num_reels=3, num_rows=3, min_cluster_size=2)
_TINY_SYMBOLS = SymbolConfig(
    standard=("A", "B", "C"),
    low_tier=("A",),
    high_tier=("B", "C"),
    payout_rank=(("A", 1), ("B", 2), ("C", 3)),
)
_TINY_POLICY = PolicyConfig(
    board_channels=4,
    cnn_filters=8,
    cnn_layers=1,
    trunk_hidden=16,
    archetype_embedding_dim=4,
    phase_embedding_dim=4,
    entropy_coefficient=0.01,
)
_NUM_SYMBOLS = 3
_NUM_ARCHETYPES = 5
_MAX_PHASES = 4
_BATCH = 2


# ---------------------------------------------------------------------------
# RLA-060: BoardEncoder output shape
# ---------------------------------------------------------------------------


def test_rla_060_board_encoder_output_shape() -> None:
    """RLA-060: BoardEncoder output shape matches expected flattened dimension."""
    in_channels = _NUM_SYMBOLS + _TINY_POLICY.board_channels  # 3 + 4 = 7
    encoder = BoardEncoder(in_channels, _TINY_BOARD.num_reels, _TINY_BOARD.num_rows, _TINY_POLICY)

    x = torch.randn(_BATCH, in_channels, _TINY_BOARD.num_reels, _TINY_BOARD.num_rows)
    out = encoder(x)

    # Output should be (batch, cnn_filters * num_reels * num_rows) = (2, 8*3*3) = (2, 72)
    expected_features = _TINY_POLICY.cnn_filters * _TINY_BOARD.num_reels * _TINY_BOARD.num_rows
    assert out.shape == (_BATCH, expected_features)


# ---------------------------------------------------------------------------
# RLA-061: BoardEncoder kernel adapts to BoardConfig
# ---------------------------------------------------------------------------


def test_rla_061_board_encoder_adapts_to_config() -> None:
    """RLA-061: BoardEncoder works with different board sizes."""
    # 5x5 board
    board_5 = BoardConfig(num_reels=5, num_rows=5, min_cluster_size=3)
    in_channels = _NUM_SYMBOLS + _TINY_POLICY.board_channels
    encoder = BoardEncoder(in_channels, board_5.num_reels, board_5.num_rows, _TINY_POLICY)

    x = torch.randn(1, in_channels, 5, 5)
    out = encoder(x)

    expected = _TINY_POLICY.cnn_filters * 5 * 5
    assert out.shape == (1, expected)


# ---------------------------------------------------------------------------
# RLA-062: ArchetypeEmbedding vocabulary size
# ---------------------------------------------------------------------------


def test_rla_062_archetype_embedding_vocab() -> None:
    """RLA-062: ArchetypeEmbedding vocabulary size = num_archetypes."""
    emb = ArchetypeEmbedding(_NUM_ARCHETYPES, _TINY_POLICY.archetype_embedding_dim)
    assert emb._embedding.num_embeddings == _NUM_ARCHETYPES

    # All valid indices should work
    idx = torch.arange(_NUM_ARCHETYPES)
    out = emb(idx)
    assert out.shape == (_NUM_ARCHETYPES, _TINY_POLICY.archetype_embedding_dim)


# ---------------------------------------------------------------------------
# RLA-063: PhaseEmbedding vocabulary size
# ---------------------------------------------------------------------------


def test_rla_063_phase_embedding_vocab() -> None:
    """RLA-063: PhaseEmbedding vocabulary size = max_phases."""
    emb = PhaseEmbedding(_MAX_PHASES, _TINY_POLICY.phase_embedding_dim)
    assert emb._embedding.num_embeddings == _MAX_PHASES


# ---------------------------------------------------------------------------
# RLA-064: PolicyHead log_prob returns finite values
# ---------------------------------------------------------------------------


def test_rla_064_policy_head_finite_log_prob() -> None:
    """RLA-064: PolicyHead log_prob returns finite values."""
    board_area = _TINY_BOARD.num_reels * _TINY_BOARD.num_rows
    size_range = board_area - _TINY_BOARD.min_cluster_size + 1

    head = PolicyHead(
        trunk_dim=_TINY_POLICY.trunk_hidden,
        num_symbols=_NUM_SYMBOLS,
        size_range=size_range,
        num_reels=_TINY_BOARD.num_reels,
        num_rows=_TINY_BOARD.num_rows,
    )

    trunk = torch.randn(_BATCH, _TINY_POLICY.trunk_hidden)
    dist = head(trunk)

    # Build action tensor with valid indices
    action = torch.zeros(_BATCH, 5)
    action[:, 0] = 0  # symbol
    action[:, 1] = 0  # size
    action[:, 2] = 1  # col
    action[:, 3] = 1  # row
    action[:, 4] = 0.0  # terminal

    log_prob = dist.log_prob(action)
    assert torch.isfinite(log_prob).all()


# ---------------------------------------------------------------------------
# RLA-065: PolicyHead entropy returns positive values
# ---------------------------------------------------------------------------


def test_rla_065_policy_head_positive_entropy() -> None:
    """RLA-065: PolicyHead entropy returns positive values."""
    board_area = _TINY_BOARD.num_reels * _TINY_BOARD.num_rows
    size_range = board_area - _TINY_BOARD.min_cluster_size + 1

    head = PolicyHead(
        trunk_dim=_TINY_POLICY.trunk_hidden,
        num_symbols=_NUM_SYMBOLS,
        size_range=size_range,
        num_reels=_TINY_BOARD.num_reels,
        num_rows=_TINY_BOARD.num_rows,
    )

    trunk = torch.randn(_BATCH, _TINY_POLICY.trunk_hidden)
    dist = head(trunk)
    entropy = dist.entropy()

    assert (entropy > 0).all()


# ---------------------------------------------------------------------------
# RLA-066: ValueHead output is scalar per batch element
# ---------------------------------------------------------------------------


def test_rla_066_value_head_scalar() -> None:
    """RLA-066: ValueHead output is scalar per batch element."""
    head = ValueHead(_TINY_POLICY.trunk_hidden)
    trunk = torch.randn(_BATCH, _TINY_POLICY.trunk_hidden)
    value = head(trunk)
    assert value.shape == (_BATCH, 1)


# ---------------------------------------------------------------------------
# RLA-067: CascadePolicy.act returns action with valid fields
# ---------------------------------------------------------------------------


def test_rla_067_cascade_policy_act() -> None:
    """RLA-067: CascadePolicy forward returns ActionDistribution + value."""
    obs_encoder = ObservationEncoder(_TINY_SYMBOLS, _TINY_BOARD)
    policy = CascadePolicy(
        board_config=_TINY_BOARD,
        num_symbols=_NUM_SYMBOLS,
        policy_config=_TINY_POLICY,
        num_archetypes=_NUM_ARCHETYPES,
        max_phases=_MAX_PHASES,
        obs_encoder=obs_encoder,
    )

    in_channels = _NUM_SYMBOLS + _TINY_POLICY.board_channels
    board_tensor = torch.randn(_BATCH, in_channels, 3, 3)
    scalar_tensor = torch.randn(_BATCH, 6)
    arch_idx = torch.zeros(_BATCH, dtype=torch.long)
    phase_idx = torch.zeros(_BATCH, dtype=torch.long)

    dist, value = policy(board_tensor, scalar_tensor, arch_idx, phase_idx)

    # Sample an action
    action = dist.sample()
    assert 0 <= action.cluster_symbol_index < _NUM_SYMBOLS
    assert 0 <= action.centroid_col < _TINY_BOARD.num_reels
    assert 0 <= action.centroid_row < _TINY_BOARD.num_rows
    assert value.shape == (_BATCH, 1)


# ---------------------------------------------------------------------------
# RLA-068: CascadePolicy.evaluate batch propagation
# ---------------------------------------------------------------------------


def test_rla_068_evaluate_batch() -> None:
    """RLA-068: CascadePolicy.evaluate batch dimension propagates correctly."""
    obs_encoder = ObservationEncoder(_TINY_SYMBOLS, _TINY_BOARD)
    policy = CascadePolicy(
        board_config=_TINY_BOARD,
        num_symbols=_NUM_SYMBOLS,
        policy_config=_TINY_POLICY,
        num_archetypes=_NUM_ARCHETYPES,
        max_phases=_MAX_PHASES,
        obs_encoder=obs_encoder,
    )

    in_channels = _NUM_SYMBOLS + _TINY_POLICY.board_channels
    board_tensor = torch.randn(_BATCH, in_channels, 3, 3)
    scalar_tensor = torch.randn(_BATCH, 6)
    arch_idx = torch.zeros(_BATCH, dtype=torch.long)
    phase_idx = torch.zeros(_BATCH, dtype=torch.long)
    action_tensor = torch.zeros(_BATCH, 5)

    log_prob, value, entropy = policy.evaluate(
        board_tensor, scalar_tensor, arch_idx, phase_idx, action_tensor,
    )

    assert log_prob.shape == (_BATCH,)
    assert value.shape == (_BATCH,)
    assert entropy.shape == (_BATCH,)


# ---------------------------------------------------------------------------
# RLA-069: Policy is deterministic given fixed seed
# ---------------------------------------------------------------------------


def test_rla_069_deterministic_with_seed() -> None:
    """RLA-069: Policy outputs are deterministic given fixed RNG state."""
    obs_encoder = ObservationEncoder(_TINY_SYMBOLS, _TINY_BOARD)

    def _make_policy():
        torch.manual_seed(42)
        return CascadePolicy(
            board_config=_TINY_BOARD,
            num_symbols=_NUM_SYMBOLS,
            policy_config=_TINY_POLICY,
            num_archetypes=_NUM_ARCHETYPES,
            max_phases=_MAX_PHASES,
            obs_encoder=obs_encoder,
        )

    policy1 = _make_policy()
    policy2 = _make_policy()

    in_channels = _NUM_SYMBOLS + _TINY_POLICY.board_channels
    x_board = torch.randn(1, in_channels, 3, 3)
    x_scalar = torch.randn(1, 6)
    x_arch = torch.zeros(1, dtype=torch.long)
    x_phase = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        dist1, val1 = policy1(x_board, x_scalar, x_arch, x_phase)
        dist2, val2 = policy2(x_board, x_scalar, x_arch, x_phase)

    assert torch.allclose(val1, val2)
    # Compare logits of the symbol distribution
    assert torch.allclose(dist1.symbol_dist.logits, dist2.symbol_dist.logits)
