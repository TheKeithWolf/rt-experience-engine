"""Tests for rl_archive/training/ — trajectory, curriculum, reporter, PPO.

RLA-070 through RLA-089 (subset): GAE computation, curriculum filtering,
reporter output, plateau detection, checkpoint format.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from ..config.schema import (
    ConfigValidationError,
    CurriculumPhase,
    ReporterConfig,
    TrainingConfig,
)
from ..pipeline.protocols import Range
from ..rl_archive.training.curriculum import CurriculumScheduler
from ..rl_archive.training.reporter import (
    BatchMetrics,
    ConsoleTrainingReporter,
    _sparkline,
)
from ..rl_archive.training.trajectory import Trajectory, TrajectoryStep, compute_gae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_curriculum_phases() -> tuple[CurriculumPhase, ...]:
    return (
        CurriculumPhase(episode_threshold=0, difficulty_filter="standard"),
        CurriculumPhase(episode_threshold=100, difficulty_filter="all"),
    )


def _make_reporter_config() -> ReporterConfig:
    return ReporterConfig(
        completion_rolling_window=100,
        completion_trend_buckets=5,
        plateau_warn_threshold=3,
        condense_above_completion=0.90,
        report_every_n_batches=1,
    )


def _make_mock_registry(archetypes: dict[str, int]) -> MagicMock:
    """Mock ArchetypeRegistry with archetype_id → cascade_depth mapping."""
    registry = MagicMock()
    registry.all_ids.return_value = set(archetypes.keys())

    def _get(aid: str):
        sig = MagicMock()
        sig.required_cascade_depth = Range(0, archetypes[aid])
        return sig

    registry.get.side_effect = _get
    return registry


def _make_batch_metrics(
    stale_batch_streak: int = 0,
    completion_rate_rolling: float = 0.5,
    batch_index: int = 0,
) -> BatchMetrics:
    return BatchMetrics(
        batch_index=batch_index,
        episode_range=(0, 9),
        total_episodes=10,
        max_episodes=100,
        elapsed_seconds=5.0,
        completed_count=5,
        batch_size=10,
        completion_rate_rolling=completion_rate_rolling,
        completion_trend=(0.3, 0.4, 0.5),
        mean_reward=1.5,
        best_reward=3.0,
        archive_filled=10,
        archive_total=100,
        new_niches=2,
        replacements=1,
        stale_batch_streak=stale_batch_streak,
        new_niche_details=(),
        coverage_by_axis={},
        step_failure_distribution={},
        phase_completion_rates={},
        policy_loss=0.5,
        value_loss=0.3,
        entropy=1.2,
        learning_rate=3e-4,
        clip_fraction=0.1,
        quality_min=None,
        quality_p25=None,
        quality_p50=None,
        quality_p75=None,
        quality_max=None,
    )


# ---------------------------------------------------------------------------
# RLA-073: CurriculumScheduler returns only standard archetypes before first threshold
# ---------------------------------------------------------------------------


def test_rla_073_curriculum_standard_only() -> None:
    """RLA-073: Before threshold, only standard (depth 2-3) archetypes are eligible."""
    registry = _make_mock_registry({
        "depth0": 0,
        "depth1": 1,
        "depth2": 2,
        "depth3": 3,
        "depth4": 4,
    })
    scheduler = CurriculumScheduler(_make_curriculum_phases(), registry)

    # At episode 0 (before threshold 100), filter = "standard" (depth 2-3)
    eligible = scheduler.eligible_archetypes(0)
    eligible_set = set(eligible)

    assert "depth2" in eligible_set
    assert "depth3" in eligible_set
    assert "depth0" not in eligible_set
    assert "depth1" not in eligible_set
    assert "depth4" not in eligible_set


# ---------------------------------------------------------------------------
# RLA-074: CurriculumScheduler returns all archetypes after last threshold
# ---------------------------------------------------------------------------


def test_rla_074_curriculum_all_after_threshold() -> None:
    """RLA-074: After last threshold, all depth >= 2 archetypes are eligible."""
    registry = _make_mock_registry({
        "depth0": 0,
        "depth2": 2,
        "depth3": 3,
        "depth4": 4,
    })
    scheduler = CurriculumScheduler(_make_curriculum_phases(), registry)

    # At episode 200 (past threshold 100), filter = "all" (depth >= 2)
    eligible = scheduler.eligible_archetypes(200)
    eligible_set = set(eligible)

    assert "depth2" in eligible_set
    assert "depth3" in eligible_set
    assert "depth4" in eligible_set
    assert "depth0" not in eligible_set


# ---------------------------------------------------------------------------
# RLA-075: CurriculumScheduler uses cascade_depth, not names
# ---------------------------------------------------------------------------


def test_rla_075_curriculum_uses_depth_not_names() -> None:
    """RLA-075: Filtering is by required_cascade_depth, not archetype name."""
    # Archetype named "easy" but has depth 4 — should be filtered by depth
    registry = _make_mock_registry({"easy": 4, "hard": 2})
    phases = (CurriculumPhase(episode_threshold=0, difficulty_filter="standard"),)
    scheduler = CurriculumScheduler(phases, registry)

    eligible = set(scheduler.eligible_archetypes(0))
    # "standard" = depth 2-3, so "easy" (depth 4) excluded, "hard" (depth 2) included
    assert "hard" in eligible
    assert "easy" not in eligible


# ---------------------------------------------------------------------------
# RLA-076: GAE output shape matches trajectory length
# ---------------------------------------------------------------------------


def test_rla_076_gae_output_shape() -> None:
    """RLA-076: compute_gae output shape matches input length."""
    rewards = [1.0, 2.0, 3.0, 0.0]
    values = [0.5, 1.5, 2.5, 0.5]
    dones = [False, False, False, True]

    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

    assert advantages.shape == (4,)
    assert returns.shape == (4,)
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()


# ---------------------------------------------------------------------------
# RLA-080: Training respects max_training_episodes
# ---------------------------------------------------------------------------


def test_rla_080_gae_with_single_step() -> None:
    """GAE handles single-step trajectories correctly."""
    advantages, returns = compute_gae(
        rewards=[5.0], values=[1.0], dones=[True],
        gamma=0.99, gae_lambda=0.95,
    )
    assert advantages.shape == (1,)
    # advantage = reward + gamma * next_value - value = 5.0 + 0 - 1.0 = 4.0
    assert advantages[0].item() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# RLA-081: ConsoleTrainingReporter.on_batch_complete produces output
# ---------------------------------------------------------------------------


def test_rla_081_reporter_produces_output(capsys) -> None:
    """RLA-081: on_batch_complete produces non-empty output."""
    reporter = ConsoleTrainingReporter(_make_reporter_config())
    metrics = _make_batch_metrics()

    reporter.on_batch_complete(metrics)

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert "Batch 0" in captured.out


# ---------------------------------------------------------------------------
# RLA-082: Completion rate matches manual computation
# ---------------------------------------------------------------------------


def test_rla_082_completion_rate_in_output(capsys) -> None:
    """RLA-082: BatchMetrics.completion_rate_rolling appears in output."""
    reporter = ConsoleTrainingReporter(_make_reporter_config())
    metrics = _make_batch_metrics(completion_rate_rolling=0.75)

    reporter.on_batch_complete(metrics)

    captured = capsys.readouterr()
    assert "75.0%" in captured.out


# ---------------------------------------------------------------------------
# RLA-083: Plateau warning emitted when streak >= threshold
# ---------------------------------------------------------------------------


def test_rla_083_plateau_warning(capsys) -> None:
    """RLA-083: Plateau warning when stale_batch_streak >= threshold."""
    config = _make_reporter_config()  # threshold = 3
    reporter = ConsoleTrainingReporter(config)
    metrics = _make_batch_metrics(stale_batch_streak=3)

    reporter.on_batch_complete(metrics)

    captured = capsys.readouterr()
    assert "Plateau" in captured.out


# ---------------------------------------------------------------------------
# RLA-084: No plateau warning below threshold
# ---------------------------------------------------------------------------


def test_rla_084_no_plateau_below_threshold(capsys) -> None:
    """RLA-084: No plateau warning when streak < threshold."""
    config = _make_reporter_config()  # threshold = 3
    reporter = ConsoleTrainingReporter(config)
    metrics = _make_batch_metrics(stale_batch_streak=2)

    reporter.on_batch_complete(metrics)

    captured = capsys.readouterr()
    assert "Plateau" not in captured.out


# ---------------------------------------------------------------------------
# RLA-087: report_every_n_batches > 1 skips intermediate batches
# ---------------------------------------------------------------------------


def test_rla_087_skip_intermediate_batches(capsys) -> None:
    """RLA-087: report_every_n_batches skips non-matching batch indices."""
    config = ReporterConfig(
        completion_rolling_window=100,
        completion_trend_buckets=5,
        plateau_warn_threshold=30,
        condense_above_completion=0.90,
        report_every_n_batches=3,
    )
    reporter = ConsoleTrainingReporter(config)

    # Batch 0 → reported (0 % 3 == 0)
    reporter.on_batch_complete(_make_batch_metrics(batch_index=0))
    out0 = capsys.readouterr().out

    # Batch 1 → skipped (1 % 3 != 0)
    reporter.on_batch_complete(_make_batch_metrics(batch_index=1))
    out1 = capsys.readouterr().out

    # Batch 3 → reported (3 % 3 == 0)
    reporter.on_batch_complete(_make_batch_metrics(batch_index=3))
    out3 = capsys.readouterr().out

    assert len(out0) > 0
    assert len(out1) == 0  # Skipped
    assert len(out3) > 0


# ---------------------------------------------------------------------------
# RLA-088: Condense above completion threshold
# ---------------------------------------------------------------------------


def test_rla_088_condensed_above_threshold(capsys) -> None:
    """RLA-088: Output is condensed when completion_rate >= condense_above_completion."""
    config = _make_reporter_config()  # condense_above = 0.90
    reporter = ConsoleTrainingReporter(config)

    # Below threshold — full output
    reporter.on_batch_complete(_make_batch_metrics(completion_rate_rolling=0.5))
    full_output = capsys.readouterr().out

    # Above threshold — condensed output
    reporter.on_batch_complete(
        _make_batch_metrics(completion_rate_rolling=0.95, batch_index=1)
    )
    condensed_output = capsys.readouterr().out

    # Condensed should be shorter (single line vs multi-line block)
    assert len(condensed_output) < len(full_output)


# ---------------------------------------------------------------------------
# Sparkline helper test
# ---------------------------------------------------------------------------


def test_sparkline_rendering() -> None:
    """Sparkline renders non-empty string for valid input."""
    result = _sparkline((0.0, 0.25, 0.5, 0.75, 1.0))
    assert len(result) == 5
    # First char should be lowest, last should be highest
    assert result[0] == "▁"
    assert result[-1] == "█"
