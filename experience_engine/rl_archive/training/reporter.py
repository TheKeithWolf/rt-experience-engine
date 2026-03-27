"""Training reporter — console output for training progress.

TrainingReporter protocol defines the lifecycle event interface.
ConsoleTrainingReporter renders block-style output with sparklines,
ETA, plateau detection, and adaptive detail based on completion rate.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ...config.schema import ReporterConfig


# ---------------------------------------------------------------------------
# Metrics dataclasses — frozen snapshots of training state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingStartSummary:
    """Emitted once at the start of training."""

    archetype_id: str
    seed: int
    max_episodes: int
    batch_size: int
    curriculum_phase: str
    policy_param_count: int
    archive_total_niches: int
    archive_descriptor_axes: dict[str, int]
    device: str


@dataclass(frozen=True, slots=True)
class ImitationEpochMetrics:
    """Emitted after each imitation learning epoch."""

    epoch: int
    total_epochs: int
    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class BatchMetrics:
    """Emitted after each PPO batch update."""

    batch_index: int
    episode_range: tuple[int, int]
    total_episodes: int
    max_episodes: int
    elapsed_seconds: float
    completed_count: int
    batch_size: int
    completion_rate_rolling: float
    completion_trend: tuple[float, ...]
    mean_reward: float
    best_reward: float
    archive_filled: int
    archive_total: int
    new_niches: int
    replacements: int
    stale_batch_streak: int
    new_niche_details: tuple[str, ...]
    coverage_by_axis: dict[str, dict[str, tuple[int, int]]]
    step_failure_distribution: dict[int, int]
    phase_completion_rates: dict[str, float]
    policy_loss: float
    value_loss: float
    entropy: float
    learning_rate: float
    clip_fraction: float
    quality_min: float | None
    quality_p25: float | None
    quality_p50: float | None
    quality_p75: float | None
    quality_max: float | None


@dataclass(frozen=True, slots=True)
class TrainingCompleteSummary:
    """Emitted once at the end of training."""

    archetype_id: str
    total_episodes: int
    wall_time_seconds: float
    final_completion_rate: float
    archive_filled: int
    archive_total: int
    archive_coverage: float
    mean_quality: float
    quality_range: tuple[float, float]
    coverage_by_axis: dict[str, dict[str, tuple[int, int]]]
    infeasible_niche_count: int
    infeasible_analysis: str
    archive_path: str
    checkpoint_path: str


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainingReporter(Protocol):
    """Lifecycle event interface for training progress reporting."""

    def on_training_start(self, summary: TrainingStartSummary) -> None: ...
    def on_imitation_epoch(self, metrics: ImitationEpochMetrics) -> None: ...
    def on_imitation_complete(self) -> None: ...
    def on_batch_complete(self, metrics: BatchMetrics) -> None: ...
    def on_checkpoint(self, path: Path) -> None: ...
    def on_training_complete(self, summary: TrainingCompleteSummary) -> None: ...


# ---------------------------------------------------------------------------
# Sparkline helper
# ---------------------------------------------------------------------------

_SPARK_CHARS = "▁▂▃▄▅▆▇█"
# ASCII fallback for terminals that can't render Unicode block characters
_SPARK_ASCII = "_.:-=+*#"


def _sparkline(values: tuple[float, ...]) -> str:
    """Render a sequence of [0,1] floats as a sparkline string.

    Falls back to ASCII characters if the terminal can't encode Unicode.
    """
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    chars = _SPARK_CHARS
    try:
        # Test if the terminal can encode these characters
        chars[0].encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError):
        chars = _SPARK_ASCII
    return "".join(
        chars[min(int((v - lo) / span * (len(chars) - 1)), len(chars) - 1)]
        for v in values
    )


# ---------------------------------------------------------------------------
# Console reporter
# ---------------------------------------------------------------------------


class ConsoleTrainingReporter:
    """Renders training progress as block-style console output.

    Adaptive detail: when completion_rate_rolling exceeds condense_above_completion,
    reduces verbosity. Plateau detection warns when stale_batch_streak exceeds
    the configured threshold.
    """

    __slots__ = ("_config", "_start_time")

    def __init__(self, config: ReporterConfig) -> None:
        self._config = config
        self._start_time = time.monotonic()

    def on_training_start(self, summary: TrainingStartSummary) -> None:
        print(f"\n{'='*60}")
        print(f"  Training: {summary.archetype_id}")
        print(f"  Seed: {summary.seed}  Device: {summary.device}")
        print(f"  Episodes: {summary.max_episodes:,}  Batch: {summary.batch_size}")
        print(f"  Policy params: {summary.policy_param_count:,}")
        print(f"  Archive niches: {summary.archive_total_niches:,}")
        print(f"  Curriculum: {summary.curriculum_phase}")
        print(f"{'='*60}\n")
        self._start_time = time.monotonic()

    def on_imitation_epoch(self, metrics: ImitationEpochMetrics) -> None:
        print(
            f"  Imitation {metrics.epoch}/{metrics.total_epochs} "
            f"loss={metrics.loss:.4f} acc={metrics.accuracy:.2%}"
        )

    def on_imitation_complete(self) -> None:
        print("  Imitation pre-training complete.\n")

    def on_batch_complete(self, metrics: BatchMetrics) -> None:
        # Skip intermediate batches if configured
        if (self._config.report_every_n_batches > 1
                and metrics.batch_index % self._config.report_every_n_batches != 0):
            return

        # Condense output when completion rate is high
        condensed = (
            metrics.completion_rate_rolling >= self._config.condense_above_completion
        )

        # ETA computation
        elapsed = metrics.elapsed_seconds
        progress = metrics.total_episodes / max(metrics.max_episodes, 1)
        eta = (elapsed / progress - elapsed) if progress > 0.01 else 0.0

        spark = _sparkline(metrics.completion_trend)

        if condensed:
            # Single-line condensed output
            print(
                f"  [{metrics.total_episodes:>7,}/{metrics.max_episodes:,}] "
                f"comp={metrics.completion_rate_rolling:.1%} {spark} "
                f"arch={metrics.archive_filled}/{metrics.archive_total} "
                f"ETA={eta:.0f}s"
            )
        else:
            # Full block output
            print(f"\n  Batch {metrics.batch_index} "
                  f"[{metrics.episode_range[0]:,}-{metrics.episode_range[1]:,}]")
            print(f"    Completion: {metrics.completion_rate_rolling:.1%} {spark}")
            print(f"    Reward: mean={metrics.mean_reward:.2f} best={metrics.best_reward:.2f}")
            print(f"    Archive: {metrics.archive_filled}/{metrics.archive_total} "
                  f"(+{metrics.new_niches} new, {metrics.replacements} replaced)")
            print(f"    Loss: policy={metrics.policy_loss:.4f} "
                  f"value={metrics.value_loss:.4f} entropy={metrics.entropy:.4f}")
            print(f"    ETA: {eta:.0f}s")

        # Plateau warning
        if metrics.stale_batch_streak >= self._config.plateau_warn_threshold:
            print(
                f"    ⚠ Plateau: {metrics.stale_batch_streak} batches "
                f"without new niches or quality improvement"
            )

    def on_checkpoint(self, path: Path) -> None:
        print(f"  Checkpoint saved: {path}")

    def on_training_complete(self, summary: TrainingCompleteSummary) -> None:
        print(f"\n{'='*60}")
        print(f"  Training complete: {summary.archetype_id}")
        print(f"  Episodes: {summary.total_episodes:,} "
              f"Wall time: {summary.wall_time_seconds:.1f}s")
        print(f"  Final completion: {summary.final_completion_rate:.1%}")
        print(f"  Archive: {summary.archive_filled}/{summary.archive_total} "
              f"({summary.archive_coverage:.1%} coverage)")
        print(f"  Quality: mean={summary.mean_quality:.3f} "
              f"range=[{summary.quality_range[0]:.3f}, {summary.quality_range[1]:.3f}]")
        if summary.infeasible_niche_count > 0:
            print(f"  Infeasible niches: {summary.infeasible_niche_count}")
            print(f"    {summary.infeasible_analysis}")
        print(f"  Archive: {summary.archive_path}")
        print(f"  Checkpoint: {summary.checkpoint_path}")
        print(f"{'='*60}\n")
