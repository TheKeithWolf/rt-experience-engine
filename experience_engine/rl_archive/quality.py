"""Quality scoring for MAP-Elites archive — ranks instances within a niche.

Higher-quality instances replace lower-quality ones in the same archive cell.
The score is a weighted sum of normalized [0,1] components. Weights from
QualityConfig; setting any weight to 0.0 disables that component.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config.schema import BoardConfig, GridMultiplierConfig, QualityConfig
from ..narrative.arc import NarrativeArc
from ..pipeline.data_types import GeneratedInstance
from ..primitives.board import Position


@runtime_checkable
class QualityScorer(Protocol):
    """Scores a generated instance for archive quality ranking."""

    def score(self, instance: GeneratedInstance, arc: NarrativeArc) -> float: ...


class CascadeQualityScorer:
    """Multi-component quality scorer for cascade instances.

    Components (each normalized to [0, 1]):
    1. Payout centering — proximity to midpoint of arc's payout range
    2. Escalation — fraction of consecutive step pairs with non-decreasing payout
    3. Cluster size — mean cluster size relative to board area
    4. Productivity — fraction of steps with nonzero step_payout
    5. Multiplier engagement — fraction of cluster positions on active multiplier cells
    """

    __slots__ = (
        "_quality_config", "_board_config", "_grid_multiplier_config",
    )

    def __init__(
        self,
        quality_config: QualityConfig,
        board_config: BoardConfig,
        grid_multiplier_config: GridMultiplierConfig,
    ) -> None:
        self._quality_config = quality_config
        self._board_config = board_config
        self._grid_multiplier_config = grid_multiplier_config

    def score(self, instance: GeneratedInstance, arc: NarrativeArc) -> float:
        """Compute weighted quality score from cascade steps."""
        if not instance.cascade_steps:
            return 0.0

        total = 0.0

        if self._quality_config.payout_centering_weight > 0.0:
            total += (
                self._quality_config.payout_centering_weight
                * self._payout_centering(instance.payout, arc)
            )

        if self._quality_config.escalation_weight > 0.0:
            total += (
                self._quality_config.escalation_weight
                * self._escalation(instance)
            )

        if self._quality_config.cluster_size_weight > 0.0:
            total += (
                self._quality_config.cluster_size_weight
                * self._cluster_size(instance)
            )

        if self._quality_config.productivity_weight > 0.0:
            total += (
                self._quality_config.productivity_weight
                * self._productivity(instance)
            )

        if self._quality_config.multiplier_engagement_weight > 0.0:
            total += (
                self._quality_config.multiplier_engagement_weight
                * self._multiplier_engagement(instance)
            )

        return total

    def _payout_centering(
        self, payout: float, arc: NarrativeArc,
    ) -> float:
        """Proximity to midpoint of arc's payout range, normalized [0, 1].

        Returns 1.0 when payout equals the midpoint, 0.0 at either extreme.
        """
        span = arc.payout.max_val - arc.payout.min_val
        if span <= 0.0:
            return 1.0  # Single-value range — any payout is centered

        midpoint = arc.payout.min_val + span / 2.0
        distance = abs(payout - midpoint)
        # Max possible distance is half the span
        return max(0.0, 1.0 - distance / (span / 2.0))

    def _escalation(self, instance: GeneratedInstance) -> float:
        """Fraction of consecutive step pairs with non-decreasing payout.

        Rewards trajectories where tension rises over time (payouts escalate).
        Returns 1.0 for monotonically non-decreasing payouts.
        """
        steps = instance.cascade_steps
        if len(steps) < 2:
            return 1.0  # Single step is trivially escalating

        non_decreasing = sum(
            1 for i in range(len(steps) - 1)
            if steps[i + 1].step_payout >= steps[i].step_payout
        )
        return non_decreasing / (len(steps) - 1)

    def _cluster_size(self, instance: GeneratedInstance) -> float:
        """Mean cluster size normalized by board area.

        Board area = num_reels * num_rows. Rewards larger clusters that
        create more visual impact.
        """
        board_area = self._board_config.num_reels * self._board_config.num_rows
        all_sizes = [
            cluster.size
            for step in instance.cascade_steps
            for cluster in step.clusters
        ]
        if not all_sizes:
            return 0.0

        return min(1.0, sum(all_sizes) / len(all_sizes) / board_area)

    def _productivity(self, instance: GeneratedInstance) -> float:
        """Fraction of steps that produce nonzero payout.

        Penalizes dead cascade steps that contribute nothing to the player.
        """
        steps = instance.cascade_steps
        productive = sum(1 for step in steps if step.step_payout > 0.0)
        return productive / len(steps)

    def _multiplier_engagement(self, instance: GeneratedInstance) -> float:
        """Fraction of cluster positions on cells with active grid multipliers.

        Active = multiplier value > GridMultiplierConfig.initial_value. Rewards
        instances that exploit the grid multiplier system for higher payouts.
        """
        initial = self._grid_multiplier_config.initial_value
        num_rows = self._board_config.num_rows
        total_positions = 0
        engaged_positions = 0

        for step in instance.cascade_steps:
            snapshot = step.grid_multipliers_snapshot
            for cluster in step.clusters:
                for pos in cluster.positions:
                    total_positions += 1
                    # Reel-major order: index = reel * num_rows + row
                    idx = pos.reel * num_rows + pos.row
                    if idx < len(snapshot) and snapshot[idx] > initial:
                        engaged_positions += 1

        if total_positions == 0:
            return 0.0
        return engaged_positions / total_positions
