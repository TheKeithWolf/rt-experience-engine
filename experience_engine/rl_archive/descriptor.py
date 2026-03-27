"""Behavioral descriptor — trajectory fingerprint for MAP-Elites archive cells.

Maps a completed GeneratedInstance to a discrete cell in behavior space.
Two instances in different cells are visibly different to a player: different
starting cluster position, orientation, symbol, or payout bracket.

Spatial binning uses the step-0 cluster centroid divided into equal grid
regions. Payout binning divides the archetype's payout range into equal-width
buckets. Orientation classifies the cluster shape as horizontal, vertical,
or compact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..config.schema import BoardConfig, DescriptorConfig
from ..pipeline.data_types import GeneratedInstance
from ..pipeline.protocols import RangeFloat
from ..primitives.board import Position
from ..primitives.booster_rules import BoosterRules


@dataclass(frozen=True, slots=True)
class TrajectoryDescriptor:
    """Discrete behavioral fingerprint of a completed cascade instance."""

    archetype_id: str
    step0_symbol: str
    spatial_bin: tuple[int, int]  # (col_bin, row_bin)
    cluster_orientation: str  # "H", "V", "compact"
    payout_bin: int


# Hashable flattened form for dict keys
DescriptorKey = tuple


@runtime_checkable
class DescriptorExtractor(Protocol):
    """Extracts a behavioral descriptor from a completed instance."""

    def extract(
        self, instance: GeneratedInstance, payout_range: RangeFloat,
    ) -> TrajectoryDescriptor: ...

    def to_key(self, descriptor: TrajectoryDescriptor) -> DescriptorKey: ...


class CascadeDescriptorExtractor:
    """Extracts behavioral descriptors from cascade instances.

    Reads step-0 cluster data to compute spatial centroid, orientation, and
    symbol. Payout is binned against the archetype's payout range.
    """

    __slots__ = ("_booster_rules", "_board_config", "_descriptor_config")

    def __init__(
        self,
        booster_rules: BoosterRules,
        board_config: BoardConfig,
        descriptor_config: DescriptorConfig,
    ) -> None:
        self._booster_rules = booster_rules
        self._board_config = board_config
        self._descriptor_config = descriptor_config

    def extract(
        self, instance: GeneratedInstance, payout_range: RangeFloat,
    ) -> TrajectoryDescriptor:
        """Compute behavioral descriptor from instance's step-0 cluster.

        payout_range comes from ArchetypeSignature.payout_range — it defines
        the min/max payout for this archetype, used to bin the instance's
        actual payout into equal-width buckets.
        """
        if not instance.cascade_steps or not instance.cascade_steps[0].clusters:
            raise ValueError(
                f"Instance {instance.archetype_id} has no step-0 clusters — "
                "cannot extract descriptor"
            )

        step0 = instance.cascade_steps[0]
        cluster = step0.clusters[0]
        centroid = self._booster_rules.compute_centroid(cluster.positions)

        return TrajectoryDescriptor(
            archetype_id=instance.archetype_id,
            step0_symbol=cluster.symbol.name,
            spatial_bin=self._compute_spatial_bin(centroid),
            cluster_orientation=_classify_orientation(cluster.positions),
            payout_bin=self._compute_payout_bin(instance.payout, payout_range),
        )

    def to_key(self, descriptor: TrajectoryDescriptor) -> DescriptorKey:
        """Flatten descriptor into a hashable tuple for dict keying."""
        return (
            descriptor.archetype_id,
            descriptor.step0_symbol,
            descriptor.spatial_bin[0],
            descriptor.spatial_bin[1],
            descriptor.cluster_orientation,
            descriptor.payout_bin,
        )

    def _compute_spatial_bin(self, centroid: Position) -> tuple[int, int]:
        """Bin centroid position into grid region.

        Divides board columns and rows into equal-width bins. Clamped to
        valid range so edge positions never exceed bin count.
        """
        col_bins = self._descriptor_config.spatial_col_bins
        row_bins = self._descriptor_config.spatial_row_bins

        col_bin = min(
            math.floor(centroid.reel / (self._board_config.num_reels / col_bins)),
            col_bins - 1,
        )
        row_bin = min(
            math.floor(centroid.row / (self._board_config.num_rows / row_bins)),
            row_bins - 1,
        )
        return (col_bin, row_bin)

    def _compute_payout_bin(
        self, payout: float, payout_range: RangeFloat,
    ) -> int:
        """Bin payout into equal-width divisions of the archetype's range.

        Single-value ranges (min == max) always map to bin 0.
        """
        bins = self._descriptor_config.payout_bins
        span = payout_range.max_val - payout_range.min_val

        if span <= 0.0:
            return 0

        # Normalize payout position within the range, clamp to [0, 1]
        normalized = max(0.0, min(1.0, (payout - payout_range.min_val) / span))
        return min(math.floor(normalized * bins), bins - 1)


def _classify_orientation(positions: frozenset[Position]) -> str:
    """Classify cluster shape as horizontal, vertical, or compact.

    Uses col_span vs row_span — same span measurement as rocket orientation
    but with "compact" on tie instead of a config-driven default, since this
    is a behavioral classification, not a game mechanic.
    """
    if not positions:
        return "compact"

    reels = [p.reel for p in positions]
    rows = [p.row for p in positions]
    col_span = max(reels) - min(reels) + 1
    row_span = max(rows) - min(rows) + 1

    if col_span > row_span:
        return "H"
    if row_span > col_span:
        return "V"
    return "compact"
