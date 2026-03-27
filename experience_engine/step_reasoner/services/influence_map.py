"""Gaussian demand field for future-step spatial reservation.

Per-step influence map centered on a demand centroid — tells each cell
how strongly the next step's demand affects it. Used to reserve WFC
suppression zones and weight seed positions via the UtilityScorer.

Stateless: each call to compute() produces a fresh dict. No internal
mutation between steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...primitives.board import Position

if TYPE_CHECKING:
    from ...config.schema import BoardConfig, SpatialIntelligenceConfig


@dataclass(frozen=True, slots=True)
class DemandSpec:
    """What the next cascade step needs spatially.

    Produced by strategies based on the archetype signature's next-step
    constraints. The centroid is the spatial center of the demand — e.g.,
    the predicted booster landing position for a bridge step, or the
    dormant booster position for an arming step.
    """

    centroid: Position
    cluster_size: int
    booster_type: str | None  # None = no booster involvement


class InfluenceMap:
    """Gaussian demand field — strongest at centroid, fading by distance.

    Sigma auto-scales: larger cluster_size means wider sigma (more board area
    needed). Booster type applies a multiplier — wilds need wider zones
    because bridge mechanics require adjacent refill across multiple columns.

    Stateless: each call to compute() produces a fresh dict. No internal
    mutation between steps.
    """

    __slots__ = ("_config", "_board_config")

    def __init__(
        self,
        config: SpatialIntelligenceConfig,
        board_config: BoardConfig,
    ) -> None:
        self._config = config
        self._board_config = board_config

    def compute(self, demand: DemandSpec) -> dict[Position, float]:
        """Compute influence values for all board positions.

        Returns Position -> influence (0.0–1.0), where 1.0 = centroid.
        """
        sigma = self._sigma(demand)
        two_sigma_sq = 2.0 * sigma * sigma
        result: dict[Position, float] = {}
        for reel in range(self._board_config.num_reels):
            for row in range(self._board_config.num_rows):
                pos = Position(reel, row)
                dist_sq = (
                    (pos.reel - demand.centroid.reel) ** 2
                    + (pos.row - demand.centroid.row) ** 2
                )
                result[pos] = math.exp(-dist_sq / two_sigma_sq)
        return result

    def reserve_zone(
        self, influence: dict[Position, float],
    ) -> frozenset[Position]:
        """Positions above the reserve threshold — these get WFC suppression.

        Cells in the reserve zone are where the next step needs clear space
        for cluster formation. WFC suppresses all standard symbols here
        to prevent noise contamination.
        """
        threshold = self._config.reserve_threshold
        return frozenset(
            pos for pos, val in influence.items() if val >= threshold
        )

    def _sigma(self, demand: DemandSpec) -> float:
        """Compute sigma from cluster size and booster type.

        sigma = (base + cluster_size × scale) × booster_multiplier

        The booster multiplier widens the zone for mechanics that need
        more adjacent space (e.g. W bridge spans multiple columns).
        Defaults to 1.0 for unknown or absent booster types.
        """
        raw = (
            self._config.influence_sigma_base
            + demand.cluster_size * self._config.influence_sigma_scale_per_cell
        )
        multiplier = (
            self._config.booster_sigma_multipliers.get(demand.booster_type, 1.0)
            if demand.booster_type
            else 1.0
        )
        return raw * multiplier
