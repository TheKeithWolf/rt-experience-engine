"""Paytable lookup, payout computation, centipayout conversion, and win level mapping.

All values are loaded from config — no hardcoded payout numbers.
The internal lookup dict maps (cluster_size, symbol_name) → payout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.schema import CentipayoutConfig, PaytableConfig, WinLevelConfig
from .symbols import Symbol

if TYPE_CHECKING:
    from .cluster_detection import Cluster
    from .grid_multipliers import GridMultiplierGrid


class Paytable:
    """Paytable lookup and payout computation engine."""

    __slots__ = ("_lookup", "_centipayout_config", "_win_level_config")

    def __init__(
        self,
        config: PaytableConfig,
        centipayout_config: CentipayoutConfig,
        win_level_config: WinLevelConfig,
    ) -> None:
        self._centipayout_config = centipayout_config
        self._win_level_config = win_level_config

        # Expand tier ranges into individual (size, symbol) → payout lookups
        self._lookup: dict[tuple[int, str], float] = {}
        for entry in config.entries:
            for size in range(entry.tier_min, entry.tier_max + 1):
                self._lookup[(size, entry.symbol)] = entry.payout

    def get_payout(self, size: int, symbol: Symbol) -> float:
        """Look up base payout for a cluster of given size and symbol.

        Returns 0.0 for sizes beyond the paytable range (e.g. size < min_cluster_size).
        """
        return self._lookup.get((size, symbol.name), 0.0)

    def compute_cluster_payout(
        self, cluster: Cluster, grid_mults: GridMultiplierGrid
    ) -> float:
        """Compute total cluster payout: base_payout * max(multiplier_sum, minimum_contribution).

        Grid multiplier sum is computed from all positions in the cluster
        (both standard and wild positions).
        """
        base = self.get_payout(cluster.size, cluster.symbol)
        all_positions = cluster.positions | cluster.wild_positions
        mult_sum = grid_mults.position_multiplier_sum(all_positions)
        return base * mult_sum

    def to_centipayout(self, payout: float) -> int:
        """Convert a payout multiplier to centipayout integer.

        Formula: round(payout * multiplier / rounding_base) * rounding_base
        Example: 2.6x → round(2.6 * 100 / 10) * 10 = round(26) * 10 = 260
        """
        cfg = self._centipayout_config
        return round(payout * cfg.multiplier / cfg.rounding_base) * cfg.rounding_base

    def get_win_level(self, payout: float) -> int:
        """Map a payout multiplier to its win level tier.

        Searches tiers for the first where min_payout <= payout < max_payout.
        The final tier uses inf as max_payout to capture all remaining values.
        Returns 0 if no tier matches (should not happen with well-formed config).
        """
        for tier in self._win_level_config.tiers:
            if tier.min_payout <= payout < tier.max_payout:
                return tier.level
        # Edge case: payout exactly equals the final tier's min_payout
        # (handled by the >= in the last tier)
        if self._win_level_config.tiers:
            last = self._win_level_config.tiers[-1]
            if payout >= last.min_payout:
                return last.level
        return 0
