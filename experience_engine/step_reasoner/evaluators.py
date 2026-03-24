"""Shared rule evaluators — migrated from sequence_planner.

Game rules exist once — in config and shared primitives. Evaluators wrap those
rules into a query interface for spawn checks, chain validity, payout estimation,
and terminal state validation.

No evaluator contains hardcoded game values. All thresholds, tier definitions,
and payout data come from injected config slices.
"""

from __future__ import annotations

import statistics

from ..archetypes.registry import TerminalNearMissSpec
from ..boosters.state_machine import BoosterState
from ..boosters.tracker import BoosterTracker
from ..config.schema import (
    BoosterConfig,
    CentipayoutConfig,
    GridMultiplierConfig,
    MasterConfig,
    PaytableConfig,
    SpawnThreshold,
    SymbolConfig,
    WinLevelConfig,
)
from ..primitives.board import Board
from ..primitives.cluster_detection import detect_clusters, detect_components
from ..primitives.paytable import Paytable
from ..primitives.symbols import SymbolTier, symbols_in_tier

# ASP integer encoding for SymbolTier — must match asp_rules.lp conventions
_TIER_TO_ASP: dict[SymbolTier, int] = {
    SymbolTier.LOW: 1,
    SymbolTier.HIGH: 2,
    SymbolTier.ANY: 3,
}


class SpawnEvaluator:
    """Determines what booster spawns from a given cluster size.

    Single source of truth: delegates to config.boosters.spawn_thresholds.
    """

    __slots__ = ("_config", "_booster_to_range")

    def __init__(self, config: BoosterConfig) -> None:
        self._config = config
        # Precompute O(1) lookup for size_range_for_booster
        self._booster_to_range: dict[str, tuple[int, int]] = {
            t.booster: (t.min_size, t.max_size)
            for t in config.spawn_thresholds
        }

    def booster_for_size(self, size: int) -> str | None:
        """Which booster spawns from a cluster of this size.

        Returns None if the cluster is too small or falls in a gap.
        Iterates thresholds from config — ordering matches spawn hierarchy.
        """
        for threshold in self._config.spawn_thresholds:
            if threshold.min_size <= size <= threshold.max_size:
                return threshold.booster
        return None

    def size_range_for_booster(self, booster: str) -> tuple[int, int] | None:
        """What cluster size range spawns this booster type.

        O(1) dict lookup from precomputed map.
        """
        return self._booster_to_range.get(booster)

    def all_thresholds(self) -> tuple[SpawnThreshold, ...]:
        """All spawn thresholds from config."""
        return self._config.spawn_thresholds


class ChainEvaluator:
    """Determines valid chain trigger relationships.

    Single source of truth: config.boosters.chain_initiators and
    config.boosters.spawn_order (for deriving all booster types).
    """

    __slots__ = ("_initiators", "_all_boosters", "_chain_pairs")

    def __init__(self, config: BoosterConfig) -> None:
        self._initiators: frozenset[str] = frozenset(config.chain_initiators)
        # Derive all chainable booster types from spawn_order, excluding wilds
        # (wilds are cluster-forming symbols, not standalone boosters that chain)
        self._all_boosters: tuple[str, ...] = tuple(
            b for b in config.spawn_order if b != "W"
        )
        # Precompute all valid (source, target) chain pairs
        self._chain_pairs: tuple[tuple[str, str], ...] = tuple(
            (s, t)
            for s in self._initiators
            for t in self._all_boosters
            if s != t
        )

    def can_initiate_chain(self, booster_type: str) -> bool:
        """Can this booster type trigger other boosters via chain reaction?"""
        return booster_type in self._initiators

    def valid_chain_pairs(self) -> tuple[tuple[str, str], ...]:
        """All valid (source, target) chain pairs — precomputed at init."""
        return self._chain_pairs

    @property
    def chain_initiators(self) -> frozenset[str]:
        """Booster types that can initiate chains."""
        return self._initiators

    @property
    def all_booster_types(self) -> tuple[str, ...]:
        """All chainable booster types derived from config.spawn_order."""
        return self._all_boosters


class PayoutEstimator:
    """Estimates payout ranges from cluster size and symbol tier.

    Uses the tier-median payout per cluster size — a statistical approximation
    that provides representative payouts without needing individual symbols.
    """

    __slots__ = (
        "_paytable", "_symbols_config", "_tier_median_cache",
        "_tier_payout_facts_cache", "_grid_first_hit", "_grid_increment",
    )

    def __init__(
        self,
        paytable_config: PaytableConfig,
        centipayout_config: CentipayoutConfig,
        win_level_config: WinLevelConfig,
        symbols_config: SymbolConfig,
        grid_multiplier_config: GridMultiplierConfig | None = None,
    ) -> None:
        self._paytable = Paytable(paytable_config, centipayout_config, win_level_config)
        self._symbols_config = symbols_config
        # Grid multiplier growth rate — scales cascade step estimates by depth
        self._grid_first_hit: int = grid_multiplier_config.first_hit_value if grid_multiplier_config else 1
        self._grid_increment: int = grid_multiplier_config.increment if grid_multiplier_config else 0
        # Precompute tier-median centipayouts: (tier_asp_int, size) → centipayout
        self._tier_median_cache: dict[tuple[int, int], int] = {}
        # Precompute ASP fact string for tier_size_payout atoms
        self._tier_payout_facts_cache: str = ""
        self._build_tier_median_table(paytable_config)

    def _build_tier_median_table(self, paytable_config: PaytableConfig) -> None:
        """Precompute tier-median centipayout for every (tier, size) combination.

        For each tier (LOW/HIGH) and each valid cluster size, computes the
        median payout across all symbols in that tier, then converts to
        centipayout. This gives a representative payout per tier+size
        without needing to reason about individual symbols.
        """
        lines: list[str] = []

        # Collect all valid cluster sizes from the paytable
        seen_sizes: set[int] = set()
        for entry in paytable_config.entries:
            for size in range(entry.tier_min, entry.tier_max + 1):
                seen_sizes.add(size)

        for tier in (SymbolTier.LOW, SymbolTier.HIGH):
            tier_symbols = symbols_in_tier(tier, self._symbols_config)
            asp_tier = _TIER_TO_ASP[tier]

            for size in sorted(seen_sizes):
                payouts = [
                    self._paytable.get_payout(size, sym)
                    for sym in tier_symbols
                    if self._paytable.get_payout(size, sym) > 0.0
                ]
                if not payouts:
                    continue
                median_payout = statistics.median(payouts)
                centipayout = self._paytable.to_centipayout(median_payout)
                self._tier_median_cache[(asp_tier, size)] = centipayout
                lines.append(f"tier_size_payout({asp_tier},{size},{centipayout}).")

        self._tier_payout_facts_cache = "\n".join(lines)

    def estimate_step_payout(
        self, cluster_size: int, symbol_tier: SymbolTier, step_index: int = 0,
    ) -> int:
        """Estimate centipayout for a cluster, scaled by grid multiplier at cascade depth.

        Grid multipliers accumulate across steps — positions hit repeatedly carry
        stacking multipliers. Step 0 uses first_hit_value (1x), subsequent steps
        grow by increment per depth.
        """
        tier = symbol_tier
        if tier is SymbolTier.ANY:
            tier = SymbolTier.LOW
        asp_tier = _TIER_TO_ASP[tier]
        base = self._tier_median_cache.get((asp_tier, cluster_size), 0)
        # Scale by expected grid multiplier at this cascade depth
        expected_mult = self._grid_first_hit + step_index * self._grid_increment
        return base * expected_mult

    def tier_payout_facts(self) -> str:
        """Precomputed ASP fact string for tier_size_payout atoms."""
        return self._tier_payout_facts_cache

    def to_centipayout(self, payout: float) -> int:
        """Convert a payout multiplier to centipayout integer.

        Delegates to the shared Paytable instance for consistent rounding.
        """
        return self._paytable.to_centipayout(payout)


class TerminalEvaluator:
    """Checks whether a board state satisfies terminal conditions.

    Used by StepAssessor (must_terminate_now), TerminalDeadStrategy,
    TerminalNearMissStrategy, and validation. All thresholds come from
    injected MasterConfig — no hardcoded values.
    """

    __slots__ = ("_config", "_board_config", "_symbols_config", "_near_miss_size")

    def __init__(self, config: MasterConfig) -> None:
        self._config = config
        self._board_config = config.board
        self._symbols_config = config.symbols
        # Near-miss = largest connected group below the cluster threshold
        self._near_miss_size = config.board.min_cluster_size - 1

    def is_dead(self, board: Board) -> bool:
        """Board has zero clusters of size >= min_cluster_size from config."""
        return len(detect_clusters(board, self._config)) == 0

    def satisfies_terminal_near_misses(
        self, board: Board, spec: TerminalNearMissSpec,
    ) -> bool:
        """Board has the required number of near-miss groups at the required tier.

        A near-miss group is a connected component of exactly
        min_cluster_size - 1 same-symbol positions (no wild awareness).
        """
        candidates = symbols_in_tier(
            spec.symbol_tier if spec.symbol_tier is not None else SymbolTier.ANY,
            self._symbols_config,
        )
        nm_count = sum(
            1
            for sym in candidates
            for comp in detect_components(board, sym, self._board_config)
            if len(comp) == self._near_miss_size
        )
        return spec.count.contains(nm_count)

    def has_dormant_boosters(
        self, required: tuple[str, ...], tracker: BoosterTracker,
    ) -> bool:
        """Every required booster type has at least one DORMANT instance."""
        dormant_types = {
            b.booster_type.name
            for b in tracker.all_boosters()
            if b.state is BoosterState.DORMANT
        }
        return all(bt in dormant_types for bt in required)
