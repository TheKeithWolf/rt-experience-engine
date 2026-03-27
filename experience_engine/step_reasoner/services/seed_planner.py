"""Seed planner — strategic future-step cell placement via backward reasoning.

Plans where to place "noise" symbols that aren't noise — gravity will carry
them to specific post-settle positions needed by the next cascade step.
Bridge seeds land adjacent to wilds, arm seeds near dormant boosters,
generic seeds set up general cascade opportunity.

Used by: InitialCluster, CascadeCluster, WildBridge.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from ...archetypes.registry import ArchetypeSignature
from ...board_filler.propagators import compute_border
from ...config.schema import BoardConfig, SymbolConfig
from ...primitives.board import Position, orthogonal_neighbors
from ...primitives.gravity import SettleResult
from ...primitives.symbols import Symbol, SymbolTier, symbols_in_tier
from ...variance.hints import VarianceHints
from ..progress import ProgressTracker
from .forward_simulator import ForwardSimulator


@dataclass(frozen=True, slots=True)
class ClusterExclusion:
    """A cluster's symbol paired with its exclusion zone (positions + 1-cell border).

    Strategic seeds matching the symbol inside this zone would merge into the
    cluster on detection — bypassing ClusterBoundaryPropagator which only
    guards WFC-filled cells, not pre-pinned strategic cells.
    """

    symbol: Symbol
    zone: frozenset[Position]


def build_cluster_exclusions(
    cluster_groups: Iterable[tuple[frozenset[Position], Symbol]],
    board_config: BoardConfig,
) -> tuple[ClusterExclusion, ...]:
    """Build exclusion zones from cluster groups — prevents strategic seeds from merging into clusters."""
    return tuple(
        ClusterExclusion(
            symbol=sym,
            zone=positions | compute_border(positions, board_config),
        )
        for positions, sym in cluster_groups
    )


class SeedPlanner:
    """Plans strategic cell placements for future cascade steps.

    Backward reasoning: given the current gravity settle result, determines
    which refill-zone positions to fill with specific symbols so that
    gravity carries them to useful post-settle locations.

    Constructed once, shared across strategies via dependency injection.
    """

    __slots__ = ("_forward_sim", "_board_config", "_symbol_config")

    def __init__(
        self,
        forward_simulator: ForwardSimulator,
        board_config: BoardConfig,
        symbol_config: SymbolConfig,
    ) -> None:
        self._forward_sim = forward_simulator
        self._board_config = board_config
        self._symbol_config = symbol_config

    @staticmethod
    def _filter_excluded(
        seeds: dict[Position, Symbol],
        exclusions: tuple[ClusterExclusion, ...],
    ) -> dict[Position, Symbol]:
        """Remove seeds that would merge into an existing cluster's exclusion zone.

        Same-symbol seeds adjacent to a cluster boundary cause oversized clusters
        because strategic cells are pinned before WFC boundary propagation runs.
        """
        if not exclusions:
            return seeds
        return {
            pos: sym for pos, sym in seeds.items()
            if not any(ex.symbol is sym and pos in ex.zone for ex in exclusions)
        }

    def plan_bridge_seeds(
        self,
        wild_post_gravity_pos: Position,
        settle_result: SettleResult,
        bridge_symbol: Symbol,
        count: int,
        variance: VarianceHints,
        rng: random.Random,
        exclusions: tuple[ClusterExclusion, ...] = (),
        utility_scores: dict[Position, float] | None = None,
    ) -> dict[Position, Symbol]:
        """Place symbols in refill zone that gravity will carry adjacent to the wild.

        When utility_scores is provided, all empty positions are eligible and
        scoring replaces column-set filtering — gravity alignment and influence
        steer seeds toward positions that actually reach the wild after settle.

        Without utility_scores, falls back to column-matching heuristic.
        """
        if utility_scores is not None:
            # Spatial intelligence path — all refill slots are candidates,
            # utility scores encode gravity alignment + influence proximity
            eligible = list(settle_result.empty_positions)
        else:
            # Legacy column-matching — only positions in target neighbor columns
            target_neighbors = set(
                orthogonal_neighbors(wild_post_gravity_pos, self._board_config)
            )
            target_columns = {n.reel for n in target_neighbors}
            eligible = [
                pos for pos in settle_result.empty_positions
                if pos.reel in target_columns
            ]

        if not eligible:
            return {}

        selected = _weighted_select(
            eligible, count, variance, rng, utility_scores,
        )
        return self._filter_excluded(
            {pos: bridge_symbol for pos in selected}, exclusions,
        )

    def plan_arm_seeds(
        self,
        booster_post_gravity_pos: Position,
        settle_result: SettleResult,
        variance: VarianceHints,
        rng: random.Random,
        exclusions: tuple[ClusterExclusion, ...] = (),
        utility_scores: dict[Position, float] | None = None,
    ) -> dict[Position, Symbol]:
        """Place symbols near a dormant booster to enable a future arming cluster.

        When utility_scores is provided, all empty positions are eligible and
        scoring replaces column-adjacency filtering — booster adjacency factor
        and gravity alignment steer seeds toward the arming zone.

        Without utility_scores, falls back to column-adjacency heuristic.
        """
        if utility_scores is not None:
            # Spatial intelligence path — all refill slots are candidates,
            # utility scores encode booster adjacency + gravity alignment
            eligible = list(settle_result.empty_positions)
        else:
            # Legacy column-adjacency — only booster column + neighbor columns
            booster_neighbors = orthogonal_neighbors(
                booster_post_gravity_pos, self._board_config,
            )
            target_columns = {n.reel for n in booster_neighbors}
            target_columns.add(booster_post_gravity_pos.reel)
            eligible = [
                pos for pos in settle_result.empty_positions
                if pos.reel in target_columns
            ]

        if not eligible:
            return {}

        # Select a standard symbol weighted by variance
        arm_symbol = self._select_weighted_symbol(variance, rng)

        # Place seeds in eligible positions — enough to form a cluster seed
        # but not so many that we constrain future steps excessively
        seed_count = min(len(eligible), self._board_config.min_cluster_size - 1)
        selected = _weighted_select(
            eligible, seed_count, variance, rng, utility_scores,
        )
        return self._filter_excluded(
            {pos: arm_symbol for pos in selected}, exclusions,
        )

    def plan_generic_seeds(
        self,
        settle_result: SettleResult,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
        rng: random.Random,
        exclusions: tuple[ClusterExclusion, ...] = (),
    ) -> dict[Position, Symbol]:
        """Place symbols in empty positions for general cascade opportunity.

        Fills up to half the refill zone with variance-weighted symbols,
        leaving room for future specific placements. Respects the signature's
        symbol_tier_per_step constraint for the upcoming step if specified.
        """
        empty = list(settle_result.empty_positions)
        if not empty:
            return {}

        # Use at most half the empty positions — preserve flexibility
        max_seeds = max(1, len(empty) // 2)
        selected = _weighted_select(empty, max_seeds, variance, rng)

        # Resolve tier for the next step — from arc phase or legacy map
        tier = SymbolTier.ANY
        next_phase = progress.peek_next_phase()
        if next_phase is not None and next_phase.cluster_symbol_tier is not None:
            tier = next_phase.cluster_symbol_tier
        elif signature.symbol_tier_per_step:
            next_step = progress.steps_completed + 1
            if next_step in signature.symbol_tier_per_step:
                tier = signature.symbol_tier_per_step[next_step]

        candidates = list(symbols_in_tier(tier, self._symbol_config))
        if not candidates:
            candidates = list(symbols_in_tier(SymbolTier.ANY, self._symbol_config))

        weights = [variance.symbol_weights.get(s, 1.0) for s in candidates]

        return self._filter_excluded(
            {pos: rng.choices(candidates, weights=weights, k=1)[0] for pos in selected},
            exclusions,
        )

    def plan_chain_aware_seeds(
        self,
        booster_post_gravity_pos: Position,
        settle_result: SettleResult,
        effect_zone: frozenset[Position],
        variance: VarianceHints,
        rng: random.Random,
        exclusions: tuple[ClusterExclusion, ...] = (),
    ) -> dict[Position, Symbol]:
        """Place seeds that gravity will carry into both the booster's adjacency
        AND the chain source's effect zone.

        Composes plan_arm_seeds (adjacent to booster) with a column filter
        (must settle into the effect zone). Gravity preserves columns, so
        filtering by reel is sufficient — a seed in a column that intersects
        the effect zone will land in that zone after gravity.
        """
        arm_seeds = self.plan_arm_seeds(
            booster_post_gravity_pos, settle_result, variance, rng, exclusions,
        )
        # Column-level filter: keep seeds whose reel intersects the effect zone
        effect_columns = {p.reel for p in effect_zone}
        return {
            pos: sym for pos, sym in arm_seeds.items()
            if pos.reel in effect_columns
        }

    # -- Private helpers ---------------------------------------------------

    def _select_weighted_symbol(
        self, variance: VarianceHints, rng: random.Random,
    ) -> Symbol:
        """Pick a standard symbol weighted by variance.symbol_weights."""
        candidates = list(symbols_in_tier(SymbolTier.ANY, self._symbol_config))
        weights = [variance.symbol_weights.get(s, 1.0) for s in candidates]
        return rng.choices(candidates, weights=weights, k=1)[0]


def _weighted_select(
    candidates: list[Position],
    count: int,
    variance: VarianceHints,
    rng: random.Random,
    utility_scores: dict[Position, float] | None = None,
) -> list[Position]:
    """Select up to `count` positions weighted by variance.spatial_bias.

    When utility_scores is provided, each position's weight is multiplied
    by its utility score — higher-utility positions are selected more often.
    This replaces column-matching heuristics with continuous spatial scoring.

    Returns fewer than `count` if fewer candidates are available.
    Does not return duplicates.
    """
    count = min(count, len(candidates))
    if count == 0:
        return []

    # Build weighted selection without replacement
    remaining = list(candidates)
    selected: list[Position] = []
    for _ in range(count):
        if not remaining:
            break
        weights = [
            variance.spatial_bias.get(p, 1.0)
            * (utility_scores.get(p, 1.0) if utility_scores else 1.0)
            for p in remaining
        ]
        chosen = rng.choices(remaining, weights=weights, k=1)[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected
