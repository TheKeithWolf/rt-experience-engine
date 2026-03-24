"""Near-miss placement service — shared by initial_dead and initial_cluster strategies.

Places connected components of size (min_cluster_size - 1) on the board to
create "almost won" visual tension. Symbol selection respects the archetype's
tier constraint and variance-driven preferences.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ...board_filler.propagators import NearMissGroup
from ...config.schema import MasterConfig
from ...primitives.board import Position, orthogonal_neighbors
from ...primitives.symbols import Symbol, SymbolTier, symbols_in_tier
from ...archetypes.registry import ArchetypeSignature
from ...variance.hints import VarianceHints
from ..context import BoardContext
from .cluster_builder import ClusterBuilder


@dataclass(frozen=True, slots=True)
class NearMissResult:
    """Output of near-miss placement — constrained cells and WFC-aware groups."""

    constrained_cells: dict[Position, Symbol]
    groups: list[NearMissGroup]


class NearMissPlanner:
    """Plans near-miss group placement on the board.

    Resolves count from the signature, selects symbols from the tier constraint,
    and uses ClusterBuilder.find_positions() for connected placement. The
    returned NearMissGroups feed into NearMissAwareDeadPropagator so WFC
    doesn't accidentally grow them into real clusters.
    """

    __slots__ = ("_config", "_cluster_builder", "_rng")

    def __init__(
        self,
        config: MasterConfig,
        cluster_builder: ClusterBuilder,
        rng: random.Random,
    ) -> None:
        self._config = config
        self._cluster_builder = cluster_builder
        self._rng = rng

    def place(
        self,
        context: BoardContext,
        signature: ArchetypeSignature,
        variance: VarianceHints,
        avoid: frozenset[Position] = frozenset(),
        cluster_symbols: frozenset[Symbol] = frozenset(),
        cluster_positions: frozenset[Position] = frozenset(),
    ) -> NearMissResult:
        """Place near-miss groups on the board per the signature's requirements.

        Returns empty result if the signature doesn't require near-misses.
        Avoids positions already occupied by clusters, scatters, or other placements.

        cluster_symbols/cluster_positions prevent near-miss groups from merging
        with existing clusters: symbols are preferentially different, and if
        forced to match, positions must not be adjacent to the cluster.
        """
        count = self._resolve_count(signature)
        if count == 0:
            return NearMissResult(constrained_cells={}, groups=[])

        near_miss_size = self._config.board.min_cluster_size - 1
        symbols = self._select_symbols(
            signature, count, variance, exclude=cluster_symbols,
        )

        # Expand avoid zone for near-miss symbols that match a cluster symbol —
        # include cluster positions + their neighbors to prevent merging
        adjacency_buffer = self._build_adjacency_buffer(
            cluster_symbols, cluster_positions,
        )

        constrained: dict[Position, Symbol] = {}
        groups: list[NearMissGroup] = []
        occupied = set(avoid)

        for symbol in symbols:
            # If this near-miss uses a cluster symbol, avoid adjacency
            step_avoid = frozenset(occupied)
            if symbol in cluster_symbols:
                step_avoid = step_avoid | adjacency_buffer

            result = self._cluster_builder.find_positions(
                context,
                near_miss_size,
                self._rng,
                variance,
                avoid_positions=step_avoid,
            )
            positions = result.planned_positions
            for pos in positions:
                constrained[pos] = symbol
            groups.append(NearMissGroup(symbol=symbol, positions=positions))
            occupied.update(positions)

        return NearMissResult(constrained_cells=constrained, groups=groups)

    def _build_adjacency_buffer(
        self,
        cluster_symbols: frozenset[Symbol],
        cluster_positions: frozenset[Position],
    ) -> frozenset[Position]:
        """Expand cluster positions to include orthogonal neighbors.

        Used when a near-miss symbol matches a cluster symbol — the buffer
        prevents find_positions from placing adjacent to the cluster, which
        would merge the two into one oversized component.
        """
        if not cluster_symbols or not cluster_positions:
            return frozenset()
        buffer: set[Position] = set(cluster_positions)
        for pos in cluster_positions:
            buffer.update(orthogonal_neighbors(pos, self._config.board))
        return frozenset(buffer)

    def _resolve_count(self, signature: ArchetypeSignature) -> int:
        """Pick a near-miss count within the signature's required range."""
        rng = signature.required_near_miss_count
        if rng.max_val <= 0:
            return 0
        return self._rng.randint(rng.min_val, rng.max_val)

    def _select_symbols(
        self,
        signature: ArchetypeSignature,
        count: int,
        variance: VarianceHints,
        exclude: frozenset[Symbol] = frozenset(),
    ) -> list[Symbol]:
        """Select symbols for near-miss groups from the signature's tier constraint.

        Prefers symbols not in `exclude` (typically cluster symbols) to prevent
        near-miss groups from merging with existing clusters. Falls back to
        excluded symbols only when the tier constraint leaves no alternatives.
        """
        tier = (
            signature.required_near_miss_symbol_tier
            if signature.required_near_miss_symbol_tier is not None
            else SymbolTier.ANY
        )
        candidates = list(symbols_in_tier(tier, self._config.symbols))
        if not candidates:
            candidates = list(symbols_in_tier(SymbolTier.ANY, self._config.symbols))

        # Prefer symbols that won't merge with existing clusters
        filtered = [s for s in candidates if s not in exclude]
        if filtered:
            candidates = filtered

        pref = variance.near_miss_symbol_preference
        pref_index = {sym: idx for idx, sym in enumerate(pref)}
        pref_len = len(pref)

        result: list[Symbol] = []
        for _ in range(count):
            weights = [
                (pref_len - pref_index[s] if s in pref_index else 1)
                for s in candidates
            ]
            chosen = self._rng.choices(candidates, weights=weights, k=1)[0]
            result.append(chosen)
        return result
