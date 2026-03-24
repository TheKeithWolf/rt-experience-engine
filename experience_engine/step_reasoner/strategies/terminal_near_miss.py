"""Terminal near-miss strategy — dead board with deliberate near-miss groups.

Places groups of exactly min_cluster_size - 1 same-symbol connected cells
from the required tier, ensures isolation (no same-symbol neighbor outside
the group), then fills the rest dead via WFC with NearMissAwareDeadPropagator.
"""

from __future__ import annotations

import random

from ...board_filler.propagators import (
    NearMissAwareDeadPropagator,
    NearMissGroup,
    NoSpecialSymbolPropagator,
)
from ...config.schema import MasterConfig
from ...primitives.board import Position
from ...primitives.symbols import Symbol, SymbolTier, symbols_in_tier
from ..context import BoardContext
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.cluster_builder import ClusterBuilder
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


class TerminalNearMissStrategy:
    """Dead board with deliberate near-miss groups visible to the player.

    Near-miss groups are connected components of exactly min_cluster_size - 1,
    isolated so no 5th same-symbol neighbor exists. The rest of the board is
    filled dead. Uses ClusterBuilder.find_positions for connected placement,
    reusing the existing BFS growth logic.
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

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        nm_spec = signature.terminal_near_misses
        near_miss_size = self._config.board.min_cluster_size - 1

        # Determine how many NM groups to place within the spec range
        nm_count = self._rng.randint(nm_spec.count.min_val, nm_spec.count.max_val)

        # Select NM symbols — weighted by variance preference (least-used first)
        nm_symbols = self._select_nm_symbols(nm_spec, nm_count, variance)

        # Place near-miss groups — each is a connected region of near_miss_size
        constrained: dict[Position, Symbol] = {}
        placed_groups: list[NearMissGroup] = []
        occupied: frozenset[Position] = frozenset()

        for symbol in nm_symbols:
            result = self._cluster_builder.find_positions(
                context,
                near_miss_size,
                self._rng,
                variance,
                avoid_positions=occupied,
            )
            positions = result.planned_positions
            for pos in positions:
                constrained[pos] = symbol
            placed_groups.append(NearMissGroup(symbol=symbol, positions=positions))
            occupied = occupied | positions

        # Propagators: dead fill with NM isolation awareness
        propagators = [
            NoSpecialSymbolPropagator(self._config.symbols),
            NearMissAwareDeadPropagator(
                max_component=near_miss_size,
                protected_groups=placed_groups,
                board_config=self._config.board,
            ),
        ]

        return StepIntent(
            step_type=StepType.TERMINAL_NEAR_MISS,
            constrained_cells=constrained,
            strategic_cells={},
            expected_cluster_count=Range(0, 0),
            expected_cluster_sizes=[],
            expected_cluster_tier=None,
            expected_spawns=[],
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=propagators,
            wfc_symbol_weights=variance.symbol_weights,
            predicted_post_gravity=None,
            terminal_near_misses=nm_spec,
            terminal_dormant_boosters=None,
            is_terminal=True,
        )

    def _select_nm_symbols(
        self,
        nm_spec,
        count: int,
        variance: VarianceHints,
    ) -> list[Symbol]:
        """Select symbols for near-miss groups, weighted by variance preference.

        Uses the tier from the spec (or ANY). Prefers least-used symbols
        from variance.near_miss_symbol_preference.
        """
        tier = nm_spec.symbol_tier if nm_spec.symbol_tier is not None else SymbolTier.ANY
        candidates = list(symbols_in_tier(tier, self._config.symbols))
        if not candidates:
            candidates = list(symbols_in_tier(SymbolTier.ANY, self._config.symbols))

        # Weight by near_miss_symbol_preference — earlier = least-used = higher weight
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
