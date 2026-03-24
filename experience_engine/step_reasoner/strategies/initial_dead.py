"""Initial dead strategy — step 0 for dead archetypes.

Places scatters and near-miss groups on an empty board, then WFC fills
the rest dead via MaxComponentPropagator (or NearMissAwareDeadPropagator
when near-miss groups are present).
"""

from __future__ import annotations

import random

from ...board_filler.propagators import (
    MaxComponentPropagator,
    NearMissAwareDeadPropagator,
    NoSpecialSymbolPropagator,
)
from ...config.schema import MasterConfig
from ...primitives.board import Position
from ...primitives.symbols import Symbol
from ..context import BoardContext
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.near_miss_planner import NearMissPlanner
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


class InitialDeadStrategy:
    """Step 0 for dead archetypes — scatters, near-misses, dead fill.

    The board starts empty. This strategy places scatters (Symbol.S) and
    optional near-miss groups via NearMissPlanner, then configures
    propagators so WFC fills the rest without forming clusters.
    """

    __slots__ = ("_config", "_near_miss_planner", "_rng")

    def __init__(
        self,
        config: MasterConfig,
        near_miss_planner: NearMissPlanner,
        rng: random.Random,
    ) -> None:
        self._config = config
        self._near_miss_planner = near_miss_planner
        self._rng = rng

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        constrained: dict[Position, Symbol] = {}
        occupied: frozenset[Position] = frozenset()

        # Place scatters if required by the signature
        scatter_count = self._resolve_scatter_count(signature)
        if scatter_count > 0:
            scatter_positions = self._place_scatters(context, scatter_count, variance)
            for pos in scatter_positions:
                constrained[pos] = Symbol.S
            occupied = occupied | frozenset(scatter_positions)

        # Place near-miss groups via shared planner
        nm_result = self._near_miss_planner.place(
            context, signature, variance, avoid=occupied,
        )
        constrained.update(nm_result.constrained_cells)

        # Select propagators — use NM-aware version when groups are present
        max_component = (
            signature.max_component_size
            or self._config.reasoner.terminal_dead_default_max_component
        )
        propagators = [NoSpecialSymbolPropagator(self._config.symbols)]
        if nm_result.groups:
            propagators.append(
                NearMissAwareDeadPropagator(
                    max_component=max_component,
                    protected_groups=nm_result.groups,
                    board_config=self._config.board,
                )
            )
        else:
            propagators.append(MaxComponentPropagator(max_component))

        return StepIntent(
            step_type=StepType.INITIAL,
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
            terminal_near_misses=signature.terminal_near_misses,
            terminal_dormant_boosters=None,
            is_terminal=True,
        )

    def _resolve_scatter_count(self, signature: ArchetypeSignature) -> int:
        """Pick a scatter count within the signature's range."""
        rng = signature.required_scatter_count
        if rng.max_val <= 0:
            return 0
        return self._rng.randint(rng.min_val, rng.max_val)

    def _place_scatters(
        self,
        context: BoardContext,
        count: int,
        variance: VarianceHints,
    ) -> list[Position]:
        """Select scatter positions weighted by spatial bias."""
        candidates = context.board.all_positions()
        weights = [variance.spatial_bias.get(p, 1.0) for p in candidates]
        selected: list[Position] = []
        remaining = list(candidates)
        remaining_weights = list(weights)
        for _ in range(min(count, len(remaining))):
            chosen = self._rng.choices(remaining, weights=remaining_weights, k=1)[0]
            idx = remaining.index(chosen)
            selected.append(chosen)
            remaining.pop(idx)
            remaining_weights.pop(idx)
        return selected
