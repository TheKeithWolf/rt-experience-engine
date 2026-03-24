"""Terminal dead strategy — fills the board dead with zero clusters.

Uses WFC with MaxComponentPropagator to prevent any connected component
from reaching the cluster threshold. Dormant boosters specified by the
signature are preserved in constrained_cells so they remain visible.
"""

from __future__ import annotations

import random

from ...board_filler.propagators import (
    MaxComponentPropagator,
    NoSpecialSymbolPropagator,
)
from ...config.schema import MasterConfig
from ...primitives.symbols import Symbol
from ..context import BoardContext
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


class TerminalDeadStrategy:
    """Fills the board to produce zero clusters — the cascade ends dead.

    Max component size comes from the signature when specified, otherwise
    falls back to config.reasoner.terminal_dead_default_max_component.
    Dormant boosters that the signature requires on the terminal board
    are pinned as constrained cells.
    """

    __slots__ = ("_config", "_rng")

    def __init__(self, config: MasterConfig, rng: random.Random) -> None:
        self._config = config
        self._rng = rng

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        max_component = (
            signature.max_component_size
            or self._config.reasoner.terminal_dead_default_max_component
        )

        propagators = [
            NoSpecialSymbolPropagator(self._config.symbols),
            MaxComponentPropagator(max_component),
        ]

        # Pin dormant boosters that the signature requires visible on terminal
        constrained: dict = {}
        if signature.dormant_boosters_on_terminal:
            for booster in context.dormant_boosters:
                if booster.booster_type in signature.dormant_boosters_on_terminal:
                    constrained[booster.position] = Symbol[booster.booster_type]

        return StepIntent(
            step_type=StepType.TERMINAL_DEAD,
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
            terminal_near_misses=None,
            terminal_dormant_boosters=(
                list(signature.dormant_boosters_on_terminal)
                if signature.dormant_boosters_on_terminal
                else None
            ),
            is_terminal=True,
        )
