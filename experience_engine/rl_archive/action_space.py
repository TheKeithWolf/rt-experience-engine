"""Action space for the cascade RL environment.

CascadeAction is the discrete action the policy outputs. ActionInterpreter
converts it into a StepIntent that the existing pipeline can execute.
This is the sole bridge between RL-world and pipeline-world — all game-rule
enforcement stays in the existing executor/validator/simulator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config.schema import BoardConfig, SymbolConfig
from ..pipeline.protocols import Range
from ..primitives.board import Position
from ..primitives.symbols import Symbol
from ..step_reasoner.intent import StepIntent, StepType

if TYPE_CHECKING:
    from ..step_reasoner.context import BoardContext
    from ..step_reasoner.progress import ProgressTracker
    from ..step_reasoner.services.cluster_builder import ClusterBuilder
    from ..variance.hints import VarianceHints


@dataclass(frozen=True, slots=True)
class CascadeAction:
    """Discrete action output from the policy network.

    All fields are integer indices into config-derived ranges, except
    is_terminal which signals the end of the cascade.
    """

    cluster_symbol_index: int  # index into SymbolConfig.standard
    cluster_size: int  # min_cluster_size to board_area
    centroid_col: int  # 0..num_reels-1
    centroid_row: int  # 0..num_rows-1
    is_terminal: bool


class ActionInterpreter:
    """Converts CascadeAction into StepIntent for the existing pipeline.

    Maps symbol indices back to Symbol enums and uses ClusterBuilder
    for BFS position growth from the specified centroid.
    """

    __slots__ = (
        "_symbol_config", "_board_config", "_cluster_builder",
        "_index_to_symbol",
    )

    def __init__(
        self,
        symbol_config: SymbolConfig,
        board_config: BoardConfig,
        cluster_builder: ClusterBuilder,
    ) -> None:
        self._symbol_config = symbol_config
        self._board_config = board_config
        self._cluster_builder = cluster_builder
        # Build reverse mapping: index → Symbol enum
        self._index_to_symbol: dict[int, Symbol] = {
            i: Symbol[name] for i, name in enumerate(symbol_config.standard)
        }

    def interpret(
        self,
        action: CascadeAction,
        context: BoardContext,
        progress: ProgressTracker,
        variance: VarianceHints,
        rng: random.Random,
    ) -> StepIntent:
        """Convert a CascadeAction into a StepIntent the pipeline can execute.

        Terminal actions produce a TERMINAL_DEAD intent with no cluster
        constraints. Non-terminal actions use ClusterBuilder.find_positions
        for BFS growth from the centroid.
        """
        if action.is_terminal:
            return self._build_terminal_intent()

        symbol = self._index_to_symbol.get(
            action.cluster_symbol_index,
            Symbol[self._symbol_config.standard[0]],  # fallback to first symbol
        )
        centroid = Position(action.centroid_col, action.centroid_row)

        # Clamp cluster size to available empty cells so BFS never requests
        # more positions than the board can provide
        available = len(context.empty_cells)
        size = max(
            self._board_config.min_cluster_size,
            min(action.cluster_size, available),
        )

        # Use ClusterBuilder for BFS growth from the centroid
        cluster_result = self._cluster_builder.find_positions(
            context=context,
            size=size,
            rng=rng,
            variance=variance,
            symbol=symbol,
            centroid_target=centroid,
        )

        # Build constrained cells from the planned positions
        constrained: dict[Position, Symbol] = {
            pos: symbol for pos in cluster_result.planned_positions
        }

        return StepIntent(
            step_type=StepType.CASCADE_CLUSTER,
            constrained_cells=constrained,
            strategic_cells={},
            expected_cluster_count=Range(1, 1),
            expected_cluster_sizes=[Range(size, size)],
            expected_cluster_tier=None,
            expected_spawns=[],
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=[],
            wfc_symbol_weights={},
            predicted_post_gravity=None,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            planned_explosion=cluster_result.planned_positions,
        )

    def _build_terminal_intent(self) -> StepIntent:
        """Build a terminal StepIntent that ends the cascade."""
        return StepIntent(
            step_type=StepType.TERMINAL_DEAD,
            constrained_cells={},
            strategic_cells={},
            expected_cluster_count=Range(0, 0),
            expected_cluster_sizes=[],
            expected_cluster_tier=None,
            expected_spawns=[],
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=[],
            wfc_symbol_weights={},
            predicted_post_gravity=None,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            is_terminal=True,
        )
