"""Population-level accumulators for variance steering.

Tracks running statistics across the generated population. Updated after
each valid instance (CONSTRAINT-VE-4). Reset at the start of each population
run (CONSTRAINT-VE-5). NOT frozen — accumulators are explicitly mutable counters.

All initial dimensions come from MasterConfig — zero hardcoded values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config.schema import MasterConfig
from ..primitives.board import Position
from ..primitives.symbols import Symbol, is_standard, symbol_from_name

if TYPE_CHECKING:
    from ..pipeline.data_types import GeneratedInstance


@dataclass(slots=True)
class PopulationAccumulators:
    """Mutable accumulators tracking population-level statistics.

    Updated after each valid instance. Reset at population run start.
    """

    # How often each standard symbol appears in winning clusters
    symbol_win_frequency: dict[Symbol, int]
    # How often each board position participates in a winning cluster
    position_win_participation: dict[Position, int]
    # Distribution of cluster sizes generated
    cluster_size_histogram: dict[int, int]
    # How often scatters land on each position
    scatter_position_frequency: dict[Position, int]
    # Per-position, per-symbol fill frequency (for WFC noise fill diversity)
    symbol_frequency_per_position: dict[Position, dict[Symbol, int]]
    # Rocket orientation balance tracker
    rocket_hv_ratio: dict[str, int]
    # Lightball target symbol distribution
    lb_target_histogram: dict[Symbol, int]
    # Near-miss symbol usage frequency
    near_miss_symbol_frequency: dict[Symbol, int]
    # Total instances processed (denominator for rate calculations)
    total_instances: int = field(default=0)

    @classmethod
    def create(cls, config: MasterConfig) -> PopulationAccumulators:
        """Factory — initializes all counters to zero using config dimensions."""
        num_reels = config.board.num_reels
        num_rows = config.board.num_rows

        # All valid board positions
        all_positions = [
            Position(reel, row)
            for reel in range(num_reels)
            for row in range(num_rows)
        ]

        # Standard symbols from config
        standard_symbols = tuple(
            symbol_from_name(name) for name in config.symbols.standard
        )

        return cls(
            symbol_win_frequency={sym: 0 for sym in standard_symbols},
            position_win_participation={pos: 0 for pos in all_positions},
            cluster_size_histogram={},
            scatter_position_frequency={pos: 0 for pos in all_positions},
            symbol_frequency_per_position={
                pos: {sym: 0 for sym in standard_symbols}
                for pos in all_positions
            },
            rocket_hv_ratio={"H": 0, "V": 0},
            lb_target_histogram={sym: 0 for sym in standard_symbols},
            near_miss_symbol_frequency={sym: 0 for sym in standard_symbols},
            total_instances=0,
        )

    def reset(self) -> None:
        """Zero all counters (CONSTRAINT-VE-5)."""
        for key in self.symbol_win_frequency:
            self.symbol_win_frequency[key] = 0
        for key in self.position_win_participation:
            self.position_win_participation[key] = 0
        self.cluster_size_histogram.clear()
        for key in self.scatter_position_frequency:
            self.scatter_position_frequency[key] = 0
        for pos_dict in self.symbol_frequency_per_position.values():
            for key in pos_dict:
                pos_dict[key] = 0
        self.rocket_hv_ratio["H"] = 0
        self.rocket_hv_ratio["V"] = 0
        for key in self.lb_target_histogram:
            self.lb_target_histogram[key] = 0
        for key in self.near_miss_symbol_frequency:
            self.near_miss_symbol_frequency[key] = 0
        self.total_instances = 0

    def update(self, instance: GeneratedInstance) -> None:
        """Increment accumulators from a validated instance (CONSTRAINT-VE-4)."""
        self.total_instances += 1

        step = instance.spatial_step

        # Cluster statistics
        for cluster in step.clusters:
            self.symbol_win_frequency[cluster.symbol] = (
                self.symbol_win_frequency.get(cluster.symbol, 0) + 1
            )
            self.cluster_size_histogram[cluster.size] = (
                self.cluster_size_histogram.get(cluster.size, 0) + 1
            )
            for pos in cluster.positions:
                self.position_win_participation[pos] = (
                    self.position_win_participation.get(pos, 0) + 1
                )

        # Scatter position tracking
        for pos in step.scatter_positions:
            self.scatter_position_frequency[pos] = (
                self.scatter_position_frequency.get(pos, 0) + 1
            )

        # Near-miss symbol tracking
        for nm in step.near_misses:
            self.near_miss_symbol_frequency[nm.symbol] = (
                self.near_miss_symbol_frequency.get(nm.symbol, 0) + 1
            )

        # Board fill symbol-per-position tracking
        board = instance.board
        for pos in self.symbol_frequency_per_position:
            sym = board.get(pos)
            if sym is not None and sym in self.symbol_frequency_per_position[pos]:
                self.symbol_frequency_per_position[pos][sym] += 1
