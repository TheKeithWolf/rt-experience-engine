"""Variance hints — solver-agnostic guidance derived from accumulators.

VarianceHints is a frozen snapshot computed once per instance from the
current accumulators. Passed to CSP (spatial_bias) and WFC (symbol_weights)
to steer the population toward uniform coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..primitives.board import Position
from ..primitives.symbols import Symbol


@dataclass(frozen=True, slots=True)
class VarianceHints:
    """Immutable variance guidance for the solver pipeline.

    spatial_bias: position → weight, sums to 1.0 (CONSTRAINT-VE-2)
    symbol_weights: symbol → weight, floored to min (CONSTRAINT-VE-3)
    near_miss_symbol_preference: sorted least-used first (for CSP symbol selection)
    cluster_size_preference: sorted least-used first (for within-range size selection)
    """

    spatial_bias: dict[Position, float]
    symbol_weights: dict[Symbol, float]
    near_miss_symbol_preference: tuple[Symbol, ...]
    cluster_size_preference: tuple[int, ...]
