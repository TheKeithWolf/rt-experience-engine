"""Bias computation — converts raw accumulators into solver-ready variance hints.

Inverse-frequency approach: symbols/positions used LESS get HIGHER weights,
steering the population toward uniform coverage. When accumulators are all
zero (first instance), returns uniform weights.

All minimum weight floors come from MasterConfig — zero hardcoded values.
"""

from __future__ import annotations

from ..config.schema import MasterConfig
from ..primitives.board import Position
from ..primitives.symbols import Symbol, symbol_from_name
from .accumulators import PopulationAccumulators
from .hints import VarianceHints


def compute_hints(
    accumulators: PopulationAccumulators,
    config: MasterConfig,
) -> VarianceHints:
    """Convert raw accumulators into solver-ready variance hints.

    Guarantees:
    - spatial_bias sums to 1.0 (CONSTRAINT-VE-2)
    - symbol_weights floored to config.solvers.wfc_min_symbol_weight (CONSTRAINT-VE-3)
    - When all accumulators are zero, returns uniform weights
    """
    min_weight = config.solvers.wfc_min_symbol_weight

    # Spatial bias — inverse frequency of position win participation
    spatial_bias = _inverse_frequency_normalized(
        accumulators.position_win_participation,
    )

    # Symbol weights — inverse frequency of symbol win usage
    symbol_weights = _inverse_frequency_floored(
        accumulators.symbol_win_frequency,
        min_weight,
    )

    # Near-miss symbol preference — least-used symbols first
    near_miss_symbol_preference = _sorted_by_ascending_count(
        accumulators.near_miss_symbol_frequency,
    )

    # Cluster size preference — least-used sizes first
    cluster_size_preference = _sorted_by_ascending_count(
        accumulators.cluster_size_histogram,
    )

    return VarianceHints(
        spatial_bias=spatial_bias,
        symbol_weights=symbol_weights,
        near_miss_symbol_preference=near_miss_symbol_preference,
        cluster_size_preference=cluster_size_preference,
    )


def _inverse_frequency_normalized(
    counts: dict[Position, int],
) -> dict[Position, float]:
    """Convert count dict to inverse-frequency weights normalized to sum to 1.0.

    Zero counts get a weight equal to (max_count + 1) to maximally prefer
    unused positions. If all counts are zero, returns uniform 1/N weights.
    """
    if not counts:
        return {}

    total_count = sum(counts.values())
    n = len(counts)

    # Cold start — all zeros → uniform
    if total_count == 0:
        uniform = 1.0 / n
        return {k: uniform for k in counts}

    # Inverse frequency: higher count → lower weight
    max_count = max(counts.values())
    ceiling = max_count + 1
    raw: dict[Position, float] = {}
    for k, v in counts.items():
        raw[k] = float(ceiling - v) if v > 0 else float(ceiling)

    # Normalize to sum to 1.0
    raw_sum = sum(raw.values())
    return {k: v / raw_sum for k, v in raw.items()}


def _inverse_frequency_floored(
    counts: dict[Symbol, int],
    min_weight: float,
) -> dict[Symbol, float]:
    """Convert count dict to inverse-frequency weights with minimum floor.

    Ensures no weight falls below min_weight (CONSTRAINT-VE-3).
    If all counts are zero, returns uniform 1/N weights.
    """
    if not counts:
        return {}

    total_count = sum(counts.values())
    n = len(counts)

    # Cold start — all zeros → uniform
    if total_count == 0:
        uniform = 1.0 / n
        return {k: max(uniform, min_weight) for k in counts}

    # Inverse frequency
    max_count = max(counts.values())
    ceiling = max_count + 1
    raw: dict[Symbol, float] = {}
    for k, v in counts.items():
        raw[k] = float(ceiling - v) if v > 0 else float(ceiling)

    # Normalize, then floor
    raw_sum = sum(raw.values())
    result: dict[Symbol, float] = {}
    for k, v in raw.items():
        result[k] = max(v / raw_sum, min_weight)

    return result


def _sorted_by_ascending_count(
    counts: dict,
) -> tuple:
    """Sort keys by ascending count value — least-used first.

    Ties broken by key value for determinism.
    """
    # Sort by (count, key) for stable ordering
    items = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    return tuple(k for k, _ in items)
