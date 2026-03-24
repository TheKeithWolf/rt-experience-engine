"""Tests for variance engine — accumulators, hints, and bias computation.

Covers TEST-P4-005 through P4-007.
"""

from __future__ import annotations

import pytest

from ..config.schema import MasterConfig
from ..primitives.board import Position
from ..primitives.symbols import Symbol, symbol_from_name
from ..variance.accumulators import PopulationAccumulators
from ..variance.bias_computation import compute_hints


# ---------------------------------------------------------------------------
# TEST-P4-005: Accumulators initialize to zeros
# ---------------------------------------------------------------------------

def test_accumulators_initialize_to_zeros(default_config: MasterConfig) -> None:
    acc = PopulationAccumulators.create(default_config)

    assert acc.total_instances == 0

    # All symbol frequencies at zero
    for sym, count in acc.symbol_win_frequency.items():
        assert count == 0, f"{sym} should be 0"

    # All position participation at zero
    for pos, count in acc.position_win_participation.items():
        assert count == 0, f"{pos} should be 0"

    # Position count matches board size (7x7 = 49)
    expected_positions = (
        default_config.board.num_reels * default_config.board.num_rows
    )
    assert len(acc.position_win_participation) == expected_positions

    # Empty cluster size histogram
    assert len(acc.cluster_size_histogram) == 0

    # Rocket ratio at zero
    assert acc.rocket_hv_ratio == {"H": 0, "V": 0}


def test_accumulators_reset_clears_all(default_config: MasterConfig) -> None:
    acc = PopulationAccumulators.create(default_config)
    # Manually set some values
    acc.total_instances = 42
    acc.symbol_win_frequency[Symbol.L1] = 10
    acc.cluster_size_histogram[5] = 20
    acc.rocket_hv_ratio["H"] = 5

    acc.reset()

    assert acc.total_instances == 0
    assert acc.symbol_win_frequency[Symbol.L1] == 0
    assert len(acc.cluster_size_histogram) == 0
    assert acc.rocket_hv_ratio["H"] == 0


# ---------------------------------------------------------------------------
# TEST-P4-006: Spatial bias — uniform accumulators → uniform distribution
# ---------------------------------------------------------------------------

def test_uniform_accumulators_produce_uniform_spatial_bias(
    default_config: MasterConfig,
) -> None:
    acc = PopulationAccumulators.create(default_config)
    hints = compute_hints(acc, default_config)

    # All spatial biases should be equal (uniform)
    values = list(hints.spatial_bias.values())
    assert len(values) > 0
    first = values[0]
    for v in values:
        assert abs(v - first) < 1e-10, "spatial bias should be uniform"

    # Sum to 1.0
    total = sum(values)
    assert abs(total - 1.0) < 1e-10


def test_uniform_accumulators_produce_uniform_symbol_weights(
    default_config: MasterConfig,
) -> None:
    acc = PopulationAccumulators.create(default_config)
    hints = compute_hints(acc, default_config)

    # All symbol weights should be equal (uniform)
    values = list(hints.symbol_weights.values())
    assert len(values) > 0
    first = values[0]
    for v in values:
        assert abs(v - first) < 1e-10, "symbol weights should be uniform"


# ---------------------------------------------------------------------------
# TEST-P4-007: Symbol weights never below config minimum
# ---------------------------------------------------------------------------

def test_symbol_weights_never_below_minimum(
    default_config: MasterConfig,
) -> None:
    acc = PopulationAccumulators.create(default_config)
    # Skew one symbol heavily so others get low weights
    acc.symbol_win_frequency[Symbol.L1] = 10000
    acc.total_instances = 10000

    hints = compute_hints(acc, default_config)

    min_weight = default_config.solvers.wfc_min_symbol_weight
    for sym, weight in hints.symbol_weights.items():
        assert weight >= min_weight, (
            f"{sym} weight {weight} < minimum {min_weight}"
        )


def test_biased_accumulators_produce_nonuniform_hints(
    default_config: MasterConfig,
) -> None:
    """After skewing accumulators, underrepresented items get higher weights."""
    acc = PopulationAccumulators.create(default_config)
    # Use L1 heavily, leave L4 unused
    acc.symbol_win_frequency[Symbol.L1] = 100
    acc.total_instances = 100

    hints = compute_hints(acc, default_config)

    # L4 (unused) should have higher weight than L1 (overused)
    assert hints.symbol_weights[Symbol.L4] > hints.symbol_weights[Symbol.L1]
