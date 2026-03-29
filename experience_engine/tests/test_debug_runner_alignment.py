"""Tests that debug_archetype.diagnostic_attempt builds structurally correct
CascadeStepRecords — specifically that the D1–D7 fixes produce records with
gravity data, booster fire records, and the post-fire merge.

The debug runner includes extra logic (phase advancement, step assessment)
not present in _attempt_generation, so exact instance equality isn't expected.
Instead, these tests verify the structural correctness of the records that
were broken by D1–D7.
"""

from __future__ import annotations

import io
import random
import sys

import pytest

from ..debug_archetype import diagnostic_attempt
from ..run import _build_full_registry, _build_pipeline
from ..variance.accumulators import PopulationAccumulators
from ..variance.bias_computation import compute_hints


@pytest.fixture(scope="module")
def pipeline_components(default_config):
    """Build full registry and pipeline once per module."""
    registry = _build_full_registry(default_config)
    _static_gen, cascade_gen, _validator = _build_pipeline(default_config, registry)
    return default_config, registry, cascade_gen


def _run_debug_silent(sig, hints, rng, gen, registry):
    """Run diagnostic_attempt with stdout suppressed."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        success, reason, step, instance = diagnostic_attempt(
            sig, sim_id=0, hints=hints, rng=rng,
            gen=gen, registry=registry, attempt_num=1,
        )
    finally:
        sys.stdout = old_stdout
    return success, reason, step, instance


def _find_valid_seed(sig, hints, gen, registry, max_seeds=200):
    """Search for a seed that produces a valid instance in the debug runner."""
    for seed in range(max_seeds):
        rng = random.Random(seed)
        success, _reason, _step, instance = _run_debug_silent(
            sig, hints, rng, gen, registry,
        )
        if success:
            return seed, instance
    return None, None


def test_cascade_records_have_gravity(pipeline_components):
    """Non-terminal steps must carry gravity records (fixes D1, D5)."""
    config, registry, cascade_gen = pipeline_components
    sig = registry.get("t1_multi_cascade")
    accumulators = PopulationAccumulators.create(config)
    hints = compute_hints(accumulators, config)

    seed, instance = _find_valid_seed(sig, hints, cascade_gen, registry)
    if instance is None:
        pytest.skip("No valid seed found for t1_multi_cascade")

    assert instance.cascade_steps is not None
    # Multi-cascade has ≥2 steps; non-terminal steps must have gravity_record
    non_terminal = instance.cascade_steps[:-1]
    for i, step_rec in enumerate(non_terminal):
        assert step_rec.gravity_record is not None, (
            f"Step {i} (non-terminal) has gravity_record=None — "
            f"transition data not collected (D1/D5)"
        )
        assert step_rec.gravity_record.refill_entries, (
            f"Step {i} gravity_record has empty refill_entries"
        )


def test_terminal_step_has_no_gravity(pipeline_components):
    """Terminal step must have gravity_record=None (transition_data=None)."""
    config, registry, cascade_gen = pipeline_components
    sig = registry.get("t1_multi_cascade")
    accumulators = PopulationAccumulators.create(config)
    hints = compute_hints(accumulators, config)

    seed, instance = _find_valid_seed(sig, hints, cascade_gen, registry)
    if instance is None:
        pytest.skip("No valid seed found for t1_multi_cascade")

    assert instance.cascade_steps is not None
    terminal = instance.cascade_steps[-1]
    assert terminal.gravity_record is None, (
        "Terminal step should have gravity_record=None"
    )


def test_instance_returned_on_success(pipeline_components):
    """diagnostic_attempt returns GeneratedInstance on success (Change 7a)."""
    config, registry, cascade_gen = pipeline_components
    sig = registry.get("t1_multi_cascade")
    accumulators = PopulationAccumulators.create(config)
    hints = compute_hints(accumulators, config)

    seed, instance = _find_valid_seed(sig, hints, cascade_gen, registry)
    if instance is None:
        pytest.skip("No valid seed found for t1_multi_cascade")

    assert instance.archetype_id == sig.id
    assert instance.cascade_steps is not None
    assert len(instance.cascade_steps) >= sig.required_cascade_depth.min_val


def test_instance_none_on_failure(pipeline_components):
    """diagnostic_attempt returns None for instance on failure."""
    config, registry, cascade_gen = pipeline_components
    sig = registry.get("t1_multi_cascade")
    accumulators = PopulationAccumulators.create(config)
    hints = compute_hints(accumulators, config)

    # Use a seed likely to fail (seed 9999 with complex archetype)
    rng = random.Random(9999)
    success, _reason, _step, instance = _run_debug_silent(
        sig, hints, rng, cascade_gen, registry,
    )
    if success:
        pytest.skip("Seed 9999 unexpectedly succeeded")

    assert instance is None
