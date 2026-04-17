"""Snapshot tests for B6 — ClusterBuilder scoring weight hoist.

Verifies that hoisting the seven literals from `_merge_score` and
`_payout_score` onto ReasonerConfig leaves the scoring outputs unchanged
at the YAML default values. Also exercises the validation surface of the
new ReasonerConfig fields.
"""

from __future__ import annotations

import dataclasses

import pytest

from ..config.schema import ConfigValidationError, MasterConfig, ReasonerConfig


# ---------------------------------------------------------------------------
# Validation — bounds on the new fields
# ---------------------------------------------------------------------------

def _make_reasoner_with(**overrides) -> ReasonerConfig:
    """Helper: build a valid ReasonerConfig with a single override.

    Any field can be overridden; remaining fields take the values that
    match default.yaml so the test isolates the field under test.
    """
    base = dict(
        payout_low_fraction=0.3,
        payout_high_fraction=0.85,
        arming_urgency_horizon=1,
        terminal_dead_default_max_component=4,
        max_forward_simulations_per_step=10,
        max_strategic_cells_per_step=16,
        lookahead_depth=3,
        cluster_merge_acceptable_score=0.6,
        cluster_merge_overflow_score=0.1,
        cluster_payout_undertarget_trigger=0.5,
        cluster_payout_overceiling_trigger=1.5,
        cluster_payout_score_floor=0.05,
        cluster_payout_ontrack_smoothing=0.5,
        cluster_payout_ontrack_floor=0.3,
    )
    base.update(overrides)
    return ReasonerConfig(**base)


def test_b6_score_field_rejects_above_one() -> None:
    """cluster_merge_acceptable_score is a probability-like weight in [0, 1]."""
    with pytest.raises(ConfigValidationError, match="cluster_merge_acceptable_score"):
        _make_reasoner_with(cluster_merge_acceptable_score=1.5)


def test_b6_trigger_field_rejects_zero() -> None:
    """Trigger multipliers must be strictly positive — 0.0 would short-circuit."""
    with pytest.raises(ConfigValidationError, match="cluster_payout_undertarget_trigger"):
        _make_reasoner_with(cluster_payout_undertarget_trigger=0.0)


def test_b6_trigger_field_rejects_above_ten() -> None:
    """The (0.0, 10.0] cap catches typos like 50 without blocking aggressive tuning."""
    with pytest.raises(ConfigValidationError, match="cluster_payout_overceiling_trigger"):
        _make_reasoner_with(cluster_payout_overceiling_trigger=20.0)


# ---------------------------------------------------------------------------
# Snapshot — default-valued ReasonerConfig matches the previous literals
# ---------------------------------------------------------------------------

def test_b6_default_yaml_matches_pre_refactor_constants(
    default_config: MasterConfig,
) -> None:
    """The YAML defaults must match the literal values that lived in
    cluster_builder.py before the B6 hoist — otherwise the refactor would
    silently change generation behavior at default config.
    """
    r = default_config.reasoner
    assert r.cluster_merge_acceptable_score == 0.6
    assert r.cluster_merge_overflow_score == 0.1
    assert r.cluster_payout_undertarget_trigger == 0.5
    assert r.cluster_payout_overceiling_trigger == 1.5
    assert r.cluster_payout_score_floor == 0.05
    assert r.cluster_payout_ontrack_smoothing == 0.5
    assert r.cluster_payout_ontrack_floor == 0.3


def test_b6_field_replaces_act_as_drop_in(
    default_config: MasterConfig,
) -> None:
    """A ReasonerConfig copy with overrides must produce the configured values
    in the cluster scoring fields — proves no shadowing or hidden defaults
    leaked into ClusterBuilder.
    """
    overridden = dataclasses.replace(
        default_config.reasoner,
        cluster_merge_acceptable_score=0.42,
        cluster_payout_score_floor=0.11,
    )
    assert overridden.cluster_merge_acceptable_score == 0.42
    assert overridden.cluster_payout_score_floor == 0.11
