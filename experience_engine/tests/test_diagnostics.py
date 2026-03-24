"""Tests for the diagnostics engine.

Covers TEST-P4-022 and P4-023.
"""

from __future__ import annotations

import pytest

from ..config.schema import MasterConfig
from ..diagnostics.engine import DiagnosticsEngine
from ..primitives.board import Position
from ..validation.metrics import InstanceMetrics


def _make_dead_metrics(sim_id: int = 0) -> InstanceMetrics:
    """Create metrics for a dead spin (payout=0)."""
    return InstanceMetrics(
        archetype_id="dead_empty",
        family="dead",
        criteria="0",
        sim_id=sim_id,
        payout=0.0,
        centipayout=0,
        win_level=1,
        cluster_count=0,
        cluster_sizes=(),
        cluster_symbols=(),
        scatter_count=0,
        near_miss_count=0,
        near_miss_symbols=(),
        max_component_size=3,
        is_valid=True,
        validation_errors=(),
    )


def _make_win_metrics(
    sim_id: int = 0,
    payout: float = 0.5,
    archetype_id: str = "t1_single",
) -> InstanceMetrics:
    """Create metrics for a winning spin."""
    return InstanceMetrics(
        archetype_id=archetype_id,
        family="t1",
        criteria="basegame",
        sim_id=sim_id,
        payout=payout,
        centipayout=int(payout * 100),
        win_level=2,
        cluster_count=1,
        cluster_sizes=(5,),
        cluster_symbols=("L1",),
        scatter_count=0,
        near_miss_count=0,
        near_miss_symbols=(),
        max_component_size=5,
        is_valid=True,
        validation_errors=(),
    )


# ---------------------------------------------------------------------------
# TEST-P4-022: dead_spin_rate = 1.0 for dead-only population
# ---------------------------------------------------------------------------

def test_dead_spin_rate_all_dead(default_config: MasterConfig) -> None:
    engine = DiagnosticsEngine(default_config)
    metrics = tuple(_make_dead_metrics(i) for i in range(100))
    report = engine.analyze(metrics)

    # Find dead_spin_rate in results
    dead_rate = next(
        r for r in report.results if r.metric == "dead_spin_rate"
    )
    assert dead_rate.value == 1.0, f"Expected 1.0, got {dead_rate.value}"

    # hit_rate should be 0
    hit_rate = next(r for r in report.results if r.metric == "hit_rate")
    assert hit_rate.value == 0.0


def test_mixed_population_rates(default_config: MasterConfig) -> None:
    """Verify hit_rate and dead_spin_rate are complementary."""
    engine = DiagnosticsEngine(default_config)
    # 70 dead + 30 winning
    metrics = (
        tuple(_make_dead_metrics(i) for i in range(70))
        + tuple(_make_win_metrics(i + 70) for i in range(30))
    )
    report = engine.analyze(metrics)

    dead_rate = next(r for r in report.results if r.metric == "dead_spin_rate")
    hit_rate = next(r for r in report.results if r.metric == "hit_rate")
    assert abs(dead_rate.value - 0.70) < 1e-10
    assert abs(hit_rate.value - 0.30) < 1e-10


# ---------------------------------------------------------------------------
# TEST-P4-023: spatial_cv decreases over instances (proxy via archetype CV)
# ---------------------------------------------------------------------------

def test_spatial_cv_uniform_distribution(default_config: MasterConfig) -> None:
    """Perfectly uniform archetype distribution → CV near 0."""
    engine = DiagnosticsEngine(default_config)
    # 50 of each archetype = perfectly uniform
    metrics = (
        tuple(_make_dead_metrics(i) for i in range(50))
        + tuple(_make_win_metrics(i + 50, archetype_id="t1_single") for i in range(50))
    )
    report = engine.analyze(metrics)
    cv = next(r for r in report.results if r.metric == "spatial_cv")
    assert cv.value == 0.0, f"Expected CV=0 for uniform, got {cv.value}"


def test_spatial_cv_skewed_distribution(default_config: MasterConfig) -> None:
    """Highly skewed archetype distribution → higher CV."""
    engine = DiagnosticsEngine(default_config)
    # 90 dead + 10 t1_single = skewed
    metrics = (
        tuple(_make_dead_metrics(i) for i in range(90))
        + tuple(_make_win_metrics(i + 90, archetype_id="t1_single") for i in range(10))
    )
    report = engine.analyze(metrics)
    cv = next(r for r in report.results if r.metric == "spatial_cv")
    assert cv.value > 0, "Expected positive CV for skewed distribution"


def test_diagnostics_payout_percentiles(default_config: MasterConfig) -> None:
    engine = DiagnosticsEngine(default_config)
    metrics = tuple(_make_win_metrics(i, payout=float(i)) for i in range(100))
    report = engine.analyze(metrics)
    assert report.payout_percentiles["p50"] > 0
    assert report.payout_percentiles["p99"] > report.payout_percentiles["p50"]


def test_diagnostics_empty_metrics(default_config: MasterConfig) -> None:
    engine = DiagnosticsEngine(default_config)
    report = engine.analyze(())
    assert len(report.results) == 0
