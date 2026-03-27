"""Tests for rl_archive/diagnostics.py — archive health reporting.

RLA-110 through RLA-116: Coverage thresholds, quality metrics, empty archive,
generation source distribution, config-driven thresholds.
"""

from __future__ import annotations

import pytest

from ..config.schema import (
    ConfigValidationError,
    DescriptorConfig,
    RLArchiveDiagnosticsConfig,
)
from ..pipeline.data_types import GeneratedInstance
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..rl_archive.archive import MAPElitesArchive
from ..rl_archive.descriptor import TrajectoryDescriptor
from ..rl_archive.diagnostics import ArchiveHealthReport, compute_archive_health


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DESCRIPTOR_CONFIG = DescriptorConfig(
    spatial_col_bins=2, spatial_row_bins=2, payout_bins=2,
)
_NUM_SYMBOLS = 3  # Small for manageable total_cells
_DIAG_CONFIG = RLArchiveDiagnosticsConfig(
    coverage_warn_threshold=0.2,
    coverage_fail_threshold=0.05,
)


def _make_archive(num_entries: int = 0) -> MAPElitesArchive:
    archive = MAPElitesArchive(_DESCRIPTOR_CONFIG, _NUM_SYMBOLS)
    for i in range(num_entries):
        desc = TrajectoryDescriptor(
            archetype_id="test",
            step0_symbol=f"S{i % _NUM_SYMBOLS}",
            spatial_bin=(i % 2, 0),
            cluster_orientation="H",
            payout_bin=i % 2,
        )
        # Use a placeholder instance (diagnostics only needs descriptor + quality)
        archive.try_insert(
            {"sim_id": i, "payout": float(i)},  # type: ignore[arg-type]
            desc,
            quality=float(i + 1),
        )
    return archive


# ---------------------------------------------------------------------------
# RLA-110: Coverage above warn → "pass"
# ---------------------------------------------------------------------------


def test_rla_110_coverage_above_warn_pass() -> None:
    """RLA-110: Coverage above warn_threshold produces 'pass' status."""
    # Total cells = 2*2*2*3*3 = 72, need >= 0.2 * 72 = 14.4 → 15 unique entries
    archive = MAPElitesArchive(_DESCRIPTOR_CONFIG, _NUM_SYMBOLS)
    total = archive.total_cells()  # 72
    needed = int(total * 0.25) + 1  # 19

    # Enumerate unique descriptor keys systematically
    symbols = ["S0", "S1", "S2"]
    orientations = ["H", "V", "compact"]
    count = 0
    for sym in symbols:
        for col_bin in range(2):
            for row_bin in range(2):
                for orient in orientations:
                    for pbin in range(2):
                        if count >= needed:
                            break
                        desc = TrajectoryDescriptor(
                            archetype_id="test",
                            step0_symbol=sym,
                            spatial_bin=(col_bin, row_bin),
                            cluster_orientation=orient,
                            payout_bin=pbin,
                        )
                        archive.try_insert({"id": count}, desc, quality=1.0)  # type: ignore[arg-type]
                        count += 1

    assert archive.filled_count() >= needed
    assert archive.coverage() >= _DIAG_CONFIG.coverage_warn_threshold

    reports = compute_archive_health({"test": archive}, _DIAG_CONFIG)
    assert len(reports) == 1
    assert reports[0].status == "pass"


# ---------------------------------------------------------------------------
# RLA-111: Coverage between warn and fail → "warn"
# ---------------------------------------------------------------------------


def test_rla_111_coverage_between_warn_fail() -> None:
    """RLA-111: Coverage between fail and warn thresholds → 'warn' status."""
    archive = MAPElitesArchive(_DESCRIPTOR_CONFIG, _NUM_SYMBOLS)
    total = archive.total_cells()  # 72

    # Insert ~10% coverage (between 5% and 20%)
    needed = max(int(total * 0.10), 1)
    for i in range(needed):
        desc = TrajectoryDescriptor(
            archetype_id="test",
            step0_symbol=["S0", "S1", "S2"][i % 3],
            spatial_bin=(i % 2, (i // 2) % 2),
            cluster_orientation=["H", "V", "compact"][i % 3],
            payout_bin=i % 2,
        )
        archive.try_insert({"id": i}, desc, quality=1.0)  # type: ignore[arg-type]

    coverage = archive.coverage()
    # Verify we're in the warn range
    assert _DIAG_CONFIG.coverage_fail_threshold < coverage < _DIAG_CONFIG.coverage_warn_threshold

    reports = compute_archive_health({"test": archive}, _DIAG_CONFIG)
    assert reports[0].status == "warn"


# ---------------------------------------------------------------------------
# RLA-112: Coverage below fail → "fail"
# ---------------------------------------------------------------------------


def test_rla_112_coverage_below_fail() -> None:
    """RLA-112: Coverage below fail_threshold → 'fail' status."""
    archive = MAPElitesArchive(_DESCRIPTOR_CONFIG, _NUM_SYMBOLS)
    # Only 1 entry in 72 total = ~1.4% < 5%
    desc = TrajectoryDescriptor(
        archetype_id="test", step0_symbol="S0",
        spatial_bin=(0, 0), cluster_orientation="H", payout_bin=0,
    )
    archive.try_insert({"id": 0}, desc, quality=1.0)  # type: ignore[arg-type]

    reports = compute_archive_health({"test": archive}, _DIAG_CONFIG)
    assert reports[0].status == "fail"


# ---------------------------------------------------------------------------
# RLA-113: mean_quality computed correctly
# ---------------------------------------------------------------------------


def test_rla_113_mean_quality() -> None:
    """RLA-113: mean_quality is the average of all entry qualities."""
    archive = MAPElitesArchive(_DESCRIPTOR_CONFIG, _NUM_SYMBOLS)
    qualities = [1.0, 3.0, 5.0]
    for i, q in enumerate(qualities):
        desc = TrajectoryDescriptor(
            archetype_id="test", step0_symbol=f"S{i}",
            spatial_bin=(0, 0), cluster_orientation="H", payout_bin=0,
        )
        archive.try_insert({"id": i}, desc, quality=q)  # type: ignore[arg-type]

    reports = compute_archive_health({"test": archive}, _DIAG_CONFIG)
    assert reports[0].mean_quality == pytest.approx(3.0)
    assert reports[0].min_quality == pytest.approx(1.0)
    assert reports[0].max_quality == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# RLA-114: Empty archive → coverage 0.0, status "fail"
# ---------------------------------------------------------------------------


def test_rla_114_empty_archive_fail() -> None:
    """RLA-114: Empty archive has coverage 0.0 and status 'fail'."""
    archive = MAPElitesArchive(_DESCRIPTOR_CONFIG, _NUM_SYMBOLS)

    reports = compute_archive_health({"test": archive}, _DIAG_CONFIG)
    assert reports[0].coverage == 0.0
    assert reports[0].status == "fail"
    assert reports[0].mean_quality == 0.0


# ---------------------------------------------------------------------------
# RLA-115: generation_source_distribution field exists on DiagnosticsReport
# ---------------------------------------------------------------------------


def test_rla_115_generation_source_field() -> None:
    """RLA-115: DiagnosticsReport has generation_source_distribution field with default."""
    from ..diagnostics.engine import DiagnosticsReport

    # Construct with only required fields — new defaulted fields should work
    report = DiagnosticsReport(
        results=(),
        spatial_heatmap={},
        cluster_size_distribution={},
        symbol_win_contribution={},
        payout_percentiles={},
        archetype_distribution={},
        win_level_distribution={},
        failure_rates={},
        tumble_depth_distribution={},
        grid_multiplier_distribution={},
        booster_spawn_rate={},
        booster_fire_rate={},
        rocket_orientation_balance={},
        chain_trigger_rate=0.0,
        lb_target_distribution={},
        slb_target_distribution={},
        freespin_trigger_rate=0.0,
        wincap_hit_rate=0.0,
    )

    # Default values
    assert report.archive_health == ()
    assert report.generation_source_distribution == {}

    # With values
    health = (ArchiveHealthReport(
        archetype_id="test", coverage=0.5, mean_quality=2.0,
        min_quality=1.0, max_quality=3.0, empty_niche_count=36, status="pass",
    ),)
    report2 = DiagnosticsReport(
        results=(),
        spatial_heatmap={},
        cluster_size_distribution={},
        symbol_win_contribution={},
        payout_percentiles={},
        archetype_distribution={},
        win_level_distribution={},
        failure_rates={},
        tumble_depth_distribution={},
        grid_multiplier_distribution={},
        booster_spawn_rate={},
        booster_fire_rate={},
        rocket_orientation_balance={},
        chain_trigger_rate=0.0,
        lb_target_distribution={},
        slb_target_distribution={},
        freespin_trigger_rate=0.0,
        wincap_hit_rate=0.0,
        archive_health=health,
        generation_source_distribution={"static": 50, "cascade": 30, "rl_archive": 20},
    )
    assert len(report2.archive_health) == 1
    assert report2.generation_source_distribution["rl_archive"] == 20


# ---------------------------------------------------------------------------
# RLA-116: Thresholds from config, not hardcoded
# ---------------------------------------------------------------------------


def test_rla_116_thresholds_from_config() -> None:
    """RLA-116: Different config thresholds produce different status."""
    archive = MAPElitesArchive(_DESCRIPTOR_CONFIG, _NUM_SYMBOLS)
    # Insert 2 entries in 72 total = ~2.8%
    for i in range(2):
        desc = TrajectoryDescriptor(
            archetype_id="test", step0_symbol=f"S{i}",
            spatial_bin=(0, 0), cluster_orientation="H", payout_bin=i,
        )
        archive.try_insert({"id": i}, desc, quality=1.0)  # type: ignore[arg-type]

    # With default config (fail=0.05, warn=0.2): 2.8% < 5% → "fail"
    reports_strict = compute_archive_health({"test": archive}, _DIAG_CONFIG)
    assert reports_strict[0].status == "fail"

    # With lenient config (fail=0.01, warn=0.03): 2.8% > 0.01 and < 0.03 → "warn"
    lenient = RLArchiveDiagnosticsConfig(
        coverage_warn_threshold=0.03,
        coverage_fail_threshold=0.01,
    )
    reports_lenient = compute_archive_health({"test": archive}, lenient)
    assert reports_lenient[0].status == "warn"
