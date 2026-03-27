"""Archive health diagnostics — coverage, quality distributions, and status.

Computes per-archetype health reports from MAP-Elites archives.
Status thresholds are config-driven, not hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import RLArchiveDiagnosticsConfig
from .archive import MAPElitesArchive


@dataclass(frozen=True, slots=True)
class ArchiveHealthReport:
    """Health report for one archetype's MAP-Elites archive."""

    archetype_id: str
    coverage: float
    mean_quality: float
    min_quality: float
    max_quality: float
    empty_niche_count: int
    status: str  # "pass", "warn", "fail"


def compute_archive_health(
    archives: dict[str, MAPElitesArchive],
    diag_config: RLArchiveDiagnosticsConfig,
) -> tuple[ArchiveHealthReport, ...]:
    """Compute health reports for all archetype archives.

    Status derivation from config thresholds:
    - coverage >= warn_threshold → "pass"
    - fail_threshold < coverage < warn_threshold → "warn"
    - coverage <= fail_threshold → "fail"
    """
    reports: list[ArchiveHealthReport] = []

    for archetype_id in sorted(archives.keys()):
        archive = archives[archetype_id]
        coverage = archive.coverage()
        filled = archive.filled_count()
        total = archive.total_cells()
        empty_count = total - filled

        # Quality statistics from occupied cells
        qualities = [
            entry.quality
            for key in archive.occupied_keys()
            if (entry := archive.get(key)) is not None
        ]

        if qualities:
            mean_q = sum(qualities) / len(qualities)
            min_q = min(qualities)
            max_q = max(qualities)
        else:
            mean_q = 0.0
            min_q = 0.0
            max_q = 0.0

        status = _derive_status(coverage, diag_config)

        reports.append(ArchiveHealthReport(
            archetype_id=archetype_id,
            coverage=coverage,
            mean_quality=mean_q,
            min_quality=min_q,
            max_quality=max_q,
            empty_niche_count=empty_count,
            status=status,
        ))

    return tuple(reports)


def _derive_status(
    coverage: float,
    config: RLArchiveDiagnosticsConfig,
) -> str:
    """Derive archive health status from coverage and config thresholds."""
    if coverage >= config.coverage_warn_threshold:
        return "pass"
    if coverage > config.coverage_fail_threshold:
        return "warn"
    return "fail"
