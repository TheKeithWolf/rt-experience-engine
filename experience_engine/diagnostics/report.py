"""Diagnostics report formatting — human-readable text output.

Formats the DiagnosticsReport into a readable text report with
pass/fail markers for each metric.
"""

from __future__ import annotations

from .engine import DiagnosticsReport


# Status markers for report output
_STATUS_MARKERS: dict[str, str] = {
    "pass": "[PASS]",
    "warn": "[WARN]",
    "fail": "[FAIL]",
}


def format_diagnostics(report: DiagnosticsReport) -> str:
    """Format diagnostics report as human-readable text."""
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("  DIAGNOSTICS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Scalar metrics with targets
    lines.append("--- Metrics ---")
    for result in report.results:
        marker = _STATUS_MARKERS.get(result.status, "[????]")
        target_str = _format_target(result.target_min, result.target_max)
        lines.append(
            f"  {marker} {result.metric}: {result.value:.4f}"
            f"  target: {target_str}"
        )
    lines.append("")

    # Cluster size distribution
    if report.cluster_size_distribution:
        lines.append("--- Cluster Size Distribution ---")
        for size in sorted(report.cluster_size_distribution.keys()):
            count = report.cluster_size_distribution[size]
            lines.append(f"  size {size}: {count}")
        lines.append("")

    # Symbol win contribution
    if report.symbol_win_contribution:
        lines.append("--- Symbol Win Contribution ---")
        for sym in sorted(report.symbol_win_contribution.keys()):
            frac = report.symbol_win_contribution[sym]
            lines.append(f"  {sym}: {frac:.4f}")
        lines.append("")

    # Win level distribution
    if report.win_level_distribution:
        lines.append("--- Win Level Distribution ---")
        for level in sorted(report.win_level_distribution.keys()):
            count = report.win_level_distribution[level]
            lines.append(f"  level {level}: {count}")
        lines.append("")

    # Payout percentiles
    if report.payout_percentiles:
        lines.append("--- Payout Percentiles ---")
        for key in ("p50", "p90", "p95", "p99"):
            if key in report.payout_percentiles:
                lines.append(
                    f"  {key}: {report.payout_percentiles[key]:.4f}x"
                )
        lines.append("")

    # Archetype distribution
    if report.archetype_distribution:
        lines.append("--- Archetype Distribution ---")
        for arch_id in sorted(report.archetype_distribution.keys()):
            count = report.archetype_distribution[arch_id]
            lines.append(f"  {arch_id}: {count}")
        lines.append("")

    # Tumble depth distribution (cascade-specific)
    if report.tumble_depth_distribution:
        lines.append("--- Tumble Depth Distribution ---")
        for depth in sorted(report.tumble_depth_distribution.keys()):
            count = report.tumble_depth_distribution[depth]
            lines.append(f"  depth {depth}: {count}")
        lines.append("")

    # Grid multiplier distribution (cascade-specific)
    if report.grid_multiplier_distribution:
        lines.append("--- Grid Multiplier Distribution ---")
        for val in sorted(report.grid_multiplier_distribution.keys()):
            count = report.grid_multiplier_distribution[val]
            lines.append(f"  value {val}: {count}")
        lines.append("")

    # Booster spawn rates (Phase 9+)
    if report.booster_spawn_rate:
        lines.append("--- Booster Spawn Rate ---")
        for btype in sorted(report.booster_spawn_rate.keys()):
            rate = report.booster_spawn_rate[btype]
            lines.append(f"  {btype}: {rate:.4f}")
        lines.append("")

    # Booster fire rates (Phase 9+)
    if report.booster_fire_rate:
        lines.append("--- Booster Fire Rate ---")
        for btype in sorted(report.booster_fire_rate.keys()):
            rate = report.booster_fire_rate[btype]
            lines.append(f"  {btype}: {rate:.4f}")
        lines.append("")

    # Rocket orientation balance (Phase 9+)
    if report.rocket_orientation_balance:
        lines.append("--- Rocket Orientation Balance ---")
        for orient in sorted(report.rocket_orientation_balance.keys()):
            frac = report.rocket_orientation_balance[orient]
            lines.append(f"  {orient}: {frac:.4f}")
        lines.append("")

    # Chain trigger rate (Phase 9+)
    if report.chain_trigger_rate > 0:
        lines.append(f"--- Chain Trigger Rate: {report.chain_trigger_rate:.4f} ---")
        lines.append("")

    # Failure rates
    if report.failure_rates:
        lines.append("--- Failure Rates ---")
        for arch_id in sorted(report.failure_rates.keys()):
            count = report.failure_rates[arch_id]
            lines.append(f"  {arch_id}: {count} failures")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def _format_target(
    min_val: float | None, max_val: float | None,
) -> str:
    """Format target range as readable string."""
    if min_val is not None and max_val is not None:
        return f"[{min_val:.4f}, {max_val:.4f}]"
    if min_val is not None:
        return f">= {min_val:.4f}"
    if max_val is not None:
        return f"<= {max_val:.4f}"
    return "none"
