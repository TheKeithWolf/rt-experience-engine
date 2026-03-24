"""Post-optimization archetype survival audit.

Joins the Rust optimizer's output (adjusted weights) with the archetype lookup
table to compute per-archetype survival rates. Flags archetypes where the optimizer
zeroed out most books — indicating poor fit with the target distribution.

The archetype LUT is produced by LookupTableWriter.write_archetype_table() and maps
each sim_id to its archetype_id and criteria.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArchetypeSurvival:
    """Survival metrics for one archetype after weight optimization."""

    archetype_id: str
    total_books: int
    # Books with weight > 0 — the optimizer kept these in the distribution
    surviving_books: int
    survival_rate: float
    mean_weight: float
    # Flagged when survival_rate < threshold — optimizer dropped most instances
    flagged: bool


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Complete post-optimization audit report."""

    archetypes: tuple[ArchetypeSurvival, ...]
    total_survival_rate: float
    flagged_archetypes: tuple[str, ...]
    # Weighted RTP: sum(weight * payout) / sum(weight) across all books
    rtp_weighted: float


def run_audit(
    optimized_lut_path: Path,
    archetype_lut_path: Path,
    survival_threshold: float,
) -> AuditReport:
    """Join optimized weights with archetype LUT, compute per-archetype survival.

    Reads:
    - optimized LUT: sim_id,weight,payout (CSV, no header)
    - archetype LUT: sim_id,archetype_id,criteria (CSV, no header)

    Joins on sim_id. Computes survival_rate = books_with_nonzero_weight / total_books
    per archetype. Flags archetypes below the survival threshold.
    """
    # Read optimized LUT — sim_id → (weight, payout)
    weights: dict[int, tuple[int, int]] = {}
    with open(optimized_lut_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or not row[0].strip():
                continue
            sim_id = int(row[0])
            weight = int(row[1])
            payout = int(row[2])
            weights[sim_id] = (weight, payout)

    # Read archetype LUT — sim_id → archetype_id
    archetype_map: dict[int, str] = {}
    with open(archetype_lut_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or not row[0].strip():
                continue
            sim_id = int(row[0])
            archetype_id = row[1]
            archetype_map[sim_id] = archetype_id

    # Aggregate per archetype
    arch_total: dict[str, int] = defaultdict(int)
    arch_surviving: dict[str, int] = defaultdict(int)
    arch_weight_sum: dict[str, int] = defaultdict(int)

    total_weight_sum = 0
    weighted_payout_sum = 0

    for sim_id, archetype_id in archetype_map.items():
        weight, payout = weights.get(sim_id, (0, 0))
        arch_total[archetype_id] += 1
        if weight > 0:
            arch_surviving[archetype_id] += 1
        arch_weight_sum[archetype_id] += weight
        total_weight_sum += weight
        weighted_payout_sum += weight * payout

    # Build per-archetype survival records
    survivals: list[ArchetypeSurvival] = []
    flagged: list[str] = []

    for archetype_id in sorted(arch_total.keys()):
        total = arch_total[archetype_id]
        surviving = arch_surviving.get(archetype_id, 0)
        rate = surviving / total if total > 0 else 0.0
        mean_w = arch_weight_sum[archetype_id] / total if total > 0 else 0.0
        is_flagged = rate < survival_threshold

        if is_flagged:
            flagged.append(archetype_id)

        survivals.append(ArchetypeSurvival(
            archetype_id=archetype_id,
            total_books=total,
            surviving_books=surviving,
            survival_rate=rate,
            mean_weight=mean_w,
            flagged=is_flagged,
        ))

    total_books = sum(arch_total.values())
    total_surviving = sum(arch_surviving.values())
    total_rate = total_surviving / total_books if total_books > 0 else 0.0
    rtp = weighted_payout_sum / total_weight_sum if total_weight_sum > 0 else 0.0

    return AuditReport(
        archetypes=tuple(survivals),
        total_survival_rate=total_rate,
        flagged_archetypes=tuple(flagged),
        rtp_weighted=rtp,
    )


def format_audit_report(report: AuditReport) -> str:
    """Human-readable audit report text."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  POST-OPTIMIZATION AUDIT REPORT")
    lines.append("=" * 60)
    lines.append(f"  Total survival rate: {report.total_survival_rate:.1%}")
    lines.append(f"  Weighted RTP: {report.rtp_weighted:.2f}")
    lines.append(f"  Flagged archetypes: {len(report.flagged_archetypes)}")
    lines.append("")

    for surv in report.archetypes:
        flag = " [FLAGGED]" if surv.flagged else ""
        lines.append(
            f"  {surv.archetype_id:<40s} "
            f"{surv.surviving_books:>5d}/{surv.total_books:<5d} "
            f"({surv.survival_rate:>5.1%}) "
            f"avg_w={surv.mean_weight:>8.1f}"
            f"{flag}"
        )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def write_audit_report(report: AuditReport, path: Path) -> None:
    """Write formatted audit report to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_audit_report(report), encoding="utf-8")
