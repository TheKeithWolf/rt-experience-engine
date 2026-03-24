"""Summary report — run manifest mapping every book to its archetype and family.

Produces text output matching spec section 15.3 format. Allows tracing
any book back to its archetype and family via sim_id ranges.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..archetypes.registry import ArchetypeRegistry, FAMILY_CRITERIA
from ..config.schema import MasterConfig
from ..population.controller import PopulationResult


@dataclass(frozen=True, slots=True)
class ArchetypeSummary:
    """Summary for one archetype within a family."""

    archetype_id: str
    family: str
    criteria: str
    book_count: int
    sim_id_start: int
    sim_id_end: int
    failure_count: int
    payout_range_actual: tuple[float, float]


@dataclass(frozen=True, slots=True)
class FamilySummary:
    """Summary for one archetype family."""

    family: str
    criteria: str
    total_books: int
    archetypes: tuple[ArchetypeSummary, ...]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Complete run manifest."""

    run_timestamp: str
    config_hash: str
    seed: int
    total_budget: int
    total_generated: int
    total_failed: int
    families: tuple[FamilySummary, ...]
    # (archetype_id, attempt_count, failure_reason) for each failed slot
    failure_log: tuple[tuple[str, int, str], ...] = ()


def generate_summary(
    population: PopulationResult,
    config: MasterConfig,
    registry: ArchetypeRegistry,
    config_path: Path | None = None,
    seed: int = 42,
) -> RunSummary:
    """Build RunSummary from population result.

    Groups instances by family and archetype, computes sim_id ranges,
    failure counts, and actual payout ranges.
    """
    # Compute config hash from the config object's repr
    config_hash = hashlib.md5(repr(config).encode()).hexdigest()[:8]

    # Group instances by archetype
    archetype_instances: dict[str, list] = {}
    for inst in population.instances:
        archetype_instances.setdefault(inst.archetype_id, []).append(inst)

    # Count failures per archetype
    failure_counts: dict[str, int] = {}
    for arch_id, _, _ in population.failure_log:
        failure_counts[arch_id] = failure_counts.get(arch_id, 0) + 1

    # Build family summaries
    family_summaries: list[FamilySummary] = []
    for family in sorted(registry.registered_families()):
        family_archetypes = registry.get_family(family)
        if not family_archetypes:
            continue

        arch_summaries: list[ArchetypeSummary] = []
        family_total = 0

        for sig in family_archetypes:
            instances = archetype_instances.get(sig.id, [])
            count = len(instances)
            family_total += count
            failures = failure_counts.get(sig.id, 0)

            if instances:
                sim_ids = [inst.sim_id for inst in instances]
                sim_id_start = min(sim_ids)
                sim_id_end = max(sim_ids)
                payouts = [inst.payout for inst in instances]
                payout_range = (min(payouts), max(payouts))
            else:
                sim_id_start = 0
                sim_id_end = 0
                payout_range = (0.0, 0.0)

            arch_summaries.append(ArchetypeSummary(
                archetype_id=sig.id,
                family=family,
                criteria=sig.criteria,
                book_count=count,
                sim_id_start=sim_id_start,
                sim_id_end=sim_id_end,
                failure_count=failures,
                payout_range_actual=payout_range,
            ))

        criteria = FAMILY_CRITERIA.get(family, "unknown")
        family_summaries.append(FamilySummary(
            family=family,
            criteria=criteria,
            total_books=family_total,
            archetypes=tuple(arch_summaries),
        ))

    return RunSummary(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        seed=seed,
        total_budget=config.population.total_budget,
        total_generated=population.total_generated,
        total_failed=population.total_failed,
        families=tuple(family_summaries),
        failure_log=population.failure_log,
    )


def format_summary(summary: RunSummary) -> str:
    """Format RunSummary as text matching spec section 15.3."""
    lines: list[str] = []

    # Header
    sep = "=" * 54
    lines.append(sep)
    lines.append("  EXPERIENCE ENGINE RUN SUMMARY")
    lines.append(f"  Timestamp: {summary.run_timestamp}")
    lines.append(f"  Config Hash: {summary.config_hash}")
    lines.append(f"  Seed: {summary.seed}")
    lines.append(
        f"  Budget: {summary.total_budget} | "
        f"Generated: {summary.total_generated} | "
        f"Failed: {summary.total_failed}"
    )
    lines.append(sep)
    lines.append("")

    # Per-family sections
    for fam in summary.families:
        lines.append(
            f"Family: {fam.family} ({fam.total_books} books) "
            f'[criteria: "{fam.criteria}"]'
        )
        for arch in fam.archetypes:
            # Pad archetype name to 30 chars for alignment
            name_padded = arch.archetype_id.ljust(30)
            sim_range = f"sim_ids: {arch.sim_id_start}-{arch.sim_id_end}"
            fail_str = f"failures: {arch.failure_count}"
            lines.append(
                f"  {name_padded}{arch.book_count:>5} books  "
                f"{sim_range:<25}{fail_str}"
            )
        lines.append("")

    # Failure details — show exact reasons for each failed instance
    if summary.failure_log:
        lines.append("Failure Details:")
        for arch_id, attempts, reason in summary.failure_log:
            lines.append(f"  {arch_id} (attempts={attempts}): {reason}")
        lines.append("")

    return "\n".join(lines)


def write_summary(summary: RunSummary, path: Path) -> None:
    """Write formatted summary to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_summary(summary), encoding="utf-8")
