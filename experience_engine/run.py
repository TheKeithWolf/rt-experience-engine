"""Royal Tumble Experience Engine — CLI entry point.

Orchestrates: config load → component init → generation → events → output.
All thresholds from MasterConfig YAML — zero hardcoded values.

Usage:
    python -m games.royal_tumble.experience_engine.run
    python -m games.royal_tumble.experience_engine.run --config config/custom.yaml
    python -m games.royal_tumble.experience_engine.run --count 100 --family dead,t1 --seed 42
    python -m games.royal_tumble.experience_engine.run --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from .archetypes.bomb import register_bomb_archetypes
from .archetypes.chain import register_chain_archetypes
from .archetypes.dead import register_dead_archetypes
from .archetypes.lb import register_lb_archetypes
from .archetypes.registry import ArchetypeRegistry, ArchetypeSignature
from .archetypes.rocket import register_rocket_archetypes
from .archetypes.slb import register_slb_archetypes
from .archetypes.tier1 import register_cascade_t1_archetypes, register_static_t1_archetypes
from .archetypes.trigger import register_trigger_archetypes
from .archetypes.wincap import register_wincap_archetypes
from .archetypes.wild import register_wild_archetypes
from .config.loader import load_config
from .config.schema import MasterConfig, OutputConfig, PopulationConfig
from .diagnostics.engine import DiagnosticsEngine
from .output.book_record import BookRecord, book_record_from_instance
from .output.book_writer import BookWriter, BookWriterConfig
from .output.event_stream import EventStreamGenerator
from .output.lookup_writer import LookupTableWriter
from .output.summary_report import format_summary, generate_summary, write_summary
from .pipeline.cascade_generator import CascadeInstanceGenerator
from .pipeline.data_types import GeneratedInstance
from .pipeline.instance_generator import StaticInstanceGenerator
from .population.controller import PopulationController, PopulationResult
from .primitives.gravity import GravityDAG
from .primitives.paytable import Paytable
from .validation.validator import InstanceValidator

# Resolved relative to this file — config/default.yaml lives alongside this module
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

# All family registration functions — called in order to populate the registry
_FAMILY_REGISTRARS = (
    register_dead_archetypes,
    register_static_t1_archetypes,
    register_cascade_t1_archetypes,
    register_wild_archetypes,
    register_rocket_archetypes,
    register_bomb_archetypes,
    register_lb_archetypes,
    register_slb_archetypes,
    register_chain_archetypes,
    register_trigger_archetypes,
    register_wincap_archetypes,
)


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments — all optional, config YAML provides defaults."""
    parser = argparse.ArgumentParser(
        description="Royal Tumble Experience Engine — generate simulation books",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to master config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (overrides config.output.output_dir)",
    )
    parser.add_argument(
        "--archetype", type=str, default=None,
        help="Generate only these archetypes (comma-separated IDs)",
    )
    parser.add_argument(
        "--family", type=str, default=None,
        help="Generate only these families (comma-separated, e.g. dead,t1)",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Override population.total_budget",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and archetypes without generating",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override output.num_workers for parallel generation",
    )
    parser.add_argument(
        "--audit", type=Path, default=None,
        help="Path to optimized LUT for post-optimization audit",
    )
    return parser


def _build_full_registry(config: MasterConfig) -> ArchetypeRegistry:
    """Create and populate a registry with all 90 archetypes."""
    registry = ArchetypeRegistry(config)
    for registrar in _FAMILY_REGISTRARS:
        registrar(registry)
    return registry


def _filter_registry(
    config: MasterConfig,
    full_registry: ArchetypeRegistry,
    archetype_csv: str | None,
    family_csv: str | None,
) -> ArchetypeRegistry:
    """Build a filtered registry containing only the requested archetypes/families.

    When both --archetype and --family are given, the union of both selections is used.
    Returns a new registry with only the matching signatures registered.
    """
    if archetype_csv is None and family_csv is None:
        return full_registry

    selected_ids: set[str] = set()

    if archetype_csv:
        for arch_id in archetype_csv.split(","):
            arch_id = arch_id.strip()
            # Validate that the archetype exists in the full registry
            full_registry.get(arch_id)
            selected_ids.add(arch_id)

    if family_csv:
        for family_name in family_csv.split(","):
            family_name = family_name.strip()
            family_sigs = full_registry.get_family(family_name)
            if not family_sigs:
                raise KeyError(f"No archetypes registered for family '{family_name}'")
            for sig in family_sigs:
                selected_ids.add(sig.id)

    # Build a new registry with only the selected archetypes
    filtered = ArchetypeRegistry(config)
    for arch_id in selected_ids:
        sig = full_registry.get(arch_id)
        filtered.register(sig)
    return filtered


def _apply_overrides(config: MasterConfig, args: argparse.Namespace) -> MasterConfig:
    """Build new frozen MasterConfig with CLI overrides applied.

    Uses dataclasses.replace() to swap individual fields on the frozen hierarchy.
    """
    # Ensure output config exists — use defaults if YAML didn't include it
    output_cfg = config.output
    if output_cfg is None:
        output_cfg = OutputConfig(
            compression=False,
            output_dir="library",
            num_workers=1,
            mode_name="base",
            audit_survival_threshold=0.10,
        )

    # Apply --count → population.total_budget
    pop_cfg = config.population
    if args.count is not None:
        pop_cfg = replace(pop_cfg, total_budget=args.count)

    # Apply --output → output.output_dir
    if args.output is not None:
        output_cfg = replace(output_cfg, output_dir=str(args.output))

    # Apply --workers → output.num_workers
    if args.workers is not None:
        output_cfg = replace(output_cfg, num_workers=args.workers)

    return replace(config, population=pop_cfg, output=output_cfg)


def _build_pipeline(
    config: MasterConfig,
    registry: ArchetypeRegistry,
) -> tuple[StaticInstanceGenerator, CascadeInstanceGenerator, InstanceValidator]:
    """Construct the generation pipeline components — shared primitives built once."""
    gravity_dag = GravityDAG(config.board, config.gravity)

    static_gen = StaticInstanceGenerator(config, registry, gravity_dag)

    # Build StepReasoner components for cascade generation
    from .step_reasoner.evaluators import ChainEvaluator, PayoutEstimator, SpawnEvaluator
    from .step_reasoner.assessor import StepAssessor
    from .step_reasoner.selector import StrategySelector, DEFAULT_SELECTION_RULES
    from .step_reasoner.reasoner import StepReasoner
    from .step_reasoner.registry import build_default_registry
    from .pipeline.step_executor import StepExecutor
    from .pipeline.step_validator import StepValidator
    from .pipeline.simulator import StepTransitionSimulator
    from .spatial_solver.solver import CSPSpatialSolver

    spawn_eval = SpawnEvaluator(config.boosters)
    chain_eval = ChainEvaluator(config.boosters)
    payout_eval = PayoutEstimator(
        config.paytable, config.centipayout, config.win_levels,
        config.symbols, config.grid_multiplier,
    )
    csp_solver = CSPSpatialSolver(config)
    strategy_registry = build_default_registry(
        config, gravity_dag, csp_solver,
        spawn_eval, chain_eval, payout_eval,
    )
    assessor = StepAssessor(spawn_eval, chain_eval, payout_eval, config.reasoner)
    selector = StrategySelector(DEFAULT_SELECTION_RULES)
    reasoner = StepReasoner(strategy_registry, selector, assessor)
    executor = StepExecutor(config)
    step_validator = StepValidator(config)
    simulator = StepTransitionSimulator(gravity_dag, config)

    cascade_gen = CascadeInstanceGenerator(
        config, registry, gravity_dag,
        reasoner, executor, step_validator, simulator,
    )
    validator = InstanceValidator(config, registry)

    return static_gen, cascade_gen, validator


def _generate_books(
    config: MasterConfig,
    result: PopulationResult,
) -> tuple[BookRecord, ...]:
    """Convert all generated instances to BookRecords with event streams."""
    paytable = Paytable(config.paytable, config.centipayout, config.win_levels)
    event_gen = EventStreamGenerator(config, paytable)

    books: list[BookRecord] = []
    for instance in result.instances:
        events = event_gen.generate(instance)
        book = book_record_from_instance(instance, events)
        books.append(book)
    return tuple(books)


def _print_dry_run(config: MasterConfig, registry: ArchetypeRegistry) -> None:
    """Print config summary and registered archetypes for validation."""
    output_cfg = config.output
    print("=" * 60)
    print("  EXPERIENCE ENGINE — DRY RUN")
    print("=" * 60)
    print(f"  Board: {config.board.num_reels}×{config.board.num_rows}")
    print(f"  Min cluster size: {config.board.min_cluster_size}")
    print(f"  Population budget: {config.population.total_budget}")
    if output_cfg:
        print(f"  Output dir: {output_cfg.output_dir}")
        print(f"  Compression: {output_cfg.compression}")
        print(f"  Workers: {output_cfg.num_workers}")
        print(f"  Mode: {output_cfg.mode_name}")
    print(f"  Wincap: {config.wincap.max_payout}x")
    print()

    families = registry.registered_families()
    total = 0
    for family in sorted(families):
        sigs = registry.get_family(family)
        total += len(sigs)
        print(f"  Family: {family} ({len(sigs)} archetypes)")
        for sig in sigs:
            print(f"    {sig.id}")
    print()
    print(f"  Total archetypes: {total}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """Main orchestration — the entire pipeline in sequence.

    Returns 0 on success, 1 on error.
    """
    args = build_parser().parse_args(argv)

    # 1. Load and validate config
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(config_path)

    # 2. Apply CLI overrides
    config = _apply_overrides(config, args)
    output_cfg = config.output
    assert output_cfg is not None  # guaranteed by _apply_overrides

    # 3. Build full registry, then optionally filter
    full_registry = _build_full_registry(config)
    registry = _filter_registry(config, full_registry, args.archetype, args.family)

    # 4. Dry run — print summary and exit
    if args.dry_run:
        _print_dry_run(config, registry)
        return 0

    output_dir = Path(output_cfg.output_dir)
    num_workers = output_cfg.num_workers

    print(f"Generating {config.population.total_budget} books "
          f"({len(registry.all_ids())} archetypes, {num_workers} worker(s))...")
    start = time.monotonic()

    # 5. Generate population (serial or parallel)
    if num_workers > 1:
        from .parallel import run_parallel_generation
        result = run_parallel_generation(config, registry, num_workers, args.seed)
    else:
        static_gen, cascade_gen, validator = _build_pipeline(config, registry)
        controller = PopulationController(
            config, registry, static_gen, cascade_gen, validator,
        )
        result = controller.run(args.seed)

    elapsed_gen = time.monotonic() - start
    print(f"Generated {result.total_generated} books "
          f"({result.total_failed} failed) in {elapsed_gen:.1f}s")

    # 6. Convert instances to book records with event streams
    books = _generate_books(config, result)

    # 7. Write book JSONL
    writer_config = BookWriterConfig(
        output_dir=output_dir,
        mode_name=output_cfg.mode_name,
        compression=output_cfg.compression,
    )
    book_writer = BookWriter(writer_config)
    books_path = book_writer.write_books(books)
    print(f"Wrote books: {books_path}")

    # 8. Write lookup tables
    lut_writer = LookupTableWriter(output_dir, output_cfg.mode_name)
    lut_path = lut_writer.write_initial_table(books)
    print(f"Wrote lookup table: {lut_path}")

    # 9. Write archetype lookup table (for post-opt audit)
    archetype_ids = tuple(inst.archetype_id for inst in result.instances)
    arch_lut_path = lut_writer.write_archetype_table(books, archetype_ids)
    print(f"Wrote archetype lookup table: {arch_lut_path}")

    # 10. Diagnostics
    diagnostics = DiagnosticsEngine(config)
    diag_report = diagnostics.analyze(result.metrics, result.failure_log)

    # 11. Summary report
    summary = generate_summary(result, config, registry, config_path=config_path, seed=args.seed)
    summary_path = output_dir / "summary.txt"
    write_summary(summary, summary_path)
    print(f"Wrote summary: {summary_path}")

    # 12. Diagnostics report
    diag_path = output_dir / "diagnostics.txt"
    from .diagnostics.report import format_diagnostics
    diag_path.write_text(format_diagnostics(diag_report), encoding="utf-8")
    print(f"Wrote diagnostics: {diag_path}")

    # 13. Optional post-optimization audit
    if args.audit is not None:
        from .output.audit import format_audit_report, run_audit, write_audit_report
        audit_report = run_audit(
            optimized_lut_path=args.audit,
            archetype_lut_path=arch_lut_path,
            survival_threshold=output_cfg.audit_survival_threshold,
        )
        audit_path = output_dir / "audit_report.txt"
        write_audit_report(audit_report, audit_path)
        print(f"Wrote audit report: {audit_path}")
        if audit_report.flagged_archetypes:
            print(f"WARNING: {len(audit_report.flagged_archetypes)} archetypes flagged "
                  f"(survival < {output_cfg.audit_survival_threshold:.0%})")

    total_elapsed = time.monotonic() - start
    print(f"Done in {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
