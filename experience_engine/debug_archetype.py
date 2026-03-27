"""Diagnostic: step-through archetype generation attempts.

Shows board state, strategy decisions, and failure points at each cascade step
so you can visually identify where the generation pipeline breaks down.

Usage:
    python -m games.royal_tumble.experience_engine.debug_archetype --archetype t1_multi_cascade
    python -m games.royal_tumble.experience_engine.debug_archetype --archetype t1_low_cascade --seed 42 --attempts 5
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .archetypes.registry import ArchetypeRegistry, ArchetypeSignature
from .board_filler.wfc_solver import FillFailed
from .boosters.tracker import BoosterTracker
from .config.loader import load_config
from .pipeline.cascade_generator import CascadeInstanceGenerator
from .pipeline.data_types import GeneratedInstance
from .pipeline.step_validator import StepValidationFailed, StepValidator
from .pipeline.step_executor import StepExecutor
from .pipeline.simulator import StepTransitionSimulator
from .primitives.board import Board, Position
from .primitives.cluster_detection import detect_clusters
from .primitives.grid_multipliers import GridMultiplierGrid
from .primitives.symbols import Symbol, SymbolTier, symbol_from_name, tier_of
from .step_reasoner.context import BoardContext
from .step_reasoner.progress import ProgressTracker
from .step_reasoner.results import StepResult
from .validation.validator import InstanceValidator
from .variance.hints import VarianceHints
from .variance.accumulators import PopulationAccumulators
from .narrative.transitions import build_transition_rules, try_advance_phase
from .variance.bias_computation import compute_hints

# Reuse pipeline construction from run.py
from .run import _build_full_registry, _build_pipeline, DEFAULT_CONFIG_PATH


def _setup_utf8() -> None:
    """Force UTF-8 output on Windows to support box-drawing characters."""
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace",
        )
        # Enable ANSI escape processing on Windows 10+
        os.system("")


class TeeWriter:
    """Duplicates writes to two streams — stdout + capture buffer.

    Allows all existing print() calls to write to both the console and a
    StringIO buffer without any changes to the diagnostic functions.
    """

    __slots__ = ("_primary", "_secondary")

    def __init__(self, primary: io.TextIOBase, secondary: io.StringIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, text: str) -> int:
        self._primary.write(text)
        return self._secondary.write(text)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()


def _write_debug_report(
    archetype_id: str,
    seed: int,
    captured_output: str,
) -> Path:
    """Write captured diagnostic output to a markdown file.

    Output path: library/debug/debug_{archetype_id}.md
    The diagnostic output is wrapped in a fenced code block to preserve
    ASCII board rendering and fixed-width alignment.
    """
    debug_dir = Path(__file__).parent / "library" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    output_path = debug_dir / f"debug_{archetype_id}.md"
    timestamp = datetime.now(timezone.utc).isoformat()

    content = (
        f"# Debug: {archetype_id}\n\n"
        f"- **Seed:** {seed}\n"
        f"- **Timestamp:** {timestamp}\n\n"
        f"```\n{captured_output}```\n"
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Board rendering — pure ASCII to avoid encoding issues
# ---------------------------------------------------------------------------

def render_board(
    board: Board,
    cluster_positions: frozenset[Position] | None = None,
    strategic_positions: frozenset[Position] | None = None,
    label: str = "",
) -> str:
    """Render a 7x7 board as an ASCII grid with highlighted positions.

    cluster_positions shown with *asterisks*, strategic with [brackets],
    empty cells as ' .. '. Standard symbols shown as their name (L1, H2, etc).
    """
    cluster_positions = cluster_positions or frozenset()
    strategic_positions = strategic_positions or frozenset()
    num_reels = board.num_reels
    num_rows = board.num_rows

    lines: list[str] = []
    if label:
        lines.append(f"  {label}")

    # Header row with reel indices
    header = "     " + "  ".join(f"R{r}" for r in range(num_reels))
    lines.append(header)

    # Top border
    lines.append("  +" + "----+" * num_reels)

    for row in range(num_rows):
        cells: list[str] = []
        for reel in range(num_reels):
            pos = Position(reel, row)
            sym = board.get(pos)

            if sym is None:
                cell = " .. "
            elif pos in cluster_positions:
                cell = f"*{sym.name:>2}*"
            elif pos in strategic_positions:
                cell = f"[{sym.name:>2}]"
            else:
                cell = f" {sym.name:>2} "

            cells.append(cell)

        row_str = f"{row} |" + "|".join(cells) + "|"
        lines.append(row_str)

        # Row separator or bottom border
        lines.append("  +" + "----+" * num_reels)

    return "\n".join(lines)


def render_empty_cells(board: Board) -> str:
    """Show which cells are empty on the board as a compact list."""
    empty = board.empty_positions()
    if not empty:
        return "  Empty cells: none"
    grouped: dict[int, list[int]] = {}
    for pos in empty:
        grouped.setdefault(pos.reel, []).append(pos.row)
    parts = []
    for reel in sorted(grouped):
        rows = ",".join(str(r) for r in sorted(grouped[reel]))
        parts.append(f"R{reel}:[{rows}]")
    return f"  Empty cells ({len(empty)}): " + " ".join(parts)


def render_booster_tracker_state(
    tracker: BoosterTracker,
    fresh_positions: frozenset[Position] | None = None,
) -> str:
    """Render all tracked boosters with type, position, orientation, and state.

    fresh_positions labels boosters spawned this step — they cannot arm until a
    future step's cluster is adjacent (Issue 1 exclusion logic).
    """
    all_boosters = tracker.all_boosters()
    if not all_boosters:
        return "  Booster tracker: empty"
    fresh = fresh_positions or frozenset()
    lines = [f"  Booster tracker ({len(all_boosters)} tracked):"]
    for b in all_boosters:
        orient = f" orientation={b.orientation}" if b.orientation else ""
        tag = " [FRESH — will not arm this step]" if b.position in fresh else ""
        lines.append(
            f"    {b.booster_type.name} at ({b.position.reel},{b.position.row})"
            f"{orient} state={b.state.name}{tag}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diagnostic generator -- replicates _attempt_generation with verbose logging
# ---------------------------------------------------------------------------

def diagnostic_attempt(
    sig: ArchetypeSignature,
    sim_id: int,
    hints: VarianceHints,
    rng: random.Random,
    gen: CascadeInstanceGenerator,
    registry: ArchetypeRegistry,
    attempt_num: int,
) -> tuple[bool, str, int]:
    """Run one generation attempt with step-by-step diagnostic output.

    Returns (success, failure_reason, step_failed_at).
    """
    config = gen._config
    board = Board.empty(config.board)
    grid_mults = GridMultiplierGrid(config.grid_multiplier, config.board)
    booster_tracker = BoosterTracker(config.board)
    progress = ProgressTracker(sig, config.centipayout.multiplier)
    transition_rules = build_transition_rules(config.board, config.symbols)

    # Build booster phase executor if the archetype fires boosters
    phase_executor = (
        gen._make_phase_executor(booster_tracker)
        if sig.required_booster_fires else None
    )

    step_results: list[StepResult] = []
    # Track board snapshots per step for building CascadeStepRecords
    step_boards: list[tuple[Board, Board]] = []
    max_steps = sig.required_cascade_depth.max_val + 1

    print(f"\n{'='*70}")
    print(f"  ATTEMPT {attempt_num}")
    print(f"{'='*70}")
    print(f"  Archetype: {sig.id}")
    print(f"  Required cascade depth: {sig.required_cascade_depth.min_val}-{sig.required_cascade_depth.max_val}")
    print(f"  Required cluster sizes: {[(r.min_val, r.max_val) for r in sig.required_cluster_sizes]}")
    print(f"  Required cluster symbols: {sig.required_cluster_symbols}")
    print(f"  Required booster fires: {sig.required_booster_fires}")
    if sig.cascade_steps:
        print(f"  Cascade steps: {len(sig.cascade_steps)} step constraints defined")
    print(f"  Payout range: {sig.payout_range.min_val}-{sig.payout_range.max_val}x")
    print(f"  Max steps: {max_steps}")

    filled_board = board  # Track last filled board for failure reporting

    for step_idx in range(max_steps):
        print(f"\n  {'-'*60}")
        print(f"  Step {step_idx}")
        print(f"  {'-'*60}")

        # Snapshot board before this step for CascadeStepRecord construction
        board_before = board.copy()

        # Show board state entering this step
        if step_idx == 0:
            print("  Board: empty (7x7)")
        else:
            print(render_empty_cells(board))
            empty_count = len(board.empty_positions())
            print(f"  Available cells for cluster: {empty_count}")
            # Show the board with empty cells visible
            print(render_board(board, label=f"Board entering step {step_idx}:"))

        # Board context -- what the reasoner sees
        context = BoardContext.from_board(
            board, grid_mults,
            progress.dormant_boosters,
            progress.active_wilds,
            config.board,
        )

        # Assess -- show what the assessor derives
        from .step_reasoner.assessor import StepAssessor
        from .step_reasoner.evaluators import ChainEvaluator, PayoutEstimator, SpawnEvaluator
        spawn_eval = SpawnEvaluator(config.boosters)
        chain_eval = ChainEvaluator(config.boosters)
        payout_eval = PayoutEstimator(
            config.paytable, config.centipayout, config.win_levels,
            config.symbols, config.grid_multiplier,
        )
        assessor = StepAssessor(spawn_eval, chain_eval, payout_eval, config.reasoner)
        assessment = assessor.assess(context, progress, sig)

        print(f"  Assessment:")
        print(f"    steps_remaining: {assessment.steps_remaining.min_val}-{assessment.steps_remaining.max_val}")
        print(f"    must_terminate_now: {assessment.must_terminate_now}")
        print(f"    is_first_step: {assessment.is_first_step}")
        print(f"    required_tier_this_step: {assessment.required_tier_this_step}")
        print(f"    payout_remaining: {assessment.payout_remaining.min_val:.2f}-{assessment.payout_remaining.max_val:.2f}x")
        print(f"    payout_running_low: {assessment.payout_running_low}")
        print(f"    payout_running_high: {assessment.payout_running_high}")

        # Reason -- produce intent
        try:
            intent = gen._reasoner.reason(context, progress, sig, hints)
        except (ValueError, FillFailed) as exc:
            print(f"\n  >> FAIL at reasoning: {exc}")
            return False, f"Step {step_idx} reasoning: {exc}", step_idx

        print(f"\n  Intent:")
        print(f"    step_type: {intent.step_type.value}")
        print(f"    is_terminal: {intent.is_terminal}")
        print(f"    constrained_cells: {len(intent.constrained_cells)}")
        print(f"    strategic_cells: {len(intent.strategic_cells)}")
        print(f"    expected_cluster_count: {intent.expected_cluster_count.min_val}-{intent.expected_cluster_count.max_val}")
        if intent.expected_cluster_sizes:
            sizes_str = ", ".join(f"{r.min_val}-{r.max_val}" for r in intent.expected_cluster_sizes)
            print(f"    expected_cluster_sizes: [{sizes_str}]")
        print(f"    expected_cluster_tier: {intent.expected_cluster_tier}")
        print(f"    expected_spawns: {intent.expected_spawns}")
        print(f"    wfc_propagators: {len(intent.wfc_propagators)}")
        for p in intent.wfc_propagators:
            print(f"      - {type(p).__name__}")

        # Show which symbols are being placed as constrained cells
        if intent.constrained_cells:
            sym_counts: Counter[str] = Counter()
            for pos, sym in intent.constrained_cells.items():
                sym_counts[sym.name] += 1
            print(f"    constrained symbols: {dict(sym_counts)}")
            for sym_name in sorted(sym_counts):
                positions = [
                    f"({p.reel},{p.row})"
                    for p, s in intent.constrained_cells.items()
                    if s.name == sym_name
                ]
                print(f"      {sym_name}: {' '.join(positions)}")

        if intent.strategic_cells:
            strat_counts: Counter[str] = Counter()
            for pos, sym in intent.strategic_cells.items():
                strat_counts[sym.name] += 1
            print(f"    strategic symbols: {dict(strat_counts)}")
            for sym_name in sorted(strat_counts):
                positions = [
                    f"({p.reel},{p.row})"
                    for p, s in intent.strategic_cells.items()
                    if s.name == sym_name
                ]
                print(f"      {sym_name}: {' '.join(positions)}")

        # Execute -- WFC fill
        try:
            filled = gen._executor.execute(intent, board, rng)
            filled_board = filled
        except FillFailed as exc:
            print(f"\n  >> FAIL at WFC fill: {exc}")
            # Show the partial board state before WFC failed
            partial = board.copy()
            for pos, sym in intent.constrained_cells.items():
                partial.set(pos, sym)
            for pos, sym in intent.strategic_cells.items():
                partial.set(pos, sym)
            print(render_board(
                partial,
                cluster_positions=frozenset(intent.constrained_cells.keys()),
                strategic_positions=frozenset(intent.strategic_cells.keys()),
                label="Board before WFC (pinned cells shown):",
            ))
            return False, f"Step {step_idx} WFC: {exc}", step_idx

        # Show the filled board
        cluster_pos = frozenset(intent.constrained_cells.keys())
        strat_pos = frozenset(intent.strategic_cells.keys())
        print(render_board(
            filled, cluster_positions=cluster_pos, strategic_positions=strat_pos,
            label=f"Filled board (step {step_idx}):",
        ))

        # Detect clusters on filled board for diagnostics
        detected = detect_clusters(filled, config)
        if detected:
            print(f"\n  Detected clusters: {len(detected)}")
            for i, cl in enumerate(detected):
                try:
                    t = tier_of(cl.symbol, config.symbols)
                except (ValueError, KeyError):
                    t = "?"
                pos_str = " ".join(
                    f"({p.reel},{p.row})"
                    for p in sorted(cl.positions, key=lambda p: (p.reel, p.row))
                )
                wild_str = ""
                if cl.wild_positions:
                    wild_str = f" + {len(cl.wild_positions)} wilds"
                print(f"    [{i}] {cl.symbol.name}(size={cl.size}{wild_str}) tier={t}  pos: {pos_str}")
        else:
            print(f"\n  Detected clusters: 0 (dead board)")

        # Validate step
        try:
            step_result = gen._validator.validate_step(
                filled, intent, progress, grid_mults,
            )
        except StepValidationFailed as exc:
            print(f"\n  >> FAIL at step validation: {exc}")
            return False, f"Step {step_idx} validation: {exc}", step_idx

        # Show validation result
        payout_this_step = step_result.step_payout / config.centipayout.multiplier
        print(f"\n  PASS Step {step_idx} validated:")
        print(f"    clusters: {len(step_result.clusters)}")
        for cr in step_result.clusters:
            print(f"      {cr.symbol.name}(size={cr.size}) payout={cr.payout} centipayout")
        print(f"    step_payout: {step_result.step_payout} centipayout ({payout_this_step:.2f}x)")
        print(f"    symbol_tier: {step_result.symbol_tier}")
        print(f"    spawns: {[s.booster_type for s in step_result.spawns]}")

        step_results.append(step_result)
        step_boards.append((board_before, filled))
        progress.update(step_result)

        # Phase advancement — pre-transition context from filled board
        fill_context = BoardContext.from_board(
            filled, grid_mults,
            progress.dormant_boosters, progress.active_wilds,
            config.board,
        )
        advanced = try_advance_phase(
            progress, step_result, transition_rules, fill_context,
        )
        if advanced:
            phase = progress.current_phase()
            phase_name = phase.id if phase else "(past end)"
            print(f"\n  >> Phase advanced → {phase_name}")

        # Show cumulative progress
        cum_payout = progress.cumulative_payout / config.centipayout.multiplier
        spent = cum_payout
        payout_max = sig.payout_range.max_val
        payout_min = sig.payout_range.min_val
        steps_min = sig.required_cascade_depth.min_val
        steps_max = sig.required_cascade_depth.max_val
        steps_done = progress.steps_completed
        print(f"\n  Progress after step {step_idx}:")
        print(f"    steps_completed: {steps_done} / {steps_min}-{steps_max}")
        print(f"    cumulative_payout: {progress.cumulative_payout} centipayout ({cum_payout:.2f}x / {payout_min}-{payout_max}x)")
        print(f"    remaining_steps: {max(0, steps_min - steps_done)}-{steps_max - steps_done}")
        print(f"    remaining_payout: {max(0, payout_min - spent):.2f}-{payout_max - spent:.2f}x")
        try:
            print(f"    is_satisfied: {progress.is_satisfied()}")
        except ValueError:
            # Ranges can go negative when progress exceeds signature bounds
            print(f"    is_satisfied: (exceeded bounds)")

        # Transition to next step (unless terminal)
        if not intent.is_terminal:
            try:
                # Route to booster-aware transition when the archetype fires boosters
                if phase_executor is not None:
                    transition_result = gen._simulator.transition_and_arm(
                        filled, step_result, booster_tracker, grid_mults,
                        phase_executor,
                    )
                else:
                    transition_result = gen._simulator.transition(
                        filled, step_result, booster_tracker, grid_mults,
                    )
            except Exception as exc:
                print(f"\n  >> FAIL at transition: {exc}")
                return False, f"Step {step_idx} transition: {exc}", step_idx

            board = transition_result.board
            print(f"\n  Transition: exploded clusters, gravity settled")

            # Grid multiplier state — makes payout contribution visible per step
            nonzero = grid_mults.nonzero_positions()
            if nonzero:
                print(f"\n  Grid multipliers ({len(nonzero)} active):")
                for pos, val in nonzero:
                    print(f"    ({pos.reel},{pos.row}): {val}")

            # Show booster spawns from this transition
            if transition_result.spawns:
                print(f"\n  Spawns from transition:")
                for s in transition_result.spawns:
                    print(f"    {s.booster_type} at ({s.position.reel},{s.position.row})")

            # Label freshly-spawned non-wild boosters — makes exclusion logic visible
            fresh_positions = frozenset(
                s.position for s in transition_result.spawns
                if s.booster_type != Symbol.W.name
            )

            # Show booster tracker state after spawn + gravity
            print(render_booster_tracker_state(booster_tracker, fresh_positions))

            print(render_empty_cells(board))
        else:
            print(f"\n  Terminal step -- cascade ends here")
            break

    # Post-terminal booster phase — fire armed boosters after cascade exhaustion
    if phase_executor is not None and booster_tracker.get_armed():
        print(f"\n  {'='*60}")
        print(f"  POST-TERMINAL BOOSTER PHASE")
        print(f"  {'='*60}")

        booster_cycle = 0
        while booster_tracker.get_armed():
            booster_cycle += 1
            armed = booster_tracker.get_armed()
            print(f"\n  Booster fire cycle {booster_cycle} ({len(armed)} armed boosters):")

            fire_result = gen._simulator.execute_terminal_booster_phase(
                board, booster_tracker, grid_mults, phase_executor,
            )
            board = fire_result.board

            for fr in fire_result.booster_fire_records:
                orient = f" orientation={fr.orientation}" if fr.orientation else ""
                print(f"    FIRE: {fr.booster_type} at ({fr.position_reel},{fr.position_row}){orient}")
                print(f"      Cleared: {fr.affected_count} cells, chains: {fr.chain_target_count}")
                if fr.target_symbols:
                    print(f"      Targets: {fr.target_symbols}")

                # Track fires in progress for validation
                progress.boosters_fired[fr.booster_type] = (
                    progress.boosters_fired.get(fr.booster_type, 0) + 1
                )

            print(render_board(board, label="Board after booster fire + gravity:"))

            # Refill empty cells
            standard_names = tuple(config.symbols.standard)
            for reel in range(config.board.num_reels):
                for row in range(config.board.num_rows):
                    pos = Position(reel, row)
                    if board.get(pos) is None:
                        board.set(pos, symbol_from_name(rng.choice(standard_names)))

            # Detect clusters on refilled board
            clusters = detect_clusters(board, config)
            if clusters:
                print(f"\n  Refill produced {len(clusters)} cluster(s) — would re-cascade")
            else:
                print(f"\n  Refill produced no clusters — board is truly terminal")
            break  # Debug runner shows one cycle; real pipeline loops

    # Instance-level validation — uses the same InstanceValidator as the real
    # pipeline (run.py → PopulationController) so SUCCESS here guarantees the
    # instance would also pass in batch mode.
    print(f"\n  {'-'*60}")
    print(f"  INSTANCE VALIDATION (full — matches batch pipeline)")
    print(f"  {'-'*60}")

    # Reuse CascadeInstanceGenerator._build_step_record() (DRY — cascade_generator.py:237)
    cascade_step_records = tuple(
        gen._build_step_record(sr, bb, ba, grid_mults)
        for sr, (bb, ba) in zip(step_results, step_boards)
    )

    total_centipayout = progress.cumulative_payout
    total_payout = total_centipayout / config.centipayout.multiplier

    # Build the same GeneratedInstance the real pipeline constructs
    from .pipeline.step_validator import _EMPTY_SPATIAL_STEP
    instance = GeneratedInstance(
        sim_id=0,
        archetype_id=sig.id,
        family=sig.family,
        criteria=sig.criteria,
        board=filled_board,
        spatial_step=_EMPTY_SPATIAL_STEP,
        payout=total_payout,
        centipayout=total_centipayout,
        win_level=0,
        cascade_steps=cascade_step_records,
    )

    # Full validation — identical checks to InstanceValidator.validate() in the batch pipeline
    validator = InstanceValidator(config, registry)
    metrics = validator.validate(instance)

    # Show key metrics so the developer can see exactly what the validator detected
    print(f"  cluster_count: {metrics.cluster_count} "
          f"(required: {sig.required_cluster_count.min_val}-{sig.required_cluster_count.max_val})")
    print(f"  cluster_sizes: {metrics.cluster_sizes}")
    print(f"  payout: {metrics.payout:.4f}x "
          f"(range: {sig.payout_range.min_val}-{sig.payout_range.max_val})")
    print(f"  cascade_depth: {metrics.cascade_depth} "
          f"(range: {sig.required_cascade_depth.min_val}-{sig.required_cascade_depth.max_val})")
    print(f"  scatter_count: {metrics.scatter_count}")
    print(f"  near_miss_count: {metrics.near_miss_count}")

    if metrics.is_valid:
        print(f"\n  ** SUCCESS -- valid {sig.id} instance! **")
        return True, "", -1

    for err in metrics.validation_errors:
        print(f"  >> FAIL: {err}")
    return False, "; ".join(metrics.validation_errors), -1


# ---------------------------------------------------------------------------
# Failure statistics
# ---------------------------------------------------------------------------

def print_failure_summary(
    failures: list[tuple[str, int]],
    total_attempts: int,
    successes: int,
) -> None:
    """Print aggregated failure statistics."""
    print(f"\n{'='*70}")
    print(f"  FAILURE SUMMARY")
    print(f"{'='*70}")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Successes: {successes}")
    print(f"  Failures: {len(failures)}")

    if not failures:
        return

    # Group by step
    step_failures: Counter[int] = Counter()
    reason_by_step: dict[int, Counter[str]] = {}
    for reason, step in failures:
        step_failures[step] += 1
        reason_by_step.setdefault(step, Counter())[reason] += 1

    print()
    for step in sorted(step_failures):
        label = "Instance validation" if step == -1 else f"Step {step}"
        count = step_failures[step]
        pct = count / len(failures) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label}: {count} ({pct:.0f}%) {bar}")
        for reason, cnt in reason_by_step[step].most_common(5):
            short = reason[:80] + "..." if len(reason) > 80 else reason
            print(f"    - {short}: {cnt}")

    # Most common failure reasons overall
    print(f"\n  Top failure reasons:")
    all_reasons: Counter[str] = Counter()
    for reason, _ in failures:
        all_reasons[reason] += 1
    for reason, cnt in all_reasons.most_common(10):
        short = reason[:90] + "..." if len(reason) > 90 else reason
        print(f"    {cnt:3d}x {short}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostic(archetype_id: str, seed: int = 42, max_attempts: int = 5) -> int:
    """Generate an archetype with diagnostic output.

    All diagnostic output is printed to stdout AND captured to a StringIO
    buffer via TeeWriter. On completion, the captured output is written to
    library/debug/debug_{archetype_id}.md for persistent review.
    """
    config = load_config(DEFAULT_CONFIG_PATH)

    # Build registry and pipeline
    registry = _build_full_registry(config)
    _, cascade_gen, _ = _build_pipeline(config, registry)

    # Get the archetype signature, listing available IDs on failure
    try:
        sig = registry.get(archetype_id)
    except KeyError:
        available = sorted(registry.all_ids())
        print(f"  Unknown archetype: '{archetype_id}'")
        print(f"  Available archetypes ({len(available)}):")
        for aid in available:
            print(f"    - {aid}")
        return 1

    # Capture all diagnostic output to both console and buffer
    capture_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, capture_buffer)

    try:
        result = _run_diagnostic_inner(
            sig, seed, max_attempts, config, registry, cascade_gen,
        )
    finally:
        sys.stdout = original_stdout

    # Write markdown report from captured output
    report_path = _write_debug_report(archetype_id, seed, capture_buffer.getvalue())
    print(f"\n  Report written to: {report_path}")

    return result


def _run_diagnostic_inner(
    sig: ArchetypeSignature,
    seed: int,
    max_attempts: int,
    config,
    registry: ArchetypeRegistry,
    cascade_gen: CascadeInstanceGenerator,
) -> int:
    """Core diagnostic loop — separated so TeeWriter can capture all output."""
    print(f"{'='*70}")
    print(f"  {sig.id} DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"  Seed: {seed}")
    print(f"  Max attempts: {max_attempts}")
    print(f"  Archetype: {sig.id}")
    print(f"  Family: {sig.family}, Criteria: {sig.criteria}")
    print(f"  Cascade depth: {sig.required_cascade_depth.min_val}-{sig.required_cascade_depth.max_val}")
    print(f"  Cluster count: {sig.required_cluster_count.min_val}-{sig.required_cluster_count.max_val}")
    print(f"  Cluster sizes: {[(r.min_val, r.max_val) for r in sig.required_cluster_sizes]}")
    print(f"  Cluster symbols: {sig.required_cluster_symbols}")
    print(f"  Payout range: {sig.payout_range.min_val}-{sig.payout_range.max_val}x")
    print(f"  symbol_tier_per_step: {sig.symbol_tier_per_step}")
    print(f"  Config: max_retries={config.solvers.max_retries_per_instance}, "
          f"wfc_backtracks={config.solvers.wfc_max_backtracks}")

    # Create fresh accumulators for default hints (no prior population bias)
    accumulators = PopulationAccumulators.create(config)
    hints = compute_hints(accumulators, config)

    # Show variance hints
    print(f"\n  Variance hints (fresh -- no prior population):")
    low_weights = {
        s.name: f"{w:.3f}"
        for s, w in hints.symbol_weights.items()
        if s.name.startswith("L")
    }
    high_weights = {
        s.name: f"{w:.3f}"
        for s, w in hints.symbol_weights.items()
        if s.name.startswith("H")
    }
    print(f"    LOW symbol weights: {low_weights}")
    print(f"    HIGH symbol weights: {high_weights}")
    print(f"    cluster_size_preference: {hints.cluster_size_preference}")

    failures: list[tuple[str, int]] = []
    successes = 0

    for attempt in range(1, max_attempts + 1):
        # Reseed RNG each attempt -- mirrors PopulationController behavior
        instance_rng = random.Random(seed + attempt * 10000)

        try:
            success, reason, step = diagnostic_attempt(
                sig, sim_id=0, hints=hints, rng=instance_rng,
                gen=cascade_gen, registry=registry, attempt_num=attempt,
            )
        except Exception as exc:
            # Catch anything unexpected -- print full traceback
            print(f"\n  >> UNEXPECTED ERROR:")
            traceback.print_exc()
            success = False
            reason = f"Unexpected: {type(exc).__name__}: {exc}"
            step = -2

        if success:
            successes += 1
            print(f"\n  First success on attempt {attempt} -- stopping.")
            break
        else:
            failures.append((reason, step))

    print_failure_summary(failures, attempt, successes)
    return 0 if successes > 0 else 1


def main(argv: list[str] | None = None) -> int:
    _setup_utf8()

    parser = argparse.ArgumentParser(
        description="Diagnostic step-through for archetype generation",
    )
    parser.add_argument(
        "--archetype", required=True,
        help="Archetype ID to diagnose (e.g. t1_multi_cascade, t1_low_cascade)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--attempts", type=int, default=5,
        help="Max attempts to try (default: 5, use higher to see failure patterns)",
    )
    args = parser.parse_args(argv)
    return run_diagnostic(
        archetype_id=args.archetype, seed=args.seed, max_attempts=args.attempts,
    )


if __name__ == "__main__":
    sys.exit(main())
