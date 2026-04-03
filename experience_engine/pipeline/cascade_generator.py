"""Cascade instance generator — reason → execute → validate → transition loop.

Generates instances for archetypes with cascade_depth > 0 using the StepReasoner.
Each step: the reasoner observes real board state, produces a StepIntent, the
executor fills the board, the validator checks constraints, and the simulator
handles cluster explosion + gravity for the next step.

All thresholds from MasterConfig — zero hardcoded values.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ..archetypes.registry import ArchetypeRegistry, ArchetypeSignature
from ..board_filler.wfc_solver import FillFailed
from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules, place_booster
from ..primitives.gravity import GravityDAG
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.paytable import Paytable
from ..primitives.symbols import Symbol, symbol_from_name
from ..step_reasoner.context import BoardContext
from ..step_reasoner.progress import ProgressTracker
from ..step_reasoner.reasoner import StepReasoner
from ..step_reasoner.results import StepResult
from ..variance.hints import VarianceHints
from .data_types import (
    BoosterArmRecord,
    BoosterFireRecord,
    CascadeStepOutcome,
    CascadeStepRecord,
    GravityRecord,
    GeneratedInstance,
    GenerationResult,
    TransitionData,
    merge_post_terminal_fires,
)
from .step_executor import StepExecutor
from .step_validator import StepValidator, StepValidationFailed
from .simulator import StepTransitionSimulator
from ..narrative.transitions import build_transition_rules, try_advance_phase

if TYPE_CHECKING:
    from ..boosters.phase_executor import BoosterPhaseExecutor

from ..primitives.cluster_detection import Cluster, detect_clusters

from ..spatial_solver.data_types import BoosterPlacement, SpatialStep


class CascadeInstanceGenerator:
    """Generates instances for archetypes with cascade_depth > 0.

    Pipeline per instance:
      reason → execute → validate → transition → repeat until terminal

    The StepReasoner observes real board state each step and decides
    what to do next. The executor fills, the validator checks, and
    the simulator handles cluster explosion + gravity between steps.
    """

    __slots__ = (
        "_config", "_registry", "_gravity_dag", "_paytable",
        "_booster_rules", "_reasoner", "_executor", "_validator",
        "_simulator", "_transition_rules",
    )

    def __init__(
        self,
        config: MasterConfig,
        registry: ArchetypeRegistry,
        gravity_dag: GravityDAG,
        reasoner: StepReasoner,
        executor: StepExecutor,
        validator: StepValidator,
        simulator: StepTransitionSimulator,
    ) -> None:
        self._config = config
        self._registry = registry
        self._gravity_dag = gravity_dag
        self._paytable = Paytable(
            config.paytable, config.centipayout, config.win_levels,
        )
        self._booster_rules = BoosterRules(
            config.boosters, config.board, config.symbols,
        )
        self._reasoner = reasoner
        self._executor = executor
        self._validator = validator
        self._simulator = simulator
        self._transition_rules = build_transition_rules(config.board, config.symbols)

    def generate(
        self,
        archetype_id: str,
        sim_id: int,
        hints: VarianceHints,
        rng: random.Random,
    ) -> GenerationResult:
        """Generate a single cascade instance.

        Retries up to max_retries_per_instance on solver/fill/validation failures.
        """
        sig = self._registry.get(archetype_id)
        max_retries = self._config.solvers.max_retries_per_instance

        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                instance = self._attempt_generation(sig, sim_id, hints, rng)
                return GenerationResult(
                    instance=instance,
                    success=True,
                    attempts=attempt,
                    failure_reason=None,
                )
            except (FillFailed, StepValidationFailed, ValueError) as exc:
                last_error = str(exc)
                continue

        return GenerationResult(
            instance=None,
            success=False,
            attempts=max_retries,
            failure_reason=f"exhausted {max_retries} retries: {last_error}",
        )

    def _attempt_generation(
        self,
        sig: ArchetypeSignature,
        sim_id: int,
        hints: VarianceHints,
        rng: random.Random,
    ) -> GeneratedInstance:
        """Single attempt at generating a cascade instance via the StepReasoner.

        Loop: reason → execute → validate → transition → repeat until terminal.
        Bounded by cascade_depth.max_val + 1 to prevent infinite loops.
        """
        board = Board.empty(self._config.board)
        grid_mults = GridMultiplierGrid(
            self._config.grid_multiplier, self._config.board,
        )
        booster_tracker = BoosterTracker(self._config.board)
        progress = ProgressTracker(sig, self._config.centipayout.multiplier)

        # Build a BoosterPhaseExecutor if the archetype requires booster fires
        phase_executor = (
            self._make_phase_executor(booster_tracker)
            if sig.required_booster_fires else None
        )

        step_results: list[StepResult] = []
        cascade_step_records: list[CascadeStepRecord] = []
        # +1 because the terminal step itself counts as a step
        max_steps = sig.required_cascade_depth.max_val + 1

        # Phase 1: Run the cascade loop, collecting raw step data.
        # Gravity refill symbols come from the NEXT step's WFC fill, so
        # record building is deferred to phase 2 where all data is available.
        # Each entry: (step_result, board_before, filled, grid_mults, transition_data)
        # transition_data: (gravity_record, empty_positions, spawns, fire_recs, booster_grav)
        # or None for terminal steps
        raw_steps: list[tuple] = []

        for _ in range(max_steps):
            outcome = self._execute_cascade_step(
                board, grid_mults, booster_tracker, phase_executor,
                progress, sig, hints, rng,
            )
            step_results.append(outcome.step_result)
            raw_steps.append((
                outcome.step_result, outcome.board_before, outcome.filled,
                grid_mults, outcome.transition_data,
            ))
            if outcome.is_terminal:
                filled = outcome.filled
                break
            board = outcome.next_board

        # Post-terminal booster phase: fire armed boosters, refill, re-cascade.
        # Uses the terminal filled board (executor copies input, so `board` is stale).
        post_fire_recs: tuple[BoosterFireRecord, ...] = ()
        post_fire_grav: GravityRecord | None = None
        if phase_executor is not None and booster_tracker.get_armed():
            post_terminal_board = filled if outcome.is_terminal else board
            board, raw_steps, post_fire_recs, post_fire_grav = (
                self._run_post_terminal_booster_phase(
                    post_terminal_board, booster_tracker, grid_mults,
                    phase_executor, progress, sig, hints, rng,
                    raw_steps, step_results,
                )
            )

        # Phase 2: Build CascadeStepRecords with actual refill symbols.
        # Step N's gravity refill reads from step N+1's filled board — the
        # symbols the WFC executor actually placed at empty positions.
        for i, (sr, bb, ba, gm, td) in enumerate(raw_steps):
            gravity_record = None
            spawns = ()
            fire_recs: tuple[BoosterFireRecord, ...] = ()
            booster_grav: GravityRecord | None = None
            arm_types: tuple[str, ...] = ()
            arm_records: tuple[BoosterArmRecord, ...] = ()
            if td is not None:
                gr_base = td.gravity_record
                empty_positions = td.empty_positions
                spawns = td.spawns
                fire_recs = td.fire_records
                booster_grav = td.booster_gravity_record
                arm_types = td.arm_types
                arm_records = td.arm_records
                next_filled = raw_steps[i + 1][2]
                refill_entries = tuple(
                    (pos.reel, pos.row, next_filled.get(pos).name)
                    for pos in empty_positions
                )
                gravity_record = GravityRecord(
                    exploded_positions=gr_base.exploded_positions,
                    move_steps=gr_base.move_steps,
                    refill_entries=refill_entries,
                )
            cascade_step_records.append(self._build_step_record(
                sr, bb, ba, gm,
                gravity_record=gravity_record,
                transition_spawns=spawns,
                booster_fire_records=fire_recs,
                booster_gravity_record=booster_grav,
                booster_arm_types=arm_types,
                booster_arm_records=arm_records,
            ))

        # Merge post-terminal booster fire records into the last step record.
        # These fires happen after the terminal dead board, so they attach to
        # the terminal step — the step that caused the cascade to end.
        merge_post_terminal_fires(cascade_step_records, post_fire_recs, post_fire_grav)

        # FINAL VALIDATION: check full instance against archetype constraints
        # Use the last filled board (terminal step) as final_board
        final_board = filled if outcome.is_terminal else board
        instance = self._validator.validate_instance(
            step_results, sig, final_board, progress,
        )
        # Stamp sim_id and cascade_steps onto the instance
        instance = GeneratedInstance(
            sim_id=sim_id,
            archetype_id=instance.archetype_id,
            family=instance.family,
            criteria=instance.criteria,
            board=instance.board,
            spatial_step=instance.spatial_step,
            payout=instance.payout,
            centipayout=instance.centipayout,
            win_level=instance.win_level,
            cascade_steps=tuple(cascade_step_records),
        )
        return instance

    # ------------------------------------------------------------------
    # Post-terminal booster phase
    # ------------------------------------------------------------------

    def _run_post_terminal_booster_phase(
        self,
        board: Board,
        booster_tracker: BoosterTracker,
        grid_mults: GridMultiplierGrid,
        phase_executor: BoosterPhaseExecutor,
        progress: ProgressTracker,
        sig: ArchetypeSignature,
        hints: VarianceHints,
        rng: random.Random,
        raw_steps: list[tuple],
        step_results: list[StepResult],
    ) -> tuple[Board, list[tuple], tuple[BoosterFireRecord, ...], GravityRecord | None]:
        """Fire armed boosters post-terminal, refill, re-cascade if clusters emerge.

        Loop:
        1. Fire all armed boosters → clear → gravity settle
        2. Refill empty cells with random standard symbols
        3. Detect clusters on refilled board
        4. If clusters → re-enter cascade loop via _execute_cascade_step()
        5. If no clusters and no armed boosters → truly terminal

        Budget is derived from the archetype: 1 fire round + max chain depth
        re-cascades. Independent of the main cascade loop ceiling.

        Returns fire records separately so they can be merged into the last
        CascadeStepRecord after Phase 2 building — avoids dependency on
        transition_data which is None for terminal steps.
        """
        standard_symbol_names = tuple(self._config.symbols.standard)
        all_fire_records: list[BoosterFireRecord] = []
        all_booster_grav: GravityRecord | None = None

        # Budget: 1 fire round + max chain re-cascade depth from archetype
        post_terminal_max = 1 + sig.required_chain_depth.max_val
        post_steps = 0

        while booster_tracker.get_armed() and post_steps < post_terminal_max:
            # 1. Fire armed boosters → clear affected → gravity settle
            fire_result = self._simulator.execute_terminal_booster_phase(
                board, booster_tracker, grid_mults, phase_executor,
            )
            board = fire_result.board

            # Track booster fires in progress for instance validation
            for fire_rec in fire_result.booster_fire_records:
                progress.boosters_fired[fire_rec.booster_type] = (
                    progress.boosters_fired.get(fire_rec.booster_type, 0) + 1
                )

            # Accumulate fire records for caller to merge into CascadeStepRecord
            all_fire_records.extend(fire_result.booster_fire_records)
            all_booster_grav = fire_result.booster_gravity_record

            # 2. Refill empty cells with random standard symbols
            for reel in range(self._config.board.num_reels):
                for row in range(self._config.board.num_rows):
                    pos = Position(reel, row)
                    if board.get(pos) is None:
                        sym = symbol_from_name(rng.choice(standard_symbol_names))
                        board.set(pos, sym)

            # 3. Detect clusters on refilled board
            clusters = detect_clusters(board, self._config)

            if not clusters:
                # No clusters and while-loop will check for armed boosters
                break

            # 4. Clusters found — re-enter cascade loop via shared step method
            for _ in range(post_terminal_max - post_steps):
                outcome = self._execute_cascade_step(
                    board, grid_mults, booster_tracker, phase_executor,
                    progress, sig, hints, rng,
                )
                step_results.append(outcome.step_result)
                raw_steps.append((
                    outcome.step_result, outcome.board_before, outcome.filled,
                    grid_mults, outcome.transition_data,
                ))
                if outcome.is_terminal:
                    break
                board = outcome.next_board

            post_steps += 1
            # After re-cascade, check if new armed boosters appeared — while loop continues

        return board, raw_steps, tuple(all_fire_records), all_booster_grav

    # ------------------------------------------------------------------
    # Shared cascade step — used by main loop and post-terminal re-cascade
    # ------------------------------------------------------------------

    def _execute_cascade_step(
        self,
        board: Board,
        grid_mults: GridMultiplierGrid,
        booster_tracker: BoosterTracker,
        phase_executor: BoosterPhaseExecutor | None,
        progress: ProgressTracker,
        sig: ArchetypeSignature,
        hints: VarianceHints,
        rng: random.Random,
    ) -> CascadeStepOutcome:
        """One reason→execute→validate→advance→transition cycle.

        Shared by the main cascade loop and post-terminal re-cascade loop.
        Callers handle appending to step_results/raw_steps and breaking on
        terminal — this method only executes one step and returns its outcome.
        """
        board_before = board.copy()

        context = BoardContext.from_board(
            board, grid_mults,
            progress.dormant_boosters, progress.active_wilds,
            self._config.board,
        )

        intent = self._reasoner.reason(context, progress, sig, hints)
        filled = self._executor.execute(intent, board, rng)

        step_result = self._validator.validate_step(
            filled, intent, progress, grid_mults,
        )

        progress.update(step_result)
        # Sync wilds from the filled board — transition hasn't spawned new
        # wilds yet, so active_wilds reflects only wilds present on the board.
        progress.sync_active_wilds(filled, self._config.board)

        # Phase advancement — pre-transition, matching RL environment ordering.
        # Context from the FILLED board so predicates see the step's outcome,
        # not the explosion's aftermath.
        fill_context = BoardContext.from_board(
            filled, grid_mults,
            progress.dormant_boosters, progress.active_wilds,
            self._config.board,
        )
        try_advance_phase(
            progress, step_result, self._transition_rules, fill_context,
        )

        transition_data = None
        next_board = filled
        if not intent.is_terminal:
            # Route to booster-aware transition when the archetype fires boosters
            if phase_executor is not None:
                transition_result = self._simulator.transition_and_arm(
                    filled, step_result, booster_tracker, grid_mults,
                    phase_executor,
                )
            else:
                transition_result = self._simulator.transition(
                    filled, step_result, booster_tracker, grid_mults,
                )

            transition_data = TransitionData(
                gravity_record=transition_result.gravity_record,
                empty_positions=transition_result.board.empty_positions(),
                spawns=transition_result.spawns,
                fire_records=transition_result.booster_fire_records,
                booster_gravity_record=transition_result.booster_gravity_record,
                arm_types=transition_result.booster_arm_types,
                arm_records=transition_result.booster_arm_records,
            )

            # Board is truth — sync wild positions after gravity + consumption
            progress.sync_active_wilds(
                transition_result.board, self._config.board,
            )
            next_board = transition_result.board

        return CascadeStepOutcome(
            step_result=step_result,
            board_before=board_before,
            filled=filled,
            transition_data=transition_data,
            is_terminal=intent.is_terminal,
            next_board=next_board,
        )

    # ------------------------------------------------------------------
    # Step record builder
    # ------------------------------------------------------------------

    def _build_step_record(
        self,
        step_result: StepResult,
        board_before: Board,
        board_after: Board,
        grid_mults: GridMultiplierGrid,
        gravity_record: GravityRecord | None = None,
        transition_spawns: tuple = (),
        booster_fire_records: tuple[BoosterFireRecord, ...] = (),
        booster_gravity_record: GravityRecord | None = None,
        booster_arm_types: tuple[str, ...] = (),
        booster_arm_records: tuple[BoosterArmRecord, ...] = (),
    ) -> CascadeStepRecord:
        """Convert a StepResult + board snapshots into a CascadeStepRecord.

        gravity_record is the settle from exploding this step's clusters —
        None for terminal steps (dead board, no clusters to explode).
        booster_fire_records/booster_gravity_record carry the booster phase
        results when boosters fired at this step.
        """
        from ..spatial_solver.data_types import ClusterAssignment

        cluster_assignments = tuple(
            ClusterAssignment(
                symbol=cr.symbol,
                positions=cr.positions,
                size=cr.size,
            )
            for cr in step_result.clusters
        )

        step_payout = (
            step_result.step_payout / self._config.centipayout.multiplier
        )

        return CascadeStepRecord(
            step_index=step_result.step_index,
            board_before=board_before,
            board_after=board_after.copy(),
            clusters=cluster_assignments,
            step_payout=step_payout,
            grid_multipliers_snapshot=self._snapshot_grid_mults(grid_mults),
            booster_spawn_types=tuple(
                s.booster_type for s in step_result.spawns
            ),
            # Actual positions from TransitionResult (post-collision-resolution),
            # not StepResult centroids — events must reflect real board state.
            booster_spawn_positions=tuple(
                (s.booster_type, s.position.reel, s.position.row, s.orientation)
                for s in transition_spawns
            ),
            booster_arm_types=booster_arm_types,
            booster_arm_records=booster_arm_records,
            booster_fire_records=booster_fire_records,
            gravity_record=gravity_record,
            booster_gravity_record=booster_gravity_record,
        )

    # ------------------------------------------------------------------
    # Booster phase executor factory
    # ------------------------------------------------------------------

    def _make_phase_executor(
        self, tracker: BoosterTracker,
    ) -> BoosterPhaseExecutor:
        """Construct a BoosterPhaseExecutor with real fire handlers for one attempt.

        Built per-attempt because the executor holds a reference to the tracker,
        which is per-attempt state. BoosterRules and handler functions are shared.
        """
        from ..boosters.phase_executor import BoosterPhaseExecutor
        from ..boosters.fire_handlers import (
            fire_rocket, fire_bomb, fire_lightball, fire_superlightball,
        )

        rules = self._simulator._booster_rules
        executor = BoosterPhaseExecutor(tracker, rules, rules.chain_initiators)
        executor.register_fire_handler(Symbol.R, fire_rocket)
        executor.register_fire_handler(Symbol.B, fire_bomb)
        executor.register_fire_handler(Symbol.LB, fire_lightball)
        executor.register_fire_handler(Symbol.SLB, fire_superlightball)
        return executor

    # ------------------------------------------------------------------
    # Utility methods (kept for backward compatibility and tests)
    # ------------------------------------------------------------------

    def _apply_spatial_step(
        self,
        board: Board,
        step: SpatialStep,
    ) -> frozenset[Position]:
        """Apply CSP spatial assignments to the board, return pinned positions."""
        pinned: set[Position] = set()

        for cluster in step.clusters:
            for pos in cluster.positions:
                board.set(pos, cluster.symbol)
                pinned.add(pos)

        for nm in step.near_misses:
            for pos in nm.positions:
                board.set(pos, nm.symbol)
                pinned.add(pos)

        for pos in step.scatter_positions:
            board.set(pos, Symbol.S)
            pinned.add(pos)

        # Place wild symbols from CSP wild placements
        for wp in step.wild_placements:
            board.set(wp.position, Symbol.W)
            pinned.add(wp.position)

        return frozenset(pinned)

    def _compute_step_payout(
        self,
        clusters: list[Cluster],
        grid_mults: GridMultiplierGrid,
    ) -> float:
        """Compute total payout for one cascade step's clusters."""
        total = 0.0
        for cluster in clusters:
            total += self._paytable.compute_cluster_payout(cluster, grid_mults)
        return total

    def _spawn_boosters(
        self,
        clusters: list[Cluster],
        tracker: BoosterTracker,
        board: Board,
    ) -> list[BoosterPlacement]:
        """Spawn boosters from winning clusters in config spawn order.

        Wilds are written directly to the board at the centroid position.
        Non-wild boosters (R, B, LB, SLB) are added to the tracker.
        Occupied tracking prevents position conflicts between spawned boosters.
        """
        rules = self._booster_rules
        # Seed with existing tracker positions to avoid collisions with
        # boosters from previous cascade steps
        occupied: set[Position] = {b.position for b in tracker.all_boosters()}
        placements: list[BoosterPlacement] = []

        for booster_name in self._config.boosters.spawn_order:
            booster_sym = symbol_from_name(booster_name)

            for cluster_idx, cluster in enumerate(clusters):
                candidate_type = rules.booster_type_for_size(cluster.size)
                if candidate_type is not booster_sym:
                    continue

                centroid = rules.compute_centroid(cluster.positions)
                position = rules.resolve_collision(
                    centroid, cluster.positions, frozenset(occupied),
                )

                orientation: str | None = None
                if booster_sym is Symbol.R:
                    orientation = rules.compute_rocket_orientation(cluster.positions)
                place_booster(
                    booster_sym, position, board, tracker,
                    orientation=orientation,
                    source_cluster_index=cluster_idx,
                )

                occupied.add(position)
                placements.append(BoosterPlacement(
                    booster_type=booster_sym, position=position,
                ))

        return placements

    def _snapshot_grid_mults(self, grid_mults: GridMultiplierGrid) -> tuple[int, ...]:
        """Flatten grid multiplier values to a tuple in reel-major order."""
        values: list[int] = []
        for reel in range(self._config.board.num_reels):
            for row in range(self._config.board.num_rows):
                values.append(grid_mults.get(Position(reel, row)))
        return tuple(values)
