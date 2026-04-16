"""ReelStripGenerator — populates the board by spinning a circular reel strip.

Mirrors the reference game's `run_spin` sequence:

    draw_board → evaluate (detect + arm + payout + grid_mults)
              → while clusters: tumble (explode + spawn + gravity + refill)
              →                 evaluate
              → if armed boosters: fire phase + evaluate
              → else: terminate

The strip, not a solver, determines the outcome — this class owns the
physical cascade loop. Every primitive operation (cluster detection, gravity
settle, paytable, grid multipliers, spawn loop, phase executor) is reused
from the existing pipeline; nothing is re-implemented.
"""

from __future__ import annotations

import random

from ..archetypes.registry import ArchetypeRegistry
from ..boosters.fire_handlers import (
    fire_bomb,
    fire_lightball,
    fire_rocket,
    fire_superlightball,
)
from ..boosters.phase_executor import BoosterPhaseExecutor
from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.cluster_detection import Cluster, detect_clusters
from ..primitives.gravity import GravityDAG, settle
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.paytable import Paytable
from ..primitives.reel_strip import ReelStrip, ReelStripCursor
from ..primitives.symbols import Symbol
from ..spatial_solver.data_types import ClusterAssignment, SpatialStep
from ..variance.hints import VarianceHints
from ..boosters.state_machine import BoosterInstance
from .booster_spawning import spawn_boosters_from_clusters
from .data_types import (
    BoosterArmRecord,
    BoosterFireRecord,
    CascadeStepRecord,
    GeneratedInstance,
    GenerationResult,
    GravityRecord,
    build_gravity_record,
    fire_result_to_record,
)
from .reel_refill import ReelStripRefill
from .simulator import _build_position_map

# Empty SpatialStep — reel instances have no CSP plan, but GeneratedInstance
# requires a non-None spatial_step for shape compatibility with other families.
_EMPTY_SPATIAL_STEP = SpatialStep(
    clusters=(),
    near_misses=(),
    scatter_positions=frozenset(),
    boosters=(),
)

# Sentinel RNG passed to ReelStripRefill.fill to satisfy the RefillStrategy
# protocol signature. The strip refill ignores rng — symbol selection is
# entirely determined by the cursor state.
_UNUSED_RNG = random.Random(0)


class ReelStripGenerator:
    """Generates instances whose board and cascades come from a reel strip.

    One strip is injected at construction (loaded once per worker). Each
    `generate` call draws fresh per-reel stop positions from `rng`, builds a
    `ReelStripCursor`, and runs the cascade loop until both clusters and
    armed boosters are exhausted (O-DEAD-1) or the wincap is hit.
    """

    __slots__ = (
        "_config", "_registry", "_gravity_dag", "_strip",
        "_paytable", "_booster_rules",
    )

    def __init__(
        self,
        config: MasterConfig,
        registry: ArchetypeRegistry,
        gravity_dag: GravityDAG,
        strip: ReelStrip,
    ) -> None:
        self._config = config
        self._registry = registry
        self._gravity_dag = gravity_dag
        self._strip = strip
        self._paytable = Paytable(
            config.paytable, config.centipayout, config.win_levels,
        )
        self._booster_rules = BoosterRules(
            config.boosters, config.board, config.symbols,
        )

    def generate(
        self,
        archetype_id: str,
        sim_id: int,
        hints: VarianceHints,  # noqa: ARG002 — strip is the source of variance
        rng: random.Random,
    ) -> GenerationResult:
        """Draw stops, run the cascade loop, return a GeneratedInstance.

        The strip is deterministic given rng state; no retries are needed
        because there's no solver failure mode — any stop vector is valid.
        """
        sig = self._registry.get(archetype_id)
        board_cfg = self._config.board

        stops = tuple(
            rng.randint(0, self._strip.strip_length - 1)
            for _ in range(self._strip.num_reels)
        )
        cursor = ReelStripCursor(self._strip, stops, board_cfg.num_rows)
        strip_refill = ReelStripRefill(cursor)

        board = Board.empty(board_cfg)
        for reel, column in enumerate(cursor.initial_board()):
            for row, sym in enumerate(column):
                board.set(Position(reel, row), sym)

        grid_mults = GridMultiplierGrid(
            self._config.grid_multiplier, board_cfg,
        )
        tracker = BoosterTracker(board_cfg)
        phase_executor = self._make_phase_executor(tracker)

        wincap_centipayout = self._paytable.to_centipayout(
            self._config.wincap.max_payout,
        )

        # Cascade budget derived from game rules — the most cluster cycles
        # physically possible when every cascade clears exactly the minimum
        # cluster. +1 for the initial evaluation step.
        max_steps = (
            (board_cfg.num_reels * board_cfg.num_rows)
            // board_cfg.min_cluster_size
            + 1
        )

        cascade_records: list[CascadeStepRecord] = []
        cumulative_centipayout = 0
        wincap_triggered = False

        # `armed_instances` carries the boosters armed during each evaluation
        # forward to the next _tumble call — arming happens alongside cluster
        # detection but the event belongs with the step record built after the
        # explosion/spawn for that same cluster.
        clusters, step_payout, armed_instances = self._evaluate(
            board, grid_mults, tracker,
        )
        cumulative_centipayout += self._paytable.to_centipayout(step_payout)

        step_count = 0
        cascade_active = True

        while cascade_active and not wincap_triggered and step_count < max_steps:

            while clusters and not wincap_triggered and step_count < max_steps:
                step_count += 1
                record = self._tumble(
                    board, clusters, tracker, grid_mults,
                    strip_refill, step_count, armed_instances,
                )
                cascade_records.append(record)

                if cumulative_centipayout >= wincap_centipayout:
                    cumulative_centipayout = wincap_centipayout
                    wincap_triggered = True
                    break

                clusters, step_payout, armed_instances = self._evaluate(
                    board, grid_mults, tracker,
                )
                cumulative_centipayout += self._paytable.to_centipayout(
                    step_payout,
                )

            if tracker.get_armed() and not wincap_triggered:
                fire_records, fire_gravity = self._execute_booster_phase(
                    board, tracker, phase_executor, strip_refill,
                )
                if cascade_records and (fire_records or fire_gravity):
                    cascade_records[-1] = _merge_fires_into_last(
                        cascade_records[-1], fire_records, fire_gravity,
                    )
                clusters, step_payout, armed_instances = self._evaluate(
                    board, grid_mults, tracker,
                )
                cumulative_centipayout += self._paytable.to_centipayout(
                    step_payout,
                )
            else:
                cascade_active = False

        payout = (
            cumulative_centipayout / self._config.centipayout.multiplier
        )
        win_level = self._paytable.get_win_level(payout)

        instance = GeneratedInstance(
            sim_id=sim_id,
            archetype_id=sig.id,
            family=sig.family,
            criteria=sig.criteria,
            board=board,
            spatial_step=_EMPTY_SPATIAL_STEP,
            payout=payout,
            centipayout=cumulative_centipayout,
            win_level=win_level,
            cascade_steps=tuple(cascade_records) if cascade_records else None,
        )
        return GenerationResult(
            instance=instance, success=True, attempts=1, failure_reason=None,
        )

    # ------------------------------------------------------------------
    # Per-step helpers (single definition each; SRP)
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        board: Board,
        grid_mults: GridMultiplierGrid,
        tracker: BoosterTracker,
    ) -> tuple[list[Cluster], float, list[BoosterInstance]]:
        """Detect clusters, arm adjacent boosters, compute payout, bump multipliers.

        Mirrors the reference game block: get_clusters_update_wins +
        _check_*_adjacency (all types) + emit_tumble_win_events +
        update_grid_mults. Returns the newly-armed booster instances so the
        caller can attach boosterArmInfo records to the step being built.
        """
        clusters = detect_clusters(board, self._config)
        if not clusters:
            return clusters, 0.0, []

        all_positions = frozenset(
            pos for c in clusters for pos in c.positions | c.wild_positions
        )
        armed_instances = tracker.arm_adjacent(all_positions)

        step_payout = sum(
            self._paytable.compute_cluster_payout(c, grid_mults)
            for c in clusters
        )
        for c in clusters:
            for pos in c.positions:
                grid_mults.increment(pos)

        return clusters, step_payout, armed_instances

    def _tumble(
        self,
        board: Board,
        clusters: list[Cluster],
        tracker: BoosterTracker,
        grid_mults: GridMultiplierGrid,
        strip_refill: ReelStripRefill,
        step_index: int,
        armed_instances: list[BoosterInstance],
    ) -> CascadeStepRecord:
        """Explode, spawn boosters, settle gravity, refill from strip."""
        # board_after follows the cascade_generator convention: the
        # pre-explosion filled board the player sees during THIS step.
        # event_stream.py and validator.py both treat steps[0].board_after
        # as the initial reveal board. Post-refill state is reconstructed
        # from gravity_record.refill_entries and surfaces as the next
        # step's board_after through the in-place board mutation below.
        pre_explosion_snapshot = board.copy()

        all_positions = frozenset(
            pos for c in clusters for pos in c.positions | c.wild_positions
        )

        for pos in all_positions:
            board.set(pos, None)

        spawn_events = spawn_boosters_from_clusters(
            clusters, board, tracker, self._booster_rules,
            self._config.boosters.spawn_order,
        )
        freshly_spawned = frozenset(e.position for e in spawn_events)

        gravity_exploded = all_positions - freshly_spawned
        settle_result = settle(
            self._gravity_dag, board, gravity_exploded, self._config.gravity,
        )
        _copy_board_contents(settle_result.board, board)

        position_map = _build_position_map(settle_result.move_steps)
        tracker.update_positions_after_gravity(position_map)

        refill_entries = strip_refill.fill(
            board, settle_result.empty_positions, rng=_UNUSED_RNG,
        )
        for reel, row, sym_name in refill_entries:
            board.set(Position(reel, row), Symbol[sym_name])

        gravity_record = build_gravity_record(
            gravity_exploded, settle_result, refill_entries,
        )

        arm_records, arm_types = BoosterArmRecord.tuple_from_instances(
            armed_instances,
        )

        # Convert detection-layer Cluster into the pipeline-layer
        # ClusterAssignment that CascadeStepRecord and EventStreamGenerator
        # consume. Fields map 1:1 — no new detection work.
        cluster_assignments = tuple(
            ClusterAssignment(
                symbol=c.symbol,
                positions=c.positions,
                size=c.size,
                wild_positions=c.wild_positions,
            )
            for c in clusters
        )

        return CascadeStepRecord(
            step_index=step_index,
            board_before=pre_explosion_snapshot,
            board_after=pre_explosion_snapshot,
            clusters=cluster_assignments,
            step_payout=sum(
                self._paytable.compute_cluster_payout(c, grid_mults)
                for c in clusters
            ),
            grid_multipliers_snapshot=_snapshot_grid(
                grid_mults, self._config.board,
            ),
            booster_spawn_types=tuple(e.booster_type.name for e in spawn_events),
            booster_spawn_positions=tuple(
                (
                    e.booster_type.name,
                    e.position.reel, e.position.row,
                    e.orientation,
                )
                for e in spawn_events
            ),
            booster_arm_types=arm_types,
            booster_arm_records=arm_records,
            gravity_record=gravity_record,
        )

    def _execute_booster_phase(
        self,
        board: Board,
        tracker: BoosterTracker,
        phase_executor: BoosterPhaseExecutor,
        strip_refill: ReelStripRefill,
    ) -> tuple[tuple[BoosterFireRecord, ...], GravityRecord | None]:
        """Fire armed boosters, clear cells, settle gravity, refill from strip."""
        fire_results = phase_executor.execute_booster_phase(board)
        if not fire_results:
            return (), None

        all_affected: set[Position] = set()
        for fr in fire_results:
            all_affected.update(fr.affected_positions)
            board.set(fr.booster.position, None)
        for pos in all_affected:
            board.set(pos, None)

        affected_frozen = frozenset(all_affected)
        fire_settle = settle(
            self._gravity_dag, board, affected_frozen, self._config.gravity,
        )
        _copy_board_contents(fire_settle.board, board)

        tracker.update_positions_after_gravity(
            _build_position_map(fire_settle.move_steps),
        )

        fire_refill = strip_refill.fill(
            board, fire_settle.empty_positions, rng=_UNUSED_RNG,
        )
        for reel, row, sym_name in fire_refill:
            board.set(Position(reel, row), Symbol[sym_name])

        fire_gravity = build_gravity_record(
            affected_frozen, fire_settle, fire_refill,
        )
        fire_records = tuple(fire_result_to_record(r) for r in fire_results)
        return fire_records, fire_gravity

    def _make_phase_executor(
        self, tracker: BoosterTracker,
    ) -> BoosterPhaseExecutor:
        """Same pattern as CascadeInstanceGenerator._make_phase_executor."""
        executor = BoosterPhaseExecutor(
            tracker, self._booster_rules, self._booster_rules.chain_initiators,
        )
        executor.register_fire_handler(Symbol.R, fire_rocket)
        executor.register_fire_handler(Symbol.B, fire_bomb)
        executor.register_fire_handler(Symbol.LB, fire_lightball)
        executor.register_fire_handler(Symbol.SLB, fire_superlightball)
        return executor


# ---------------------------------------------------------------------------
# Module-level helpers (no state; small enough to not warrant a class)
# ---------------------------------------------------------------------------

def _snapshot_grid(
    grid_mults: GridMultiplierGrid, board_config,
) -> tuple[int, ...]:
    """Flatten grid multipliers in reel-major order to match CascadeStepRecord."""
    return tuple(
        grid_mults.get(Position(reel, row))
        for reel in range(board_config.num_reels)
        for row in range(board_config.num_rows)
    )


def _copy_board_contents(src: Board, dst: Board) -> None:
    """Overwrite dst's cells from src in place — keeps dst identity stable."""
    for pos in src.all_positions():
        dst.set(pos, src.get(pos))


def _merge_fires_into_last(
    record: CascadeStepRecord,
    fire_records: tuple[BoosterFireRecord, ...],
    fire_gravity: GravityRecord | None,
) -> CascadeStepRecord:
    """Attach booster fire outputs to the preceding tumble record."""
    import dataclasses
    return dataclasses.replace(
        record,
        booster_fire_records=record.booster_fire_records + fire_records,
        booster_gravity_record=fire_gravity or record.booster_gravity_record,
    )
