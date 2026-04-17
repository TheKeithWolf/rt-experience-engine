"""Step transition simulator — handles the transition between cascade steps.

After a step is validated, the simulator:
1. Increments grid multipliers at cluster positions
2. Explodes cluster positions (sets to None)
3. Spawns boosters into emptied cells (wilds survive explosion)
4. Runs gravity settle
5. Updates booster positions after gravity
6. Returns the post-transition board state with spawn and gravity records

All thresholds from MasterConfig — zero hardcoded values.
Booster spawn logic mirrors CascadeInstanceGenerator._spawn_boosters() using
BoosterRules directly — single source of truth for spawn thresholds, centroid
computation, collision resolution, and orientation rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.gravity import GravityDAG, settle
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.symbols import Symbol
from ..step_reasoner.progress import ClusterRecord
from ..step_reasoner.results import SpawnRecord, StepResult
from .booster_spawning import spawn_boosters_from_clusters
from .data_types import (
    BoosterArmRecord,
    BoosterFireRecord,
    GravityRecord,
    TransitionResult,
    build_gravity_record,
    fire_result_to_record,
)

if TYPE_CHECKING:
    from ..boosters.phase_executor import BoosterPhaseExecutor


class StepTransitionSimulator:
    """Simulates the physical transition between cascade steps.

    Handles cluster explosion, booster spawning, gravity settle, and
    booster position updates. Owns no game logic beyond the mechanical
    cascade transition — all rules come from BoosterRules and GravityDAG.
    """

    __slots__ = ("_gravity_dag", "_config", "_booster_rules")

    def __init__(
        self,
        gravity_dag: GravityDAG,
        config: MasterConfig,
    ) -> None:
        self._gravity_dag = gravity_dag
        self._config = config
        self._booster_rules = BoosterRules(
            config.boosters, config.board, config.symbols,
        )

    def transition(
        self,
        board: Board,
        step_result: StepResult,
        booster_tracker: BoosterTracker,
        grid_mults: GridMultiplierGrid,
        preferred_landings: dict[int, Position] | None = None,
    ) -> TransitionResult:
        """Execute the transition after a validated step.

        1. Increment grid multipliers at all cluster positions
        2. Explode cluster positions on a board copy
        3. Spawn boosters into emptied cells (wilds survive explosion)
        4. Run gravity settle
        5. Update booster tracker positions after gravity
        6. Return post-transition board + records

        `preferred_landings` — atlas-derived landings keyed by cluster index,
        forwarded into booster spawning so post-gravity placement aligns with
        armable cells AtlasQuery already validated.
        """
        result_board = board.copy()

        # Collect all positions from all clusters for explosion
        all_cluster_positions: set[Position] = set()
        for cluster in step_result.clusters:
            all_cluster_positions.update(cluster.positions)

        # 1. Increment grid multipliers at cluster positions
        for pos in all_cluster_positions:
            grid_mults.increment(pos)

        # 2. Explode cluster positions (set to None) — must precede spawning
        # so wilds placed at cluster centroids survive the explosion
        for pos in all_cluster_positions:
            result_board.set(pos, None)

        # 3. Spawn boosters into the freshly-emptied cells
        spawn_records = self._spawn_boosters(
            step_result.clusters, booster_tracker, step_result.step_index,
            result_board,
            preferred_landings=preferred_landings,
        )

        # Exclude positions where boosters were just placed — settle() re-marks
        # all "exploded" positions as None, which would destroy spawned symbols
        booster_spawn_positions = {sr.position for sr in spawn_records}
        gravity_exploded = frozenset(all_cluster_positions - booster_spawn_positions)

        # 4. Run gravity settle
        settle_result = settle(
            self._gravity_dag,
            result_board,
            gravity_exploded,
            self._config.gravity,
        )
        result_board = settle_result.board

        # 5. Update booster positions after gravity
        position_map = _build_position_map(settle_result.move_steps)
        booster_tracker.update_positions_after_gravity(position_map)

        # 6. Build gravity record for event stream replay
        gravity_record = build_gravity_record(
            all_cluster_positions, settle_result,
        )

        return TransitionResult(
            board=result_board,
            spawns=tuple(spawn_records),
            gravity_record=gravity_record,
        )

    def transition_and_arm(
        self,
        board: Board,
        step_result: StepResult,
        booster_tracker: BoosterTracker,
        grid_mults: GridMultiplierGrid,
        phase_executor: BoosterPhaseExecutor,
        preferred_landings: dict[int, Position] | None = None,
    ) -> TransitionResult:
        """Execute transition with booster spawn and arming — fires are deferred.

        Handles cluster explosion, booster spawning, gravity settle, and arming
        of dormant boosters adjacent to the exploding cluster. Armed boosters
        sit on the board until the post-terminal booster phase fires them.

        `preferred_landings` — atlas-derived landings keyed by cluster index,
        used by `_spawn_boosters` to place bomb/rocket at armable cells.
        """
        result_board = board.copy()

        # Collect all cluster positions for explosion
        all_cluster_positions: set[Position] = set()
        for cluster in step_result.clusters:
            all_cluster_positions.update(cluster.positions)

        # 1. Increment grid multipliers at cluster positions
        for pos in all_cluster_positions:
            grid_mults.increment(pos)

        # 2. Explode cluster positions
        for pos in all_cluster_positions:
            result_board.set(pos, None)

        # 3. Spawn boosters into emptied cells
        spawn_records = self._spawn_boosters(
            step_result.clusters, booster_tracker, step_result.step_index,
            result_board,
            preferred_landings=preferred_landings,
        )

        # Exclude booster spawn positions from gravity — they'd be destroyed
        booster_spawn_positions = {sr.position for sr in spawn_records}
        gravity_exploded = frozenset(all_cluster_positions - booster_spawn_positions)

        # 4. Arm dormant boosters adjacent to the exploding cluster positions
        # Spec: arm adjacency is checked BEFORE gravity (step 8 precedes step 9).
        # Freshly-spawned non-wild boosters are excluded — they sit at the source
        # cluster's centroid and are inherently adjacent.
        freshly_spawned_positions = frozenset(
            sr.position
            for sr in spawn_records
            if sr.booster_type != Symbol.W.name
        )
        armed_instances = booster_tracker.arm_adjacent(
            frozenset(all_cluster_positions),
            exclude_positions=freshly_spawned_positions,
        )

        # 5. Gravity settle (cluster explosion)
        settle_result = settle(
            self._gravity_dag, result_board, gravity_exploded, self._config.gravity,
        )
        result_board = settle_result.board

        # 6. Update booster positions after cluster-gravity
        position_map = _build_position_map(settle_result.move_steps)
        booster_tracker.update_positions_after_gravity(position_map)

        # Build cluster gravity record
        gravity_record = build_gravity_record(all_cluster_positions, settle_result)

        # Build arm records for event stream replay (shared helper keeps the
        # reel-strip family in lockstep with this cascade family)
        arm_records, arm_types = BoosterArmRecord.tuple_from_instances(
            armed_instances,
        )

        # Armed boosters sit until the post-terminal booster phase fires them
        return TransitionResult(
            board=result_board,
            spawns=tuple(spawn_records),
            gravity_record=gravity_record,
            booster_arm_types=arm_types,
            booster_arm_records=arm_records,
        )

    # ------------------------------------------------------------------
    # Booster fire phase — shared by mid-cascade and post-terminal paths
    # ------------------------------------------------------------------

    def _fire_and_settle(
        self,
        board: Board,
        phase_executor: BoosterPhaseExecutor,
        booster_tracker: BoosterTracker,
        grid_mults: GridMultiplierGrid,
    ) -> tuple[Board, tuple[BoosterFireRecord, ...], GravityRecord | None]:
        """Fire all armed boosters, clear affected cells, gravity settle.

        Delegates to BoosterPhaseExecutor.execute_booster_phase() which handles
        row-major fire order and depth-first chain propagation (R/B initiate
        chains; LB/SLB receive but do not propagate).

        Returns (settled_board, fire_records, booster_gravity_record).
        Returns (board, (), None) when no boosters are armed.
        Shared by transition_and_arm() and execute_terminal_booster_phase().
        """
        fire_results = phase_executor.execute_booster_phase(board)

        if not fire_results:
            return board, (), None

        # Clear fire-affected positions and fired booster positions on the board
        all_fire_affected: set[Position] = set()
        for fr in fire_results:
            all_fire_affected.update(fr.affected_positions)
            # Remove the fired booster itself from the board
            board.set(fr.booster.position, None)

        for pos in all_fire_affected:
            board.set(pos, None)

        # Gravity settle for booster fire clearings
        booster_gravity_exploded = frozenset(all_fire_affected)
        booster_settle = settle(
            self._gravity_dag, board, booster_gravity_exploded,
            self._config.gravity,
        )
        board = booster_settle.board

        # Update booster positions after booster-gravity
        booster_pos_map = _build_position_map(booster_settle.move_steps)
        booster_tracker.update_positions_after_gravity(booster_pos_map)

        # Convert fire results to lightweight records for event stream
        booster_gravity_record = build_gravity_record(
            all_fire_affected, booster_settle,
        )
        booster_fire_records = tuple(
            fire_result_to_record(fr) for fr in fire_results
        )

        return board, booster_fire_records, booster_gravity_record

    def execute_terminal_booster_phase(
        self,
        board: Board,
        booster_tracker: BoosterTracker,
        grid_mults: GridMultiplierGrid,
        phase_executor: BoosterPhaseExecutor,
    ) -> TransitionResult:
        """Fire armed boosters on a terminal board, then gravity settle.

        Called after the cascade loop when the board is dead but armed
        boosters remain. Returns a TransitionResult with fire records
        and booster gravity for event stream replay. No spawns occur
        (no clusters to spawn from).
        """
        result_board = board.copy()
        settled_board, fire_records, booster_gravity = self._fire_and_settle(
            result_board, phase_executor, booster_tracker, grid_mults,
        )

        return TransitionResult(
            board=settled_board,
            spawns=(),
            # No cluster gravity in the booster phase — only booster-fire gravity
            gravity_record=GravityRecord(
                exploded_positions=(),
                move_steps=(),
                refill_entries=(),
            ),
            booster_fire_records=fire_records,
            booster_gravity_record=booster_gravity,
        )

    # ------------------------------------------------------------------
    # Booster spawn logic
    # ------------------------------------------------------------------

    def _spawn_boosters(
        self,
        clusters: tuple[ClusterRecord, ...],
        tracker: BoosterTracker,
        step_index: int,
        board: Board,
        preferred_landings: dict[int, Position] | None = None,
    ) -> list[SpawnRecord]:
        """Adapt the shared spawn loop to this simulator's SpawnRecord output.

        `preferred_landings` — atlas-derived landings keyed by cluster index.
        Forwarded into `spawn_boosters_from_clusters` so bomb/rocket spawn
        at positions that AtlasQuery already validated as armable.
        """
        events = spawn_boosters_from_clusters(
            clusters, board, tracker, self._booster_rules,
            self._config.boosters.spawn_order,
            preferred_landings=preferred_landings,
        )
        return [
            SpawnRecord(
                booster_type=e.booster_type.name,
                position=e.position,
                source_cluster_index=e.source_cluster_index,
                step_index=step_index,
                orientation=e.orientation,
            )
            for e in events
        ]


def _build_position_map(
    move_steps: tuple[tuple[tuple[str, Position, Position], ...], ...],
) -> dict[Position, Position]:
    """Build a cumulative position map from gravity move passes.

    Traces each source position through all passes to its final destination.
    Used by BoosterTracker.update_positions_after_gravity().
    """
    # Collect all unique source positions from the first pass they appear
    tracked: dict[Position, Position] = {}
    for pass_moves in move_steps:
        for _sym, src, dst in pass_moves:
            if src not in tracked:
                tracked[src] = dst
            elif tracked[src] == src:
                # Position was stationary in earlier passes, now it moves
                tracked[src] = dst

    # Re-trace: walk each original position through all passes
    result: dict[Position, Position] = {}
    all_sources: set[Position] = set()
    for pass_moves in move_steps:
        for _sym, src, _dst in pass_moves:
            all_sources.add(src)

    for original_src in all_sources:
        current = original_src
        for pass_moves in move_steps:
            for _sym, src, dst in pass_moves:
                if src == current:
                    current = dst
                    break
        result[original_src] = current

    return result
