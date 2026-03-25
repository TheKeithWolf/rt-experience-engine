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

from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.gravity import GravityDAG, settle
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.symbols import Symbol, is_wild, symbol_from_name
from ..step_reasoner.progress import ClusterRecord
from ..step_reasoner.results import SpawnRecord, StepResult
from .data_types import GravityRecord, TransitionResult, build_gravity_record


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
    ) -> TransitionResult:
        """Execute the transition after a validated step.

        1. Increment grid multipliers at all cluster positions
        2. Explode cluster positions on a board copy
        3. Spawn boosters into emptied cells (wilds survive explosion)
        4. Run gravity settle
        5. Update booster tracker positions after gravity
        6. Return post-transition board + records
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
        )

        # 4. Run gravity settle
        settle_result = settle(
            self._gravity_dag,
            result_board,
            frozenset(all_cluster_positions),
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

    def _spawn_boosters(
        self,
        clusters: tuple[ClusterRecord, ...],
        tracker: BoosterTracker,
        step_index: int,
    ) -> list[SpawnRecord]:
        """Spawn boosters from qualifying clusters in config spawn order.

        Processes clusters checking each against the spawn thresholds.
        Wild spawns are handled by the wild lifecycle — only non-wild
        boosters (R, B, LB, SLB) are added to the tracker.

        Occupied tracking prevents position conflicts between spawned boosters.
        """
        rules = self._booster_rules
        # Seed occupied set with existing tracker positions to avoid collisions
        occupied: set[Position] = {
            b.position for b in tracker.all_boosters()
        }
        spawn_records: list[SpawnRecord] = []

        for booster_name in self._config.boosters.spawn_order:
            booster_sym = symbol_from_name(booster_name)

            # Wild spawns handled by wild lifecycle, not the booster tracker
            if is_wild(booster_sym):
                continue

            for cluster_idx, cluster in enumerate(clusters):
                candidate_type = rules.booster_type_for_size(cluster.size)
                if candidate_type is not booster_sym:
                    continue

                centroid = rules.compute_centroid(cluster.positions)
                position = rules.resolve_collision(
                    centroid, cluster.positions, frozenset(occupied),
                )

                # Determine orientation for rockets
                orientation: str | None = None
                if booster_sym is Symbol.R:
                    orientation = rules.compute_rocket_orientation(
                        cluster.positions,
                    )

                tracker.add(
                    booster_sym, position,
                    orientation=orientation,
                    source_cluster_index=cluster_idx,
                )
                occupied.add(position)

                spawn_records.append(SpawnRecord(
                    booster_type=booster_name,
                    position=position,
                    source_cluster_index=cluster_idx,
                    step_index=step_index,
                ))

        return spawn_records


def _build_position_map(
    move_steps: tuple[tuple[tuple[Position, Position], ...], ...],
) -> dict[Position, Position]:
    """Build a cumulative position map from gravity move passes.

    Traces each source position through all passes to its final destination.
    Used by BoosterTracker.update_positions_after_gravity().
    """
    # Collect all unique source positions from the first pass they appear
    tracked: dict[Position, Position] = {}
    for pass_moves in move_steps:
        for src, dst in pass_moves:
            if src not in tracked:
                tracked[src] = dst
            elif tracked[src] == src:
                # Position was stationary in earlier passes, now it moves
                tracked[src] = dst

    # Re-trace: walk each original position through all passes
    result: dict[Position, Position] = {}
    all_sources: set[Position] = set()
    for pass_moves in move_steps:
        for src, _ in pass_moves:
            all_sources.add(src)

    for original_src in all_sources:
        current = original_src
        for pass_moves in move_steps:
            for src, dst in pass_moves:
                if src == current:
                    current = dst
                    break
        result[original_src] = current

    return result
