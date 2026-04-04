"""Pipeline output data types — shared between generator, validator, and controller.

Frozen dataclasses representing generation results. These carry the complete
state needed for validation, accumulator updates, event stream generation,
and output writing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..spatial_solver.data_types import ClusterAssignment, SpatialStep

if TYPE_CHECKING:
    from ..step_reasoner.results import SpawnRecord, StepResult


@dataclass(frozen=True, slots=True)
class BoosterFireRecord:
    """Lightweight snapshot of one booster fire for validation/diagnostics.

    Avoids importing BoosterFireResult directly to prevent circular imports.
    Captures the essential facts: which booster type fired, where, its
    orientation (rockets only), and how many cells/chains it produced.
    """

    booster_type: str           # "R", "B", etc.
    position_reel: int
    position_row: int
    # "H"/"V" for rockets, None for bombs — used by orientation balance diagnostics
    orientation: str | None
    affected_count: int         # number of cells cleared by this fire
    chain_target_count: int     # number of boosters hit for potential chaining
    # Symbols targeted by LB (1 symbol) or SLB (2 symbols) — empty for R/B
    target_symbols: tuple[str, ...] = ()
    # Cleared positions with symbol identity — event stream needs symbol for
    # boosterFireInfo.clearedCells[].symbol. Each entry: (reel, row, symbol_name).
    affected_positions_list: tuple[tuple[int, int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class BoosterArmRecord:
    """Snapshot of one booster arm event for event stream replay."""

    booster_type: str
    position_reel: int
    position_row: int
    orientation: str | None


@dataclass(frozen=True, slots=True)
class GravityRecord:
    """Gravity settle snapshot — positions exploded, per-pass moves, and refill symbols.

    Captured during cascade generation so the event stream can emit gravitySettle
    events without re-deriving gravity from board diffs.
    """

    # Positions removed (clusters exploded) that triggered this gravity settle
    exploded_positions: tuple[tuple[int, int], ...]
    # Per-pass movement data: each move is (symbol_name, (reel, row), (reel, row))
    move_steps: tuple[tuple[tuple[str, tuple[int, int], tuple[int, int]], ...], ...]
    # New symbols placed at empty positions after gravity + WFC refill
    # Each entry: (reel, row, symbol_name) — frontend needs symbol + position for animation
    refill_entries: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True, slots=True)
class CascadeStepRecord:
    """Snapshot of one cascade step — used by the validator, diagnostics, and event stream.

    Captures board state before/after cluster removal and gravity, along with
    which clusters were detected and the payout for this step. Grid multiplier
    state is stored as a flat tuple of all position values (reel-major order)
    so the record stays frozen and serializable.

    Gravity records capture the settle that preceded this step (exploding the
    previous step's clusters) and the post-booster settle if boosters fired.
    Step 0 has no preceding gravity (gravity_record is None).
    """

    step_index: int
    board_before: Board
    board_after: Board
    clusters: tuple[ClusterAssignment, ...]
    # Payout contribution from this step's clusters (bet multiplier)
    step_payout: float
    # Flattened grid multiplier values after this step, reel-major order
    grid_multipliers_snapshot: tuple[int, ...]
    # Booster types spawned at this step (e.g., ("R",) or ("R", "B"))
    booster_spawn_types: tuple[str, ...] = ()
    # Resolved spawn positions — (booster_type, reel, row, orientation) per spawn.
    # Populated from TransitionResult.spawns (post-collision-resolution).
    # Parallel to booster_spawn_types; carries position + orientation data event_stream needs.
    booster_spawn_positions: tuple[tuple[str, int, int, str | None], ...] = ()
    # Booster types armed at this step (e.g., ("R",)) — populated from
    # BoosterTracker.arm_adjacent() results during transition.
    booster_arm_types: tuple[str, ...] = ()
    # Full arm records with position/orientation — event stream needs these
    # for boosterArmInfo events. Populated alongside booster_arm_types.
    booster_arm_records: tuple[BoosterArmRecord, ...] = ()
    # Booster fires that occurred during this step's booster phase
    booster_fire_records: tuple[BoosterFireRecord, ...] = ()
    # Gravity settle that preceded this step — None for step 0 (initial board)
    gravity_record: GravityRecord | None = None
    # Post-booster gravity settle — None if no boosters fired this step
    booster_gravity_record: GravityRecord | None = None


@dataclass(frozen=True, slots=True)
class GeneratedInstance:
    """A single generated board instance — the output of the generation pipeline.

    Contains everything needed for validation, accumulator update, and output.
    For cascade instances (cascade_depth > 0), cascade_steps carries the per-step
    breakdown. Static instances leave cascade_steps as None.
    """

    sim_id: int
    archetype_id: str
    family: str
    criteria: str
    board: Board
    spatial_step: SpatialStep
    # Payout as bet multiplier (e.g., 2.6x)
    payout: float
    # Integer centipayout (e.g., 260) for RGS output
    centipayout: int
    # Win level tier from config.win_levels
    win_level: int
    # Per-step cascade data — None for static (cascade_depth=0) instances
    cascade_steps: tuple[CascadeStepRecord, ...] | None = None
    # Gravity settle after cluster explosion — used by static instances for the
    # post-win animation (explode → gravity → refill). Cascade instances carry
    # gravity data on each CascadeStepRecord instead.
    gravity_record: GravityRecord | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result of attempting to generate one instance — success or failure.

    On failure, instance is None and failure_reason describes what went wrong.
    attempts tracks how many tries were needed (even on success).
    """

    instance: GeneratedInstance | None
    success: bool
    attempts: int
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Post-transition board state with gravity and booster spawn records.

    Produced by StepTransitionSimulator.transition() after exploding clusters,
    running gravity, and spawning boosters. The settled board becomes the input
    for the next cascade step. When boosters fire, also carries fire records
    and a second gravity settle from the booster clearings.
    """

    board: Board
    # Boosters spawned from this step's qualifying clusters
    spawns: tuple[SpawnRecord, ...]
    # Gravity settle record for event stream replay (cluster explosion)
    gravity_record: GravityRecord
    # Booster phase results — empty when no boosters fired this step
    booster_fire_records: tuple[BoosterFireRecord, ...] = ()
    # Post-booster gravity settle — None if no boosters fired
    booster_gravity_record: GravityRecord | None = None
    # Booster types armed during this transition (DORMANT -> ARMED)
    booster_arm_types: tuple[str, ...] = ()
    # Full arm records with position/orientation for event stream
    booster_arm_records: tuple[BoosterArmRecord, ...] = ()


class TransitionData(NamedTuple):
    """Structured container for per-step transition data in the cascade loop.

    Replaces a positional tuple that is packed in 3 locations and unpacked in 2
    across cascade_generator.py and debug_archetype.py.  Named fields prevent
    silent positional misalignment when fields are added.
    """

    gravity_record: GravityRecord
    empty_positions: tuple                          # from board.empty_positions()
    spawns: tuple[SpawnRecord, ...]
    fire_records: tuple[BoosterFireRecord, ...]
    booster_gravity_record: GravityRecord | None
    arm_types: tuple[str, ...] = ()
    arm_records: tuple[BoosterArmRecord, ...] = ()


class CascadeStepOutcome(NamedTuple):
    """Result of a single reason→execute→validate→advance→transition cycle.

    Returned by CascadeInstanceGenerator._execute_cascade_step() so the
    main cascade loop and post-terminal re-cascade loop share a single
    implementation. Callers handle appending to step_results/raw_steps
    and breaking on terminal.
    """

    step_result: StepResult
    board_before: Board
    filled: Board
    transition_data: TransitionData | None
    is_terminal: bool
    # transition_result.board when non-terminal; filled when terminal
    next_board: Board


def merge_post_terminal_fires(
    records: list[CascadeStepRecord],
    fire_recs: tuple[BoosterFireRecord, ...],
    gravity_rec: GravityRecord | None,
) -> list[CascadeStepRecord]:
    """Merge post-terminal booster fire records into the last step record.

    Post-terminal fires happen after the cascade dies but before final
    validation.  They attach to the terminal step because that step caused
    the cascade to end.

    Returns the records list (mutated in place for efficiency).
    """
    if not fire_recs or not records:
        return records

    last = records[-1]
    records[-1] = dataclasses.replace(
        last,
        booster_fire_records=last.booster_fire_records + fire_recs,
        booster_gravity_record=gravity_rec or last.booster_gravity_record,
    )
    return records


# ---------------------------------------------------------------------------
# Shared gravity record builder — used by both cascade and static generators
# ---------------------------------------------------------------------------

def build_gravity_record(
    exploded_positions: set[Position] | frozenset[Position],
    settle_result: object,
    refill_entries: tuple[tuple[int, int, str], ...] = (),
) -> GravityRecord:
    """Convert a SettleResult into a GravityRecord for event stream replay.

    Translates Position objects into (reel, row) tuples for serialization.
    Callers provide refill_entries via a RefillStrategy for the frontend
    gravity settle animation.
    """
    exploded_tuples = tuple(
        (pos.reel, pos.row) for pos in sorted(
            exploded_positions, key=lambda p: (p.reel, p.row),
        )
    )

    # Convert move_steps: tuple of passes, each move is (symbol, src, dst) with Position→tuple
    move_tuples: list[tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]] = []
    for pass_moves in settle_result.move_steps:  # type: ignore[attr-defined]
        pass_tuples = tuple(
            (sym, (src.reel, src.row), (dst.reel, dst.row))
            for sym, src, dst in pass_moves
        )
        move_tuples.append(pass_tuples)

    return GravityRecord(
        exploded_positions=exploded_tuples,
        move_steps=tuple(move_tuples),
        refill_entries=refill_entries,
    )



# ---------------------------------------------------------------------------
# Booster fire result conversion
# ---------------------------------------------------------------------------

def fire_result_to_record(result: object) -> BoosterFireRecord:
    """Convert a BoosterFireResult (boosters layer) to a BoosterFireRecord (pipeline layer).

    Takes ``object`` to avoid importing BoosterFireResult at module level
    (would create a circular dependency: pipeline → boosters → pipeline).
    The caller is responsible for passing a valid BoosterFireResult instance.
    """
    return BoosterFireRecord(
        booster_type=result.booster.booster_type.name,  # type: ignore[attr-defined]
        position_reel=result.booster.position.reel,  # type: ignore[attr-defined]
        position_row=result.booster.position.row,  # type: ignore[attr-defined]
        orientation=result.booster.orientation,  # type: ignore[attr-defined]
        affected_count=len(result.affected_positions),  # type: ignore[attr-defined]
        chain_target_count=len(result.chain_targets),  # type: ignore[attr-defined]
        target_symbols=result.target_symbols,  # type: ignore[attr-defined]
        affected_positions_list=tuple(
            (pos.reel, pos.row, name)
            for pos, name in sorted(
                result.affected_symbols,  # type: ignore[attr-defined]
                key=lambda pair: (pair[0].reel, pair[0].row),
            )
        ),
    )
