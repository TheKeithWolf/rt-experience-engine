"""Pipeline output data types — shared between generator, validator, and controller.

Frozen dataclasses representing generation results. These carry the complete
state needed for validation, accumulator updates, event stream generation,
and output writing.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..spatial_solver.data_types import ClusterAssignment, SpatialStep

if TYPE_CHECKING:
    from ..step_reasoner.results import SpawnRecord


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
    # Actual board positions cleared — needed by event stream for clearedCells in boosterPhase
    affected_positions_list: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class GravityRecord:
    """Gravity settle snapshot — positions exploded, per-pass moves, and refill symbols.

    Captured during cascade generation so the event stream can emit gravitySettle
    events without re-deriving gravity from board diffs.
    """

    # Positions removed (clusters exploded) that triggered this gravity settle
    exploded_positions: tuple[tuple[int, int], ...]
    # Per-pass movement data: each pass is a tuple of (source, dest) position pairs
    move_steps: tuple[tuple[tuple[tuple[int, int], tuple[int, int]], ...], ...]
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
    # Resolved spawn positions — (booster_type, reel, row) per spawn.
    # Populated from TransitionResult.spawns (post-collision-resolution).
    # Parallel to booster_spawn_types; carries the position data event_stream needs.
    booster_spawn_positions: tuple[tuple[str, int, int], ...] = ()
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
    for the next cascade step.
    """

    board: Board
    # Boosters spawned from this step's qualifying clusters
    spawns: tuple[SpawnRecord, ...]
    # Gravity settle record for event stream replay
    gravity_record: GravityRecord


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
    Callers provide refill_entries via compute_refill_entries() for the
    frontend gravity settle animation.
    """
    exploded_tuples = tuple(
        (pos.reel, pos.row) for pos in sorted(
            exploded_positions, key=lambda p: (p.reel, p.row),
        )
    )

    # Convert move_steps: tuple of passes, each pass is tuple of (src, dst) Position pairs
    move_tuples: list[tuple[tuple[tuple[int, int], tuple[int, int]], ...]] = []
    for pass_moves in settle_result.move_steps:  # type: ignore[attr-defined]
        pass_tuples = tuple(
            ((src.reel, src.row), (dst.reel, dst.row))
            for src, dst in pass_moves
        )
        move_tuples.append(pass_tuples)

    return GravityRecord(
        exploded_positions=exploded_tuples,
        move_steps=tuple(move_tuples),
        refill_entries=refill_entries,
    )


def compute_refill_entries(
    empty_positions: Iterable[Position],
    standard_symbols: tuple[str, ...],
    rng: random.Random,
) -> tuple[tuple[int, int, str], ...]:
    """Generate refill symbols for empty positions after gravity settles.

    Frontend animates these symbols dropping into empty cells.
    Used by both static (instance_generator) and cascade (cascade_generator) paths.
    """
    return tuple(
        (pos.reel, pos.row, rng.choice(standard_symbols))
        for pos in empty_positions
    )
