"""Shared booster-spawn loop — DRY rule-of-three extraction.

Three pipelines now need to spawn boosters from qualifying clusters in the
configured spawn order:

  - `CascadeInstanceGenerator._spawn_boosters` (static/cascade generator)
  - `StepTransitionSimulator._spawn_boosters`  (ASP-step simulator)
  - `ReelStripGenerator._tumble`                (reel-strip generator)

The loop body is identical across all three; only the output record type
differs (BoosterPlacement vs SpawnRecord). This module owns the loop and
returns a neutral per-spawn record (`SpawnEvent`) that callers adapt to
their preferred return shape.

No behavior change relative to the two existing implementations — every
operation (centroid, collision resolution, rocket orientation, place_booster,
occupied tracking) is kept in its original order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..boosters.tracker import BoosterTracker
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules, place_booster
from ..primitives.symbols import Symbol, symbol_from_name


class _ClusterLike(Protocol):
    """Structural view of the cluster attributes the spawn loop reads.

    Both `primitives.cluster_detection.Cluster` and
    `step_reasoner.progress.ClusterRecord` satisfy this — they each expose
    `.size` (int) and `.positions` (frozenset[Position]). Using a Protocol
    means the loop accepts either without shipping a shared base class.
    """

    @property
    def size(self) -> int: ...  # pragma: no cover - protocol

    @property
    def positions(self) -> frozenset[Position]: ...  # pragma: no cover - protocol


@dataclass(frozen=True, slots=True)
class SpawnEvent:
    """Neutral per-spawn record returned by `spawn_boosters_from_clusters`.

    Callers map this to their domain record (BoosterPlacement, SpawnRecord,
    etc.). The Symbol form of booster_type is preserved so callers can route
    by type without re-resolving the name.
    """

    booster_type: Symbol
    position: Position
    source_cluster_index: int
    orientation: str | None


def spawn_boosters_from_clusters(
    clusters: list[_ClusterLike] | tuple[_ClusterLike, ...],
    board: Board,
    tracker: BoosterTracker,
    rules: BoosterRules,
    spawn_order: tuple[str, ...],
) -> tuple[SpawnEvent, ...]:
    """Spawn boosters from qualifying clusters in config spawn order.

    Shared loop semantics (unchanged from the two original copies):
      - Iterate `spawn_order` so wilds spawn before rockets before bombs, etc.
      - For each spawn symbol, scan clusters in detection order, spawning when
        the rules' size-based `booster_type_for_size` matches.
      - Seed the occupied set with existing tracker positions so new spawns
        never collide with previously-placed boosters.
      - For rockets only, compute orientation from cluster positions.
      - Place each booster via `place_booster` (board + tracker).
      - Track placed positions in the occupied set for subsequent iterations.
    """
    occupied: set[Position] = {b.position for b in tracker.all_boosters()}
    events: list[SpawnEvent] = []

    for booster_name in spawn_order:
        booster_sym = symbol_from_name(booster_name)

        for cluster_idx, cluster in enumerate(clusters):
            if rules.booster_type_for_size(cluster.size) is not booster_sym:
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
            events.append(SpawnEvent(
                booster_type=booster_sym,
                position=position,
                source_cluster_index=cluster_idx,
                orientation=orientation,
            ))

    return tuple(events)
