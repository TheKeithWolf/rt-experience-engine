"""BoosterSpawnCapPropagator — budget-aware cluster-size prevention.

Prunes placements whose projected component size would spawn a booster
whose remaining spawn budget has hit zero. Derives the cap dynamically
from `BoosterRules.booster_type_for_size` and `ProgressTracker`'s
`remaining_booster_spawns()` — no hardcoded thresholds.

Wild-aware: delegates to the same `max_component_size` BFS used by
`detect_clusters()` and the validator so wild-bridged merges count toward
the projected size, matching what validation will see after the step runs.

Closes the signature gap the validator reports as `booster_spawn(W)=N
outside [X, Y]` when WFC accidentally forms 7-8 clusters during cascade
refill beyond the archetype's wild-spawn budget.
"""

from __future__ import annotations

from collections.abc import Callable

from ..pipeline.protocols import Range
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.cluster_detection import max_component_size
from ..primitives.symbols import Symbol
from .cell_state import CellState


class BoosterSpawnCapPropagator:
    """Block placements whose projected size would spawn an exhausted booster.

    For every uncollapsed neighbor of a just-collapsed cell:
      1. For each candidate symbol, project the post-placement component size.
      2. Ask `size_to_booster(size)` which booster name (if any) spawns.
      3. If that booster's remaining `max_val == 0`, remove the candidate.

    Wild-aware through `wild_positions`, matching detect_clusters() semantics
    so wild-bridged merges count toward the projected size.

    Implements the Propagator protocol (LSP) — the WFC solver treats this
    identically to NoClusterPropagator / MaxComponentPropagator.

    The `size_to_booster` callable abstracts the size → booster-name lookup
    so either `BoosterRules.booster_type_for_size` (returns Symbol, compose
    via `.name`) or `SpawnEvaluator.booster_for_size` (returns str) works
    without coupling the propagator to either class.
    """

    __slots__ = ("_size_to_booster", "_remaining_spawns", "_wild_positions")

    def __init__(
        self,
        size_to_booster: Callable[[int], str | None],
        remaining_spawns: dict[str, Range],
        wild_positions: frozenset[Position] | None = None,
    ) -> None:
        # Size → booster-name lookup — reuses config.boosters.spawn_thresholds
        # via whatever evaluator the caller injects
        self._size_to_booster = size_to_booster
        # Live budget from ProgressTracker.remaining_booster_spawns() —
        # budget of 0 means "no more spawns of this type allowed"
        self._remaining_spawns = remaining_spawns
        # Forwarded to max_component_size so wild-bridged merges count.
        # Empty frozenset when no wilds present; None normalized to frozenset
        # for consistent downstream handling.
        self._wild_positions = wild_positions or frozenset()

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config,
    ) -> set[Position]:
        """Prune neighbor possibilities that would spawn an exhausted booster."""
        changed: set[Position] = set()
        for neighbor in orthogonal_neighbors(position, board_config):
            if neighbor not in cells or cells[neighbor].collapsed:
                continue
            to_remove: list[Symbol] = []
            for sym in cells[neighbor].possibilities:
                if self._would_spawn_exhausted_booster(
                    board, neighbor, sym, board_config,
                ):
                    to_remove.append(sym)
            for sym in to_remove:
                if cells[neighbor].remove(sym):
                    changed.add(neighbor)
        return changed

    def validate_placement(
        self,
        board: Board,
        position: Position,
        board_config,
    ) -> bool:
        """Reject collapsed placements that would spawn an exhausted booster."""
        sym = board.get(position)
        if sym is None:
            return True
        return not self._would_spawn_exhausted_booster_at(
            board, position, sym, board_config,
        )

    def allows(
        self,
        board: Board,
        position: Position,
        sym: Symbol,
        board_config,
    ) -> bool:
        """Public predicate: True iff placing sym at position is budget-safe.

        Projects the post-placement component via the same BFS the WFC
        propagator uses, then consults the live remaining-budget map. This
        is the single source of truth shared between WFC propagation
        (via `propagate`) and non-WFC refill paths (e.g. ClusterSeekingRefill)
        so both code paths reject the same placements.
        """
        return not self._would_spawn_exhausted_booster(
            board, position, sym, board_config,
        )

    def _would_spawn_exhausted_booster(
        self,
        board: Board,
        position: Position,
        sym: Symbol,
        board_config,
    ) -> bool:
        """Check if placing sym at position would spawn an exhausted booster.

        Pre-collapse check: `extra={position}` treats the candidate as if
        already placed, so the BFS sees the full projected component.
        """
        projected_size = max_component_size(
            board, sym, board_config,
            extra=frozenset({position}),
            wild_positions=self._wild_positions,
        )
        return self._size_is_exhausted(projected_size)

    def _would_spawn_exhausted_booster_at(
        self,
        board: Board,
        position: Position,
        sym: Symbol,
        board_config,
    ) -> bool:
        """Post-collapse variant — sym is already on the board."""
        projected_size = max_component_size(
            board, sym, board_config,
            wild_positions=self._wild_positions,
        )
        return self._size_is_exhausted(projected_size)

    def _size_is_exhausted(self, size: int) -> bool:
        """True when a component of `size` would spawn an out-of-budget booster.

        Returns False for sizes below any spawn threshold (no booster spawns)
        and for booster types with positive remaining max. Only exhausted
        types (max_val == 0) trigger pruning — so archetypes with positive
        budgets are unaffected until their quota fills.
        """
        booster_name = self._size_to_booster(size)
        if booster_name is None:
            return False
        remaining = self._remaining_spawns.get(booster_name)
        if remaining is None:
            # Signature doesn't mention this booster type — no cap to enforce
            return False
        return remaining.max_val == 0
