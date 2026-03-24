"""WFC constraint propagators — remove impossible symbols from cell possibilities.

Each propagator encodes one constraint type. The WFC solver runs all propagators
after each cell collapse to prune the search space via a BFS work queue.

Propagators are pluggable — new constraints can be added without modifying the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..config.schema import BoardConfig, SymbolConfig
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.cluster_detection import max_component_size
from ..primitives.symbols import Symbol, SymbolTier, symbol_from_name, symbols_in_tier
from .cell_state import CellState


@runtime_checkable
class Propagator(Protocol):
    """Constraint propagator interface for WFC board filling.

    Each propagator prunes symbol possibilities from cells near a just-resolved
    position. Returns the set of positions whose possibilities changed, so the
    solver can enqueue them for cascading propagation.
    """

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]: ...

    def validate_placement(
        self,
        board: Board,
        position: Position,
        board_config: BoardConfig,
    ) -> bool:
        """Post-collapse safety net — check if the symbol at position is valid.

        Catches stale possibilities where a same-symbol chain grew from the
        far end since this cell was last pruned by neighbor propagation.
        Returns True if valid, False if the placement violates the constraint.
        """
        ...


def _count_adjacent_same(
    board: Board,
    position: Position,
    symbol: Symbol,
    board_config: BoardConfig,
) -> int:
    """Count orthogonal neighbors of position that hold the given symbol.

    Fast O(1) pre-check — if a cell has fewer than threshold-2 same-symbol
    neighbors, it cannot possibly form a component of size >= threshold when
    placed, so we can skip the expensive full-board BFS.
    """
    count = 0
    for neighbor in orthogonal_neighbors(position, board_config):
        if board.get(neighbor) is symbol:
            count += 1
    return count


def _propagate_cluster_constraint(
    threshold: int,
    board: Board,
    cells: dict[Position, CellState],
    position: Position,
    board_config: BoardConfig,
) -> set[Position]:
    """Shared logic for cluster-size propagators (DRY).

    For each uncollapsed neighbor of `position`, checks whether placing any of
    its remaining possibilities would create a connected component >= threshold.
    If so, removes that symbol from the neighbor's possibilities.

    Uses a fast adjacency pre-check before the expensive BFS — if a symbol has
    zero same-symbol neighbors at a position, it cannot form a component >= threshold
    (since isolated placements start at size 1).

    Reuses cluster_detection.max_component_size — no reimplementation (CONSTRAINT-WFC-2).
    """
    changed: set[Position] = set()
    for neighbor in orthogonal_neighbors(position, board_config):
        if neighbor not in cells or cells[neighbor].collapsed:
            continue
        # Check each remaining possibility at this neighbor
        to_remove: list[Symbol] = []
        for sym in cells[neighbor].possibilities:
            # Fast pre-check: if no same-symbol neighbors exist, placing this
            # symbol creates an isolated component of size 1 — safe when
            # threshold > 1 (skips expensive BFS for the common case)
            if threshold > 1 and _count_adjacent_same(board, neighbor, sym, board_config) == 0:
                continue
            # Full BFS check: would placing sym at neighbor exceed threshold?
            component_size = max_component_size(
                board, sym, board_config, extra=frozenset({neighbor})
            )
            if component_size >= threshold:
                to_remove.append(sym)
        for sym in to_remove:
            if cells[neighbor].remove(sym):
                changed.add(neighbor)
    return changed


def _validate_cluster_placement(
    threshold: int,
    board: Board,
    position: Position,
    board_config: BoardConfig,
) -> bool:
    """Check if the symbol at position forms a component below threshold.

    Post-collapse safety net — catches stale possibilities where a same-symbol
    chain grew from the far end since this cell's possibilities were last pruned.
    Reuses cluster_detection.max_component_size (DRY — single BFS algorithm).
    """
    sym = board.get(position)
    if sym is None:
        return True
    return max_component_size(board, sym, board_config) < threshold


class NoSpecialSymbolPropagator:
    """Strips non-standard symbols from cell possibilities (CONSTRAINT-WFC-1).

    Standard symbols are the only valid fill choices — special symbols (scatter,
    wild, boosters) are placed by the CSP solver, never by WFC. The allowed set
    is derived from config.symbols.standard at construction.
    """

    def __init__(self, symbol_config: SymbolConfig) -> None:
        # Precompute allowed symbols once — all standard symbols from config
        self._allowed: frozenset[Symbol] = frozenset(
            symbol_from_name(name) for name in symbol_config.standard
        )

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Remove non-standard symbols from the cell at position.

        Effective during init pass only — once specials are stripped they
        never reappear, so subsequent calls are no-ops.
        """
        if position not in cells or cells[position].collapsed:
            return set()

        changed: set[Position] = set()
        # Remove any symbol not in the allowed (standard) set
        to_remove = cells[position].possibilities - self._allowed
        for sym in to_remove:
            if cells[position].remove(sym):
                changed.add(position)
        return changed

    def validate_placement(
        self, board: Board, position: Position, board_config: BoardConfig,
    ) -> bool:
        """Init-only propagator — placement is always valid post-collapse."""
        return True


class NoClusterPropagator:
    """Prevents any standard-symbol component from reaching the cluster threshold.

    After a cell is collapsed, checks each uncollapsed neighbor: if placing a
    remaining possibility would create a connected component >= threshold, that
    possibility is removed. Threshold comes from config.board.min_cluster_size.
    """

    def __init__(self, threshold: int) -> None:
        # Cluster prevention threshold — from config.board.min_cluster_size
        self._threshold = threshold

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Prune neighbor possibilities that would form clusters >= threshold."""
        return _propagate_cluster_constraint(
            self._threshold, board, cells, position, board_config
        )

    def validate_placement(
        self, board: Board, position: Position, board_config: BoardConfig,
    ) -> bool:
        """Check collapsed cell doesn't form a cluster >= threshold."""
        return _validate_cluster_placement(
            self._threshold, board, position, board_config
        )


class MaxComponentPropagator:
    """Caps the maximum connected component size — stricter than NoClusterPropagator.

    Used for dead-spin boards where no cluster of any appreciable size should form.
    max_size is the hard upper bound (e.g., 3 means no component of 4+ allowed).
    The threshold passed to the caller is archetype-specific, not hardcoded.
    """

    def __init__(self, max_size: int) -> None:
        # Prevent component >= max_size + 1 (reuse same threshold logic)
        self._threshold = max_size + 1

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Prune neighbor possibilities that would form components > max_size."""
        return _propagate_cluster_constraint(
            self._threshold, board, cells, position, board_config
        )

    def validate_placement(
        self, board: Board, position: Position, board_config: BoardConfig,
    ) -> bool:
        """Check collapsed cell doesn't form a component > max_size."""
        return _validate_cluster_placement(
            self._threshold, board, position, board_config
        )


class TierConstraintPropagator:
    """Restricts WFC fill to symbols from a specific tier (narrative arc constraint).

    When a cascade step requires a specific symbol tier (e.g., LOW for
    t1_low_cascade), this propagator strips non-tier symbols from cell
    possibilities during the init pass — same mechanism as NoSpecialSymbolPropagator.
    Reusable for any archetype with symbol_tier_per_step constraints.
    """

    def __init__(self, tier: SymbolTier, symbol_config: SymbolConfig) -> None:
        # Precompute allowed symbols from the target tier
        self._allowed: frozenset[Symbol] = frozenset(
            symbols_in_tier(tier, symbol_config)
        )

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Remove non-tier symbols from the cell at position.

        Effective during init pass only — once non-tier symbols are stripped
        they never reappear, so subsequent calls are no-ops.
        """
        if position not in cells or cells[position].collapsed:
            return set()

        changed: set[Position] = set()
        to_remove = cells[position].possibilities - self._allowed
        for sym in to_remove:
            if cells[position].remove(sym):
                changed.add(position)
        return changed

    def validate_placement(
        self, board: Board, position: Position, board_config: BoardConfig,
    ) -> bool:
        """Check collapsed cell's symbol belongs to the allowed tier."""
        sym = board.get(position)
        if sym is None:
            return True
        return sym in self._allowed


class WildBridgePropagator:
    """Prevents WFC from bridging clusters through wild positions.

    detect_clusters() merges same-symbol components that share a wild
    neighbor via union-find. WFC doesn't know about wilds and can place
    symbols that create paths through them, merging separate clusters
    into oversized ones. This propagator receives wild positions from
    the CSP step and blocks placements that would bridge through a wild.
    """

    def __init__(self, wild_positions: frozenset[Position]) -> None:
        self._wild_positions = wild_positions

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Prune symbols from neighbors that would bridge through an adjacent wild.

        For each uncollapsed neighbor of the just-collapsed position, checks
        if placing any remaining symbol there would connect to a wild that
        also neighbors another component of the same symbol — creating a
        wild-mediated merge that detect_clusters() would count as one cluster.
        """
        changed: set[Position] = set()
        for neighbor in orthogonal_neighbors(position, board_config):
            if neighbor not in cells or cells[neighbor].collapsed:
                continue
            to_remove: list[Symbol] = []
            for sym in cells[neighbor].possibilities:
                if self._would_bridge_through_wild(board, neighbor, sym, board_config):
                    to_remove.append(sym)
            for sym in to_remove:
                if cells[neighbor].remove(sym):
                    changed.add(neighbor)
        return changed

    def validate_placement(
        self, board: Board, position: Position, board_config: BoardConfig,
    ) -> bool:
        """Check if the just-placed symbol bridges through any adjacent wild."""
        sym = board.get(position)
        if sym is None:
            return True
        return not self._would_bridge_through_wild(board, position, sym, board_config)

    def _would_bridge_through_wild(
        self,
        board: Board,
        position: Position,
        sym: Symbol,
        board_config: BoardConfig,
    ) -> bool:
        """Check if placing sym at position would merge components through a wild.

        Returns True if any adjacent wild also neighbors an existing instance
        of sym — detect_clusters() would merge them through the wild.
        """
        for neighbor in orthogonal_neighbors(position, board_config):
            if neighbor not in self._wild_positions:
                continue
            # neighbor is a wild — check if it also touches sym elsewhere
            for wild_neighbor in orthogonal_neighbors(neighbor, board_config):
                if wild_neighbor == position:
                    continue
                if board.get(wild_neighbor) is sym:
                    return True
        return False


@dataclass(frozen=True, slots=True)
class NearMissGroup:
    """A protected near-miss group: symbol + connected positions.

    Used by NearMissAwareDeadPropagator to enforce isolation —
    no WFC cell adjacent to a group edge may hold the group's symbol.
    """

    symbol: Symbol
    positions: frozenset[Position]


def _compute_border(
    positions: frozenset[Position],
    board_config: BoardConfig,
) -> frozenset[Position]:
    """Positions orthogonally adjacent to a group but not in the group itself.

    Shared by NearMissAwareDeadPropagator (isolation around near-miss groups)
    and ClusterBoundaryPropagator (isolation around cluster cores).
    """
    border: set[Position] = set()
    for pos in positions:
        for neighbor in orthogonal_neighbors(pos, board_config):
            if neighbor not in positions:
                border.add(neighbor)
    return frozenset(border)


class ClusterBoundaryPropagator:
    """Forbids the cluster symbol from cells adjacent to cluster positions.

    After a cluster is pinned on the board, WFC must not place the same symbol
    adjacent to it — otherwise explosion + gravity leaves same-symbol survivors
    at the refill zone boundary, causing impossible merges on the next cascade step.

    Uses the shared _compute_border() helper (DRY with NearMissAwareDeadPropagator).
    """

    __slots__ = ("_forbidden_symbol", "_border")

    def __init__(
        self,
        cluster_positions: frozenset[Position],
        cluster_symbol: Symbol,
        board_config: BoardConfig,
    ) -> None:
        self._forbidden_symbol = cluster_symbol
        # Border = cells adjacent to cluster but not in the cluster
        self._border = _compute_border(cluster_positions, board_config)

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Remove the cluster symbol from uncollapsed cells in the border zone."""
        changed: set[Position] = set()
        for neighbor in orthogonal_neighbors(position, board_config):
            if neighbor not in self._border:
                continue
            if neighbor not in cells or cells[neighbor].collapsed:
                continue
            if cells[neighbor].remove(self._forbidden_symbol):
                changed.add(neighbor)
        return changed

    def validate_placement(
        self,
        board: Board,
        position: Position,
        board_config: BoardConfig,
    ) -> bool:
        """Reject the cluster symbol at border positions."""
        if position not in self._border:
            return True
        return board.get(position) is not self._forbidden_symbol


class NearMissAwareDeadPropagator:
    """Dead-board propagator that protects near-miss group isolation.

    Combines MaxComponentPropagator behavior (caps component size) with an
    isolation zone around near-miss groups — WFC cannot place a symbol
    adjacent to a near-miss group if it matches that group's symbol.

    The component cap uses the shared _propagate_cluster_constraint helper (DRY).
    The isolation map is precomputed at init for O(1) lookup during propagation.
    """

    def __init__(
        self,
        max_component: int,
        protected_groups: list[NearMissGroup],
        board_config: BoardConfig,
    ) -> None:
        # Component cap threshold — components of max_component+1 or more are blocked
        self._threshold = max_component + 1
        # Precompute isolation map: for each position bordering a NM group,
        # track which symbols are forbidden (the group's symbol)
        self._forbidden: dict[Position, set[Symbol]] = {}
        for group in protected_groups:
            border = _compute_border(group.positions, board_config)
            for pos in border:
                if pos not in self._forbidden:
                    self._forbidden[pos] = set()
                self._forbidden[pos].add(group.symbol)

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Enforce component cap and near-miss isolation around resolved position."""
        # Component cap — reuse shared logic (DRY with MaxComponentPropagator)
        changed = _propagate_cluster_constraint(
            self._threshold, board, cells, position, board_config,
        )

        # Isolation — strip forbidden symbols from cells in the NM border zone
        for neighbor in orthogonal_neighbors(position, board_config):
            if neighbor not in cells or cells[neighbor].collapsed:
                continue
            forbidden = self._forbidden.get(neighbor)
            if not forbidden:
                continue
            for sym in forbidden:
                if cells[neighbor].remove(sym):
                    changed.add(neighbor)

        return changed

    def validate_placement(
        self,
        board: Board,
        position: Position,
        board_config: BoardConfig,
    ) -> bool:
        """Check component cap and isolation constraint for the placed symbol."""
        if not _validate_cluster_placement(
            self._threshold, board, position, board_config,
        ):
            return False
        # Check isolation — placed symbol must not be forbidden at this position
        forbidden = self._forbidden.get(position)
        if forbidden:
            sym = board.get(position)
            if sym in forbidden:
                return False
        return True
