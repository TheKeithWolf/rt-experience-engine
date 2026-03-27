"""WFC constraint propagators — remove impossible symbols from cell possibilities.

Each propagator encodes one constraint type. The WFC solver runs all propagators
after each cell collapse to prune the search space via a BFS work queue.

Propagators are pluggable — new constraints can be added without modifying the solver.
"""

from __future__ import annotations

from collections import deque
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
    wild_positions: frozenset[Position] | None = None,
) -> set[Position]:
    """Shared logic for cluster-size propagators (DRY).

    For each uncollapsed neighbor of `position`, checks whether placing any of
    its remaining possibilities would create a connected component >= threshold.
    If so, removes that symbol from the neighbor's possibilities.

    Uses a fast adjacency pre-check before the expensive BFS — if a symbol has
    zero same-symbol neighbors at a position, it cannot form a component >= threshold
    (since isolated placements start at size 1).

    wild_positions — forwarded to max_component_size so wilds count as same-symbol
    during BFS, matching detect_clusters() semantics.

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
                board, sym, board_config, extra=frozenset({neighbor}),
                wild_positions=wild_positions,
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
    wild_positions: frozenset[Position] | None = None,
) -> bool:
    """Check if the symbol at position forms a component below threshold.

    Post-collapse safety net — catches stale possibilities where a same-symbol
    chain grew from the far end since this cell's possibilities were last pruned.
    Reuses cluster_detection.max_component_size (DRY — single BFS algorithm).
    """
    sym = board.get(position)
    if sym is None:
        return True
    return max_component_size(
        board, sym, board_config, wild_positions=wild_positions,
    ) < threshold


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

    wild_positions — when provided, wilds count as same-symbol during BFS,
    preventing wild-mediated extensions past the threshold. This subsumes
    WildBridgePropagator for terminal fills (bridging is just extension past
    threshold via a wild neighbor).
    """

    def __init__(
        self,
        max_size: int,
        wild_positions: frozenset[Position] | None = None,
    ) -> None:
        # Prevent component >= max_size + 1 (reuse same threshold logic)
        self._threshold = max_size + 1
        self._wild_positions = wild_positions

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Prune neighbor possibilities that would form components > max_size."""
        return _propagate_cluster_constraint(
            self._threshold, board, cells, position, board_config,
            wild_positions=self._wild_positions,
        )

    def validate_placement(
        self, board: Board, position: Position, board_config: BoardConfig,
    ) -> bool:
        """Check collapsed cell doesn't form a component > max_size."""
        return _validate_cluster_placement(
            self._threshold, board, position, board_config,
            wild_positions=self._wild_positions,
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


def compute_border(
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

    Uses the shared compute_border() helper (DRY with NearMissAwareDeadPropagator).
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
        self._border = compute_border(cluster_positions, board_config)

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

    Combines MaxComponentPropagator behavior (caps component size) with a
    two-layer isolation zone around near-miss groups:

    1-hop border: WFC cannot place the NM symbol directly adjacent to the group.
    2-hop border: WFC cannot place the NM symbol at a 2-hop position if doing so
    would bridge through an intermediate 1-hop cell already holding the NM symbol.

    The 2-hop layer prevents the absorption scenario where WFC fills a cell two
    hops out, then later fills the intermediate cell with the same symbol,
    merging the near-miss group into a cluster-sized component.

    The component cap uses the shared _propagate_cluster_constraint helper (DRY).
    All isolation maps are precomputed at init for O(1) lookup during propagation.
    """

    def __init__(
        self,
        max_component: int,
        protected_groups: list[NearMissGroup],
        board_config: BoardConfig,
    ) -> None:
        # Component cap threshold — components of max_component+1 or more are blocked
        self._threshold = max_component + 1
        # 1-hop border: positions directly adjacent to any NM group → forbidden symbols
        self._forbidden: dict[Position, set[Symbol]] = {}
        # 2-hop border: positions two hops from any NM group → symbols that could bridge
        self._extended_border: dict[Position, set[Symbol]] = {}
        # 1-hop border positions per symbol — for bridge detection in validate_placement
        self._border_positions: dict[Symbol, frozenset[Position]] = {}

        for group in protected_groups:
            border = compute_border(group.positions, board_config)

            # 1-hop isolation map
            for pos in border:
                if pos not in self._forbidden:
                    self._forbidden[pos] = set()
                self._forbidden[pos].add(group.symbol)

            # Track border positions per symbol for bridge detection
            existing = self._border_positions.get(group.symbol, frozenset())
            self._border_positions[group.symbol] = existing | border

            # 2-hop border: cells adjacent to the 1-hop border but not in the
            # group or the 1-hop border itself — these are where bridge formation
            # can start if the intermediate 1-hop cell holds the NM symbol
            group_and_border = group.positions | border
            outer_border = compute_border(group_and_border, board_config) - group_and_border
            for pos in outer_border:
                if pos not in self._extended_border:
                    self._extended_border[pos] = set()
                self._extended_border[pos].add(group.symbol)

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Enforce component cap, 1-hop isolation, and 2-hop bridge prevention."""
        # Component cap — reuse shared logic (DRY with MaxComponentPropagator)
        changed = _propagate_cluster_constraint(
            self._threshold, board, cells, position, board_config,
        )

        # 1-hop isolation — strip forbidden symbols from cells in the NM border zone
        for neighbor in orthogonal_neighbors(position, board_config):
            if neighbor not in cells or cells[neighbor].collapsed:
                continue
            forbidden = self._forbidden.get(neighbor)
            if not forbidden:
                continue
            for sym in forbidden:
                if cells[neighbor].remove(sym):
                    changed.add(neighbor)

        # 2-hop bridge prevention: if the just-collapsed cell is at the 1-hop
        # border and holds an NM symbol, prune that symbol from its uncollapsed
        # neighbors in the 2-hop ring to prevent bridge formation through this cell
        placed_sym = board.get(position)
        if placed_sym is not None and position in self._forbidden:
            if placed_sym in self._forbidden[position]:
                for neighbor in orthogonal_neighbors(position, board_config):
                    if neighbor not in cells or cells[neighbor].collapsed:
                        continue
                    extended = self._extended_border.get(neighbor)
                    if extended and placed_sym in extended:
                        if cells[neighbor].remove(placed_sym):
                            changed.add(neighbor)

        return changed

    def validate_placement(
        self,
        board: Board,
        position: Position,
        board_config: BoardConfig,
    ) -> bool:
        """Check component cap, 1-hop isolation, and 2-hop bridge constraint."""
        if not _validate_cluster_placement(
            self._threshold, board, position, board_config,
        ):
            return False
        # 1-hop isolation — placed symbol must not be forbidden at this position
        forbidden = self._forbidden.get(position)
        if forbidden:
            sym = board.get(position)
            if sym in forbidden:
                return False
        # 2-hop bridge detection — reject if placing the NM symbol here would
        # connect through an intermediate 1-hop border cell that already holds
        # the same symbol, creating a path into the protected NM group
        extended = self._extended_border.get(position)
        if extended:
            sym = board.get(position)
            if sym is not None and sym in extended:
                border_for_sym = self._border_positions.get(sym, frozenset())
                for neighbor in orthogonal_neighbors(position, board_config):
                    if neighbor in border_for_sym and board.get(neighbor) is sym:
                        return False
        return True


# ---------------------------------------------------------------------------
# Post-Gravity Propagator
# ---------------------------------------------------------------------------


def _virtual_component_size(
    board: Board,
    start: Position,
    symbol: Symbol,
    virtual_neighbors_fn,
    wild_positions: frozenset[Position] | None = None,
) -> int:
    """BFS component size using virtual (post-gravity) adjacency.

    Counts how many collapsed cells hold ``symbol`` and are reachable from
    ``start`` via virtual_neighbors. Wild positions are treated as
    same-symbol — matching detect_clusters() union-find semantics where
    wilds bridge adjacent standard-symbol groups (C-COMPAT-2).

    Includes ``start`` itself in the count (assumes it would hold ``symbol``).

    Separate from max_component_size in cluster_detection.py because that
    function operates on physical adjacency — this one uses the post-gravity
    virtual graph via a neighbor function parameter.
    """
    wild_set = wild_positions or frozenset()
    visited: set[Position] = {start}
    queue = deque([start])
    count = 1  # start position counted

    while queue:
        pos = queue.popleft()
        for neighbor in virtual_neighbors_fn(pos):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            if board.get(neighbor) is symbol or neighbor in wild_set:
                count += 1
                queue.append(neighbor)

    return count


class PostGravityPropagator:
    """Prevents unintended clusters in post-gravity space.

    Standard propagators check physical (pre-gravity) adjacency. This
    propagator checks virtual adjacency — which cells will be neighbors
    AFTER gravity settles. If placing a symbol at a cell would create a
    virtual component >= threshold, that symbol is pruned.

    Implements the Propagator protocol (LSP) — the WFC solver treats it
    identically to NoClusterPropagator.
    """

    __slots__ = ("_virtual_neighbors_fn", "_threshold", "_wild_positions")

    def __init__(
        self,
        virtual_neighbors_fn,
        threshold: int,
        wild_positions: frozenset[Position] | None = None,
    ) -> None:
        # Callable: Position → list[Position] — from PostGravityAdjacency
        self._virtual_neighbors_fn = virtual_neighbors_fn
        # Cluster prevention threshold — from config.board.min_cluster_size
        self._threshold = threshold
        # Predicted wild landings — wilds count as same-symbol during BFS,
        # matching detect_clusters() semantics (C-COMPAT-2)
        self._wild_positions = wild_positions or frozenset()

    def propagate(
        self,
        board: Board,
        cells: dict[Position, CellState],
        position: Position,
        board_config: BoardConfig,
    ) -> set[Position]:
        """Prune symbols from virtual neighbors that would form post-gravity clusters.

        For each uncollapsed virtual neighbor of the just-collapsed position,
        checks whether placing any remaining possibility would create a
        virtual component >= threshold. If so, removes that symbol.
        """
        changed: set[Position] = set()
        for neighbor in self._virtual_neighbors_fn(position):
            if neighbor not in cells or cells[neighbor].collapsed:
                continue

            to_remove: list[Symbol] = []
            for sym in cells[neighbor].possibilities:
                # Fast pre-check: if no same-symbol or wild virtual neighbors
                # exist, placing this symbol creates an isolated component of
                # size 1 — skip the expensive BFS
                has_same = any(
                    board.get(vn) is sym or vn in self._wild_positions
                    for vn in self._virtual_neighbors_fn(neighbor)
                )
                if self._threshold > 1 and not has_same:
                    continue

                # Full BFS: would placing sym at neighbor exceed threshold
                # in post-gravity space?
                component = _virtual_component_size(
                    board, neighbor, sym, self._virtual_neighbors_fn,
                    self._wild_positions,
                )
                if component >= self._threshold:
                    to_remove.append(sym)

            for sym in to_remove:
                if cells[neighbor].remove(sym):
                    changed.add(neighbor)

        return changed

    def validate_placement(
        self,
        board: Board,
        position: Position,
        board_config: BoardConfig,
    ) -> bool:
        """Check if the symbol at position forms a virtual component below threshold."""
        sym = board.get(position)
        if sym is None:
            return True
        return _virtual_component_size(
            board, position, sym, self._virtual_neighbors_fn,
            self._wild_positions,
        ) < self._threshold
