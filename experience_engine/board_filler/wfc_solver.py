"""Wave Function Collapse board filler — fills free cells with standard symbols.

Given a board with some cells pre-filled (pinned by the CSP solver), WFC
populates remaining cells so that no unintended clusters form. The algorithm:
min-entropy cell selection → weighted symbol collapse → constraint propagation
→ backtrack on contradiction.

All tweakable constants come from MasterConfig — zero hardcoded values.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from ..config.schema import MasterConfig
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.symbols import Symbol, symbol_from_name
from .cell_state import CellState
from .propagators import (
    MaxComponentPropagator,
    NoClusterPropagator,
    NoSpecialSymbolPropagator,
    Propagator,
)


class FillFailed(Exception):
    """Raised when WFC exhausts its backtrack budget without finding a valid fill."""


@dataclass(frozen=True, slots=True)
class _DecisionRecord:
    """Snapshot of solver state at a decision point — used for backtracking."""

    position: Position
    chosen_symbol: Symbol
    board_snapshot: Board
    cells_snapshot: dict[Position, set[Symbol]]


class WFCBoardFiller:
    """Wave Function Collapse board filler implementing the BoardFiller protocol.

    Fills unconstrained cells with standard symbols while preventing unintended
    clusters via pluggable propagators. All configuration from MasterConfig.
    """

    def __init__(self, config: MasterConfig, use_defaults: bool = True) -> None:
        # All values sourced from config — zero hardcoded constants
        self._board_config = config.board
        self._symbol_config = config.symbols
        self._max_backtracks = config.solvers.wfc_max_backtracks
        self._min_weight = config.solvers.wfc_min_symbol_weight

        # Standard symbols derived from config — the only valid WFC fill choices
        self._standard_symbols: frozenset[Symbol] = frozenset(
            symbol_from_name(name) for name in config.symbols.standard
        )

        if use_defaults:
            # Default propagators with thresholds from config — used when
            # the caller doesn't provide step-specific propagators.
            # nm_fill_cap: one below near-miss size — prevents WFC from
            # accidentally creating NM-sized components.
            nm_fill_cap = config.board.min_cluster_size - 2
            self._propagators: list[Propagator] = [
                NoSpecialSymbolPropagator(config.symbols),
                NoClusterPropagator(config.board.min_cluster_size),
                MaxComponentPropagator(nm_fill_cap),
            ]
        else:
            # Caller will add all propagators via add_propagator() —
            # used by StepExecutor where strategies specify the complete set
            self._propagators: list[Propagator] = []

    def add_propagator(self, propagator: Propagator) -> None:
        """Add an additional propagator (e.g., MaxComponentPropagator for dead boards).

        Threshold for MaxComponentPropagator comes from the archetype signature,
        not hardcoded — the caller is responsible for passing the correct value.
        """
        self._propagators.append(propagator)

    def fill(
        self,
        board: Board,
        pinned: frozenset[Position],
        constraints: dict[str, object] | None = None,
        rng: random.Random | None = None,
        weights: dict[Symbol, float] | None = None,
    ) -> Board:
        """Fill all empty cells on the board via WFC.

        Args:
            board: Input board with some cells pre-filled (pinned).
            pinned: Positions that must not be modified (CONSTRAINT-WFC-5).
            constraints: Reserved for future phases — additional constraint hints.
            rng: Random instance for deterministic reproducibility.
            weights: Per-symbol selection weights from variance engine
                     (CONSTRAINT-WFC-3). Floored to config.solvers.wfc_min_symbol_weight.
                     Defaults to uniform if None.

        Returns:
            A new Board with all cells filled.

        Raises:
            FillFailed: If backtrack limit exceeded (CONSTRAINT-WFC-4).
        """
        if rng is None:
            rng = random.Random()

        result = board.copy()

        # Build cell states for every empty, non-pinned position
        cells: dict[Position, CellState] = {}
        for pos in result.all_positions():
            if result.get(pos) is None and pos not in pinned:
                cells[pos] = CellState(set(self._standard_symbols))

        # No free cells — nothing to fill
        if not cells:
            return result

        # Prepare effective weights — floor to min_weight from config
        effective_weights = self._compute_effective_weights(weights)

        # Initial propagation from all already-filled positions
        self._initial_propagation(result, cells)

        # Solve loop with backtracking
        return self._solve(result, cells, effective_weights, rng)

    def _compute_effective_weights(
        self, weights: dict[Symbol, float] | None
    ) -> dict[Symbol, float]:
        """Build per-symbol weights, flooring to config minimum (never zero).

        When weights is None, uniform weight derived from config symbol count.
        When provided, each value is floored to self._min_weight.
        """
        if weights is None:
            # Uniform weight derived from number of standard symbols in config
            uniform = 1.0 / len(self._standard_symbols)
            return {sym: uniform for sym in self._standard_symbols}

        # Floor provided weights to config minimum
        return {
            sym: max(w, self._min_weight) for sym, w in weights.items()
            if sym in self._standard_symbols
        }

    def _initial_propagation(
        self, board: Board, cells: dict[Position, CellState]
    ) -> None:
        """Run all propagators from every pre-filled position.

        Seeds the BFS work queue with all filled positions — their symbols
        constrain adjacent free cells. Raises FillFailed on contradiction.
        """
        work_queue: deque[Position] = deque()

        # Seed with all positions that are already filled on the board
        for pos in board.all_positions():
            if board.get(pos) is not None:
                work_queue.append(pos)

        # Also seed with all free positions for the NoSpecialSymbol init pass
        for pos in cells:
            work_queue.append(pos)

        visited: set[Position] = set()
        while work_queue:
            pos = work_queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)

            for propagator in self._propagators:
                changed = propagator.propagate(
                    board, cells, pos, self._board_config
                )
                for changed_pos in changed:
                    if cells[changed_pos].entropy == 0:
                        raise FillFailed(
                            "Contradiction during initial propagation"
                        )
                    work_queue.append(changed_pos)

    def _solve(
        self,
        board: Board,
        cells: dict[Position, CellState],
        weights: dict[Symbol, float],
        rng: random.Random,
    ) -> Board:
        """Main WFC solve loop with min-entropy selection and backtracking."""
        decision_stack: list[_DecisionRecord] = []
        backtrack_count = 0

        while True:
            # Select uncollapsed cell with minimum entropy — random tie-breaking
            # gives each retry a different search path, improving success rate
            target = self._select_min_entropy(cells, board, rng)
            if target is None:
                # All cells collapsed — success
                return board

            # Snapshot state for potential backtrack
            board_snapshot = board.copy()
            cells_snapshot = {
                pos: cs.snapshot()
                for pos, cs in cells.items()
                if not cs.collapsed
            }

            # Weighted collapse — pick a symbol from remaining possibilities
            symbol = self._weighted_select(
                cells[target].possibilities, weights, rng
            )
            cells[target].collapse_to(symbol)
            board.set(target, symbol)

            # Post-collapse safety net — catches stale possibilities where a
            # same-symbol chain grew from the far end since this cell was last pruned
            if not self._validate_placement(board, target):
                contradiction = True
            else:
                # Propagate constraints from the collapsed cell to neighbors
                contradiction = self._propagate_from(board, cells, target)

            if contradiction:
                backtrack_count += 1
                if backtrack_count > self._max_backtracks:
                    raise FillFailed(
                        f"Exceeded max backtracks ({self._max_backtracks})"
                    )

                # Restore state from snapshot
                board = self._restore_board(board, board_snapshot)
                self._restore_cells(cells, cells_snapshot)

                # Remove the failed symbol from the target cell
                cells[target].remove(symbol)

                # If target cell is exhausted, backtrack further up the stack
                while cells[target].entropy == 0 and decision_stack:
                    record = decision_stack.pop()
                    board = self._restore_board(board, record.board_snapshot)
                    self._restore_cells(cells, record.cells_snapshot)
                    target = record.position
                    cells[target].remove(record.chosen_symbol)

                if cells[target].entropy == 0:
                    raise FillFailed(
                        "All possibilities exhausted during backtracking"
                    )
                continue

            # Success — push decision to stack for potential future backtrack
            decision_stack.append(
                _DecisionRecord(target, symbol, board_snapshot, cells_snapshot)
            )

    def _select_min_entropy(
        self,
        cells: dict[Position, CellState],
        board: Board,
        rng: random.Random,
    ) -> Position | None:
        """Select the uncollapsed cell with the fewest remaining possibilities.

        Among cells tied at minimum entropy, biases toward cells adjacent to
        filled positions (higher constraint density) — resolving heavily
        constrained regions first catches contradictions early (fail-fast).
        Random tie-breaking across retries gives diverse search paths.
        """
        # Find minimum entropy among uncollapsed cells
        min_entropy = float("inf")
        for cs in cells.values():
            if not cs.collapsed and cs.entropy < min_entropy:
                min_entropy = cs.entropy

        if min_entropy == float("inf"):
            return None

        # Collect all cells at min entropy
        tied: list[Position] = [
            pos for pos, cs in cells.items()
            if not cs.collapsed and cs.entropy == min_entropy
        ]

        if len(tied) == 1:
            return tied[0]

        # Bias toward cells near filled positions — more constrained regions
        # should be resolved first to catch contradictions early
        neighbors_fn = orthogonal_neighbors
        weights = []
        for pos in tied:
            filled_neighbor_count = sum(
                1 for n in neighbors_fn(pos, self._board_config)
                if board.get(n) is not None
            )
            # Base weight 1, +1 per filled neighbor — cells deep in constrained
            # regions get higher selection probability
            weights.append(1 + filled_neighbor_count)

        return rng.choices(tied, weights=weights, k=1)[0]

    def _weighted_select(
        self,
        possibilities: frozenset[Symbol],
        weights: dict[Symbol, float],
        rng: random.Random,
    ) -> Symbol:
        """Choose a symbol from possibilities using weighted random selection.

        Symbols sorted by integer value for deterministic ordering regardless
        of frozenset iteration order. Weights floored to config minimum.
        """
        # Sort by symbol value for deterministic iteration
        sorted_syms = sorted(possibilities, key=lambda s: s.value)

        # Build cumulative weight list
        cumulative: list[float] = []
        total = 0.0
        for sym in sorted_syms:
            w = weights.get(sym, self._min_weight)
            total += w
            cumulative.append(total)

        # Weighted random selection
        r = rng.random() * total
        for i, cum_w in enumerate(cumulative):
            if r <= cum_w:
                return sorted_syms[i]

        # Floating-point edge case — return last symbol
        return sorted_syms[-1]

    def _validate_placement(self, board: Board, position: Position) -> bool:
        """Check all propagators accept the just-collapsed cell's placement.

        Standard WFC propagation only prunes NEIGHBORS of a collapsed cell.
        If a same-symbol chain grows from the far end, previously-approved
        possibilities become stale. This post-collapse check catches the gap.
        """
        return all(
            p.validate_placement(board, position, self._board_config)
            for p in self._propagators
        )

    def _propagate_from(
        self,
        board: Board,
        cells: dict[Position, CellState],
        origin: Position,
    ) -> bool:
        """BFS propagation from a just-collapsed cell.

        Returns True if a contradiction was detected (any cell reaches entropy 0).
        """
        work_queue: deque[Position] = deque([origin])
        visited: set[Position] = set()

        while work_queue:
            pos = work_queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)

            for propagator in self._propagators:
                changed = propagator.propagate(
                    board, cells, pos, self._board_config
                )
                for changed_pos in changed:
                    if cells[changed_pos].entropy == 0:
                        return True  # Contradiction
                    work_queue.append(changed_pos)

        return False

    @staticmethod
    def _restore_board(current: Board, snapshot: Board) -> Board:
        """Restore board state from a snapshot.

        Returns the snapshot board directly — the solver replaces its reference.
        """
        return snapshot

    @staticmethod
    def _restore_cells(
        cells: dict[Position, CellState],
        snapshot: dict[Position, set[Symbol]],
    ) -> None:
        """Restore cell states from snapshot possibilities."""
        for pos, possibilities in snapshot.items():
            cells[pos].restore(possibilities)
