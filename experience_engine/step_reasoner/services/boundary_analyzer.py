"""Boundary analyzer — determines merge risk between empty cells and survivors.

When a cluster is placed in empty cells on a settled board, surviving symbols
adjacent to the empty region may merge with the planned cluster, creating
larger-than-intended clusters. This module computes the merge risk ONCE per
step so that symbol selection, position selection, and merge handling all
reference the same analysis (DRY).

Used by ClusterBuilder for merge-aware cluster placement.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

from ...config.schema import BoardConfig, SymbolConfig
from ...primitives.board import Position, orthogonal_neighbors
from ...primitives.symbols import Symbol, is_standard
from ..context import BoardContext


@dataclass(frozen=True, slots=True)
class SurvivorComponent:
    """A connected group of surviving same-symbol cells adjacent to the empty region.

    contact_points are the EMPTY cells that touch this component — these are
    the positions where placing the component's symbol would trigger a merge.
    """

    symbol: Symbol
    positions: frozenset[Position]
    contact_points: frozenset[Position]
    size: int


@dataclass(frozen=True, slots=True)
class BoundaryAnalysis:
    """What surviving symbols are adjacent to a set of empty cells.

    Computed once for the empty region. Symbol selection, position selection,
    and merge handling all reference this single analysis.
    """

    # Per-symbol: which empty cells are adjacent to a survivor of that symbol?
    merge_risk: dict[Symbol, frozenset[Position]]

    # Per-symbol: connected survivor groups reachable from the empty region
    survivor_components: dict[Symbol, list[SurvivorComponent]]

    # Symbols with zero merge risk — completely safe to use
    safe_symbols: frozenset[Symbol]

    # Symbols that would merge — maps to total survivor count (caller adds planned size)
    acceptable_merge_symbols: dict[Symbol, int]


class BoundaryAnalyzer:
    """Analyzes the boundary between empty cells and surviving symbols.

    Single source of truth for merge risk computation.
    Injected into ClusterBuilder — strategies never call this directly.
    """

    __slots__ = ("_board_config", "_symbol_config")

    def __init__(
        self,
        board_config: BoardConfig,
        symbol_config: SymbolConfig,
    ) -> None:
        self._board_config = board_config
        self._symbol_config = symbol_config

    def analyze(
        self,
        context: BoardContext,
        empty_cells: frozenset[Position],
        extra_occupied: frozenset[Position] | None = None,
        extra_symbols: dict[Position, Symbol] | None = None,
    ) -> BoundaryAnalysis:
        """Compute boundary analysis for an empty region.

        Scans all empty cells' neighbors to identify surviving symbols,
        then BFS-traces each symbol's connected components from the boundary.

        extra_occupied: positions to exclude from empty_cells — e.g., cluster 1's
        cells when analyzing merge risk for cluster 2 in multi-cluster placement.
        extra_symbols: position→symbol for cells not yet on the real board (e.g.,
        previously-placed clusters in the same step). These are treated as
        survivors for merge risk computation so cluster 2 sees cluster 1's
        symbols as merge hazards.
        """
        # Effective empty region excludes extra_occupied (previously placed clusters)
        effective_empty = empty_cells - extra_occupied if extra_occupied else empty_cells
        effective_extras = extra_symbols or {}

        # Per-symbol: which empty cells neighbor a survivor of that symbol?
        merge_risk: dict[Symbol, set[Position]] = {}

        for empty_pos in effective_empty:
            for neighbor in orthogonal_neighbors(empty_pos, self._board_config):
                if neighbor in effective_empty:
                    continue
                # Check real board first, then extra_symbols for pending placements
                sym = context.board.get(neighbor)
                if sym is None:
                    sym = effective_extras.get(neighbor)
                if sym is not None and is_standard(sym, self._symbol_config):
                    merge_risk.setdefault(sym, set()).add(empty_pos)

        # BFS to find connected survivor components for each risky symbol
        survivor_components: dict[Symbol, list[SurvivorComponent]] = {}
        for sym, contact_empties in merge_risk.items():
            components = self._find_reachable_survivors(
                context, sym, contact_empties, effective_empty,
                extra_symbols=effective_extras,
            )
            survivor_components[sym] = components

        # Classify: standard symbols not in merge_risk are safe
        all_standard = frozenset(
            s for s in Symbol if is_standard(s, self._symbol_config)
        )
        safe_symbols = all_standard - frozenset(merge_risk.keys())

        # For risky symbols, total survivor count (caller adds planned size to get merged total)
        acceptable: dict[Symbol, int] = {}
        for sym, components in survivor_components.items():
            acceptable[sym] = sum(c.size for c in components)

        return BoundaryAnalysis(
            merge_risk={s: frozenset(p) for s, p in merge_risk.items()},
            survivor_components=survivor_components,
            safe_symbols=safe_symbols,
            acceptable_merge_symbols=acceptable,
        )

    def _find_reachable_survivors(
        self,
        context: BoardContext,
        symbol: Symbol,
        contact_empties: set[Position],
        empty_region: frozenset[Position],
        extra_symbols: dict[Position, Symbol] | None = None,
    ) -> list[SurvivorComponent]:
        """BFS from contact empties into surviving same-symbol cells.

        Finds all connected components of `symbol` that touch the empty region.
        Each component tracks which empty cells it contacts.

        extra_symbols positions are treated as survivors — previously-placed
        clusters in multi-cluster placement appear as occupied same-symbol
        cells that BFS can traverse into.
        """
        extras = extra_symbols or {}
        visited: set[Position] = set()
        components: list[SurvivorComponent] = []

        def _symbol_at(pos: Position) -> Symbol | None:
            """Check real board first, then extra_symbols for pending placements."""
            sym = context.board.get(pos)
            if sym is None:
                return extras.get(pos)
            return sym

        for empty_pos in contact_empties:
            for neighbor in orthogonal_neighbors(empty_pos, self._board_config):
                if neighbor in visited or neighbor in empty_region:
                    continue
                if _symbol_at(neighbor) is not symbol:
                    continue

                # BFS to find the full connected component
                component_positions: set[Position] = set()
                contacts: set[Position] = set()
                queue: collections.deque[Position] = collections.deque([neighbor])

                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)
                    component_positions.add(current)

                    for adj in orthogonal_neighbors(current, self._board_config):
                        if adj in empty_region:
                            contacts.add(adj)
                        elif (
                            adj not in visited
                            and _symbol_at(adj) is symbol
                        ):
                            queue.append(adj)

                if component_positions:
                    components.append(SurvivorComponent(
                        symbol=symbol,
                        positions=frozenset(component_positions),
                        contact_points=frozenset(contacts),
                        size=len(component_positions),
                    ))

        return components
