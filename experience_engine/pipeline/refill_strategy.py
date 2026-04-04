"""Refill strategies for post-gravity empty cell population.

Protocol + two concrete strategies:
- ClusterSeekingRefill — biases symbol selection toward extending existing
  formations (used by cascade generator post-terminal booster phase)
- TerminalRefill — ensures no accidental clusters form in cosmetic refill
  (used by static instance generator)

Strategy selection is the caller's responsibility via composition — no
if/else dispatch inside this module.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ..config.schema import BoardConfig, RefillConfig
from ..primitives.board import Board, Position
from ..primitives.cluster_detection import max_component_size
from ..primitives.symbols import Symbol, is_wild, symbol_from_name
from .refill_scoring import score_symbols_by_adjacency


@runtime_checkable
class RefillStrategy(Protocol):
    """Produces refill entries for empty positions after gravity settles.

    Implementations choose symbols with different objectives
    (cluster-seeking vs terminal-safe) while returning the same
    tuple format consumed by GravityRecord and the event stream.
    """

    def fill(
        self,
        board: Board,
        empty_positions: Iterable[Position],
        rng: random.Random,
    ) -> tuple[tuple[int, int, str], ...]: ...


class ClusterSeekingRefill:
    """Biases refill toward extending existing board formations.

    Bottom-up fill order ensures deeper cells are placed first, so their
    symbols become neighbors for cells scored next — creating a chain
    effect that compounds adjacency bias across multiple rows.

    Probabilistic, not guaranteed — the caller (cascade generator) checks
    detect_clusters() after refill and handles both outcomes.
    """

    __slots__ = ("_board_config", "_standard_symbols", "_config")

    def __init__(
        self,
        board_config: BoardConfig,
        standard_symbols: tuple[str, ...],
        refill_config: RefillConfig,
    ) -> None:
        self._board_config = board_config
        # Convert once at construction — avoids per-fill repeated lookups
        self._standard_symbols = tuple(
            symbol_from_name(name) for name in standard_symbols
        )
        self._config = refill_config

    def fill(
        self,
        board: Board,
        empty_positions: Iterable[Position],
        rng: random.Random,
    ) -> tuple[tuple[int, int, str], ...]:
        empties = sorted(empty_positions, key=lambda p: -p.row)
        if not empties:
            return ()

        # Mutations track placements within this pass so each placed
        # symbol influences the scoring of subsequent cells
        working = board.copy()
        result: list[tuple[int, int, str]] = []

        for pos in empties:
            weights = score_symbols_by_adjacency(
                working, pos, self._standard_symbols,
                self._board_config,
                self._config.adjacency_boost,
                self._config.depth_scale,
            )
            symbols = list(weights.keys())
            symbol_weights = [weights[s] for s in symbols]
            chosen = rng.choices(symbols, weights=symbol_weights, k=1)[0]
            working.set(pos, chosen)
            result.append((pos.reel, pos.row, chosen.name))

        return tuple(result)


class TerminalRefill:
    """Refills empty cells while ensuring no cluster forms.

    For each empty cell, draws a random candidate and checks whether it
    creates a connected component reaching the cluster threshold. Retries
    up to a config-driven budget; if exhausted, scans all symbols and picks
    the one producing the smallest component (guaranteed termination).

    Reuses max_component_size from cluster_detection — same BFS used by
    NoClusterPropagator and _validate_cluster_placement.
    """

    __slots__ = (
        "_board_config", "_standard_symbols", "_min_cluster_size", "_config",
    )

    def __init__(
        self,
        board_config: BoardConfig,
        standard_symbols: tuple[str, ...],
        min_cluster_size: int,
        refill_config: RefillConfig,
    ) -> None:
        self._board_config = board_config
        self._standard_symbols = tuple(
            symbol_from_name(name) for name in standard_symbols
        )
        self._min_cluster_size = min_cluster_size
        self._config = refill_config

    def fill(
        self,
        board: Board,
        empty_positions: Iterable[Position],
        rng: random.Random,
    ) -> tuple[tuple[int, int, str], ...]:
        empties = list(empty_positions)
        if not empties:
            return ()

        working = board.copy()

        # Scan once for wild positions — wilds bridge groups so a refill
        # symbol adjacent to a wild could form a cluster through it
        wild_set = frozenset(
            Position(reel, row)
            for reel in range(self._board_config.num_reels)
            for row in range(self._board_config.num_rows)
            if working.get(Position(reel, row)) is not None
            and is_wild(working.get(Position(reel, row)))  # type: ignore[arg-type]
        )

        result: list[tuple[int, int, str]] = []

        for pos in empties:
            chosen = self._pick_safe_symbol(working, pos, wild_set, rng)
            working.set(pos, chosen)
            result.append((pos.reel, pos.row, chosen.name))

        return tuple(result)

    def _pick_safe_symbol(
        self,
        board: Board,
        pos: Position,
        wild_positions: frozenset[Position],
        rng: random.Random,
    ) -> Symbol:
        """Select a symbol that does not create a cluster at *pos*."""
        max_retries = self._config.terminal_max_retries

        for _ in range(max_retries):
            candidate = rng.choice(self._standard_symbols)
            board.set(pos, candidate)
            component = max_component_size(
                board, candidate, self._board_config,
                wild_positions=wild_positions,
            )
            if component < self._min_cluster_size:
                # Undo placement — caller will set the final symbol
                board.set(pos, None)
                return candidate
            # Undo failed candidate before next retry
            board.set(pos, None)

        # Retries exhausted — deterministic fallback: pick symbol with
        # smallest component (guaranteed to terminate with a valid result)
        best_sym = self._standard_symbols[0]
        best_size = self._min_cluster_size + 1  # sentinel above threshold

        for candidate in self._standard_symbols:
            board.set(pos, candidate)
            component = max_component_size(
                board, candidate, self._board_config,
                wild_positions=wild_positions,
            )
            if component < best_size:
                best_size = component
                best_sym = candidate
            board.set(pos, None)

        return best_sym
