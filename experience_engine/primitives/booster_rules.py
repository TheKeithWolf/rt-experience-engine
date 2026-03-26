"""Booster spawn, placement, and effect rules.

All thresholds, blast radii, immunity sets, and orientation tie-breaking
come from config. No hardcoded values.

Booster spawn hierarchy (from smallest to largest cluster):
  Wild (W): 7-8, Rocket (R): 9-10, Bomb (B): 11-12,
  Lightball (LB): 13-14, Super Lightball (SLB): 15-49
"""

from __future__ import annotations

import math

from collections import Counter
from typing import TYPE_CHECKING

from ..config.schema import BoardConfig, BoosterConfig, SymbolConfig
from .board import Board, Position, is_valid
from .symbols import Symbol, is_wild, symbol_from_name

if TYPE_CHECKING:
    from ..boosters.tracker import BoosterTracker


class BoosterRules:
    """Booster spawn, placement, and effect calculations.

    Constructed once with config, reused across spins.
    """

    __slots__ = (
        "_config",
        "_board_config",
        "_symbol_config",
        "_immune_to_rocket",
        "_immune_to_bomb",
        "_chain_initiators",
        "_standard_symbols",
        "_payout_rank",
    )

    def __init__(
        self,
        config: BoosterConfig,
        board_config: BoardConfig,
        symbol_config: SymbolConfig,
    ) -> None:
        self._config = config
        self._board_config = board_config
        self._symbol_config = symbol_config
        self._immune_to_rocket: frozenset[Symbol] = frozenset(
            symbol_from_name(n) for n in config.immune_to_rocket
        )
        self._immune_to_bomb: frozenset[Symbol] = frozenset(
            symbol_from_name(n) for n in config.immune_to_bomb
        )
        self._chain_initiators: frozenset[Symbol] = frozenset(
            symbol_from_name(n) for n in config.chain_initiators
        )
        # Pre-compute standard symbol set and payout rank for LB/SLB targeting
        self._standard_symbols: frozenset[Symbol] = frozenset(
            symbol_from_name(n) for n in symbol_config.standard
        )
        self._payout_rank: dict[Symbol, int] = {
            symbol_from_name(name): rank
            for name, rank in symbol_config.payout_rank
        }

    @property
    def immune_to_rocket(self) -> frozenset[Symbol]:
        """Symbols that rockets cannot clear."""
        return self._immune_to_rocket

    @property
    def immune_to_bomb(self) -> frozenset[Symbol]:
        """Symbols that bombs cannot clear."""
        return self._immune_to_bomb

    @property
    def chain_initiators(self) -> frozenset[Symbol]:
        """Only these booster types can initiate chain reactions."""
        return self._chain_initiators

    def booster_type_for_size(self, size: int) -> Symbol | None:
        """Determine which booster spawns for a cluster of the given size.

        Returns None if the cluster is too small to spawn any booster.
        """
        for threshold in self._config.spawn_thresholds:
            if threshold.min_size <= size <= threshold.max_size:
                return symbol_from_name(threshold.booster)
        return None

    def compute_centroid(self, positions: frozenset[Position]) -> Position:
        """Arithmetic mean of cluster positions, snapped to the nearest member.

        If the mean coordinates land exactly on a member position, return it.
        Otherwise return the member closest by Euclidean distance.
        """
        if not positions:
            raise ValueError("Cannot compute centroid of empty position set")

        mean_reel = sum(p.reel for p in positions) / len(positions)
        mean_row = sum(p.row for p in positions) / len(positions)

        # Snap to integer
        ideal = Position(round(mean_reel), round(mean_row))

        # If the rounded mean is a member, use it directly
        if ideal in positions:
            return ideal

        # Otherwise find the nearest member by Euclidean distance
        return min(
            positions,
            key=lambda p: math.sqrt(
                (p.reel - mean_reel) ** 2 + (p.row - mean_row) ** 2
            ),
        )

    def resolve_collision(
        self,
        centroid: Position,
        cluster_positions: frozenset[Position],
        occupied: frozenset[Position],
    ) -> Position:
        """Place the booster at centroid if unoccupied, else next-nearest cluster member.

        Iterates cluster members by distance from centroid to find the
        first unoccupied position.
        """
        if centroid not in occupied:
            return centroid

        mean_reel = sum(p.reel for p in cluster_positions) / len(cluster_positions)
        mean_row = sum(p.row for p in cluster_positions) / len(cluster_positions)

        # Sort by distance from the arithmetic mean (not from centroid)
        # to get a stable, predictable ordering
        sorted_members = sorted(
            cluster_positions,
            key=lambda p: math.sqrt(
                (p.reel - mean_reel) ** 2 + (p.row - mean_row) ** 2
            ),
        )
        for candidate in sorted_members:
            if candidate not in occupied:
                return candidate

        # All positions occupied — shouldn't happen in practice
        raise ValueError("All cluster positions are occupied — cannot place booster")

    def compute_rocket_orientation(self, positions: frozenset[Position]) -> str:
        """Determine rocket firing direction from cluster shape.

        Fires perpendicular to the dominant axis:
        - Taller than wide (row_span > col_span) → fires Horizontal ("H")
        - Wider than tall (col_span > row_span) → fires Vertical ("V")
        - Tie → config default (rocket_tie_orientation)
        """
        if not positions:
            return self._config.rocket_tie_orientation

        reels = [p.reel for p in positions]
        rows = [p.row for p in positions]
        col_span = max(reels) - min(reels) + 1
        row_span = max(rows) - min(rows) + 1

        if row_span > col_span:
            return "H"
        if col_span > row_span:
            return "V"
        return self._config.rocket_tie_orientation

    def rocket_path(self, pos: Position, orientation: str) -> frozenset[Position]:
        """All positions cleared by a rocket at the given position and orientation.

        "H" clears all cells in the same row. "V" clears all cells in the same column.
        """
        if orientation == "H":
            return frozenset(
                Position(reel, pos.row)
                for reel in range(self._board_config.num_reels)
            )
        # "V"
        return frozenset(
            Position(pos.reel, row)
            for row in range(self._board_config.num_rows)
        )

    def bomb_blast(self, pos: Position) -> frozenset[Position]:
        """All positions within Manhattan distance of bomb_blast_radius, clipped to board.

        radius=1 produces a 3x3 area centered on pos (minus out-of-bounds cells).
        """
        radius = self._config.bomb_blast_radius
        result: set[Position] = set()
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                candidate = Position(pos.reel + dc, pos.row + dr)
                if is_valid(candidate, self._board_config):
                    result.add(candidate)
        return frozenset(result)

    # ------------------------------------------------------------------
    # LB / SLB targeting — abundance-based symbol selection
    # ------------------------------------------------------------------

    @property
    def standard_symbols(self) -> frozenset[Symbol]:
        """Standard (cluster-forming) symbol set from config."""
        return self._standard_symbols

    def count_standard_symbols(self, board: Board) -> dict[Symbol, int]:
        """Count occurrences of each standard symbol currently on the board.

        Only counts symbols in the config's standard set — special symbols
        (wilds, scatters, boosters) are excluded.
        """
        counts: dict[Symbol, int] = Counter()
        for reel in range(self._board_config.num_reels):
            for row in range(self._board_config.num_rows):
                sym = board.get(Position(reel, row))
                if sym is not None and sym in self._standard_symbols:
                    counts[sym] += 1
        return counts

    def most_abundant_standard(
        self, board: Board, count: int = 1,
    ) -> tuple[Symbol, ...]:
        """Return the N most abundant standard symbols on the board.

        Tiebreaker: higher payout_rank wins (from config.symbols.payout_rank).
        This determines which symbol(s) LB/SLB will target when fired.

        Raises ValueError if fewer than `count` standard symbols exist on board.
        """
        symbol_counts = self.count_standard_symbols(board)
        if len(symbol_counts) < count:
            raise ValueError(
                f"Board has {len(symbol_counts)} distinct standard symbols, "
                f"but {count} requested"
            )

        # Sort by count descending, then by payout_rank descending for ties
        ranked = sorted(
            symbol_counts.keys(),
            key=lambda s: (symbol_counts[s], self._payout_rank.get(s, 0)),
            reverse=True,
        )
        return tuple(ranked[:count])


def place_booster(
    booster_sym: Symbol,
    position: Position,
    board: Board,
    tracker: BoosterTracker,
    orientation: str | None = None,
    source_cluster_index: int | None = None,
) -> None:
    """Write a spawned booster to the board and register non-wilds in the tracker.

    Wilds occupy the board only (they're regular symbols for cluster purposes).
    Non-wild boosters occupy the board AND register in the tracker — prevents
    gravity holes at spawn positions that would cascade into payout drift.
    """
    if is_wild(booster_sym):
        board.set(position, Symbol.W)
    else:
        # Non-wild boosters must occupy their cell to prevent gravity holes
        board.set(position, booster_sym)
        tracker.add(
            booster_sym, position,
            orientation=orientation,
            source_cluster_index=source_cluster_index,
        )
