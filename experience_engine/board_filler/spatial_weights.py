"""Spatial weight map — per-cell symbol weight suppression for gravity-aware WFC.

Four zone types bias WFC symbol selection away from same-tier symbols near
planned clusters, preventing unintended post-gravity merges. Each zone
applies a multiplicative suppression factor to specific symbols at specific
positions. Zones stack multiplicatively when they overlap.

All suppression values and the min-weight floor come from GravityWfcConfig —
zero hardcoded constants.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config.schema import BoardConfig, GravityWfcConfig, MasterConfig, SymbolConfig
from ..primitives.board import Position, orthogonal_neighbors
from ..primitives.symbols import Symbol, SymbolTier, symbols_in_tier, tier_of
from .propagators import compute_border

if TYPE_CHECKING:
    from ..step_reasoner.intent import StepIntent


@dataclass(frozen=True, slots=True)
class WeightZone:
    """A spatial region where specific symbols receive a weight multiplier.

    positions: cells in this zone
    adjustments: symbol → multiplier (applied to base weight)
    """

    positions: frozenset[Position]
    adjustments: dict[Symbol, float]

    def contains(self, pos: Position) -> bool:
        return pos in self.positions

    def apply(
        self, weights: dict[Symbol, float], min_weight: float
    ) -> dict[Symbol, float]:
        """Return new weights with adjustments applied, floored to min_weight.

        Only symbols present in both weights and adjustments are modified.
        """
        result = dict(weights)
        for sym, multiplier in self.adjustments.items():
            if sym in result:
                result[sym] = max(result[sym] * multiplier, min_weight)
        return result


class SpatialWeightMap:
    """Per-cell symbol weights computed from overlapping zone suppression.

    Given base weights and a list of zones, computes effective per-cell weights
    by stacking all containing zones multiplicatively. Results are cached per
    position — computed once, reused for every WFC collapse at that cell.
    """

    __slots__ = ("_base_weights", "_zones", "_min_weight", "_cache")

    def __init__(
        self,
        base_weights: dict[Symbol, float],
        zones: list[WeightZone],
        min_weight: float,
    ) -> None:
        self._base_weights = base_weights
        self._zones = zones
        self._min_weight = min_weight
        # Lazy cache — avoids computing weights for cells never collapsed
        self._cache: dict[Position, dict[Symbol, float]] = {}

    def get_weights(self, pos: Position) -> dict[Symbol, float]:
        """Return per-symbol weights for this cell, applying all containing zones."""
        cached = self._cache.get(pos)
        if cached is not None:
            return cached

        weights = dict(self._base_weights)
        for zone in self._zones:
            if zone.contains(pos):
                weights = zone.apply(weights, self._min_weight)

        self._cache[pos] = weights
        return weights


def build_weight_zones(
    intent: StepIntent,
    config: MasterConfig,
) -> list[WeightZone]:
    """Build spatial suppression zones from a StepIntent's cluster plan.

    Returns an empty list when planned_explosion is None (terminal/dead intents)
    — no spatial suppression needed for boards that won't explode.

    Four zone types (see GravityWfcConfig for multiplier semantics):
    1. Boundary: 1-cell border around cluster, suppresses same-tier symbols
    2. Extended neighborhood: BFS radius around cluster, softer suppression
    3. Compression columns: cells below cluster, prevents gravity stacking
    4. Strategic seed protection: neighbors of strategic cells, suppresses seed symbol
    """
    if intent.planned_explosion is None:
        return []

    gwfc = config.gravity_wfc
    if gwfc is None:
        return []

    zones: list[WeightZone] = []
    cluster_positions = frozenset(intent.constrained_cells.keys())
    board_config = config.board

    # Identify the primary cluster symbol and its tier
    primary_symbol = _primary_cluster_symbol(intent)
    if primary_symbol is None:
        return []

    try:
        primary_tier = tier_of(primary_symbol, config.symbols)
    except ValueError:
        return []

    # Same-tier symbols to suppress — all standard symbols in the cluster's tier
    tier_symbols = symbols_in_tier(primary_tier, config.symbols)
    tier_adjustments = {sym: gwfc.cluster_boundary_tier_suppression for sym in tier_symbols}

    # Zone 1: Boundary — 1-cell border around cluster positions
    boundary = compute_border(cluster_positions, board_config)
    if boundary:
        zones.append(WeightZone(
            positions=boundary,
            adjustments=tier_adjustments,
        ))

    # Zone 2: Extended neighborhood — BFS to radius, minus boundary (zone 1)
    extended = _compute_extended_neighborhood(
        cluster_positions, gwfc.extended_neighborhood_radius, board_config,
    )
    # Exclude zone 1 positions — they already get stronger suppression
    extended_only = extended - boundary
    if extended_only:
        extended_adjustments = {
            sym: gwfc.extended_neighborhood_suppression for sym in tier_symbols
        }
        zones.append(WeightZone(
            positions=frozenset(extended_only),
            adjustments=extended_adjustments,
        ))

    # Zone 3: Compression columns — cells below cluster in each column,
    # where gravity will stack symbols after explosion
    compression = _compute_compression_columns(cluster_positions, board_config)
    if compression:
        compression_adjustments = {
            sym: gwfc.compression_column_suppression for sym in tier_symbols
        }
        zones.append(WeightZone(
            positions=frozenset(compression),
            adjustments=compression_adjustments,
        ))

    # Zone 4: Strategic seed protection — neighbors of strategic cells,
    # suppressing each seed's specific symbol to prevent duplication
    if intent.strategic_cells:
        seed_border: set[Position] = set()
        # Aggregate all seed symbols for adjustment map
        seed_adjustments: dict[Symbol, float] = {}
        for pos, sym in intent.strategic_cells.items():
            for neighbor in orthogonal_neighbors(pos, board_config):
                if neighbor not in cluster_positions:
                    seed_border.add(neighbor)
            seed_adjustments[sym] = gwfc.strategic_cell_neighbor_suppression

        if seed_border:
            zones.append(WeightZone(
                positions=frozenset(seed_border),
                adjustments=seed_adjustments,
            ))

    return zones


def _primary_cluster_symbol(intent: StepIntent) -> Symbol | None:
    """Extract the dominant symbol from the intent's constrained cells.

    Uses the most frequent symbol — handles multi-cluster intents where
    constrained_cells may contain scatters or mixed symbols.
    """
    if not intent.constrained_cells:
        return None

    counts = Counter(intent.constrained_cells.values())
    # Filter out scatter/wild/booster symbols — only standard symbols form clusters
    standard_counts = {
        sym: count for sym, count in counts.items()
        if sym.value <= 7  # L1-H3 are values 1-7
    }
    if not standard_counts:
        return None

    return max(standard_counts, key=standard_counts.get)


def _compute_extended_neighborhood(
    positions: frozenset[Position],
    radius: int,
    board_config: BoardConfig,
) -> frozenset[Position]:
    """BFS from cluster positions to radius depth, excluding the cluster itself.

    Returns all cells reachable within `radius` orthogonal steps from any
    cluster position, minus the cluster positions themselves.
    """
    visited: set[Position] = set(positions)
    frontier: deque[tuple[Position, int]] = deque()

    # Seed BFS from all cluster edge positions
    for pos in positions:
        for neighbor in orthogonal_neighbors(pos, board_config):
            if neighbor not in visited:
                frontier.append((neighbor, 1))
                visited.add(neighbor)

    result: set[Position] = set()

    while frontier:
        pos, depth = frontier.popleft()
        result.add(pos)
        if depth < radius:
            for neighbor in orthogonal_neighbors(pos, board_config):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, depth + 1))

    return frozenset(result)


def _compute_compression_columns(
    cluster_positions: frozenset[Position],
    board_config: BoardConfig,
) -> frozenset[Position]:
    """Cells below the lowest cluster cell in each column.

    After explosion, gravity pulls symbols downward in these columns.
    Suppressing same-tier symbols here prevents post-settle merges
    where falling symbols stack into unintended clusters.
    """
    # Find the lowest cluster cell (highest row index) per column
    lowest_per_column: dict[int, int] = {}
    for pos in cluster_positions:
        current = lowest_per_column.get(pos.reel)
        if current is None or pos.row > current:
            lowest_per_column[pos.reel] = pos.row

    result: set[Position] = set()
    for reel, lowest_row in lowest_per_column.items():
        # All cells below the lowest cluster cell in this column
        for row in range(lowest_row + 1, board_config.num_rows):
            pos = Position(reel, row)
            if pos not in cluster_positions:
                result.add(pos)

    return frozenset(result)


def build_reserve_zone(
    positions: frozenset[Position],
    suppression_multiplier: float,
    symbol_config: SymbolConfig,
) -> WeightZone:
    """Reserve zone for future-step demand — suppresses all standard symbols.

    Cells in the reserve zone are where the next step needs clear space
    for cluster formation. Returns a WeightZone that applies uniform
    suppression to all standard symbols, stacking multiplicatively with
    existing zones in SpatialWeightMap.
    """
    all_symbols = list(symbols_in_tier(SymbolTier.ANY, symbol_config))
    adjustments = {s: suppression_multiplier for s in all_symbols}
    return WeightZone(positions=positions, adjustments=adjustments)
