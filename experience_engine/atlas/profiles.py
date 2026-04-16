"""Column profile enumeration — pure combinatorics over cluster sizes.

The atlas indexes one topology per (composition of cluster_size over num_reels,
depth_band). This module produces those composition tuples. All board
dimensions, cluster sizes, and depth bands are supplied by the caller —
no hardcoded sizes or reel counts.

A ColumnProfile pairs each composition with a depth band name so that two
profiles with the same counts but different depth bands key different
entries in the atlas.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..config.schema import AtlasDepthBand, BoardConfig, SpawnThreshold
from ..primitives.board import Position
from .data_types import ColumnProfile


def enumerate_compositions(
    total: int,
    num_columns: int,
    max_per_column: int,
) -> Iterator[tuple[int, ...]]:
    """Yield every composition of `total` into `num_columns` non-negative parts,
    each part <= max_per_column.

    Order is lexicographic in the generated tuple — deterministic across
    Python versions and useful for reproducible atlas builds.
    """
    if num_columns < 1:
        raise ValueError("num_columns must be >= 1")
    if max_per_column < 0:
        raise ValueError("max_per_column must be >= 0")
    if total < 0:
        raise ValueError("total must be >= 0")
    if total > num_columns * max_per_column:
        return  # Infeasible — no composition can sum that high.
    yield from _compose(total, num_columns, max_per_column, ())


def _compose(
    remaining: int,
    columns_left: int,
    cap: int,
    prefix: tuple[int, ...],
) -> Iterator[tuple[int, ...]]:
    """Recursive DFS helper — fills columns left-to-right.

    The upper bound (cap) prevents runaway stacking in a single column; the
    lower bound (remaining - (columns_left-1)*cap) prunes branches that
    cannot possibly reach `remaining` even if every remaining column is full.
    """
    if columns_left == 1:
        if 0 <= remaining <= cap:
            yield prefix + (remaining,)
        return
    lo = max(0, remaining - (columns_left - 1) * cap)
    hi = min(cap, remaining)
    for count in range(lo, hi + 1):
        yield from _compose(remaining - count, columns_left - 1, cap, prefix + (count,))


def profile_to_positions(
    profile: ColumnProfile,
    depth_bands: Iterable[AtlasDepthBand],
    num_rows: int,
) -> frozenset[Position] | None:
    """Convert a (ColumnProfile, band) into concrete exploded cell positions.

    Placement rule — bottom-up within the band: for each column with count > 0,
    cells occupy rows starting at `band.max_row` (the bottom of the band in
    board coordinates, since row 0 is the top) and work upward (decreasing
    row index). If the column's count exceeds the band span, cells continue
    past `band.min_row` toward row 0; if that would underflow row 0, the
    profile is infeasible and the function returns None.

    Why bottom-up: clusters in the engine tend to land low (gravity + seed
    depth bias), so deep explosions fill the band from its bottom upward —
    this matches the most common runtime shape and keeps the representative
    settle outcome close to what strategies actually see.
    """
    band = _resolve_depth_band(profile.depth_band, depth_bands)
    positions: set[Position] = set()
    for reel, count in enumerate(profile.counts):
        if count == 0:
            continue
        top_row = band.max_row - count + 1
        if top_row < 0 or band.max_row >= num_rows:
            return None
        for row in range(top_row, band.max_row + 1):
            positions.add(Position(reel, row))
    return frozenset(positions)


def _resolve_depth_band(
    name: str, depth_bands: Iterable[AtlasDepthBand]
) -> AtlasDepthBand:
    """Lookup helper — raises on unknown band so callers fail fast rather
    than silently emitting an empty position set."""
    for band in depth_bands:
        if band.name == name:
            return band
    raise KeyError(f"Unknown depth band: {name!r}")


def depth_band_for_row(row: int, depth_bands: Iterable[AtlasDepthBand]) -> str | None:
    """Return the band name whose inclusive [min_row, max_row] contains `row`.

    Returns None if no band matches — AtlasBuilder treats this as an invalid
    configuration, since depth_bands must cover every board row the atlas
    will be keyed on.
    """
    for band in depth_bands:
        if band.min_row <= row <= band.max_row:
            return band.name
    return None


def atlas_cluster_sizes(
    board: BoardConfig,
    spawn_thresholds: Iterable[SpawnThreshold],
) -> tuple[int, ...]:
    """Every cluster size the atlas needs to cover.

    Spans from board.min_cluster_size up to the largest size any spawn
    threshold captures. max_size is bounded by board area — an
    unbounded open range (e.g. 15-49 for SLB) would otherwise explode the
    profile space. This is the single source of truth consumed by the
    builder; nothing else should infer cluster sizes from config.
    """
    max_cells = board.num_reels * board.num_rows
    largest = board.min_cluster_size
    for threshold in spawn_thresholds:
        capped = min(threshold.max_size, max_cells)
        if capped > largest:
            largest = capped
    return tuple(range(board.min_cluster_size, largest + 1))


def enumerate_column_profiles(
    board: BoardConfig,
    depth_bands: tuple[AtlasDepthBand, ...],
    cluster_sizes: Iterable[int],
) -> Iterator[ColumnProfile]:
    """Yield every (composition × depth_band) profile the atlas will key on.

    Caller supplies cluster_sizes (typically via atlas_cluster_sizes) so the
    decision of which sizes matter stays with the builder — this function
    is a pure generator with no config-derived implicit filters.
    """
    for size in cluster_sizes:
        for composition in enumerate_compositions(
            size, board.num_reels, board.num_rows
        ):
            for band in depth_bands:
                yield ColumnProfile(
                    counts=composition, depth_band=band.name, total=size
                )
