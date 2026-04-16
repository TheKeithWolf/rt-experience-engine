"""Tests for Step 2: atlas column profile enumeration."""

from __future__ import annotations

from math import comb

import pytest

from ..atlas.data_types import ArmAdjacencyEntry, ColumnProfile, SpatialAtlas
from ..atlas.profiles import (
    atlas_cluster_sizes,
    depth_band_for_row,
    enumerate_column_profiles,
    enumerate_compositions,
    profile_to_positions,
)
from ..config.schema import (
    AtlasDepthBand,
    BoardConfig,
    MasterConfig,
    SpawnThreshold,
)
from ..primitives.board import Position


def _count_compositions_with_cap(total: int, columns: int, cap: int) -> int:
    """Closed-form count of compositions of `total` into `columns` non-negative
    parts each <= cap via inclusion-exclusion.

    Formula: sum_{k=0..columns} (-1)^k * C(columns, k) *
             C(total - k*(cap+1) + columns - 1, columns - 1)
    """
    result = 0
    for k in range(columns + 1):
        n = total - k * (cap + 1)
        if n < 0:
            break
        result += (-1) ** k * comb(columns, k) * comb(n + columns - 1, columns - 1)
    return result


def test_enumerate_compositions_counts_match_closed_form() -> None:
    """Verify enumeration against inclusion-exclusion for a handful of
    (total, columns, cap) cases."""
    for total, columns, cap in [(5, 7, 7), (9, 7, 7), (4, 3, 2), (6, 4, 3)]:
        generated = list(enumerate_compositions(total, columns, cap))
        expected = _count_compositions_with_cap(total, columns, cap)
        assert len(generated) == expected, (
            f"total={total} cols={columns} cap={cap}: "
            f"got {len(generated)} expected {expected}"
        )
        # Every tuple sums to `total` and respects the cap.
        for comp in generated:
            assert len(comp) == columns
            assert sum(comp) == total
            assert all(0 <= v <= cap for v in comp)


def test_enumerate_compositions_deterministic_order() -> None:
    """Same input must yield the same output sequence across calls — the
    atlas build depends on reproducible iteration order."""
    first = list(enumerate_compositions(4, 3, 3))
    second = list(enumerate_compositions(4, 3, 3))
    assert first == second


def test_enumerate_compositions_rejects_infeasible_total() -> None:
    """Total exceeding columns*cap yields nothing (not an exception)."""
    assert list(enumerate_compositions(10, 2, 3)) == []


def test_enumerate_compositions_validates_inputs() -> None:
    with pytest.raises(ValueError, match="num_columns"):
        list(enumerate_compositions(3, 0, 2))
    with pytest.raises(ValueError, match="max_per_column"):
        list(enumerate_compositions(3, 2, -1))
    with pytest.raises(ValueError, match="total"):
        list(enumerate_compositions(-1, 2, 2))


def test_depth_band_for_row_resolves_each_band() -> None:
    bands = (
        AtlasDepthBand(name="low", min_row=0, max_row=2),
        AtlasDepthBand(name="mid", min_row=3, max_row=4),
        AtlasDepthBand(name="deep", min_row=5, max_row=6),
    )
    assert depth_band_for_row(0, bands) == "low"
    assert depth_band_for_row(3, bands) == "mid"
    assert depth_band_for_row(6, bands) == "deep"
    assert depth_band_for_row(7, bands) is None


def test_atlas_cluster_sizes_spans_min_cluster_to_largest_spawn() -> None:
    board = BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5)
    thresholds = (
        SpawnThreshold(booster="W", min_size=7, max_size=8),
        SpawnThreshold(booster="R", min_size=9, max_size=10),
        # Large open range — should clip to board area (49).
        SpawnThreshold(booster="SLB", min_size=15, max_size=49),
    )
    sizes = atlas_cluster_sizes(board, thresholds)
    assert sizes[0] == 5
    assert sizes[-1] == 49  # board area
    assert len(sizes) == 49 - 5 + 1


def test_atlas_cluster_sizes_caps_at_board_area() -> None:
    """An unbounded or oversized max_size must collapse to num_reels*num_rows."""
    board = BoardConfig(num_reels=3, num_rows=3, min_cluster_size=3)
    thresholds = (SpawnThreshold(booster="X", min_size=3, max_size=1000),)
    sizes = atlas_cluster_sizes(board, thresholds)
    assert sizes[-1] == 9


def test_profile_to_positions_places_bottom_up_within_band() -> None:
    """A profile of (0, 3, 0, ..., 0) in band "deep" [5, 6] should emit
    rows 6, 5, 4 in column 1 — bottom of the band, then extending upward
    past min_row=5 because the count exceeds the band span."""
    bands = (
        AtlasDepthBand(name="low", min_row=0, max_row=2),
        AtlasDepthBand(name="mid", min_row=3, max_row=4),
        AtlasDepthBand(name="deep", min_row=5, max_row=6),
    )
    profile = ColumnProfile(counts=(0, 3, 0, 0, 0, 0, 0), depth_band="deep", total=3)
    positions = profile_to_positions(profile, bands, num_rows=7)
    assert positions == frozenset(
        {Position(1, 6), Position(1, 5), Position(1, 4)}
    )


def test_profile_to_positions_skips_empty_columns() -> None:
    """Columns with count 0 contribute zero positions — nothing below."""
    bands = (AtlasDepthBand(name="low", min_row=0, max_row=2),)
    profile = ColumnProfile(counts=(1, 0, 2), depth_band="low", total=3)
    positions = profile_to_positions(profile, bands, num_rows=3)
    assert positions == frozenset(
        {Position(0, 2), Position(2, 2), Position(2, 1)}
    )


def test_profile_to_positions_returns_none_when_column_underflows() -> None:
    """A column count exceeding band.max_row + 1 rows of space above row 0
    means the profile cannot fit — return None rather than raise so the
    builder can skip the profile silently."""
    bands = (AtlasDepthBand(name="low", min_row=0, max_row=2),)
    profile = ColumnProfile(counts=(5, 0, 0), depth_band="low", total=5)
    assert profile_to_positions(profile, bands, num_rows=7) is None


def test_profile_to_positions_raises_on_unknown_band() -> None:
    """An unknown band name is a builder bug, not a silent miss — fail fast."""
    bands = (AtlasDepthBand(name="low", min_row=0, max_row=2),)
    profile = ColumnProfile(counts=(1, 0), depth_band="nope", total=1)
    try:
        profile_to_positions(profile, bands, num_rows=3)
    except KeyError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected KeyError for unknown band")


def test_arm_adjacency_entry_round_trips() -> None:
    """Simple construction sanity — the entry is a data carrier."""
    entry = ArmAdjacencyEntry(
        adjacent_refill=frozenset({Position(3, 4)}),
        adjacent_count=1,
        sufficient=False,
    )
    assert entry.adjacent_count == 1
    assert entry.sufficient is False
    assert Position(3, 4) in entry.adjacent_refill


def test_spatial_atlas_accepts_new_maps() -> None:
    """SpatialAtlas construction must accept the two new dicts so the
    builder's handoff contract is backward-compatible with existing call
    sites — exercised through a minimal construction."""
    atlas = SpatialAtlas(
        topologies={},
        booster_landings={},
        arm_adjacencies={},
        fire_zones={},
        dormant_survivals={},
        bridge_feasibilities={},
    )
    assert atlas.arm_adjacencies == {}
    assert atlas.fire_zones == {}


def test_enumerate_column_profiles_pairs_each_band(default_config: MasterConfig) -> None:
    """For each composition and each depth band the enumerator must emit one
    ColumnProfile — the atlas builder relies on this pairing to key entries."""
    board = default_config.board
    bands = (
        AtlasDepthBand(name="low", min_row=0, max_row=2),
        AtlasDepthBand(name="mid", min_row=3, max_row=4),
    )
    cluster_sizes = (5,)
    profiles = list(enumerate_column_profiles(board, bands, cluster_sizes))
    composition_count = _count_compositions_with_cap(5, board.num_reels, board.num_rows)
    assert len(profiles) == composition_count * len(bands)
    # Sanity: every profile has the expected total and a known band name.
    assert all(p.total == 5 for p in profiles)
    assert {p.depth_band for p in profiles} == {"low", "mid"}
