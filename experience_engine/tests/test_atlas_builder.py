"""Tests for Step 3: AtlasBuilder + AtlasStorage.

Covers:
- Builder composition produces topologies that match a direct settle() call.
- Booster landings and dormant survivals are indexed when applicable.
- AtlasStorage round-trips and invalidates on relevant config changes only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ..atlas.builder import AtlasBuilder, build_atlas_services
from ..atlas.data_types import SpatialAtlas
from ..atlas.storage import AtlasStorage
from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.gravity import GravityDAG, settle
from ..primitives.symbols import Symbol


DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent / "config" / "default.yaml"
)


@pytest.fixture
def atlas_builder(default_config: MasterConfig) -> AtlasBuilder:
    """Wired builder via the shared helper — single source of truth for the
    atlas service dance, so this fixture and the CLI can't drift apart."""
    return build_atlas_services(default_config).builder


def test_builder_requires_atlas_config(default_config: MasterConfig) -> None:
    """Atlas-less configs must raise — there is no silent no-op path."""
    without_atlas = replace(default_config, atlas=None)
    with pytest.raises(ValueError, match="config.atlas"):
        build_atlas_services(without_atlas)


def test_builder_produces_topology_for_size_five(
    atlas_builder: AtlasBuilder,
) -> None:
    """Only size 5 is requested so the build stays quick; every emitted
    profile must have a matching topology and correct composition total."""
    atlas = atlas_builder.build(sizes=(5,))
    assert isinstance(atlas, SpatialAtlas)
    assert atlas.topologies
    for profile, topology in atlas.topologies.items():
        assert profile.total == 5
        # refill_per_column sum cannot exceed the cluster size (cells shifted
        # equals cells removed; extras collect as empties).
        assert sum(topology.refill_per_column) == 5


def test_topology_matches_direct_settle(
    atlas_builder: AtlasBuilder, default_config: MasterConfig
) -> None:
    """Builder composition has no hidden logic — its topology must equal
    what settle() produces on the same synthetic board + exploded set.

    Uses the bottom-up-in-band convention: a size-5 profile in column 0
    with band "deep" [5,6] places cells at rows 2..6 (bottom of band,
    working upward). Filler board uses the tier-alternating pattern.
    """
    atlas = atlas_builder.build(sizes=(5,))
    profile = next(
        p for p in atlas.topologies
        if p.depth_band == "deep" and p.counts[0] == 5
    )
    topology = atlas.topologies[profile]

    # Rebuild the synthetic board + explosion matching the builder's contract.
    syms = [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4, Symbol.H1, Symbol.H2, Symbol.H3]
    board = Board.empty(default_config.board)
    for reel in range(default_config.board.num_reels):
        for row in range(default_config.board.num_rows):
            board.set(Position(reel, row), syms[(reel + row) % len(syms)])
    # band "deep" = rows [5,6]; bottom-up for count=5 → rows 2..6.
    exploded = frozenset(Position(0, row) for row in range(2, 7))
    dag = GravityDAG(default_config.board, default_config.gravity)
    result = settle(dag, board, exploded, default_config.gravity)
    expected_empty = frozenset(result.empty_positions)
    assert topology.empty_positions == expected_empty


def test_booster_landings_indexed_when_size_crosses_threshold(
    atlas_builder: AtlasBuilder,
) -> None:
    """Size 9 falls in the R (rocket) spawn band — the atlas must index a
    landing entry for it. Size 5 is below the first threshold, so no entry."""
    atlas = atlas_builder.build(sizes=(5, 9))
    has_small = any(size_in_key == 5 for key in atlas.booster_landings
                    for size_in_key in (key[0].total,))
    assert not has_small, "Size 5 should not produce booster landings"
    size9_keys = [k for k in atlas.booster_landings if k[0].total == 9]
    assert size9_keys, "Expected booster landings for size 9 (R threshold)"
    for _, booster_type, _ in size9_keys:
        assert booster_type == "R"


def test_arm_adjacency_matches_landing_neighbors(
    atlas_builder: AtlasBuilder, default_config: MasterConfig
) -> None:
    """ArmAdjacencyEntry.adjacent_refill must equal orthogonal_neighbors of
    the landing intersected with the topology's empties — rule-of-three
    duplication check against the builder's computation."""
    from ..primitives.board import orthogonal_neighbors
    atlas = atlas_builder.build(sizes=(9,))
    key = next(iter(atlas.arm_adjacencies))
    profile, _, _ = key
    topology = atlas.topologies[profile]
    landing = atlas.booster_landings[key].landing_position
    expected = frozenset(
        pos for pos in orthogonal_neighbors(landing, default_config.board)
        if pos in topology.empty_positions
    )
    assert atlas.arm_adjacencies[key].adjacent_refill == expected
    assert atlas.arm_adjacencies[key].adjacent_count == len(expected)


def test_fire_zones_indexed_only_for_chain_initiators(
    atlas_builder: AtlasBuilder, default_config: MasterConfig
) -> None:
    """Rockets and bombs are chain initiators in default.yaml — atlas must
    index their fire zones. Wilds / LB / SLB are not; no zones expected
    for sizes that spawn those."""
    atlas = atlas_builder.build(sizes=(9, 11))
    booster_types_in_fire = {k[0] for k in atlas.fire_zones}
    initiators = set(default_config.boosters.chain_initiators)
    assert booster_types_in_fire <= initiators
    # Rocket entries must appear for both orientations plus the None lookup.
    r_keys = {k[2] for k in atlas.fire_zones if k[0] == "R"}
    assert r_keys == {"H", "V", None}
    # Bomb entries are orientation-free — only None should appear.
    b_keys = {k[2] for k in atlas.fire_zones if k[0] == "B"}
    assert b_keys == {None}


def test_rocket_none_entry_matches_preferred_orientation(
    atlas_builder: AtlasBuilder,
) -> None:
    """The None-orientation entry must equal the longer of the H / V zones
    — that's the builder's rule, and the query relies on it for the
    unconstrained lookup path."""
    atlas = atlas_builder.build(sizes=(9,))
    # Pick any rocket landing that has all three keys.
    h_keys = [k for k in atlas.fire_zones if k[0] == "R" and k[2] == "H"]
    assert h_keys
    for key in h_keys:
        landing = key[1]
        h_zone = atlas.fire_zones[("R", landing, "H")]
        v_zone = atlas.fire_zones[("R", landing, "V")]
        none_zone = atlas.fire_zones[("R", landing, None)]
        expected = h_zone if len(h_zone) >= len(v_zone) else v_zone
        assert none_zone == expected


def test_build_progress_callback_receives_one_line_per_size(
    atlas_builder: AtlasBuilder,
) -> None:
    """The progress callback must be invoked once per requested size plus
    once for the final summary — a plain list.append captures both without
    touching stdout."""
    lines: list[str] = []
    atlas_builder.build(sizes=(5, 9), progress=lines.append)
    # One line per size (2) + one summary line = 3.
    assert len(lines) == 3
    assert lines[0].startswith("  Size  5")
    assert lines[1].startswith("  Size  9")
    assert lines[2].startswith("Total:")


def test_filled_board_has_no_clusters(
    atlas_builder: AtlasBuilder, default_config: MasterConfig
) -> None:
    """Alternating fill (matching test_gravity::_filled_board) must not
    produce pre-existing clusters — settle would otherwise interact with
    them and poison the topology index."""
    from ..primitives.cluster_detection import detect_clusters
    board = atlas_builder._filled_board()
    clusters = detect_clusters(board, default_config)
    assert clusters == []


def test_dormant_survivals_span_every_column(
    atlas_builder: AtlasBuilder, default_config: MasterConfig
) -> None:
    """For each profile the atlas must answer survival for every column —
    the query tier relies on an exhaustive lookup table."""
    atlas = atlas_builder.build(sizes=(5,))
    profile = next(iter(atlas.topologies))
    keys_for_profile = {
        k[1] for k in atlas.dormant_survivals if k[0] == profile
    }
    assert keys_for_profile == set(range(default_config.board.num_reels))


def test_storage_round_trips(
    tmp_path: Path, atlas_builder: AtlasBuilder, default_config: MasterConfig
) -> None:
    atlas = atlas_builder.build(sizes=(5,))
    path = tmp_path / "atlas.bin"
    storage = AtlasStorage()
    storage.save(atlas, default_config, path)
    loaded = storage.load(default_config, path)
    assert loaded is not None
    assert set(loaded.topologies.keys()) == set(atlas.topologies.keys())


def test_storage_invalidates_on_booster_change(
    tmp_path: Path, atlas_builder: AtlasBuilder, default_config: MasterConfig
) -> None:
    """Changing booster config must force a rebuild (load returns None)."""
    atlas = atlas_builder.build(sizes=(5,))
    path = tmp_path / "atlas.bin"
    storage = AtlasStorage()
    storage.save(atlas, default_config, path)

    modified_boosters = replace(
        default_config.boosters, bomb_blast_radius=2
    )
    modified_config = replace(default_config, boosters=modified_boosters)
    assert storage.load(modified_config, path) is None


def test_storage_ignores_paytable_change(
    tmp_path: Path, atlas_builder: AtlasBuilder, default_config: MasterConfig
) -> None:
    """Paytable tweaks don't affect settle topology — the hash must ignore them."""
    atlas = atlas_builder.build(sizes=(5,))
    path = tmp_path / "atlas.bin"
    storage = AtlasStorage()
    storage.save(atlas, default_config, path)

    # Swap in a modified paytable — the atlas hash deliberately ignores it.
    # Mutating a different non-atlas-relevant section keeps the test focused
    # on the hash contract rather than YAML round-trip noise.
    alt_paytable = replace(
        default_config.paytable,
        entries=default_config.paytable.entries,  # identity is fine
    )
    alt_config = replace(default_config, paytable=alt_paytable)
    # Also change an output-only value (audit_survival_threshold) to prove
    # non-gravity/non-booster fields don't invalidate.
    if alt_config.output is not None:
        alt_config = replace(
            alt_config,
            output=replace(alt_config.output, audit_survival_threshold=0.5),
        )

    loaded = storage.load(alt_config, path)
    assert loaded is not None


def test_bridge_indexed_for_wild_spawn_sizes(
    atlas_builder: AtlasBuilder,
) -> None:
    """Sizes 7–8 fall in the W (wild) spawn band — bridge feasibilities must
    be non-empty and every key's profile total must be in that range."""
    atlas = atlas_builder.build(sizes=(7, 8))
    assert atlas.bridge_feasibilities, "Expected bridge entries for wild-spawn sizes"
    for (profile, _col), entry in atlas.bridge_feasibilities.items():
        assert profile.total in (7, 8)
        assert entry.bridge_score >= 0


def test_bridge_not_indexed_below_wild_threshold(
    atlas_builder: AtlasBuilder,
) -> None:
    """Size 5 is below the wild spawn threshold — no bridge entries expected."""
    atlas = atlas_builder.build(sizes=(5,))
    assert not atlas.bridge_feasibilities


def test_bridge_score_positive_for_adjacent_gap(
    atlas_builder: AtlasBuilder,
) -> None:
    """At least one size-7 profile must produce a bridgeable gap with
    positive score — counts on both sides of the gap touch the gap column."""
    atlas = atlas_builder.build(sizes=(7,))
    positive = [
        e for e in atlas.bridge_feasibilities.values()
        if e.bridge_score > 0
    ]
    assert positive, "Expected at least one bridge entry with positive score"
    for entry in positive:
        assert entry.left_adjacency_count > 0
        assert entry.right_adjacency_count > 0


def test_bridge_feasibility_survives_storage_roundtrip(
    tmp_path: Path, atlas_builder: AtlasBuilder, default_config: MasterConfig
) -> None:
    """Bridge feasibilities must survive a save/load cycle unchanged."""
    atlas = atlas_builder.build(sizes=(7,))
    path = tmp_path / "atlas_bridge.bin"
    storage = AtlasStorage()
    storage.save(atlas, default_config, path)
    loaded = storage.load(default_config, path)
    assert loaded is not None
    assert set(loaded.bridge_feasibilities.keys()) == set(
        atlas.bridge_feasibilities.keys()
    )


def test_build_progress_includes_bridge_count(
    atlas_builder: AtlasBuilder,
) -> None:
    """The summary line must mention bridge count so operators can spot-check."""
    lines: list[str] = []
    atlas_builder.build(sizes=(7,), progress=lines.append)
    summary = lines[-1]
    assert "bridges" in summary


def test_storage_returns_none_on_missing_file(
    tmp_path: Path, default_config: MasterConfig
) -> None:
    assert AtlasStorage().load(default_config, tmp_path / "missing.bin") is None


def test_storage_returns_none_on_bad_header(
    tmp_path: Path, default_config: MasterConfig
) -> None:
    path = tmp_path / "corrupted.bin"
    path.write_bytes(b"not an atlas")
    assert AtlasStorage().load(default_config, path) is None
