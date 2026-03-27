"""Tests for rl_archive/descriptor.py — behavioral descriptor extraction.

RLA-001 through RLA-009: TrajectoryDescriptor immutability, spatial binning,
payout binning, orientation classification, and DescriptorKey hashing.
"""

from __future__ import annotations

import dataclasses

import pytest

from ..config.schema import BoardConfig, ConfigValidationError, DescriptorConfig
from ..pipeline.data_types import CascadeStepRecord, GeneratedInstance
from ..pipeline.protocols import RangeFloat
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.symbols import Symbol
from ..rl_archive.descriptor import (
    CascadeDescriptorExtractor,
    TrajectoryDescriptor,
    _classify_orientation,
)
from ..spatial_solver.data_types import ClusterAssignment


# ---------------------------------------------------------------------------
# Helpers — minimal objects for testing
# ---------------------------------------------------------------------------


def _make_cluster(
    symbol: Symbol,
    positions: frozenset[Position],
) -> ClusterAssignment:
    """Build a minimal ClusterAssignment for testing."""
    return ClusterAssignment(
        symbol=symbol,
        positions=positions,
        size=len(positions),
        wild_positions=frozenset(),
    )


def _make_step_record(
    clusters: tuple[ClusterAssignment, ...],
    step_index: int = 0,
    step_payout: float = 1.0,
) -> CascadeStepRecord:
    """Build a minimal CascadeStepRecord with only the fields descriptor needs."""
    board = Board.empty(BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5))
    return CascadeStepRecord(
        step_index=step_index,
        board_before=board,
        board_after=board,
        clusters=clusters,
        step_payout=step_payout,
        grid_multipliers_snapshot=tuple(0 for _ in range(49)),
        booster_spawn_types=(),
        booster_spawn_positions=(),
        booster_fire_records=(),
        gravity_record=None,
        booster_gravity_record=None,
    )


def _make_instance(
    archetype_id: str,
    cluster_positions: frozenset[Position],
    cluster_symbol: Symbol = Symbol.L1,
    payout: float = 5.0,
) -> GeneratedInstance:
    """Build a minimal GeneratedInstance with one step-0 cluster."""
    cluster = _make_cluster(cluster_symbol, cluster_positions)
    step = _make_step_record(clusters=(cluster,))
    board = Board.empty(BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5))
    return GeneratedInstance(
        sim_id=1,
        archetype_id=archetype_id,
        family="test",
        criteria="basegame",
        board=board,
        spatial_step=None,  # type: ignore[arg-type]
        payout=payout,
        centipayout=int(payout * 100),
        win_level=1,
        cascade_steps=(step,),
        gravity_record=None,
    )


def _make_extractor(
    board_config: BoardConfig | None = None,
    descriptor_config: DescriptorConfig | None = None,
) -> CascadeDescriptorExtractor:
    """Build a CascadeDescriptorExtractor with default or custom config."""
    from ..config.schema import BoosterConfig, SpawnThreshold, SymbolConfig

    bc = board_config or BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5)

    # Minimal valid BoosterConfig for BoosterRules construction
    booster_config = BoosterConfig(
        spawn_thresholds=(SpawnThreshold(booster="W", min_size=7, max_size=49),),
        spawn_order=("W",),
        rocket_tie_orientation="H",
        bomb_blast_radius=1,
        immune_to_rocket=(),
        immune_to_bomb=(),
        chain_initiators=(),
    )
    symbol_config = SymbolConfig(
        standard=("L1", "L2", "L3", "L4", "H1", "H2", "H3"),
        low_tier=("L1", "L2", "L3", "L4"),
        high_tier=("H1", "H2", "H3"),
        payout_rank=(
            ("L1", 1), ("L2", 2), ("L3", 3), ("L4", 4),
            ("H1", 5), ("H2", 6), ("H3", 7),
        ),
    )
    dc = descriptor_config or DescriptorConfig(
        spatial_col_bins=3, spatial_row_bins=3, payout_bins=4,
    )
    booster_rules = BoosterRules(booster_config, bc, symbol_config)
    return CascadeDescriptorExtractor(booster_rules, bc, dc)


# ---------------------------------------------------------------------------
# RLA-001: TrajectoryDescriptor is frozen
# ---------------------------------------------------------------------------


def test_rla_001_trajectory_descriptor_is_frozen() -> None:
    """RLA-001: TrajectoryDescriptor is immutable."""
    desc = TrajectoryDescriptor(
        archetype_id="test",
        step0_symbol="L1",
        spatial_bin=(0, 0),
        cluster_orientation="H",
        payout_bin=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        desc.payout_bin = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RLA-002: Correct spatial bin for known centroid
# ---------------------------------------------------------------------------


def test_rla_002_spatial_bin_for_known_centroid() -> None:
    """RLA-002: CascadeDescriptorExtractor produces correct spatial bin."""
    # Cluster centered at (1, 1) on a 7x7 board with 3x3 bins
    # col_bin = floor(1 / (7/3)) = floor(0.428) = 0
    # row_bin = floor(1 / (7/3)) = floor(0.428) = 0
    positions = frozenset({
        Position(0, 0), Position(1, 0), Position(1, 1),
        Position(1, 2), Position(2, 1),
    })
    instance = _make_instance("test_arch", positions)
    extractor = _make_extractor()
    payout_range = RangeFloat(0.0, 10.0)

    desc = extractor.extract(instance, payout_range)
    assert desc.spatial_bin == (0, 0)


# ---------------------------------------------------------------------------
# RLA-003: Spatial binning respects BoardConfig dimensions
# ---------------------------------------------------------------------------


def test_rla_003_spatial_binning_respects_board_config() -> None:
    """RLA-003: Spatial binning uses BoardConfig.num_reels/num_rows, not hardcoded 7."""
    # 5x5 board with 3 col bins: bin width = 5/3 ≈ 1.667
    # Centroid at reel=3: col_bin = floor(3 / 1.667) = floor(1.8) = 1
    board_config = BoardConfig(num_reels=5, num_rows=5, min_cluster_size=3)
    positions = frozenset({
        Position(3, 2), Position(3, 3), Position(4, 2),
    })
    instance = _make_instance("test_arch", positions, payout=2.0)
    extractor = _make_extractor(board_config=board_config)
    payout_range = RangeFloat(0.0, 10.0)

    desc = extractor.extract(instance, payout_range)
    # col_bin = floor(3 / (5/3)) = floor(1.8) = 1
    # row_bin = floor(2 / (5/3)) = floor(1.2) = 1  (centroid ≈ row 2.33, snapped to 2)
    assert desc.spatial_bin[0] == 1


# ---------------------------------------------------------------------------
# RLA-004: Payout binning covers full range
# ---------------------------------------------------------------------------


def test_rla_004_payout_binning_full_range() -> None:
    """RLA-004: Min payout maps to bin 0, max maps to last bin."""
    positions = frozenset({
        Position(3, 3), Position(3, 4), Position(4, 3),
        Position(4, 4), Position(3, 2),
    })
    payout_range = RangeFloat(1.0, 10.0)
    extractor = _make_extractor()

    # Min payout → bin 0
    inst_min = _make_instance("test", positions, payout=1.0)
    desc_min = extractor.extract(inst_min, payout_range)
    assert desc_min.payout_bin == 0

    # Max payout → last bin (payout_bins - 1 = 3)
    inst_max = _make_instance("test", positions, payout=10.0)
    desc_max = extractor.extract(inst_max, payout_range)
    assert desc_max.payout_bin == 3


# ---------------------------------------------------------------------------
# RLA-005: Orientation classification
# ---------------------------------------------------------------------------


def test_rla_005_orientation_classification() -> None:
    """RLA-005: H when col_span > row_span, V vice versa, compact on tie."""
    # Horizontal: col_span=3, row_span=1
    h_positions = frozenset({Position(0, 0), Position(1, 0), Position(2, 0)})
    assert _classify_orientation(h_positions) == "H"

    # Vertical: col_span=1, row_span=3
    v_positions = frozenset({Position(0, 0), Position(0, 1), Position(0, 2)})
    assert _classify_orientation(v_positions) == "V"

    # Compact: col_span=2, row_span=2 (tie)
    c_positions = frozenset({
        Position(0, 0), Position(1, 0), Position(0, 1), Position(1, 1),
    })
    assert _classify_orientation(c_positions) == "compact"


# ---------------------------------------------------------------------------
# RLA-006: to_key produces hashable tuple usable as dict key
# ---------------------------------------------------------------------------


def test_rla_006_to_key_is_hashable() -> None:
    """RLA-006: to_key returns a hashable tuple usable as dict key."""
    desc = TrajectoryDescriptor(
        archetype_id="wild_enable_rocket",
        step0_symbol="H1",
        spatial_bin=(1, 2),
        cluster_orientation="V",
        payout_bin=3,
    )
    extractor = _make_extractor()
    key = extractor.to_key(desc)

    # Must be hashable
    assert isinstance(key, tuple)
    d: dict[tuple, int] = {key: 42}
    assert d[key] == 42

    # Key contains all descriptor fields
    assert "wild_enable_rocket" in key
    assert "H1" in key
    assert 1 in key
    assert 2 in key
    assert "V" in key
    assert 3 in key


# ---------------------------------------------------------------------------
# RLA-007: DescriptorConfig rejects bins < 1
# ---------------------------------------------------------------------------


def test_rla_007_descriptor_config_rejects_invalid_bins() -> None:
    """RLA-007: DescriptorConfig raises ConfigValidationError for bins < 1."""
    with pytest.raises(ConfigValidationError, match="spatial_col_bins"):
        DescriptorConfig(spatial_col_bins=0, spatial_row_bins=3, payout_bins=4)

    with pytest.raises(ConfigValidationError, match="spatial_row_bins"):
        DescriptorConfig(spatial_col_bins=3, spatial_row_bins=0, payout_bins=4)

    with pytest.raises(ConfigValidationError, match="payout_bins"):
        DescriptorConfig(spatial_col_bins=3, spatial_row_bins=3, payout_bins=0)


# ---------------------------------------------------------------------------
# RLA-008: Different centroids produce different spatial bins
# ---------------------------------------------------------------------------


def test_rla_008_different_centroids_different_bins() -> None:
    """RLA-008: Two instances with different step-0 centroids produce different spatial bins."""
    payout_range = RangeFloat(0.0, 10.0)
    extractor = _make_extractor()

    # Cluster in top-left (centroid near 1,1)
    positions_tl = frozenset({
        Position(0, 0), Position(1, 0), Position(0, 1),
        Position(1, 1), Position(0, 2),
    })
    inst_tl = _make_instance("test", positions_tl, payout=5.0)
    desc_tl = extractor.extract(inst_tl, payout_range)

    # Cluster in bottom-right (centroid near 5,5)
    positions_br = frozenset({
        Position(5, 5), Position(6, 5), Position(5, 6),
        Position(6, 6), Position(5, 4),
    })
    inst_br = _make_instance("test", positions_br, payout=5.0)
    desc_br = extractor.extract(inst_br, payout_range)

    assert desc_tl.spatial_bin != desc_br.spatial_bin


# ---------------------------------------------------------------------------
# RLA-009: Same centroid, different payout → different descriptors
# ---------------------------------------------------------------------------


def test_rla_009_same_centroid_different_payout() -> None:
    """RLA-009: Two instances with same centroid but different payout produce different descriptors."""
    positions = frozenset({
        Position(3, 3), Position(3, 4), Position(4, 3),
        Position(4, 4), Position(3, 2),
    })
    payout_range = RangeFloat(0.0, 20.0)  # 4 bins → width 5.0 each
    extractor = _make_extractor()

    # Payout 2.0 → bin 0 (2/20 * 4 = 0.4 → floor = 0)
    inst_low = _make_instance("test", positions, payout=2.0)
    desc_low = extractor.extract(inst_low, payout_range)

    # Payout 18.0 → bin 3 (18/20 * 4 = 3.6 → floor = 3)
    inst_high = _make_instance("test", positions, payout=18.0)
    desc_high = extractor.extract(inst_high, payout_range)

    assert desc_low.payout_bin != desc_high.payout_bin
    # Spatial bins should be identical since positions are the same
    assert desc_low.spatial_bin == desc_high.spatial_bin
