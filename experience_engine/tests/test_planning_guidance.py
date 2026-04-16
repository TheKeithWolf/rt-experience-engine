"""Tests for Step 1: shared planning types, atlas/trajectory data types, config.

Validates:
- RegionConstraint / region_for_step() with both guidance shapes and None.
- AtlasConfiguration.region_at() and TrajectorySketch.region_at() honor the
  GuidanceSource contract.
- TrajectoryConfig / AtlasConfig validation and loader integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ..atlas.data_types import (
    AtlasConfiguration,
    BoosterLandingEntry,
    ColumnProfile,
    DormantSurvivalEntry,
    PhaseGuidance,
    SettleTopology,
)
from ..config.loader import load_config
from ..config.schema import (
    AtlasConfig,
    AtlasDepthBand,
    ConfigValidationError,
    TrajectoryConfig,
)
from ..planning.region_constraint import (
    GuidanceSource,
    RegionConstraint,
    region_for_step,
)
from ..primitives.board import Position
from ..primitives.gravity import SettleResult
from ..primitives.symbols import Symbol
from ..trajectory.data_types import TrajectorySketch, TrajectoryWaypoint


DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent / "config" / "default.yaml"
)


# ---------------------------------------------------------------------------
# RegionConstraint + region_for_step
# ---------------------------------------------------------------------------


def test_region_for_step_returns_none_when_guidance_absent() -> None:
    """Unguided path — no planning active means strategies see None."""
    assert region_for_step(None, 0) is None
    assert region_for_step(None, 5) is None


def _sample_phase_guidance(columns: frozenset[int]) -> PhaseGuidance:
    profile = ColumnProfile(counts=(0, 2, 3, 0, 0, 0, 0), depth_band="mid", total=5)
    return PhaseGuidance(
        viable_columns=columns,
        preferred_row_range=(2, 4),
        column_profile=profile,
        booster_landing=None,
        dormant_survival=None,
        chain_target_zone=None,
    )


def test_atlas_configuration_region_at_matches_phase() -> None:
    cfg = AtlasConfiguration(
        phases=(
            _sample_phase_guidance(frozenset({1, 2, 3})),
            _sample_phase_guidance(frozenset({4, 5})),
        ),
        composite_score=0.8,
    )
    assert isinstance(cfg, GuidanceSource)
    first = cfg.region_at(0)
    assert first is not None
    assert first.viable_columns == frozenset({1, 2, 3})
    assert first.preferred_row_range == (2, 4)
    # Out-of-range index returns None so strategies fall back gracefully.
    assert cfg.region_at(99) is None
    # region_for_step delegates without type inspection.
    assert region_for_step(cfg, 1) is not None


def _sample_waypoint(
    phase_index: int, cluster_region: frozenset[Position]
) -> TrajectoryWaypoint:
    settle = SettleResult(board=None, move_steps=(), empty_positions=())  # type: ignore[arg-type]
    return TrajectoryWaypoint(
        phase_index=phase_index,
        cluster_region=cluster_region,
        cluster_symbol=Symbol.H1,
        booster_type=None,
        booster_spawn_pos=None,
        booster_landing_pos=None,
        settle_result=settle,
        landing_context=None,
        landing_score=0.5,
        reserve_zone=frozenset(),
        chain_target_pos=None,
        seed_hints={},
    )


def test_trajectory_sketch_region_at_uses_waypoint_columns() -> None:
    positions = frozenset({Position(2, 3), Position(2, 4), Position(3, 3)})
    sketch = TrajectorySketch(
        waypoints=(_sample_waypoint(0, positions),),
        composite_score=0.5,
        is_feasible=True,
        arc=None,  # type: ignore[arg-type]
    )
    assert isinstance(sketch, GuidanceSource)
    region = sketch.region_at(0)
    assert region is not None
    assert region.viable_columns == frozenset({2, 3})
    assert region.preferred_row_range == (3, 4)
    assert sketch.region_at(42) is None


# ---------------------------------------------------------------------------
# TrajectoryConfig / AtlasConfig validation
# ---------------------------------------------------------------------------


def test_trajectory_config_rejects_non_positive_retries() -> None:
    with pytest.raises(ConfigValidationError, match="max_sketch_retries"):
        TrajectoryConfig(
            max_sketch_retries=0,
            waypoint_feasibility_threshold=0.5,
            sketch_feasibility_threshold=0.3,
        )


def test_trajectory_config_rejects_thresholds_out_of_range() -> None:
    with pytest.raises(ConfigValidationError, match="waypoint_feasibility_threshold"):
        TrajectoryConfig(
            max_sketch_retries=1,
            waypoint_feasibility_threshold=0.0,
            sketch_feasibility_threshold=0.3,
        )
    with pytest.raises(ConfigValidationError, match="sketch_feasibility_threshold"):
        TrajectoryConfig(
            max_sketch_retries=1,
            waypoint_feasibility_threshold=0.3,
            sketch_feasibility_threshold=1.5,
        )


def test_atlas_config_rejects_empty_depth_bands() -> None:
    with pytest.raises(ConfigValidationError, match="depth_bands"):
        AtlasConfig(
            enabled=True,
            path="atlas.bin",
            depth_bands=(),
            region_falloff_per_column=0.5,
            min_composite_score=0.2,
        )


def test_atlas_config_rejects_duplicate_band_names() -> None:
    bands = (
        AtlasDepthBand(name="low", min_row=0, max_row=2),
        AtlasDepthBand(name="low", min_row=3, max_row=5),
    )
    with pytest.raises(ConfigValidationError, match="unique"):
        AtlasConfig(
            enabled=True,
            path="atlas.bin",
            depth_bands=bands,
            region_falloff_per_column=0.5,
            min_composite_score=0.2,
        )


def test_atlas_depth_band_rejects_inverted_range() -> None:
    with pytest.raises(ConfigValidationError, match="max_row"):
        AtlasDepthBand(name="mid", min_row=5, max_row=3)


# ---------------------------------------------------------------------------
# Loader integration
# ---------------------------------------------------------------------------


def test_default_config_populates_atlas_and_trajectory() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.atlas is not None
    assert cfg.atlas.enabled is True
    # Depth bands preserve declaration order for deterministic band assignment
    names = tuple(band.name for band in cfg.atlas.depth_bands)
    assert names == ("low", "mid", "deep")
    assert cfg.reasoner.trajectory is not None
    assert cfg.reasoner.trajectory.max_sketch_retries >= 1


def test_omitting_atlas_and_trajectory_sections_yields_none(tmp_path: Path) -> None:
    """Backward compatibility: configs without the new sections still load."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data.pop("atlas", None)
    data.get("reasoner", {}).pop("trajectory", None)
    path = tmp_path / "no_planning.yaml"
    path.write_text(yaml.dump(data))

    cfg = load_config(path)
    assert cfg.atlas is None
    assert cfg.reasoner.trajectory is None


def test_atlas_depth_bands_must_be_mapping(tmp_path: Path) -> None:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["atlas"]["depth_bands"] = [[0, 2], [3, 4]]
    path = tmp_path / "bad_bands.yaml"
    path.write_text(yaml.dump(data))
    with pytest.raises(ConfigValidationError, match="depth_bands"):
        load_config(path)


# ---------------------------------------------------------------------------
# AtlasConfiguration.as_region round-trip
# ---------------------------------------------------------------------------


def test_phase_guidance_as_region_includes_profile_hint() -> None:
    guidance = _sample_phase_guidance(frozenset({0, 1}))
    region = guidance.as_region()
    assert region.viable_columns == frozenset({0, 1})
    assert region.preferred_row_range == (2, 4)
    assert region.profile_hint is guidance.column_profile


def test_atlas_configuration_booster_landing_entries_construct() -> None:
    entry = BoosterLandingEntry(
        landing_position=Position(3, 5),
        adjacent_refill=frozenset({Position(3, 4), Position(2, 5)}),
        landing_score=0.9,
    )
    dormant = DormantSurvivalEntry(
        survives=True,
        post_gravity_position=Position(1, 6),
        column_shift=False,
    )
    topology = SettleTopology(
        empty_positions=frozenset({Position(4, 0)}),
        refill_per_column=(0, 0, 0, 0, 1, 0, 0),
        has_diagonal_redistribution=False,
        gravity_mapping={},
    )
    assert entry.landing_position.reel == 3
    assert dormant.survives is True
    assert topology.refill_per_column[4] == 1
