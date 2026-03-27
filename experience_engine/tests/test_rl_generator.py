"""Tests for rl_archive Phase 8 — generator, archive_io, population integration.

RLA-090 through RLA-102: Archive sampling, round-trip serialization,
population routing, diversity weighting, and fallback behavior.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ..config.schema import (
    BoardConfig,
    DescriptorConfig,
    GridMultiplierConfig,
    MasterConfig,
)
from ..pipeline.data_types import CascadeStepRecord, GeneratedInstance
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..rl_archive.archive import MAPElitesArchive
from ..rl_archive.archive_io import load_archive, save_archive
from ..rl_archive.descriptor import TrajectoryDescriptor
from ..rl_archive.generator import RLArchiveGenerator
from ..spatial_solver.data_types import ClusterAssignment
from ..variance.hints import VarianceHints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOARD_CONFIG = BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5)
_DESCRIPTOR_CONFIG = DescriptorConfig(
    spatial_col_bins=3, spatial_row_bins=3, payout_bins=4,
)
_NUM_SYMBOLS = 7


def _make_descriptor(
    archetype_id: str = "test",
    symbol: str = "L1",
    spatial: tuple[int, int] = (0, 0),
    payout_bin: int = 0,
) -> TrajectoryDescriptor:
    return TrajectoryDescriptor(
        archetype_id=archetype_id,
        step0_symbol=symbol,
        spatial_bin=spatial,
        cluster_orientation="H",
        payout_bin=payout_bin,
    )


def _make_instance(
    archetype_id: str = "test",
    sim_id: int = 0,
    payout: float = 5.0,
) -> GeneratedInstance:
    board = Board.empty(_BOARD_CONFIG)
    cluster = ClusterAssignment(
        symbol=Symbol.L1,
        positions=frozenset(Position(0, i) for i in range(5)),
        size=5,
        wild_positions=frozenset(),
    )
    step = CascadeStepRecord(
        step_index=0,
        board_before=board,
        board_after=board,
        clusters=(cluster,),
        step_payout=payout,
        grid_multipliers_snapshot=tuple(0 for _ in range(49)),
        booster_spawn_types=(),
        booster_spawn_positions=(),
        booster_fire_records=(),
        gravity_record=None,
        booster_gravity_record=None,
    )
    return GeneratedInstance(
        sim_id=sim_id,
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


def _make_populated_archive(
    archetype_id: str = "test",
    num_entries: int = 3,
) -> MAPElitesArchive:
    archive = MAPElitesArchive(_DESCRIPTOR_CONFIG, _NUM_SYMBOLS)
    for i in range(num_entries):
        desc = _make_descriptor(
            archetype_id=archetype_id, symbol=f"L{i + 1}", payout_bin=i,
        )
        inst = _make_instance(archetype_id=archetype_id, sim_id=i, payout=float(i + 1))
        archive.try_insert(inst, desc, quality=float(i + 1))
    return archive


def _make_default_hints() -> VarianceHints:
    return VarianceHints(
        spatial_bias={},
        symbol_weights={},
        near_miss_symbol_preference=(),
        cluster_size_preference=(),
    )


def _make_mock_config() -> MasterConfig:
    """Return the real default_config via conftest pattern."""
    from pathlib import Path as P
    from ..config.loader import load_config
    return load_config(P(__file__).parent.parent / "config" / "default.yaml")


def _make_mock_registry(archetypes: dict[str, int]) -> MagicMock:
    """Mock registry with archetype_id → cascade depth."""
    registry = MagicMock()
    registry.all_ids.return_value = set(archetypes.keys())

    def _get(aid):
        sig = MagicMock()
        sig.required_cascade_depth = Range(0, archetypes[aid])
        sig.payout_range = RangeFloat(0.0, 10.0)
        return sig

    registry.get.side_effect = _get
    return registry


# ---------------------------------------------------------------------------
# RLA-090: generate returns success from populated archive
# ---------------------------------------------------------------------------


def test_rla_090_generate_success() -> None:
    """RLA-090: generate returns GenerationResult(success=True) from populated archive."""
    archive = _make_populated_archive("wild_arch")
    config = _make_mock_config()
    registry = _make_mock_registry({"wild_arch": 3})

    gen = RLArchiveGenerator(config, registry, {"wild_arch": archive})
    result = gen.generate("wild_arch", sim_id=42, hints=_make_default_hints(),
                          rng=random.Random(1))

    assert result.success is True
    assert result.instance is not None


# ---------------------------------------------------------------------------
# RLA-091: Returned instance has correct archetype_id
# ---------------------------------------------------------------------------


def test_rla_091_correct_archetype_id() -> None:
    """RLA-091: Returned instance has correct archetype_id."""
    archive = _make_populated_archive("test_arch")
    config = _make_mock_config()
    registry = _make_mock_registry({"test_arch": 3})

    gen = RLArchiveGenerator(config, registry, {"test_arch": archive})
    result = gen.generate("test_arch", sim_id=1, hints=_make_default_hints(),
                          rng=random.Random(1))

    assert result.instance.archetype_id == "test_arch"


# ---------------------------------------------------------------------------
# RLA-092: Returned instance has stamped sim_id
# ---------------------------------------------------------------------------


def test_rla_092_stamped_sim_id() -> None:
    """RLA-092: Returned instance has the population's sequential sim_id."""
    archive = _make_populated_archive("test_arch")
    config = _make_mock_config()
    registry = _make_mock_registry({"test_arch": 3})

    gen = RLArchiveGenerator(config, registry, {"test_arch": archive})
    result = gen.generate("test_arch", sim_id=999, hints=_make_default_hints(),
                          rng=random.Random(1))

    assert result.instance.sim_id == 999


# ---------------------------------------------------------------------------
# RLA-094: Missing archetype returns success=False
# ---------------------------------------------------------------------------


def test_rla_094_missing_archetype() -> None:
    """RLA-094: Missing archetype returns GenerationResult(success=False)."""
    config = _make_mock_config()
    registry = _make_mock_registry({"other": 3})

    gen = RLArchiveGenerator(config, registry, {})
    result = gen.generate("nonexistent", sim_id=1, hints=_make_default_hints(),
                          rng=random.Random(1))

    assert result.success is False
    assert "no archive" in result.failure_reason


# ---------------------------------------------------------------------------
# RLA-095: save/load archive round-trip preserves entries
# ---------------------------------------------------------------------------


def test_rla_095_archive_round_trip() -> None:
    """RLA-095: save_archive / load_archive round-trip preserves entries."""
    archive = _make_populated_archive("rt_test", num_entries=3)
    assert archive.filled_count() == 3

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "rt_test.jsonl"
        save_archive(archive, path)

        # Verify file exists and has content
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Load back
        loaded = load_archive(path, _DESCRIPTOR_CONFIG, _NUM_SYMBOLS)
        assert loaded.filled_count() == 3


# ---------------------------------------------------------------------------
# RLA-096: load_archive on empty file produces empty archive
# ---------------------------------------------------------------------------


def test_rla_096_load_empty_file() -> None:
    """RLA-096: load_archive on empty file produces filled_count=0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.jsonl"
        path.write_text("")

        loaded = load_archive(path, _DESCRIPTOR_CONFIG, _NUM_SYMBOLS)
        assert loaded.filled_count() == 0


# ---------------------------------------------------------------------------
# RLA-097: Routing depth >= 2 → RL archive when available
# ---------------------------------------------------------------------------


def test_rla_097_routing_depth2_to_archive() -> None:
    """RLA-097: Depth >= 2 routes to RL archive generator when available."""
    from ..population.controller import PopulationController

    config = _make_mock_config()
    registry = _make_mock_registry({"deep": 3, "shallow": 1, "static": 0})

    mock_static = MagicMock()
    mock_cascade = MagicMock()
    mock_archive = MagicMock()

    ctrl = PopulationController(
        config, registry, mock_static, mock_cascade,
        MagicMock(), rl_archive_generator=mock_archive,
    )

    assert ctrl._select_generator("deep") is mock_archive


# ---------------------------------------------------------------------------
# RLA-098: Routing depth 1 → cascade
# ---------------------------------------------------------------------------


def test_rla_098_routing_depth1_to_cascade() -> None:
    """RLA-098: Depth 1 routes to cascade generator."""
    from ..population.controller import PopulationController

    config = _make_mock_config()
    registry = _make_mock_registry({"shallow": 1})

    mock_static = MagicMock()
    mock_cascade = MagicMock()
    mock_archive = MagicMock()

    ctrl = PopulationController(
        config, registry, mock_static, mock_cascade,
        MagicMock(), rl_archive_generator=mock_archive,
    )

    assert ctrl._select_generator("shallow") is mock_cascade


# ---------------------------------------------------------------------------
# RLA-099: Routing depth 0 → static
# ---------------------------------------------------------------------------


def test_rla_099_routing_depth0_to_static() -> None:
    """RLA-099: Depth 0 routes to static generator."""
    from ..population.controller import PopulationController

    config = _make_mock_config()
    registry = _make_mock_registry({"dead": 0})

    mock_static = MagicMock()
    mock_cascade = MagicMock()

    ctrl = PopulationController(
        config, registry, mock_static, mock_cascade, MagicMock(),
    )

    assert ctrl._select_generator("dead") is mock_static


# ---------------------------------------------------------------------------
# RLA-100: Fallback to cascade when RL archive is None
# ---------------------------------------------------------------------------


def test_rla_100_fallback_to_cascade() -> None:
    """RLA-100: Depth >= 2 falls back to cascade when rl_archive_generator is None."""
    from ..population.controller import PopulationController

    config = _make_mock_config()
    registry = _make_mock_registry({"deep": 3})

    mock_static = MagicMock()
    mock_cascade = MagicMock()

    # No rl_archive_generator (default None)
    ctrl = PopulationController(
        config, registry, mock_static, mock_cascade, MagicMock(),
    )

    assert ctrl._select_generator("deep") is mock_cascade
