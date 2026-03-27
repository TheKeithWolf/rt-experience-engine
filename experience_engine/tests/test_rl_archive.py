"""Tests for rl_archive/archive.py and rl_archive/quality.py.

RLA-010 through RLA-021: Archive insertion, replacement, sampling, coverage,
and quality scoring components.
"""

from __future__ import annotations

import random

import pytest

from ..config.schema import (
    BoardConfig,
    ConfigValidationError,
    DescriptorConfig,
    GridMultiplierConfig,
    QualityConfig,
)
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.data_types import CascadeStepRecord, GeneratedInstance
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..rl_archive.archive import ArchiveEmpty, ArchiveEntry, MAPElitesArchive
from ..rl_archive.descriptor import TrajectoryDescriptor
from ..rl_archive.quality import CascadeQualityScorer
from ..spatial_solver.data_types import ClusterAssignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOARD_CONFIG = BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5)
_DESCRIPTOR_CONFIG = DescriptorConfig(
    spatial_col_bins=3, spatial_row_bins=3, payout_bins=4,
)
_GRID_MULT_CONFIG = GridMultiplierConfig(
    initial_value=0, first_hit_value=1, increment=1, cap=100,
    minimum_contribution=1,
)
_NUM_SYMBOLS = 7  # L1-L4, H1-H3


def _make_descriptor(
    archetype_id: str = "test",
    symbol: str = "L1",
    spatial: tuple[int, int] = (0, 0),
    orientation: str = "H",
    payout_bin: int = 0,
) -> TrajectoryDescriptor:
    return TrajectoryDescriptor(
        archetype_id=archetype_id,
        step0_symbol=symbol,
        spatial_bin=spatial,
        cluster_orientation=orientation,
        payout_bin=payout_bin,
    )


def _make_cluster(
    symbol: Symbol = Symbol.L1,
    positions: frozenset[Position] | None = None,
    size: int = 5,
) -> ClusterAssignment:
    pos = positions or frozenset(
        Position(0, i) for i in range(size)
    )
    return ClusterAssignment(
        symbol=symbol, positions=pos, size=len(pos), wild_positions=frozenset(),
    )


def _make_step_record(
    step_payout: float = 1.0,
    step_index: int = 0,
    clusters: tuple[ClusterAssignment, ...] | None = None,
    grid_multipliers: tuple[int, ...] | None = None,
) -> CascadeStepRecord:
    board = Board.empty(_BOARD_CONFIG)
    return CascadeStepRecord(
        step_index=step_index,
        board_before=board,
        board_after=board,
        clusters=clusters or (_make_cluster(),),
        step_payout=step_payout,
        grid_multipliers_snapshot=grid_multipliers or tuple(0 for _ in range(49)),
        booster_spawn_types=(),
        booster_spawn_positions=(),
        booster_fire_records=(),
        gravity_record=None,
        booster_gravity_record=None,
    )


def _make_instance(
    payout: float = 5.0,
    cascade_steps: tuple[CascadeStepRecord, ...] | None = None,
) -> GeneratedInstance:
    board = Board.empty(_BOARD_CONFIG)
    steps = cascade_steps or (_make_step_record(),)
    return GeneratedInstance(
        sim_id=1,
        archetype_id="test",
        family="test",
        criteria="basegame",
        board=board,
        spatial_step=None,  # type: ignore[arg-type]
        payout=payout,
        centipayout=int(payout * 100),
        win_level=1,
        cascade_steps=steps,
        gravity_record=None,
    )


def _make_arc(
    payout_min: float = 0.0,
    payout_max: float = 10.0,
) -> NarrativeArc:
    """Minimal NarrativeArc for quality scoring tests."""
    phase = NarrativePhase(
        id="p1",
        intent="test phase",
        repetitions=Range(1, 3),
        cluster_count=Range(1, 2),
        cluster_sizes=(Range(5, 10),),
        cluster_symbol_tier=None,
        spawns=None,
        arms=None,
        fires=None,
        wild_behavior=None,
        ends_when="always",
    )
    return NarrativeArc(
        phases=(phase,),
        payout=RangeFloat(payout_min, payout_max),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )


def _make_archive() -> MAPElitesArchive:
    return MAPElitesArchive(_DESCRIPTOR_CONFIG, _NUM_SYMBOLS)


def _make_quality_scorer(
    quality_config: QualityConfig | None = None,
) -> CascadeQualityScorer:
    qc = quality_config or QualityConfig(
        payout_centering_weight=2.0,
        escalation_weight=1.0,
        cluster_size_weight=0.5,
        productivity_weight=1.0,
        multiplier_engagement_weight=0.5,
    )
    return CascadeQualityScorer(qc, _BOARD_CONFIG, _GRID_MULT_CONFIG)


# ---------------------------------------------------------------------------
# RLA-010: Insert into empty cell
# ---------------------------------------------------------------------------


def test_rla_010_insert_empty_cell() -> None:
    """RLA-010: Insert into empty cell returns True, filled_count increments."""
    archive = _make_archive()
    assert archive.filled_count() == 0

    desc = _make_descriptor()
    instance = _make_instance()
    result = archive.try_insert(instance, desc, quality=1.0)

    assert result is True
    assert archive.filled_count() == 1


# ---------------------------------------------------------------------------
# RLA-011: Insert higher quality replaces incumbent
# ---------------------------------------------------------------------------


def test_rla_011_higher_quality_replaces() -> None:
    """RLA-011: Higher quality instance replaces the incumbent."""
    archive = _make_archive()
    desc = _make_descriptor()
    inst_low = _make_instance(payout=3.0)
    inst_high = _make_instance(payout=7.0)

    archive.try_insert(inst_low, desc, quality=1.0)
    result = archive.try_insert(inst_high, desc, quality=2.0)

    assert result is True
    assert archive.filled_count() == 1
    from ..rl_archive.archive import _descriptor_to_key
    entry = archive.get(_descriptor_to_key(desc))
    assert entry is not None
    assert entry.quality == 2.0


# ---------------------------------------------------------------------------
# RLA-012: Insert lower quality returns False
# ---------------------------------------------------------------------------


def test_rla_012_lower_quality_rejected() -> None:
    """RLA-012: Lower quality instance does not replace incumbent."""
    archive = _make_archive()
    desc = _make_descriptor()
    inst = _make_instance()

    archive.try_insert(inst, desc, quality=5.0)
    result = archive.try_insert(inst, desc, quality=3.0)

    assert result is False
    assert archive.filled_count() == 1
    from ..rl_archive.archive import _descriptor_to_key
    entry = archive.get(_descriptor_to_key(desc))
    assert entry is not None
    assert entry.quality == 5.0


# ---------------------------------------------------------------------------
# RLA-013: Coverage = filled / total
# ---------------------------------------------------------------------------


def test_rla_013_coverage_ratio() -> None:
    """RLA-013: coverage() equals filled_count / total_cells."""
    archive = _make_archive()
    assert archive.coverage() == 0.0

    desc = _make_descriptor()
    archive.try_insert(_make_instance(), desc, quality=1.0)

    expected = 1 / archive.total_cells()
    assert archive.coverage() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# RLA-014: Sample from occupied cells
# ---------------------------------------------------------------------------


def test_rla_014_sample_from_occupied() -> None:
    """RLA-014: sample() with uniform weights returns from occupied cells."""
    archive = _make_archive()
    desc = _make_descriptor()
    instance = _make_instance()
    archive.try_insert(instance, desc, quality=1.0)

    rng = random.Random(42)
    sampled = archive.sample({}, rng)
    assert sampled.sim_id == instance.sim_id


# ---------------------------------------------------------------------------
# RLA-015: Zero-weight cell never sampled (statistical)
# ---------------------------------------------------------------------------


def test_rla_015_zero_weight_never_sampled() -> None:
    """RLA-015: Cell with zero weight is never returned (statistical test)."""
    archive = _make_archive()
    from ..rl_archive.archive import _descriptor_to_key

    desc_a = _make_descriptor(symbol="L1")
    desc_b = _make_descriptor(symbol="L2")
    inst_a = _make_instance(payout=1.0)
    inst_b = _make_instance(payout=2.0)

    archive.try_insert(inst_a, desc_a, quality=1.0)
    archive.try_insert(inst_b, desc_b, quality=1.0)

    key_a = _descriptor_to_key(desc_a)
    key_b = _descriptor_to_key(desc_b)

    # Give zero weight to cell A, all weight to cell B
    weights = {key_a: 0.0, key_b: 1.0}
    rng = random.Random(42)

    # Sample 100 times — none should be from cell A
    samples = [archive.sample(weights, rng) for _ in range(100)]
    assert all(s.payout == 2.0 for s in samples)


# ---------------------------------------------------------------------------
# RLA-016: Sample from empty archive raises ArchiveEmpty
# ---------------------------------------------------------------------------


def test_rla_016_sample_empty_raises() -> None:
    """RLA-016: Sampling an empty archive raises ArchiveEmpty."""
    archive = _make_archive()
    rng = random.Random(42)
    with pytest.raises(ArchiveEmpty):
        archive.sample({}, rng)


# ---------------------------------------------------------------------------
# RLA-017: all_keys returns full set
# ---------------------------------------------------------------------------


def test_rla_017_all_keys_includes_occupied() -> None:
    """RLA-017: all_keys() includes occupied keys."""
    archive = _make_archive()
    desc = _make_descriptor()
    archive.try_insert(_make_instance(), desc, quality=1.0)

    # occupied_keys should be a subset of all_keys
    assert archive.occupied_keys() <= archive.all_keys()


# ---------------------------------------------------------------------------
# RLA-018: occupied_keys ⊆ all_keys
# ---------------------------------------------------------------------------


def test_rla_018_occupied_subset_of_all() -> None:
    """RLA-018: occupied_keys() is a subset of all_keys()."""
    archive = _make_archive()

    # Insert two different cells
    desc_a = _make_descriptor(symbol="L1")
    desc_b = _make_descriptor(symbol="H1")
    archive.try_insert(_make_instance(), desc_a, quality=1.0)
    archive.try_insert(_make_instance(), desc_b, quality=1.0)

    occupied = archive.occupied_keys()
    all_keys = archive.all_keys()
    assert occupied <= all_keys
    assert len(occupied) == 2


# ---------------------------------------------------------------------------
# RLA-019: QualityConfig rejects negative weights
# ---------------------------------------------------------------------------


def test_rla_019_quality_config_rejects_negative() -> None:
    """RLA-019: QualityConfig raises on negative weights."""
    with pytest.raises(ConfigValidationError, match="payout_centering_weight"):
        QualityConfig(
            payout_centering_weight=-1.0,
            escalation_weight=1.0,
            cluster_size_weight=0.5,
            productivity_weight=1.0,
            multiplier_engagement_weight=0.5,
        )

    with pytest.raises(ConfigValidationError, match="multiplier_engagement_weight"):
        QualityConfig(
            payout_centering_weight=1.0,
            escalation_weight=1.0,
            cluster_size_weight=0.5,
            productivity_weight=1.0,
            multiplier_engagement_weight=-0.1,
        )


# ---------------------------------------------------------------------------
# RLA-020: Higher score for centered payout
# ---------------------------------------------------------------------------


def test_rla_020_centered_payout_higher_score() -> None:
    """RLA-020: Payout at arc midpoint scores higher than at edge."""
    arc = _make_arc(payout_min=0.0, payout_max=10.0)
    # Only payout centering enabled
    scorer = _make_quality_scorer(QualityConfig(
        payout_centering_weight=1.0,
        escalation_weight=0.0,
        cluster_size_weight=0.0,
        productivity_weight=0.0,
        multiplier_engagement_weight=0.0,
    ))

    inst_centered = _make_instance(payout=5.0)
    inst_edge = _make_instance(payout=0.5)

    score_centered = scorer.score(inst_centered, arc)
    score_edge = scorer.score(inst_edge, arc)
    assert score_centered > score_edge


# ---------------------------------------------------------------------------
# RLA-021: Higher score for escalating payouts
# ---------------------------------------------------------------------------


def test_rla_021_escalating_higher_score() -> None:
    """RLA-021: Escalating payouts score higher than flat/decreasing."""
    arc = _make_arc()
    # Only escalation enabled
    scorer = _make_quality_scorer(QualityConfig(
        payout_centering_weight=0.0,
        escalation_weight=1.0,
        cluster_size_weight=0.0,
        productivity_weight=0.0,
        multiplier_engagement_weight=0.0,
    ))

    # Escalating: 1.0, 2.0, 3.0
    steps_escalating = (
        _make_step_record(step_payout=1.0, step_index=0),
        _make_step_record(step_payout=2.0, step_index=1),
        _make_step_record(step_payout=3.0, step_index=2),
    )
    inst_escalating = _make_instance(cascade_steps=steps_escalating)

    # Flat: 2.0, 2.0, 2.0 — still non-decreasing, should also score 1.0
    steps_flat = (
        _make_step_record(step_payout=2.0, step_index=0),
        _make_step_record(step_payout=2.0, step_index=1),
        _make_step_record(step_payout=2.0, step_index=2),
    )
    inst_flat = _make_instance(cascade_steps=steps_flat)

    # Decreasing: 3.0, 2.0, 1.0
    steps_decreasing = (
        _make_step_record(step_payout=3.0, step_index=0),
        _make_step_record(step_payout=2.0, step_index=1),
        _make_step_record(step_payout=1.0, step_index=2),
    )
    inst_decreasing = _make_instance(cascade_steps=steps_decreasing)

    score_escalating = scorer.score(inst_escalating, arc)
    score_flat = scorer.score(inst_flat, arc)
    score_decreasing = scorer.score(inst_decreasing, arc)

    # Both escalating and flat are fully non-decreasing
    assert score_escalating == pytest.approx(score_flat)
    # Decreasing should score lower
    assert score_escalating > score_decreasing
