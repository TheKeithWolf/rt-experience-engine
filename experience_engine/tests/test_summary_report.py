"""Tests for the summary report.

Covers TEST-P4-024 and P4-025.
"""

from __future__ import annotations

import pytest

from ..archetypes.dead import register_dead_archetypes
from ..archetypes.registry import ArchetypeRegistry
from ..archetypes.tier1 import register_static_t1_archetypes
from ..config.schema import MasterConfig
from ..output.summary_report import (
    RunSummary,
    format_summary,
    generate_summary,
)
from ..pipeline.data_types import GeneratedInstance
from ..population.controller import PopulationResult
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..spatial_solver.data_types import SpatialStep


@pytest.fixture
def full_registry(default_config: MasterConfig) -> ArchetypeRegistry:
    reg = ArchetypeRegistry(default_config)
    register_dead_archetypes(reg)
    register_static_t1_archetypes(reg)
    return reg


def _make_instance(
    config: MasterConfig,
    sim_id: int,
    archetype_id: str,
    family: str,
    criteria: str,
    payout: float = 0.0,
) -> GeneratedInstance:
    board = Board.empty(config.board)
    # Fill with standard symbols
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            board.set(Position(reel, row), Symbol.L1)
    return GeneratedInstance(
        sim_id=sim_id,
        archetype_id=archetype_id,
        family=family,
        criteria=criteria,
        board=board,
        spatial_step=SpatialStep(
            clusters=(), near_misses=(), scatter_positions=frozenset(), boosters=(),
        ),
        payout=payout,
        centipayout=int(payout * 100),
        win_level=1,
    )


# ---------------------------------------------------------------------------
# TEST-P4-024: Summary lists all families with correct sim_id ranges
# ---------------------------------------------------------------------------

def test_summary_lists_families_and_sim_ids(
    default_config: MasterConfig,
    full_registry: ArchetypeRegistry,
) -> None:
    instances = [
        _make_instance(default_config, 0, "dead_empty", "dead", "0"),
        _make_instance(default_config, 1, "dead_empty", "dead", "0"),
        _make_instance(default_config, 2, "t1_single", "t1", "basegame", 0.5),
    ]
    pop_result = PopulationResult(
        instances=tuple(instances),
        metrics=(),
        total_generated=3,
        total_failed=0,
        failure_log=(),
    )

    summary = generate_summary(pop_result, default_config, full_registry)

    # Should have dead and t1 families
    family_names = {f.family for f in summary.families}
    assert "dead" in family_names
    assert "t1" in family_names

    # dead_empty sim_ids should be 0-1
    for fam in summary.families:
        if fam.family == "dead":
            for arch in fam.archetypes:
                if arch.archetype_id == "dead_empty":
                    assert arch.sim_id_start == 0
                    assert arch.sim_id_end == 1
                    assert arch.book_count == 2

    # t1_single sim_id should be 2
    for fam in summary.families:
        if fam.family == "t1":
            for arch in fam.archetypes:
                if arch.archetype_id == "t1_single":
                    assert arch.sim_id_start == 2
                    assert arch.sim_id_end == 2
                    assert arch.book_count == 1

    # Format should contain key text
    text = format_summary(summary)
    assert "EXPERIENCE ENGINE RUN SUMMARY" in text
    assert "dead_empty" in text
    assert "t1_single" in text


# ---------------------------------------------------------------------------
# TEST-P4-025: Summary failure counts match controller logs
# ---------------------------------------------------------------------------

def test_summary_failure_counts(
    default_config: MasterConfig,
    full_registry: ArchetypeRegistry,
) -> None:
    instances = [
        _make_instance(default_config, 0, "dead_empty", "dead", "0"),
    ]
    failure_log = (
        ("dead_near_miss_low", 100, "solver failed"),
        ("dead_near_miss_low", 100, "solver failed"),
        ("t1_single", 50, "fill failed"),
    )
    pop_result = PopulationResult(
        instances=tuple(instances),
        metrics=(),
        total_generated=1,
        total_failed=3,
        failure_log=failure_log,
    )

    summary = generate_summary(pop_result, default_config, full_registry)

    assert summary.total_failed == 3
    assert summary.total_generated == 1

    # Check per-archetype failure counts in family summaries
    for fam in summary.families:
        for arch in fam.archetypes:
            if arch.archetype_id == "dead_near_miss_low":
                assert arch.failure_count == 2
            elif arch.archetype_id == "t1_single":
                assert arch.failure_count == 1
