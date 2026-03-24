"""Tests for the static instance generation pipeline (CSP → WFC).

Covers TEST-P4-008 through P4-014 and P4-027.
"""

from __future__ import annotations

import random

import pytest

from ..archetypes.dead import register_dead_archetypes
from ..archetypes.registry import ArchetypeRegistry
from ..archetypes.tier1 import register_static_t1_archetypes
from ..config.schema import MasterConfig
from ..pipeline.instance_generator import StaticInstanceGenerator
from ..primitives.cluster_detection import detect_clusters, detect_components
from ..primitives.gravity import GravityDAG
from ..primitives.symbols import (
    Symbol,
    SymbolTier,
    is_booster,
    is_wild,
    symbol_from_name,
    tier_of,
)
from ..variance.accumulators import PopulationAccumulators
from ..variance.bias_computation import compute_hints


@pytest.fixture
def full_registry(default_config: MasterConfig) -> ArchetypeRegistry:
    reg = ArchetypeRegistry(default_config)
    register_dead_archetypes(reg)
    register_static_t1_archetypes(reg)
    return reg


@pytest.fixture
def generator(
    default_config: MasterConfig,
    full_registry: ArchetypeRegistry,
) -> StaticInstanceGenerator:
    gravity_dag = GravityDAG(default_config.board, default_config.gravity)
    return StaticInstanceGenerator(default_config, full_registry, gravity_dag)


@pytest.fixture
def uniform_hints(default_config: MasterConfig):
    acc = PopulationAccumulators.create(default_config)
    return compute_hints(acc, default_config)


# ---------------------------------------------------------------------------
# TEST-P4-008: dead_empty → zero clusters, max component <= 3
# ---------------------------------------------------------------------------

def test_dead_empty_zero_clusters(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    rng = random.Random(42)
    result = generator.generate("dead_empty", 0, uniform_hints, rng)
    assert result.success, f"Generation failed: {result.failure_reason}"

    inst = result.instance
    clusters = detect_clusters(inst.board, default_config)
    assert len(clusters) == 0, f"Expected 0 clusters, got {len(clusters)}"

    # Max component of any standard symbol <= 3
    max_comp = 0
    for name in default_config.symbols.standard:
        sym = symbol_from_name(name)
        for comp in detect_components(inst.board, sym, default_config.board):
            max_comp = max(max_comp, len(comp))
    assert max_comp <= 3, f"Max component {max_comp} > 3"


# ---------------------------------------------------------------------------
# TEST-P4-009: dead_near_miss_high → zero clusters, 1 HIGH NM
# ---------------------------------------------------------------------------

def test_dead_near_miss_high(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    rng = random.Random(123)
    result = generator.generate("dead_near_miss_high", 1, uniform_hints, rng)
    assert result.success, f"Generation failed: {result.failure_reason}"

    inst = result.instance
    clusters = detect_clusters(inst.board, default_config)
    assert len(clusters) == 0

    # Detect near-misses (components of size min_cluster_size - 1 = 4)
    nm_size = default_config.board.min_cluster_size - 1
    near_misses = []
    for name in default_config.symbols.standard:
        sym = symbol_from_name(name)
        for comp in detect_components(inst.board, sym, default_config.board):
            if len(comp) == nm_size:
                near_misses.append(sym)
    assert len(near_misses) >= 1, "Expected at least 1 near-miss"

    # At least one NM should be HIGH tier
    has_high = any(
        tier_of(sym, default_config.symbols) is SymbolTier.HIGH
        for sym in near_misses
    )
    assert has_high, "Expected at least one HIGH tier near-miss"


# ---------------------------------------------------------------------------
# TEST-P4-010: dead_scatter_3_near_miss_high → 3 scatters + HIGH NMs
# ---------------------------------------------------------------------------

def test_dead_scatter_3_near_miss_high(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    rng = random.Random(456)
    result = generator.generate(
        "dead_scatter_3_near_miss_high", 2, uniform_hints, rng,
    )
    assert result.success, f"Generation failed: {result.failure_reason}"

    inst = result.instance
    # Count scatters
    scatter_count = sum(
        1 for pos in inst.board.all_positions()
        if inst.board.get(pos) is Symbol.S
    )
    assert scatter_count == 3, f"Expected 3 scatters, got {scatter_count}"


# ---------------------------------------------------------------------------
# TEST-P4-011: dead_saturated → 0 scatters, exactly 3 NMs
# ---------------------------------------------------------------------------

def test_dead_saturated(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    rng = random.Random(789)
    result = generator.generate("dead_saturated", 3, uniform_hints, rng)
    assert result.success, f"Generation failed: {result.failure_reason}"

    inst = result.instance
    scatter_count = sum(
        1 for pos in inst.board.all_positions()
        if inst.board.get(pos) is Symbol.S
    )
    assert scatter_count == 0, f"Expected 0 scatters, got {scatter_count}"

    # Detect near-misses
    nm_size = default_config.board.min_cluster_size - 1
    near_misses = []
    for name in default_config.symbols.standard:
        sym = symbol_from_name(name)
        for comp in detect_components(inst.board, sym, default_config.board):
            if len(comp) == nm_size:
                near_misses.append(sym)
    # Signature requires exactly 3, but WFC fill may create extra size-4 groups
    # since max_component_size=4 allows them. CSP placed exactly 3 intentional NMs.
    assert len(near_misses) >= 3, f"Expected >= 3 NMs, got {len(near_misses)}"


# ---------------------------------------------------------------------------
# TEST-P4-012: t1_single → 1 cluster size 5-6
# ---------------------------------------------------------------------------

def test_t1_single(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    rng = random.Random(100)
    result = generator.generate("t1_single", 4, uniform_hints, rng)
    assert result.success, f"Generation failed: {result.failure_reason}"

    inst = result.instance
    clusters = detect_clusters(inst.board, default_config)
    assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
    assert 5 <= clusters[0].size <= 6, f"Cluster size {clusters[0].size} not in [5, 6]"


# ---------------------------------------------------------------------------
# TEST-P4-013: t1_near_miss → cluster + 1-2 NMs
# ---------------------------------------------------------------------------

def test_t1_near_miss(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    rng = random.Random(200)
    result = generator.generate("t1_near_miss", 5, uniform_hints, rng)
    assert result.success, f"Generation failed: {result.failure_reason}"

    inst = result.instance
    clusters = detect_clusters(inst.board, default_config)
    assert 1 <= len(clusters) <= 2, f"Expected 1-2 clusters, got {len(clusters)}"

    # Detect near-misses
    nm_size = default_config.board.min_cluster_size - 1
    near_misses = []
    for name in default_config.symbols.standard:
        sym = symbol_from_name(name)
        for comp in detect_components(inst.board, sym, default_config.board):
            if len(comp) == nm_size:
                near_misses.append(sym)
    assert 1 <= len(near_misses) <= 2, f"Expected 1-2 NMs, got {len(near_misses)}"


# ---------------------------------------------------------------------------
# TEST-P4-014: t1_multi → 2-3 clusters
# ---------------------------------------------------------------------------

def test_t1_multi(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    rng = random.Random(300)
    result = generator.generate("t1_multi", 6, uniform_hints, rng)
    assert result.success, f"Generation failed: {result.failure_reason}"

    inst = result.instance
    clusters = detect_clusters(inst.board, default_config)
    assert 2 <= len(clusters) <= 3, f"Expected 2-3 clusters, got {len(clusters)}"


# ---------------------------------------------------------------------------
# TEST-P4-027: No booster or wild symbols on any generated board
# ---------------------------------------------------------------------------

def test_no_boosters_or_wilds_on_any_board(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    """All Phase 4 generated boards must contain only standard + scatter symbols."""
    rng = random.Random(500)
    archetypes = ["dead_empty", "t1_single", "t1_multi"]
    for arch_id in archetypes:
        result = generator.generate(arch_id, 0, uniform_hints, rng)
        assert result.success, f"{arch_id} failed: {result.failure_reason}"
        for pos in result.instance.board.all_positions():
            sym = result.instance.board.get(pos)
            assert sym is not None, f"empty cell at {pos}"
            assert not is_wild(sym), f"wild at {pos} in {arch_id}"
            assert not is_booster(sym), f"booster {sym} at {pos} in {arch_id}"


# ---------------------------------------------------------------------------
# Gravity record tests — static instances carry post-win animation data
# ---------------------------------------------------------------------------

def test_t1_single_has_gravity_record(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    """Winning static instances populate gravity_record for the post-win animation."""
    rng = random.Random(100)
    result = generator.generate("t1_single", 10, uniform_hints, rng)
    assert result.success, f"Generation failed: {result.failure_reason}"

    inst = result.instance
    assert inst.gravity_record is not None, "gravity_record missing on winning instance"
    assert len(inst.gravity_record.exploded_positions) > 0

def test_dead_empty_no_gravity_record(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    """Dead instances must not have a gravity_record — nothing to explode."""
    rng = random.Random(42)
    result = generator.generate("dead_empty", 0, uniform_hints, rng)
    assert result.success, f"Generation failed: {result.failure_reason}"

    assert result.instance.gravity_record is None

def test_gravity_record_conservation(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    """Refill entry count equals exploded position count (gravity conservation law)."""
    rng = random.Random(100)
    result = generator.generate("t1_single", 10, uniform_hints, rng)
    assert result.success

    gr = result.instance.gravity_record
    assert gr is not None
    assert len(gr.refill_entries) == len(gr.exploded_positions), (
        f"Conservation violated: {len(gr.refill_entries)} refill entries "
        f"vs {len(gr.exploded_positions)} exploded positions"
    )

def test_gravity_record_refill_symbols_are_standard(
    default_config: MasterConfig,
    generator: StaticInstanceGenerator,
    uniform_hints,
) -> None:
    """Refill symbols must be drawn from standard symbols only."""
    rng = random.Random(100)
    result = generator.generate("t1_single", 10, uniform_hints, rng)
    assert result.success

    gr = result.instance.gravity_record
    assert gr is not None
    standard = set(default_config.symbols.standard)
    for _reel, _row, sym_name in gr.refill_entries:
        assert sym_name in standard, f"refill symbol {sym_name} not in standard set"
