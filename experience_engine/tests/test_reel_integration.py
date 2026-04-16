"""Pipeline integration tests for the reel family (TEST-REEL-050 through 055).

Covers:
  - Registry assembles with reel archetypes alongside existing families
  - Budget allocation includes reel_base
  - PopulationController dispatch picks the right generator per family
  - Missing reel_generator raises a clear error when needed
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ..archetypes.bomb import register_bomb_archetypes
from ..archetypes.chain import register_chain_archetypes
from ..archetypes.dead import register_dead_archetypes
from ..archetypes.lb import register_lb_archetypes
from ..archetypes.reel import register_reel_archetypes
from ..archetypes.registry import ArchetypeRegistry
from ..archetypes.rocket import register_rocket_archetypes
from ..archetypes.slb import register_slb_archetypes
from ..archetypes.tier1 import (
    register_cascade_t1_archetypes,
    register_static_t1_archetypes,
)
from ..archetypes.trigger import register_trigger_archetypes
from ..archetypes.wincap import register_wincap_archetypes
from ..archetypes.wild import register_wild_archetypes
from ..config.schema import MasterConfig
from ..pipeline.cascade_generator import CascadeInstanceGenerator
from ..pipeline.instance_generator import StaticInstanceGenerator
from ..pipeline.reel_generator import ReelStripGenerator
from ..population.allocator import allocate_budget
from ..population.controller import PopulationController
from ..primitives.gravity import GravityDAG
from ..primitives.reel_strip import load_reel_strip
from ..validation.validator import InstanceValidator

REFERENCE_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "reel_strip.csv"
)

_ALL_REGISTRARS = (
    register_dead_archetypes,
    register_static_t1_archetypes,
    register_cascade_t1_archetypes,
    register_wild_archetypes,
    register_rocket_archetypes,
    register_bomb_archetypes,
    register_lb_archetypes,
    register_slb_archetypes,
    register_chain_archetypes,
    register_trigger_archetypes,
    register_wincap_archetypes,
    register_reel_archetypes,
)


def _full_registry(config: MasterConfig) -> ArchetypeRegistry:
    registry = ArchetypeRegistry(config)
    for registrar in _ALL_REGISTRARS:
        registrar(registry)
    return registry


# ---------------------------------------------------------------------------
# TEST-REEL-050 / 051
# ---------------------------------------------------------------------------

def test_reel_050_registry_includes_reel_base(default_config: MasterConfig) -> None:
    """TEST-REEL-050: full registry includes reel_base alongside other families."""
    registry = _full_registry(default_config)
    assert "reel_base" in registry.all_ids()
    # Still contains existing families — reel registration is additive.
    assert "dead_empty" in registry.all_ids()


def test_reel_051_allocate_budget_includes_reel_base(
    default_config: MasterConfig,
) -> None:
    """TEST-REEL-051: allocate_budget gives reel_base at least one slot."""
    registry = _full_registry(default_config)
    allocations = allocate_budget(default_config, registry)
    reel_allocs = [a for a in allocations if a.archetype_id == "reel_base"]
    assert len(reel_allocs) == 1
    assert reel_allocs[0].count >= 1


# ---------------------------------------------------------------------------
# TEST-REEL-053 / 054 / 055 — controller dispatch
# ---------------------------------------------------------------------------

def _make_controller(
    default_config: MasterConfig,
    reel_generator: ReelStripGenerator | None,
) -> PopulationController:
    registry = _full_registry(default_config)
    # Stand-ins — dispatch doesn't call these; we only exercise _select_generator.
    static_gen = MagicMock(spec=StaticInstanceGenerator)
    cascade_gen = MagicMock(spec=CascadeInstanceGenerator)
    validator = MagicMock(spec=InstanceValidator)
    return PopulationController(
        default_config, registry, static_gen, cascade_gen, validator,
        reel_generator=reel_generator,
    )


def test_reel_053_reel_family_routes_to_reel_generator(
    default_config: MasterConfig,
) -> None:
    """TEST-REEL-053: _select_generator returns ReelStripGenerator for reel family."""
    dag = GravityDAG(default_config.board, default_config.gravity)
    strip = load_reel_strip(REFERENCE_CSV, default_config.board)
    registry = _full_registry(default_config)
    reel_gen = ReelStripGenerator(default_config, registry, dag, strip)

    controller = _make_controller(default_config, reel_gen)
    selected = controller._select_generator("reel_base")
    assert selected is reel_gen


def test_reel_054_dead_family_still_routes_to_static(
    default_config: MasterConfig,
) -> None:
    """TEST-REEL-054: Depth-0 archetypes still route to StaticInstanceGenerator."""
    controller = _make_controller(default_config, reel_generator=None)
    selected = controller._select_generator("dead_empty")
    # MagicMock spec means identity check is against the stored static gen.
    assert selected is controller._static_generator


def test_reel_055_reel_without_generator_raises(
    default_config: MasterConfig,
) -> None:
    """TEST-REEL-055: Reel archetype with no generator provided → RuntimeError."""
    controller = _make_controller(default_config, reel_generator=None)
    with pytest.raises(RuntimeError, match="reel family"):
        controller._select_generator("reel_base")
