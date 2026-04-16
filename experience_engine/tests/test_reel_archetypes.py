"""Reel family archetype registration tests (TEST-REEL-020 through 023)."""

from __future__ import annotations

import pytest

from ..archetypes.reel import register_reel_archetypes
from ..archetypes.registry import (
    ArchetypeRegistry,
    FAMILY_CRITERIA,
    REGISTERED_FAMILIES,
)
from ..config.schema import MasterConfig


@pytest.fixture
def registry(default_config: MasterConfig) -> ArchetypeRegistry:
    return ArchetypeRegistry(default_config)


def test_reel_020_register_without_error(registry: ArchetypeRegistry) -> None:
    """TEST-REEL-020: register_reel_archetypes runs without raising."""
    register_reel_archetypes(registry)


def test_reel_021_family_contains_one_archetype(
    registry: ArchetypeRegistry,
) -> None:
    """TEST-REEL-021: registry.get_family("reel") returns 1 archetype."""
    register_reel_archetypes(registry)
    reel_family = registry.get_family("reel")
    assert len(reel_family) == 1
    assert reel_family[0].id == "reel_base"


def test_reel_022_reel_base_family_and_criteria(
    registry: ArchetypeRegistry,
) -> None:
    """TEST-REEL-022: reel_base has family="reel", criteria="basegame"."""
    register_reel_archetypes(registry)
    sig = registry.get("reel_base")
    assert sig.family == "reel"
    assert sig.criteria == "basegame"


def test_reel_023_registered_family_constants_include_reel() -> None:
    """TEST-REEL-023: "reel" present in REGISTERED_FAMILIES and FAMILY_CRITERIA."""
    assert "reel" in REGISTERED_FAMILIES
    assert FAMILY_CRITERIA["reel"] == "basegame"
