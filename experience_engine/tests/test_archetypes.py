"""Tests for archetype registry, signature validation, and archetype definitions.

Covers TEST-P4-001 through P4-004 and P4-026.
"""

from __future__ import annotations

import pytest

from ..archetypes.dead import register_dead_archetypes
from ..archetypes.registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    SignatureValidationError,
)
from ..archetypes.tier1 import register_static_t1_archetypes
from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(default_config: MasterConfig) -> ArchetypeRegistry:
    """Empty registry initialized with default config."""
    return ArchetypeRegistry(default_config)


@pytest.fixture
def full_registry(default_config: MasterConfig) -> ArchetypeRegistry:
    """Registry with all Phase 4 archetypes registered."""
    reg = ArchetypeRegistry(default_config)
    register_dead_archetypes(reg)
    register_static_t1_archetypes(reg)
    return reg


def _make_valid_dead(id: str = "test_dead") -> dict:
    """Minimal valid dead signature kwargs."""
    return dict(
        id=id,
        family="dead",
        criteria="0",
        required_cluster_count=Range(0, 0),
        required_cluster_sizes=(),
        required_cluster_symbols=None,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=3,
        required_cascade_depth=Range(0, 0),
        cascade_steps=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.0, 0.0),
        triggers_freespin=False,
        reaches_wincap=False,
    )


# ---------------------------------------------------------------------------
# TEST-P4-001: Signature validation accepts valid dead_empty
# ---------------------------------------------------------------------------

def test_valid_dead_empty_accepted(registry: ArchetypeRegistry) -> None:
    sig = ArchetypeSignature(**_make_valid_dead("dead_empty"))
    registry.register(sig)
    assert registry.get("dead_empty") is sig


# ---------------------------------------------------------------------------
# TEST-P4-002: Signature validation rejects bad inputs
# ---------------------------------------------------------------------------

def test_rejects_missing_id_duplicate(registry: ArchetypeRegistry) -> None:
    """CONTRACT-SIG-1: duplicate id raises."""
    sig = ArchetypeSignature(**_make_valid_dead("dup"))
    registry.register(sig)
    with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-1"):
        registry.register(ArchetypeSignature(**_make_valid_dead("dup")))


def test_rejects_bad_family(registry: ArchetypeRegistry) -> None:
    """CONTRACT-SIG-2: unknown family raises."""
    kwargs = _make_valid_dead("bad_family")
    kwargs["family"] = "nonexistent"
    with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-2"):
        registry.register(ArchetypeSignature(**kwargs))


def test_rejects_dead_nonzero_payout(registry: ArchetypeRegistry) -> None:
    """CONTRACT-SIG-3: dead family with non-zero payout raises."""
    kwargs = _make_valid_dead("bad_payout")
    kwargs["payout_range"] = RangeFloat(0.0, 1.0)
    with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-3"):
        registry.register(ArchetypeSignature(**kwargs))


def test_rejects_freespin_insufficient_scatters(
    registry: ArchetypeRegistry,
) -> None:
    """CONTRACT-SIG-4: triggers_freespin without enough scatters raises."""
    kwargs = _make_valid_dead("bad_fs")
    kwargs["family"] = "trigger"
    kwargs["criteria"] = "freegame"
    kwargs["payout_range"] = RangeFloat(0.0, 100.0)
    kwargs["triggers_freespin"] = True
    # min_scatters_to_trigger is 4, but scatter count min is 0
    kwargs["required_scatter_count"] = Range(0, 5)
    with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-4"):
        registry.register(ArchetypeSignature(**kwargs))


def test_rejects_cascade_steps_when_depth_zero(
    registry: ArchetypeRegistry,
) -> None:
    """CONTRACT-SIG-5: cascade_depth.max=0 with cascade_steps raises."""
    from ..archetypes.registry import CascadeStepConstraint
    kwargs = _make_valid_dead("bad_cascade")
    kwargs["cascade_steps"] = (
        CascadeStepConstraint(
            cluster_count=Range(1, 1),
            cluster_sizes=(Range(5, 6),),
            cluster_symbol_tier=None,
            must_spawn_booster=None,
            must_arm_booster=None,
            must_fire_booster=None,
        ),
    )
    with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-5"):
        registry.register(ArchetypeSignature(**kwargs))


def test_rejects_booster_fires_without_cascade(
    registry: ArchetypeRegistry,
) -> None:
    """CONTRACT-SIG-6: booster_fires without cascade_depth >= 1 raises."""
    kwargs = _make_valid_dead("bad_booster")
    kwargs["family"] = "rocket"
    kwargs["criteria"] = "basegame"
    kwargs["payout_range"] = RangeFloat(0.0, 100.0)
    kwargs["required_booster_fires"] = {"R": Range(1, 1)}
    with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-6"):
        registry.register(ArchetypeSignature(**kwargs))


def test_rejects_chain_without_two_booster_types(
    registry: ArchetypeRegistry,
) -> None:
    """CONTRACT-SIG-7: chain_depth > 0 with < 2 booster types raises."""
    kwargs = _make_valid_dead("bad_chain")
    kwargs["family"] = "chain"
    kwargs["criteria"] = "basegame"
    kwargs["payout_range"] = RangeFloat(0.0, 100.0)
    kwargs["required_cascade_depth"] = Range(1, 3)
    kwargs["required_chain_depth"] = Range(1, 2)
    kwargs["required_booster_fires"] = {"R": Range(1, 1)}
    with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-7"):
        registry.register(ArchetypeSignature(**kwargs))


# ---------------------------------------------------------------------------
# TEST-P4-003: Registry rejects duplicate IDs
# ---------------------------------------------------------------------------

def test_duplicate_id_rejected(registry: ArchetypeRegistry) -> None:
    registry.register(ArchetypeSignature(**_make_valid_dead("unique_1")))
    with pytest.raises(SignatureValidationError):
        registry.register(ArchetypeSignature(**_make_valid_dead("unique_1")))


# ---------------------------------------------------------------------------
# TEST-P4-004: get_family("dead") returns 11 archetypes
# ---------------------------------------------------------------------------

def test_dead_family_has_11_archetypes(full_registry: ArchetypeRegistry) -> None:
    dead = full_registry.get_family("dead")
    assert len(dead) == 11


def test_t1_family_has_4_static_archetypes(
    full_registry: ArchetypeRegistry,
) -> None:
    t1 = full_registry.get_family("t1")
    assert len(t1) == 4


def test_total_registered_is_15(full_registry: ArchetypeRegistry) -> None:
    assert len(full_registry.all_ids()) == 15


# ---------------------------------------------------------------------------
# TEST-P4-026: Config validation rejects weights summing to 0
# (tested via allocator — see test_population.py)
# Here we verify the registry handles empty families gracefully.
# ---------------------------------------------------------------------------

def test_empty_family_returns_empty_tuple(
    full_registry: ArchetypeRegistry,
) -> None:
    """Families with no registered archetypes return empty tuple."""
    wild = full_registry.get_family("wild")
    assert wild == ()
