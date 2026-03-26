"""Tests for narrative signature migration (NRA-040 through NRA-049).

Validates the new narrative_arc field on ArchetypeSignature, the
build_arc_signature factory, and updated CONTRACT-SIG validations.
"""

from __future__ import annotations

import pytest

from ..archetypes.registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    SignatureValidationError,
    build_arc_signature,
)
from ..config.schema import MasterConfig
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase(**overrides) -> NarrativePhase:
    defaults = dict(
        id="p", intent="test", repetitions=Range(1, 1),
        cluster_count=Range(1, 2), cluster_sizes=(Range(5, 6),),
        cluster_symbol_tier=None, spawns=None, arms=None, fires=None,
        wild_behavior=None, ends_when="always",
    )
    defaults.update(overrides)
    return NarrativePhase(**defaults)


def _arc(*phases: NarrativePhase, **overrides) -> NarrativeArc:
    defaults = dict(
        phases=phases, payout=RangeFloat(0.1, 10.0),
        wild_count_on_terminal=Range(0, 0), terminal_near_misses=None,
        dormant_boosters_on_terminal=None, required_chain_depth=Range(0, 0),
        rocket_orientation=None, lb_target_tier=None,
    )
    defaults.update(overrides)
    return NarrativeArc(**defaults)


def _depth0_sig(**overrides) -> ArchetypeSignature:
    """Build a valid depth-0 signature for testing."""
    defaults = dict(
        id="test_depth0", family="dead", criteria="0",
        required_cluster_count=Range(0, 0),
        required_cluster_sizes=(), required_cluster_symbols=None,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None, max_component_size=3,
        required_cascade_depth=Range(0, 0),
        cascade_steps=None,
        required_booster_spawns={}, required_booster_fires={},
        required_chain_depth=Range(0, 0), rocket_orientation=None,
        lb_target_tier=None, symbol_tier_per_step=None,
        terminal_near_misses=None, dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.0, 0.0),
        triggers_freespin=False, reaches_wincap=False,
        narrative_arc=None,
    )
    defaults.update(overrides)
    return ArchetypeSignature(**defaults)


# ---------------------------------------------------------------------------
# NRA-042: ArchetypeSignature has narrative_arc field
# ---------------------------------------------------------------------------

class TestNRA042:
    def test_narrative_arc_field_exists(self):
        sig = _depth0_sig()
        assert hasattr(sig, "narrative_arc")
        assert sig.narrative_arc is None


# ---------------------------------------------------------------------------
# NRA-043: build_arc_signature computes required_cascade_depth from arc
# ---------------------------------------------------------------------------

class TestNRA043:
    def test_cascade_depth_derived(self):
        arc = _arc(_phase(repetitions=Range(2, 3)))
        sig = build_arc_signature(
            arc,
            id="test_arc", family="t1", criteria="basegame",
            required_cluster_symbols=None,
            required_scatter_count=Range(0, 0),
            required_near_miss_count=Range(0, 0),
            required_near_miss_symbol_tier=None,
            max_component_size=None,
            triggers_freespin=False, reaches_wincap=False,
        )
        assert sig.required_cascade_depth == Range(2, 3)
        assert sig.narrative_arc is arc


# ---------------------------------------------------------------------------
# NRA-044: build_arc_signature computes required_booster_spawns from arc
# ---------------------------------------------------------------------------

class TestNRA044:
    def test_booster_spawns_derived(self):
        arc = _arc(
            _phase(spawns=("R",), repetitions=Range(1, 2)),
            _phase(id="p2", spawns=("R",), repetitions=Range(1, 1)),
        )
        sig = build_arc_signature(
            arc,
            id="test_spawns", family="rocket", criteria="basegame",
            required_cluster_symbols=None,
            required_scatter_count=Range(0, 0),
            required_near_miss_count=Range(0, 0),
            required_near_miss_symbol_tier=None,
            max_component_size=None,
            triggers_freespin=False, reaches_wincap=False,
        )
        assert sig.required_booster_spawns == {"R": Range(2, 3)}


# ---------------------------------------------------------------------------
# NRA-045: build_arc_signature computes required_booster_fires from arc
# ---------------------------------------------------------------------------

class TestNRA045:
    def test_booster_fires_derived(self):
        arc = _arc(_phase(fires=("B",), repetitions=Range(1, 1)))
        sig = build_arc_signature(
            arc,
            id="test_fires", family="bomb", criteria="basegame",
            required_cluster_symbols=None,
            required_scatter_count=Range(0, 0),
            required_near_miss_count=Range(0, 0),
            required_near_miss_symbol_tier=None,
            max_component_size=None,
            triggers_freespin=False, reaches_wincap=False,
        )
        assert sig.required_booster_fires == {"B": Range(1, 1)}


# ---------------------------------------------------------------------------
# NRA-046: CONTRACT-SIG-5 updated: depth 0 + narrative_arc present raises
# ---------------------------------------------------------------------------

class TestNRA046:
    def test_depth0_with_arc_raises(self, default_config: MasterConfig):
        registry = ArchetypeRegistry(default_config)
        sig = _depth0_sig(
            narrative_arc=_arc(_phase()),
            required_cascade_depth=Range(0, 0),
        )
        with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-5"):
            registry.register(sig)


# ---------------------------------------------------------------------------
# NRA-047: CONTRACT-SIG-ARC: unknown ends_when value raises
# ---------------------------------------------------------------------------

class TestNRA047:
    def test_unknown_ends_when_raises(self, default_config: MasterConfig):
        bad_phase = _phase(ends_when="nonexistent_rule")
        arc = _arc(bad_phase)
        sig = build_arc_signature(
            arc,
            id="test_bad_rule", family="t1", criteria="basegame",
            required_cluster_symbols=None,
            required_scatter_count=Range(0, 0),
            required_near_miss_count=Range(0, 0),
            required_near_miss_symbol_tier=None,
            max_component_size=None,
            triggers_freespin=False, reaches_wincap=False,
        )
        registry = ArchetypeRegistry(default_config)
        with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-ARC"):
            registry.register(sig)


# ---------------------------------------------------------------------------
# NRA-048: Depth-0 signature with narrative_arc=None passes validation
# ---------------------------------------------------------------------------

class TestNRA048:
    def test_depth0_none_arc_passes(self, default_config: MasterConfig):
        registry = ArchetypeRegistry(default_config)
        sig = _depth0_sig()
        registry.register(sig)
        assert registry.get("test_depth0") is sig


# ---------------------------------------------------------------------------
# NRA-049: build_arc_signature carries arc's global constraints through
# ---------------------------------------------------------------------------

class TestNRA049:
    def test_global_constraints_carried(self):
        arc = _arc(
            _phase(),
            rocket_orientation="H",
            lb_target_tier=SymbolTier.LOW,
            required_chain_depth=Range(1, 2),
        )
        sig = build_arc_signature(
            arc,
            id="test_global", family="chain", criteria="basegame",
            required_cluster_symbols=None,
            required_scatter_count=Range(0, 0),
            required_near_miss_count=Range(0, 0),
            required_near_miss_symbol_tier=None,
            max_component_size=None,
            triggers_freespin=False, reaches_wincap=False,
        )
        assert sig.rocket_orientation == "H"
        assert sig.lb_target_tier is SymbolTier.LOW
        assert sig.required_chain_depth == Range(1, 2)
        assert sig.payout_range == arc.payout
