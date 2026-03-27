"""Tests for try_advance_phase (TPA-001 through TPA-008).

Validates the shared phase advancement function used by cascade generator,
RL environment, and debug archetype.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ..archetypes.registry import ArchetypeSignature
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..narrative.transitions import try_advance_phase
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier
from ..step_reasoner.progress import ProgressTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase(
    id: str = "p",
    repetitions: Range = Range(1, 1),
    ends_when: str = "always",
) -> NarrativePhase:
    return NarrativePhase(
        id=id, intent="test", repetitions=repetitions,
        cluster_count=Range(1, 2), cluster_sizes=(Range(5, 6),),
        cluster_symbol_tier=None,
        spawns=None, arms=None, fires=None,
        wild_behavior=None, ends_when=ends_when,
    )


def _arc(*phases: NarrativePhase) -> NarrativeArc:
    return NarrativeArc(
        phases=phases,
        payout=RangeFloat(0.0, 100.0),
        wild_count_on_terminal=Range(0, 10),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )


def _signature(arc: NarrativeArc | None = None, **overrides) -> ArchetypeSignature:
    defaults = dict(
        id="test_sig",
        family="t1",
        criteria="basegame",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 8),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        max_component_size=None,
        required_cascade_depth=Range(2, 5),
        cascade_steps=None,
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        symbol_tier_per_step=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        payout_range=RangeFloat(0.0, 50.0),
        triggers_freespin=False,
        reaches_wincap=False,
        narrative_arc=arc,
    )
    defaults.update(overrides)
    return ArchetypeSignature(**defaults)


def _progress(sig: ArchetypeSignature, **overrides) -> ProgressTracker:
    defaults = dict(signature=sig, centipayout_multiplier=100)
    defaults.update(overrides)
    return ProgressTracker(**defaults)


@dataclass(frozen=True)
class _FakeStepResult:
    """Minimal StepResult stand-in — only needs clusters and fires."""
    clusters: tuple = ()
    fires: tuple = ()
    step_payout: int = 0
    step_index: int = 0
    spawns: tuple = ()
    symbol_tier: SymbolTier | None = None


# Transition rules for testing — context-free versions
_TEST_RULES: dict = {
    "always": lambda _r, _c: True,
    "no_clusters": lambda r, _c: len(r.clusters) == 0,
    "no_bridges": lambda _r, _c: True,  # simplified — always fires when called
    "booster_fired": lambda r, _c: len(r.fires) > 0,
}


# ---------------------------------------------------------------------------
# TPA-001: ends_when="always", 1 rep completed, min=1 → advances
# ---------------------------------------------------------------------------

class TestTPA001:
    def test_always_advances_when_min_met(self):
        arc = _arc(_phase(id="a", repetitions=Range(1, 1), ends_when="always"))
        sig = _signature(arc=arc)
        p = _progress(sig)
        # Simulate 1 rep completed
        p.current_phase_repetitions = 1

        result = try_advance_phase(p, _FakeStepResult(), _TEST_RULES)

        assert result is True
        assert p.current_phase_index == 1


# ---------------------------------------------------------------------------
# TPA-002: repetitions=Range(2,3), 1 rep, ends_when="always" → no advance
# ---------------------------------------------------------------------------

class TestTPA002:
    def test_always_does_not_advance_below_min(self):
        arc = _arc(_phase(id="a", repetitions=Range(2, 3), ends_when="always"))
        sig = _signature(arc=arc)
        p = _progress(sig)
        p.current_phase_repetitions = 1

        result = try_advance_phase(p, _FakeStepResult(), _TEST_RULES)

        assert result is False
        assert p.current_phase_index == 0


# ---------------------------------------------------------------------------
# TPA-003: repetitions=Range(1,3), 3 reps → advances via max-rep rule
# ---------------------------------------------------------------------------

class TestTPA003:
    def test_max_reps_advances_unconditionally(self):
        # Use a predicate that never fires to prove max-rep overrides
        rules = {**_TEST_RULES, "never": lambda _r, _c: False}
        arc = _arc(_phase(id="a", repetitions=Range(1, 3), ends_when="never"))
        sig = _signature(arc=arc)
        p = _progress(sig)
        p.current_phase_repetitions = 3

        result = try_advance_phase(p, _FakeStepResult(), rules)

        assert result is True
        assert p.current_phase_index == 1


# ---------------------------------------------------------------------------
# TPA-004: ends_when="no_clusters", 0 clusters, min met → advances
# ---------------------------------------------------------------------------

class TestTPA004:
    def test_no_clusters_predicate_fires(self):
        arc = _arc(_phase(id="a", repetitions=Range(1, 3), ends_when="no_clusters"))
        sig = _signature(arc=arc)
        p = _progress(sig)
        p.current_phase_repetitions = 1

        step = _FakeStepResult(clusters=())  # 0 clusters
        result = try_advance_phase(p, step, _TEST_RULES)

        assert result is True
        assert p.current_phase_index == 1


# ---------------------------------------------------------------------------
# TPA-005: ends_when="no_clusters", 2 clusters → does not advance
# ---------------------------------------------------------------------------

class TestTPA005:
    def test_no_clusters_predicate_does_not_fire(self):
        arc = _arc(_phase(id="a", repetitions=Range(1, 3), ends_when="no_clusters"))
        sig = _signature(arc=arc)
        p = _progress(sig)
        p.current_phase_repetitions = 1

        step = _FakeStepResult(clusters=("c1", "c2"))  # 2 clusters
        result = try_advance_phase(p, step, _TEST_RULES)

        assert result is False
        assert p.current_phase_index == 0


# ---------------------------------------------------------------------------
# TPA-006: ends_when="no_bridges", context=None → does not advance
# ---------------------------------------------------------------------------

class TestTPA006:
    def test_context_dependent_rule_skips_without_context(self):
        arc = _arc(_phase(id="a", repetitions=Range(1, 3), ends_when="no_bridges"))
        sig = _signature(arc=arc)
        p = _progress(sig)
        p.current_phase_repetitions = 1

        # No context passed — context-dependent rule should skip
        result = try_advance_phase(p, _FakeStepResult(), _TEST_RULES, context=None)

        assert result is False
        assert p.current_phase_index == 0


# ---------------------------------------------------------------------------
# TPA-007: No narrative arc on signature → returns False
# ---------------------------------------------------------------------------

class TestTPA007:
    def test_no_arc_returns_false(self):
        sig = _signature(arc=None)
        p = _progress(sig)

        result = try_advance_phase(p, _FakeStepResult(), _TEST_RULES)

        assert result is False
        assert p.current_phase_index == 0


# ---------------------------------------------------------------------------
# TPA-008: current_phase_index past end of phases → returns False
# ---------------------------------------------------------------------------

class TestTPA008:
    def test_past_end_returns_false(self):
        arc = _arc(_phase(id="a", repetitions=Range(1, 1), ends_when="always"))
        sig = _signature(arc=arc)
        p = _progress(sig)
        # Manually set past end
        p.current_phase_index = 5

        result = try_advance_phase(p, _FakeStepResult(), _TEST_RULES)

        assert result is False
        assert p.current_phase_index == 5
