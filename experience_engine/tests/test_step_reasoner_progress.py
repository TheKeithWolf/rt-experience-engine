"""Tests for ProgressTracker.peek_phase_after_current.

Distinguishes the "post-current phase" query from peek_next_phase — which
returns the current phase while repetitions remain. Planning strategies
need the post-current semantics because the step they are building has
not yet been counted toward the current phase's repetitions.
"""

from __future__ import annotations

from ..archetypes.registry import ArchetypeSignature
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier
from ..step_reasoner.progress import ProgressTracker


def _phase(id: str, arms: tuple[str, ...] | None = None) -> NarrativePhase:
    return NarrativePhase(
        id=id, intent="test", repetitions=Range(1, 1),
        cluster_count=Range(1, 2), cluster_sizes=(Range(5, 6),),
        cluster_symbol_tier=None,
        spawns=None, arms=arms, fires=None,
        wild_behavior=None, ends_when="always",
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


def _signature(arc: NarrativeArc | None) -> ArchetypeSignature:
    return ArchetypeSignature(
        id="sig", family="t1", criteria="basegame",
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


def _progress(sig: ArchetypeSignature) -> ProgressTracker:
    return ProgressTracker(signature=sig, centipayout_multiplier=100)


def test_peek_phase_after_current_returns_next_phase():
    """At idx=0, reps=0 — peek_next_phase returns the current phase (reps
    remain), but peek_phase_after_current skips ahead to Phase 1. This is
    the semantic split that makes arming-cluster gates reachable at Step 0."""
    spawn = _phase("spawn", arms=None)
    fire = _phase("fire", arms=("R",))
    sig = _signature(_arc(spawn, fire))
    progress = _progress(sig)

    assert progress.peek_next_phase() is spawn
    assert progress.peek_phase_after_current() is fire


def test_peek_phase_after_current_returns_none_at_last_phase():
    """At the final phase there is no successor — must return None so
    callers can fall through to legacy cascade_steps logic."""
    only = _phase("only")
    sig = _signature(_arc(only))
    progress = _progress(sig)

    assert progress.peek_phase_after_current() is None


def test_peek_phase_after_current_returns_none_without_arc():
    """Non-arc signatures (legacy cascade_steps) have narrative_arc=None
    and must return None so the caller falls through."""
    sig = _signature(arc=None)
    progress = _progress(sig)

    assert progress.peek_phase_after_current() is None
