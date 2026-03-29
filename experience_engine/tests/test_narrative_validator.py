"""Tests for narrative arc validator (NRA-020 through NRA-038).

Validates greedy linear phase matching, transition rules, skippable phases,
and global constraint checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ..config.schema import MasterConfig
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..narrative.arc_validator import NarrativeArcValidator
from ..narrative.derivation import derive_constraints
from ..narrative.transitions import ALLOWED_TRANSITION_KEYS, build_transition_rules
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Position
from ..primitives.symbols import Symbol, SymbolTier
from ..spatial_solver.data_types import ClusterAssignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase(
    id: str = "p",
    repetitions: Range = Range(1, 1),
    cluster_count: Range = Range(1, 2),
    cluster_sizes: tuple[Range, ...] = (Range(5, 6),),
    cluster_symbol_tier: SymbolTier | None = None,
    spawns: tuple[str, ...] | None = None,
    arms: tuple[str, ...] | None = None,
    fires: tuple[str, ...] | None = None,
    wild_behavior: str | None = None,
    ends_when: str = "always",
) -> NarrativePhase:
    return NarrativePhase(
        id=id, intent="test", repetitions=repetitions,
        cluster_count=cluster_count, cluster_sizes=cluster_sizes,
        cluster_symbol_tier=cluster_symbol_tier,
        spawns=spawns, arms=arms, fires=fires,
        wild_behavior=wild_behavior, ends_when=ends_when,
    )


def _arc(*phases: NarrativePhase, **overrides) -> NarrativeArc:
    defaults = dict(
        phases=phases,
        payout=RangeFloat(0.0, 100.0),
        wild_count_on_terminal=Range(0, 10),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    defaults.update(overrides)
    return NarrativeArc(**defaults)


def _cluster(symbol: Symbol = Symbol.L1, size: int = 5) -> ClusterAssignment:
    """Minimal cluster assignment for testing."""
    positions = frozenset(Position(0, i) for i in range(size))
    return ClusterAssignment(
        symbol=symbol,
        positions=positions,
        size=size,
        wild_positions=frozenset(),
    )


@dataclass(frozen=True)
class _FakeFireRecord:
    """Minimal booster fire record for testing."""
    booster_type: str
    position_reel: int = 0
    position_row: int = 0
    orientation: str | None = None
    affected_count: int = 7
    chain_target_count: int = 0
    target_symbols: tuple[str, ...] = ()
    affected_positions_list: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _FakeStepRecord:
    """Minimal CascadeStepRecord stand-in for validator tests."""
    step_index: int = 0
    clusters: tuple[ClusterAssignment, ...] = ()
    step_payout: float = 0.0
    booster_spawn_types: tuple[str, ...] = ()
    booster_arm_types: tuple[str, ...] = ()
    booster_fire_records: tuple[_FakeFireRecord, ...] = ()
    # Fields the validator doesn't inspect but CascadeStepRecord has
    board_before: object = None
    board_after: object = None
    grid_multipliers_snapshot: tuple[int, ...] = ()
    booster_spawn_positions: tuple[tuple[str, int, int], ...] = ()
    gravity_record: object = None
    booster_gravity_record: object = None


def _step(
    step_index: int = 0,
    clusters: tuple[ClusterAssignment, ...] = (),
    step_payout: float = 1.0,
    spawn_types: tuple[str, ...] = (),
    arm_types: tuple[str, ...] = (),
    fires: tuple[_FakeFireRecord, ...] = (),
) -> _FakeStepRecord:
    return _FakeStepRecord(
        step_index=step_index,
        clusters=clusters,
        step_payout=step_payout,
        booster_spawn_types=spawn_types,
        booster_arm_types=arm_types,
        booster_fire_records=fires,
    )


@pytest.fixture
def validator(default_config: MasterConfig) -> NarrativeArcValidator:
    rules = build_transition_rules(default_config.board, default_config.symbols)
    return NarrativeArcValidator(rules, default_config.symbols)


# ---------------------------------------------------------------------------
# NRA-020: 3-step trajectory matching 3-phase arc passes
# ---------------------------------------------------------------------------

class TestNRA020:
    def test_exact_match(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="a"), _phase(id="b"), _phase(id="c"),
            payout=RangeFloat(0.0, 10.0),
        )
        steps = tuple(
            _step(step_index=i, clusters=(_cluster(),), step_payout=1.0)
            for i in range(3)
        )
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert errors == []


# ---------------------------------------------------------------------------
# NRA-021: Trajectory shorter than min total repetitions fails
# ---------------------------------------------------------------------------

class TestNRA021:
    def test_too_short(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="a"), _phase(id="b"), _phase(id="c"),
        )
        # Only 2 steps for 3 required phases
        steps = tuple(
            _step(step_index=i, clusters=(_cluster(),), step_payout=1.0)
            for i in range(2)
        )
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert len(errors) > 0
        assert any("never reached" in e for e in errors)


# ---------------------------------------------------------------------------
# NRA-022: Trajectory longer than max total repetitions fails
# ---------------------------------------------------------------------------

class TestNRA022:
    def test_too_long(self, validator: NarrativeArcValidator):
        arc = _arc(_phase(id="only", repetitions=Range(1, 1)))
        # 3 steps for a single-phase arc with max_val=1
        steps = tuple(
            _step(step_index=i, clusters=(_cluster(),), step_payout=1.0)
            for i in range(3)
        )
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert len(errors) > 0
        assert any("no phase left" in e for e in errors)


# ---------------------------------------------------------------------------
# NRA-023: Step with wrong cluster count fails phase match
# ---------------------------------------------------------------------------

class TestNRA023:
    def test_wrong_cluster_count(self, validator: NarrativeArcValidator):
        arc = _arc(_phase(id="p", cluster_count=Range(2, 3)))
        # Only 1 cluster but phase requires 2-3
        steps = (_step(clusters=(_cluster(),), step_payout=1.0),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# NRA-024: Step with cluster size outside all phase ranges fails
# ---------------------------------------------------------------------------

class TestNRA024:
    def test_wrong_cluster_size(self, validator: NarrativeArcValidator):
        arc = _arc(_phase(id="p", cluster_sizes=(Range(9, 14),)))
        # Cluster size 5 doesn't fit Range(9,14)
        steps = (_step(clusters=(_cluster(size=5),), step_payout=1.0),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# NRA-025: Repeating phase consumes 3 consecutive matching steps
# ---------------------------------------------------------------------------

class TestNRA025:
    def test_repeating_phase(self, validator: NarrativeArcValidator):
        arc = _arc(_phase(id="repeat", repetitions=Range(3, 3)),
                   payout=RangeFloat(0.0, 10.0))
        steps = tuple(
            _step(step_index=i, clusters=(_cluster(),), step_payout=1.0)
            for i in range(3)
        )
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert errors == []


# ---------------------------------------------------------------------------
# NRA-026: Phase with ends_when="no_clusters" advances on zero clusters
# ---------------------------------------------------------------------------

class TestNRA026:
    def test_no_clusters_transition(self, validator: NarrativeArcValidator):
        """Phase can repeat 1-3 times but ends when step has no clusters.

        cluster_count includes 0 so the dead step can match the phase —
        the transition predicate then fires to advance.
        """
        arc = _arc(
            _phase(id="cascade", repetitions=Range(1, 3),
                   cluster_count=Range(0, 2), ends_when="no_clusters"),
            payout=RangeFloat(0.0, 10.0),
        )
        # Step 0: has clusters (no_clusters predicate → False, phase continues)
        # Step 1: no clusters (no_clusters predicate → True, phase ends)
        steps = (
            _step(step_index=0, clusters=(_cluster(),), step_payout=1.0),
            _step(step_index=1, clusters=(), step_payout=0.0),
        )
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert errors == []


# ---------------------------------------------------------------------------
# NRA-027: Phase with ends_when="no_bridges" — tested with no board_contexts
# (defaults to True, so phase advances after min reps)
# ---------------------------------------------------------------------------

class TestNRA027:
    def test_no_bridges_without_context(self, validator: NarrativeArcValidator):
        """Without board_contexts, 'no_bridges' defaults to advancing."""
        arc = _arc(
            _phase(id="bridge", repetitions=Range(1, 3), ends_when="no_bridges"),
            payout=RangeFloat(0.0, 10.0),
        )
        steps = (
            _step(step_index=0, clusters=(_cluster(),), step_payout=1.0),
        )
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert errors == []


# ---------------------------------------------------------------------------
# NRA-028: Phase with ends_when="always" advances after exactly max_val
# ---------------------------------------------------------------------------

class TestNRA028:
    def test_always_at_max(self, validator: NarrativeArcValidator):
        arc = _arc(_phase(id="fixed", repetitions=Range(2, 2), ends_when="always"),
                   payout=RangeFloat(0.0, 10.0))
        steps = tuple(
            _step(step_index=i, clusters=(_cluster(),), step_payout=1.0)
            for i in range(2)
        )
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert errors == []


# ---------------------------------------------------------------------------
# NRA-029: Skippable phase (repetitions=Range(0,N)) can be bypassed
# ---------------------------------------------------------------------------

class TestNRA029:
    def test_skippable_phase(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="mandatory"),
            _phase(id="optional", repetitions=Range(0, 2)),
            payout=RangeFloat(0.0, 10.0),
        )
        # Only 1 step — mandatory is consumed, optional is skipped
        steps = (_step(clusters=(_cluster(),), step_payout=1.0),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert errors == []


# ---------------------------------------------------------------------------
# NRA-030: Global payout check rejects out-of-range total
# ---------------------------------------------------------------------------

class TestNRA030:
    def test_payout_out_of_range(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="p"),
            payout=RangeFloat(5.0, 10.0),
        )
        # Step payout 1.0 → total 1.0, below min 5.0
        steps = (_step(clusters=(_cluster(),), step_payout=1.0),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert any("payout" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# NRA-031: Global booster spawn count check rejects deficit
# ---------------------------------------------------------------------------

class TestNRA031:
    def test_spawn_deficit(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="p", spawns=("R",)),
            payout=RangeFloat(0.0, 100.0),
        )
        # Step has correct cluster but no R spawns
        steps = (_step(clusters=(_cluster(),), step_payout=1.0, spawn_types=()),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        # Phase match fails because spawns check requires "R" in step spawns
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# NRA-032: Terminal wild count check — placeholder (requires instance-level)
# ---------------------------------------------------------------------------

class TestNRA032:
    def test_terminal_wild_count(self, validator: NarrativeArcValidator):
        """Terminal wild count is an instance-level check — arc validator doesn't
        have wild count info from step records. This test verifies the validator
        doesn't crash on arcs with wild constraints."""
        arc = _arc(
            _phase(id="p"),
            wild_count_on_terminal=Range(1, 3),
            payout=RangeFloat(0.0, 100.0),
        )
        steps = (_step(clusters=(_cluster(),), step_payout=1.0),)
        derived = derive_constraints(arc)
        # Should not raise
        validator.validate(steps, arc, derived)


# ---------------------------------------------------------------------------
# NRA-033: Chain depth check — deferred to InstanceValidator
# ---------------------------------------------------------------------------

class TestNRA033:
    def test_chain_depth(self, validator: NarrativeArcValidator):
        """Chain depth validation is deferred to InstanceValidator — arc validator
        passes through arcs with chain constraints without error."""
        arc = _arc(
            _phase(id="p"),
            required_chain_depth=Range(1, 2),
            payout=RangeFloat(0.0, 100.0),
        )
        steps = (_step(clusters=(_cluster(),), step_payout=1.0),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        # No chain depth error from arc validator
        assert not any("chain" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# NRA-034: Rocket orientation check — deferred to InstanceValidator
# ---------------------------------------------------------------------------

class TestNRA034:
    def test_rocket_orientation(self, validator: NarrativeArcValidator):
        """Rocket orientation is an instance-level check."""
        arc = _arc(
            _phase(id="p"),
            rocket_orientation="H",
            payout=RangeFloat(0.0, 100.0),
        )
        steps = (_step(clusters=(_cluster(),), step_payout=1.0),)
        derived = derive_constraints(arc)
        # Should not raise
        validator.validate(steps, arc, derived)


# ---------------------------------------------------------------------------
# NRA-035: Phase with cluster_symbol_tier=LOW rejects HIGH cluster
# ---------------------------------------------------------------------------

class TestNRA035:
    def test_tier_mismatch(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="p", cluster_symbol_tier=SymbolTier.LOW),
            payout=RangeFloat(0.0, 100.0),
        )
        # H1 is a HIGH tier symbol
        steps = (_step(clusters=(_cluster(symbol=Symbol.H1),), step_payout=1.0),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# NRA-036: Phase spawn check — step missing required "R" fails
# ---------------------------------------------------------------------------

class TestNRA036:
    def test_missing_spawn(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="p", spawns=("R",)),
            payout=RangeFloat(0.0, 100.0),
        )
        steps = (_step(clusters=(_cluster(),), step_payout=1.0, spawn_types=()),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# NRA-037: Phase fire check — step with correct fire passes
# ---------------------------------------------------------------------------

class TestNRA037:
    def test_fire_passes(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="p", fires=("R",)),
            payout=RangeFloat(0.0, 100.0),
        )
        steps = (_step(
            clusters=(_cluster(),),
            step_payout=1.0,
            fires=(_FakeFireRecord(booster_type="R"),),
        ),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert errors == []


# ---------------------------------------------------------------------------
# NRA-038: All ends_when values used by registered arcs exist in TRANSITION_RULES
# ---------------------------------------------------------------------------

class TestNRA038:
    def test_all_transition_keys_valid(self):
        """Every key used by test arcs must exist in ALLOWED_TRANSITION_KEYS."""
        used_keys = {"always", "no_clusters", "no_bridges", "booster_fired"}
        assert used_keys.issubset(ALLOWED_TRANSITION_KEYS)


# ---------------------------------------------------------------------------
# NRA-039: Phase arm check — step with matching booster_arm_types passes
# ---------------------------------------------------------------------------

class TestNRA039:
    def test_arm_passes(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="p", arms=("R",)),
            payout=RangeFloat(0.0, 100.0),
        )
        steps = (_step(
            clusters=(_cluster(),),
            step_payout=1.0,
            arm_types=("R",),
        ),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert errors == []


# ---------------------------------------------------------------------------
# NRA-040: Phase arm check — step missing required arm type fails
# ---------------------------------------------------------------------------

class TestNRA040:
    def test_missing_arm_fails(self, validator: NarrativeArcValidator):
        arc = _arc(
            _phase(id="p", arms=("R",)),
            payout=RangeFloat(0.0, 100.0),
        )
        steps = (_step(clusters=(_cluster(),), step_payout=1.0, arm_types=()),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# NRA-041: Phase arm check uses booster_arm_types, not booster_spawn_types
# (regression — old code checked spawn_types as proxy for arms)
# ---------------------------------------------------------------------------

class TestNRA041:
    def test_arm_independent_of_spawn(self, validator: NarrativeArcValidator):
        """Phase requiring arm=("R",) passes with arm_types set, even without spawns."""
        arc = _arc(
            _phase(id="p", arms=("R",)),
            payout=RangeFloat(0.0, 100.0),
        )
        # arm_types populated but spawn_types empty — old code would fail here
        steps = (_step(
            clusters=(_cluster(),),
            step_payout=1.0,
            spawn_types=(),
            arm_types=("R",),
        ),)
        derived = derive_constraints(arc)
        errors = validator.validate(steps, arc, derived)
        assert errors == []
