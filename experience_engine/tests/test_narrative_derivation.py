"""Tests for narrative constraint derivation (NRA-010 through NRA-019).

Validates that derive_constraints() correctly computes cascade_depth,
booster_spawns, booster_fires, cluster_count, and cluster_sizes from
NarrativeArc phase definitions.
"""

from __future__ import annotations

from ..narrative.arc import NarrativeArc, NarrativePhase
from ..narrative.derivation import DerivedConstraints, derive_constraints
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.symbols import SymbolTier


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
        id=id,
        intent="test",
        repetitions=repetitions,
        cluster_count=cluster_count,
        cluster_sizes=cluster_sizes,
        cluster_symbol_tier=cluster_symbol_tier,
        spawns=spawns,
        arms=arms,
        fires=fires,
        wild_behavior=wild_behavior,
        ends_when=ends_when,
    )


def _terminal_phase() -> NarrativePhase:
    """Terminal phase — zero clusters, excluded from depth count."""
    return _phase(id="terminal", cluster_count=Range(0, 0), cluster_sizes=())


def _arc(*phases: NarrativePhase, **overrides) -> NarrativeArc:
    defaults = dict(
        phases=phases,
        payout=RangeFloat(0.0, 10.0),
        wild_count_on_terminal=Range(0, 0),
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
    )
    defaults.update(overrides)
    return NarrativeArc(**defaults)


# ---------------------------------------------------------------------------
# NRA-010: Single phase → cascade_depth matches repetitions
# ---------------------------------------------------------------------------

class TestNRA010:
    def test_single_phase_depth(self):
        arc = _arc(_phase(repetitions=Range(1, 1)))
        d = derive_constraints(arc)
        assert d.cascade_depth == Range(1, 1)


# ---------------------------------------------------------------------------
# NRA-011: Three phases → depth = sum of repetition mins and maxes
# ---------------------------------------------------------------------------

class TestNRA011:
    def test_three_phases_depth(self):
        arc = _arc(
            _phase(id="a", repetitions=Range(1, 2)),
            _phase(id="b", repetitions=Range(2, 3)),
            _phase(id="c", repetitions=Range(1, 1)),
        )
        d = derive_constraints(arc)
        # 1+2+1=4 min, 2+3+1=6 max
        assert d.cascade_depth == Range(4, 6)


# ---------------------------------------------------------------------------
# NRA-012: Terminal phase excluded from depth count
# ---------------------------------------------------------------------------

class TestNRA012:
    def test_terminal_excluded(self):
        arc = _arc(
            _phase(id="cascade", repetitions=Range(2, 3)),
            _terminal_phase(),
        )
        d = derive_constraints(arc)
        # Only the cascade phase counts — terminal has cluster_count=Range(0,0)
        assert d.cascade_depth == Range(2, 3)


# ---------------------------------------------------------------------------
# NRA-013: Phase spawning "W" with repetitions → booster_spawns
# ---------------------------------------------------------------------------

class TestNRA013:
    def test_single_booster_spawn(self):
        arc = _arc(_phase(spawns=("W",), repetitions=Range(2, 4)))
        d = derive_constraints(arc)
        assert d.booster_spawns == {"W": Range(2, 4)}


# ---------------------------------------------------------------------------
# NRA-014: Two phases spawning "R" → aggregated
# ---------------------------------------------------------------------------

class TestNRA014:
    def test_aggregated_spawns(self):
        arc = _arc(
            _phase(id="a", spawns=("R",), repetitions=Range(1, 2)),
            _phase(id="b", spawns=("R",), repetitions=Range(1, 3)),
        )
        d = derive_constraints(arc)
        assert d.booster_spawns == {"R": Range(2, 5)}


# ---------------------------------------------------------------------------
# NRA-015: Phase with no spawns contributes zero
# ---------------------------------------------------------------------------

class TestNRA015:
    def test_no_spawns_phase(self):
        arc = _arc(
            _phase(id="a", spawns=("R",), repetitions=Range(1, 1)),
            _phase(id="b", spawns=None),
        )
        d = derive_constraints(arc)
        assert d.booster_spawns == {"R": Range(1, 1)}

    def test_fires_aggregation(self):
        arc = _arc(
            _phase(id="a", fires=("B",), repetitions=Range(1, 2)),
            _phase(id="b", fires=None),
        )
        d = derive_constraints(arc)
        assert d.booster_fires == {"B": Range(1, 2)}


# ---------------------------------------------------------------------------
# NRA-016: cluster_sizes deduplicates identical Range tuples
# ---------------------------------------------------------------------------

class TestNRA016:
    def test_deduplication(self):
        arc = _arc(
            _phase(id="a", cluster_sizes=(Range(5, 6),)),
            _phase(id="b", cluster_sizes=(Range(5, 6),)),
            _phase(id="c", cluster_sizes=(Range(9, 14),)),
        )
        d = derive_constraints(arc)
        assert d.cluster_sizes == (Range(5, 6), Range(9, 14))


# ---------------------------------------------------------------------------
# NRA-017: Derived constraints for wild_enable_rocket match original values
# ---------------------------------------------------------------------------

class TestNRA017:
    def test_wild_enable_rocket_equivalent(self):
        """Simulates the wild_enable_rocket archetype migration.

        Original: 3 cascade_steps with wild bridging and rocket spawning.
        Expected derived: depth Range(3,3), booster_spawns={"R": Range(1,1)}.
        """
        arc = _arc(
            _phase(id="initial", repetitions=Range(1, 1),
                   cluster_count=Range(1, 2), cluster_sizes=(Range(5, 6),)),
            _phase(id="bridge", repetitions=Range(1, 1),
                   cluster_count=Range(1, 2), cluster_sizes=(Range(5, 8),),
                   wild_behavior="bridge", spawns=("R",)),
            _phase(id="fire", repetitions=Range(1, 1),
                   cluster_count=Range(1, 2), cluster_sizes=(Range(5, 6),),
                   fires=("R",)),
        )
        d = derive_constraints(arc)
        assert d.cascade_depth == Range(3, 3)
        assert d.booster_spawns == {"R": Range(1, 1)}
        assert d.booster_fires == {"R": Range(1, 1)}


# ---------------------------------------------------------------------------
# NRA-018: Derived constraints for t1_cascade_1 match original values
# ---------------------------------------------------------------------------

class TestNRA018:
    def test_t1_cascade_1_equivalent(self):
        """Simple cascade: 1 step, cluster Range(1,2), sizes Range(5,6)."""
        arc = _arc(
            _phase(repetitions=Range(1, 1),
                   cluster_count=Range(1, 2), cluster_sizes=(Range(5, 6),)),
        )
        d = derive_constraints(arc)
        assert d.cascade_depth == Range(1, 1)
        assert d.booster_spawns == {}
        assert d.booster_fires == {}
        assert d.cluster_sizes == (Range(5, 6),)


# ---------------------------------------------------------------------------
# NRA-019: Derived constraints for rocket_chain_bomb match original values
# ---------------------------------------------------------------------------

class TestNRA019:
    def test_rocket_chain_bomb_equivalent(self):
        """Rocket spawns, arms, then fires triggering bomb chain.

        Original: depth Range(3,4), spawns R+B, fires R+B.
        """
        arc = _arc(
            _phase(id="growth", repetitions=Range(1, 2),
                   cluster_count=Range(1, 2), cluster_sizes=(Range(9, 14),),
                   spawns=("R",)),
            _phase(id="arm_fire", repetitions=Range(1, 1),
                   cluster_count=Range(1, 2), cluster_sizes=(Range(5, 8),),
                   spawns=("B",), fires=("R",)),
            _phase(id="chain_fire", repetitions=Range(1, 1),
                   cluster_count=Range(1, 2), cluster_sizes=(Range(5, 6),),
                   fires=("B",)),
        )
        d = derive_constraints(arc)
        assert d.cascade_depth == Range(3, 4)
        assert d.booster_spawns == {"R": Range(1, 2), "B": Range(1, 1)}
        assert d.booster_fires == {"R": Range(1, 1), "B": Range(1, 1)}
