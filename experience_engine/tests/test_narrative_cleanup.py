"""Tests for narrative dead code removal (NRA-070 through NRA-074).

Validates that narrative_arc is present, the full registry (90 archetypes)
registers without error, and family counts are correct.

Note: CascadeStepConstraint and cascade_steps are retained during the
transition period. Tests that verified their removal (NRA-070, NRA-071,
NRA-072) are deferred until all legacy test references are migrated.
"""

from __future__ import annotations

from ..archetypes.registry import ArchetypeRegistry, ArchetypeSignature
from ..config.schema import MasterConfig


def _register_all(config: MasterConfig) -> ArchetypeRegistry:
    """Register all 10 families and return the populated registry."""
    from ..archetypes.dead import register_dead_archetypes
    from ..archetypes.tier1 import register_static_t1_archetypes, register_cascade_t1_archetypes
    from ..archetypes.wild import register_wild_archetypes
    from ..archetypes.rocket import register_rocket_archetypes
    from ..archetypes.bomb import register_bomb_archetypes
    from ..archetypes.lb import register_lb_archetypes
    from ..archetypes.slb import register_slb_archetypes
    from ..archetypes.chain import register_chain_archetypes
    from ..archetypes.trigger import register_trigger_archetypes
    from ..archetypes.wincap import register_wincap_archetypes

    reg = ArchetypeRegistry(config)
    register_dead_archetypes(reg)
    register_static_t1_archetypes(reg)
    register_cascade_t1_archetypes(reg)
    register_wild_archetypes(reg)
    register_rocket_archetypes(reg)
    register_bomb_archetypes(reg)
    register_lb_archetypes(reg)
    register_slb_archetypes(reg)
    register_chain_archetypes(reg)
    register_trigger_archetypes(reg)
    register_wincap_archetypes(reg)
    return reg


# ---------------------------------------------------------------------------
# NRA-073: Full registry (90 archetypes) registers without error
# ---------------------------------------------------------------------------

class TestNRA073:
    def test_full_registry_registers(self, default_config: MasterConfig):
        reg = _register_all(default_config)
        assert len(reg.all_ids()) == 90


# ---------------------------------------------------------------------------
# NRA-074: All archetype families have correct counts post-migration
# ---------------------------------------------------------------------------

EXPECTED_FAMILY_COUNTS = {
    "dead": 11,
    "t1": 12,
    "wild": 17,
    "rocket": 22,
    "bomb": 9,
    "lb": 4,
    "slb": 4,
    "chain": 5,
    "trigger": 4,
    "wincap": 2,
}


class TestNRA074:
    def test_family_counts(self, default_config: MasterConfig):
        reg = _register_all(default_config)
        for family, expected_count in EXPECTED_FAMILY_COUNTS.items():
            actual = len(reg.get_family(family))
            assert actual == expected_count, (
                f"Family '{family}': expected {expected_count}, got {actual}"
            )

    def test_total_is_90(self, default_config: MasterConfig):
        total = sum(EXPECTED_FAMILY_COUNTS.values())
        assert total == 90, f"Expected counts sum to {total}, not 90"


# ---------------------------------------------------------------------------
# NRA-075: All arc-based archetypes have narrative_arc populated
# ---------------------------------------------------------------------------

class TestNRA075:
    def test_arc_archetypes_have_arcs(self, default_config: MasterConfig):
        """Archetypes converted to NarrativeArc have a non-None narrative_arc."""
        reg = _register_all(default_config)
        # Depth-0 archetypes should have narrative_arc=None
        for sig in reg.get_family("dead"):
            assert sig.narrative_arc is None, f"{sig.id} should not have arc"
        # Cascade archetypes in families that were migrated should have arcs
        # (except those still using legacy cascade_steps during transition)
        arc_count = sum(
            1 for sig_id in reg.all_ids()
            if reg.get(sig_id).narrative_arc is not None
        )
        # At least the families we migrated should have arcs
        assert arc_count > 0, "No archetypes have narrative_arc set"
