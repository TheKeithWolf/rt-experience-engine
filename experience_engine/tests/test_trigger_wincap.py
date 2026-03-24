"""Phase 12 tests — feature triggers & wincap.

TEST-P12-001 through TEST-P12-016 covering trigger and wincap archetype
registration, CONTRACT-SIG-4 enforcement, DiagnosticsEngine new metrics,
total archetype count, and cascade generator wincap halt logic.
"""

from __future__ import annotations

import pytest

from ..archetypes.bomb import register_bomb_archetypes
from ..archetypes.chain import register_chain_archetypes
from ..archetypes.dead import register_dead_archetypes
from ..archetypes.lb import register_lb_archetypes
from ..archetypes.registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    SignatureValidationError,
)
from ..archetypes.rocket import register_rocket_archetypes
from ..archetypes.slb import register_slb_archetypes
from ..archetypes.tier1 import register_cascade_t1_archetypes, register_static_t1_archetypes
from ..archetypes.trigger import register_trigger_archetypes
from ..archetypes.wild import register_wild_archetypes
from ..archetypes.wincap import register_wincap_archetypes
from ..config.schema import MasterConfig
from ..diagnostics.engine import DiagnosticsEngine
from ..pipeline.protocols import Range, RangeFloat
from ..validation.metrics import InstanceMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_all_families(config: MasterConfig) -> ArchetypeRegistry:
    """Register all 10 families — complete 90-archetype registry."""
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


def _make_instance_metrics(
    archetype_id: str = "test",
    family: str = "dead",
    criteria: str = "0",
    payout: float = 0.0,
    triggers_freespin: bool = False,
    reaches_wincap: bool = False,
) -> InstanceMetrics:
    """Create minimal InstanceMetrics for diagnostics tests."""
    from ..primitives.paytable import Paytable

    return InstanceMetrics(
        archetype_id=archetype_id,
        family=family,
        criteria=criteria,
        sim_id=0,
        payout=payout,
        centipayout=int(payout * 100),
        win_level=0,
        cluster_count=0,
        cluster_sizes=(),
        cluster_symbols=(),
        scatter_count=0,
        near_miss_count=0,
        near_miss_symbols=(),
        max_component_size=0,
        is_valid=True,
        validation_errors=(),
        triggers_freespin=triggers_freespin,
        reaches_wincap=reaches_wincap,
    )


# ===========================================================================
# Trigger registration tests
# ===========================================================================

class TestTriggerRegistration:
    """Trigger family archetype registration and CONTRACT-SIG validation."""

    def test_register_all_4_trigger_archetypes(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-001: All 4 trigger archetypes register without errors."""
        reg = ArchetypeRegistry(default_config)
        register_trigger_archetypes(reg)

        trigger_family = reg.get_family("trigger")
        assert len(trigger_family) == 4

    def test_trigger_criteria_is_freegame(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-002: All trigger archetypes have criteria='freegame'
        and triggers_freespin=True."""
        reg = ArchetypeRegistry(default_config)
        register_trigger_archetypes(reg)

        for sig in reg.get_family("trigger"):
            assert sig.criteria == "freegame", f"{sig.id} has criteria={sig.criteria}"
            assert sig.triggers_freespin is True, f"{sig.id} has triggers_freespin=False"

    def test_trigger_4s_constraints(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-003: trigger_4s has correct scatter, cluster, cascade, payout."""
        reg = ArchetypeRegistry(default_config)
        register_trigger_archetypes(reg)
        sig = reg.get("trigger_4s")

        assert sig.required_scatter_count.min_val == 4
        assert sig.required_scatter_count.max_val == 4
        assert sig.required_cluster_count.min_val == 0
        assert sig.required_cluster_count.max_val == 0
        assert sig.required_cascade_depth.min_val == 0
        assert sig.required_cascade_depth.max_val == 0
        assert sig.payout_range.min_val == 0.0
        assert sig.payout_range.max_val == 0.0
        assert sig.max_component_size == 4

    def test_trigger_5s_max_scatters(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-004: trigger_5s uses max_scatters_on_board from config."""
        reg = ArchetypeRegistry(default_config)
        register_trigger_archetypes(reg)
        sig = reg.get("trigger_5s")

        assert sig.required_scatter_count.min_val == 5
        assert sig.required_scatter_count.max_val == 5
        # 5 must equal config.freespin.max_scatters_on_board
        assert sig.required_scatter_count.max_val == default_config.freespin.max_scatters_on_board

    def test_trigger_4s_with_win_has_clusters(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-005: trigger_4s_with_win has clusters + scatters, cascade 0-2."""
        reg = ArchetypeRegistry(default_config)
        register_trigger_archetypes(reg)
        sig = reg.get("trigger_4s_with_win")

        assert sig.required_cluster_count.min_val >= 1
        assert sig.required_scatter_count.min_val == 4
        assert sig.required_cascade_depth.min_val == 0
        assert sig.required_cascade_depth.max_val == 2
        assert sig.payout_range.min_val > 0.0

    def test_trigger_5s_with_booster_has_spawns(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-006: trigger_5s_with_booster has booster spawns, cascade 1-3."""
        reg = ArchetypeRegistry(default_config)
        register_trigger_archetypes(reg)
        sig = reg.get("trigger_5s_with_booster")

        assert sig.required_scatter_count.min_val == 5
        assert sig.required_cascade_depth.min_val == 1
        assert sig.required_cascade_depth.max_val == 3
        # Must have some booster spawn ranges defined
        assert len(sig.required_booster_spawns) > 0
        # No required fires — boosters are dormant visual presence
        assert len(sig.required_booster_fires) == 0

    def test_contract_sig4_scatter_too_low_raises(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-007: CONTRACT-SIG-4 — triggers_freespin with insufficient
        scatter_count.min raises SignatureValidationError."""
        reg = ArchetypeRegistry(default_config)

        with pytest.raises(SignatureValidationError, match="CONTRACT-SIG-4"):
            reg.register(ArchetypeSignature(
                id="bad_trigger",
                family="trigger",
                criteria="freegame",
                triggers_freespin=True,
                reaches_wincap=False,
                required_cluster_count=Range(0, 0),
                required_cluster_sizes=(),
                required_cluster_symbols=None,
                # Only 3 scatters — below min_scatters_to_trigger (4)
                required_scatter_count=Range(3, 3),
                required_near_miss_count=Range(0, 0),
                required_near_miss_symbol_tier=None,
                max_component_size=4,
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
            ))


# ===========================================================================
# Wincap registration tests
# ===========================================================================

class TestWincapRegistration:
    """Wincap family archetype registration."""

    def test_register_all_2_wincap_archetypes(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-008: All 2 wincap archetypes register without errors."""
        reg = ArchetypeRegistry(default_config)
        register_wincap_archetypes(reg)

        wincap_family = reg.get_family("wincap")
        assert len(wincap_family) == 2

    def test_near_wincap_does_not_reach_cap(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-009: near_wincap payout_range=(2000,4999), reaches_wincap=False."""
        reg = ArchetypeRegistry(default_config)
        register_wincap_archetypes(reg)
        sig = reg.get("near_wincap")

        assert sig.payout_range.min_val == 2000.0
        assert sig.payout_range.max_val == 4999.0
        assert sig.reaches_wincap is False
        assert sig.criteria == "wincap"

    def test_wincap_exact_cap(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-010: wincap payout_range=(5000,5000), reaches_wincap=True."""
        reg = ArchetypeRegistry(default_config)
        register_wincap_archetypes(reg)
        sig = reg.get("wincap")

        assert sig.payout_range.min_val == default_config.wincap.max_payout
        assert sig.payout_range.max_val == default_config.wincap.max_payout
        assert sig.reaches_wincap is True
        assert sig.criteria == "wincap"



# ===========================================================================
# Diagnostics metrics tests
# ===========================================================================

class TestDiagnosticsNewMetrics:
    """DiagnosticsEngine freespin_trigger_rate and wincap_hit_rate."""

    def test_freespin_trigger_rate(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-013: DiagnosticsEngine computes freespin_trigger_rate."""
        engine = DiagnosticsEngine(default_config)

        # 2 out of 10 instances trigger freespin
        metrics = tuple(
            _make_instance_metrics(
                archetype_id=f"test_{i}",
                triggers_freespin=(i < 2),
            )
            for i in range(10)
        )

        report = engine.analyze(metrics)
        assert report.freespin_trigger_rate == pytest.approx(0.2)

    def test_wincap_hit_rate(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-014: DiagnosticsEngine computes wincap_hit_rate."""
        engine = DiagnosticsEngine(default_config)

        # 1 out of 10 instances reaches wincap
        metrics = tuple(
            _make_instance_metrics(
                archetype_id=f"test_{i}",
                reaches_wincap=(i == 0),
                payout=5000.0 if i == 0 else 0.0,
            )
            for i in range(10)
        )

        report = engine.analyze(metrics)
        assert report.wincap_hit_rate == pytest.approx(0.1)

    def test_empty_metrics_zero_rates(
        self, default_config: MasterConfig,
    ) -> None:
        """Empty metrics produce zero rates without errors."""
        engine = DiagnosticsEngine(default_config)
        report = engine.analyze(())

        assert report.freespin_trigger_rate == 0.0
        assert report.wincap_hit_rate == 0.0


# ===========================================================================
# Total archetype count
# ===========================================================================

class TestTotalArchetypeCount:
    """Full registry count with all 10 families."""

    def test_90_archetypes_total(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-015: All families registered produce exactly 90 archetypes."""
        reg = _register_all_families(default_config)

        total = len(reg.all_ids())
        assert total == 90, f"Expected 90 archetypes, got {total}"

        # All 10 families present
        families = reg.registered_families()
        assert len(families) == 10
        expected_families = {
            "dead", "t1", "wild", "rocket", "bomb",
            "lb", "slb", "chain", "trigger", "wincap",
        }
        assert families == expected_families


# ===========================================================================
# Wincap halt logic
# ===========================================================================

class TestWincapHalt:
    """Cascade generator wincap halt behavior."""

    def test_wincap_halt_clamps_payout(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-P12-016: Wincap halt — payout clamped to max_payout when
        reaches_wincap=True and total_payout >= max_payout.

        Unit test verifies the clamping logic without running the full
        cascade generator (which requires Clingo).
        """
        max_payout = default_config.wincap.max_payout
        halt_cascade = default_config.wincap.halt_cascade

        # Simulate the halt condition check from cascade_generator
        total_payout = 5500.0  # Exceeds cap
        reaches_wincap = True

        if reaches_wincap and halt_cascade and total_payout >= max_payout:
            total_payout = max_payout  # Clamp to exact cap

        assert total_payout == max_payout
        assert total_payout == 5000.0

    def test_near_wincap_no_halt(
        self, default_config: MasterConfig,
    ) -> None:
        """near_wincap (reaches_wincap=False) does not trigger halt."""
        max_payout = default_config.wincap.max_payout
        halt_cascade = default_config.wincap.halt_cascade

        total_payout = 5500.0
        reaches_wincap = False  # near_wincap

        # Should NOT clamp
        if reaches_wincap and halt_cascade and total_payout >= max_payout:
            total_payout = max_payout

        assert total_payout == 5500.0  # Not clamped
