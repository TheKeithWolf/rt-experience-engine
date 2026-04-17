"""Tests for FeasibilityEstimator — defect fixes and regression coverage.

Validates:
- Only the next phase is checked (Defect 1 fix: no cross-phase summing)
- Bridge phases use min_cluster_size floor (Defect 2 fix: no bridge inflation)
- Category C archetypes (passing today) remain feasible
- Category A/B archetypes (previously blocked) now report feasible
"""

from __future__ import annotations

import pytest

from ..archetypes.registry import ArchetypeSignature
from ..config.schema import BoardConfig, MasterConfig
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol, SymbolTier
from ..step_reasoner.context import BoardContext
from ..step_reasoner.progress import ProgressTracker
from ..step_reasoner.services.feasibility_estimator import FeasibilityEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase(**overrides) -> NarrativePhase:
    """Construct a NarrativePhase with sensible defaults."""
    defaults = dict(
        id="test_phase",
        intent="A test phase.",
        repetitions=Range(1, 1),
        cluster_count=Range(1, 1),
        cluster_sizes=(Range(5, 6),),
        cluster_symbol_tier=None,
        spawns=None,
        arms=None,
        fires=None,
        wild_behavior=None,
        ends_when="always",
    )
    defaults.update(overrides)
    return NarrativePhase(**defaults)


def _make_arc(phases: tuple[NarrativePhase, ...], **overrides) -> NarrativeArc:
    """Construct a NarrativeArc with sensible defaults."""
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


def _make_signature(**overrides) -> ArchetypeSignature:
    """Build a minimal ArchetypeSignature with sensible defaults."""
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
        payout_range=RangeFloat(0.5, 5.0),
        triggers_freespin=False,
        reaches_wincap=False,
    )
    defaults.update(overrides)
    return ArchetypeSignature(**defaults)


def _make_context(
    board: Board,
    config: MasterConfig,
    active_wilds: list[Position] | None = None,
) -> BoardContext:
    """Build a BoardContext from fixtures."""
    from ..primitives.grid_multipliers import GridMultiplierGrid

    return BoardContext.from_board(
        board=board,
        grid_multipliers=GridMultiplierGrid(config.grid_multiplier, config.board),
        dormant_boosters=[],
        active_wilds=active_wilds or [],
        board_config=config.board,
    )


def _make_estimator(config: MasterConfig) -> FeasibilityEstimator:
    return FeasibilityEstimator(config.board)


def _board_with_n_empties(
    n: int, config: MasterConfig,
) -> Board:
    """Create a board where exactly *n* cells are empty (None).

    Fills the board with Symbol.L1, then clears *n* cells
    starting from reel 0, row 0 onward.
    """
    board = Board(config.board.num_reels, config.board.num_rows)

    # Fill every cell
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            board.set(Position(reel, row), Symbol.L1)

    # Clear first n cells to make them empty
    cleared = 0
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            if cleared >= n:
                return board
            board.set(Position(reel, row), None)
            cleared += 1
    return board


# ---------------------------------------------------------------------------
# Unit Tests — FeasibilityEstimator
# ---------------------------------------------------------------------------

class TestFeasibilityEstimator:
    """Core unit tests for estimate() and _required_cells_for_phase()."""

    def test_no_arc_always_feasible(
        self, default_config: MasterConfig,
    ) -> None:
        """Signature without narrative_arc → True regardless of board."""
        sig = _make_signature(narrative_arc=None)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        board = _board_with_n_empties(0, default_config)
        context = _make_context(board, default_config)

        est = _make_estimator(default_config)
        assert est.estimate(progress, context) is True

    def test_no_remaining_phases_feasible(
        self, default_config: MasterConfig,
    ) -> None:
        """current_phase_index past end of phases → True."""
        phase = _make_phase()
        arc = _make_arc((phase,))
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        progress.current_phase_index = 1  # past the single phase

        board = _board_with_n_empties(0, default_config)
        context = _make_context(board, default_config)

        est = _make_estimator(default_config)
        assert est.estimate(progress, context) is True

    def test_fire_phase_needs_zero_cells(
        self, default_config: MasterConfig,
    ) -> None:
        """Phase with empty cluster_sizes (fire/terminal) → True on full board."""
        fire_phase = _make_phase(id="fire", cluster_sizes=(), cluster_count=Range(0, 0))
        arc = _make_arc((fire_phase,))
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)

        board = _board_with_n_empties(0, default_config)
        context = _make_context(board, default_config)

        est = _make_estimator(default_config)
        assert est.estimate(progress, context) is True

    def test_cluster_phase_full_size_required(
        self, default_config: MasterConfig,
    ) -> None:
        """Regular cluster phase: size 9 × count 1 needs 9 empties."""
        phase = _make_phase(cluster_sizes=(Range(9, 12),), cluster_count=Range(1, 1))
        arc = _make_arc((phase,))
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)

        est = _make_estimator(default_config)

        # 8 empties → infeasible
        board_8 = _board_with_n_empties(8, default_config)
        ctx_8 = _make_context(board_8, default_config)
        assert est.estimate(progress, ctx_8) is False

        # 10 empties → feasible
        board_10 = _board_with_n_empties(10, default_config)
        ctx_10 = _make_context(board_10, default_config)
        assert est.estimate(progress, ctx_10) is True

    def test_bridge_phase_uses_min_cluster_size_floor(
        self, default_config: MasterConfig,
    ) -> None:
        """Bridge phase with active wild uses min_cluster_size, not full cluster size."""
        bridge_phase = _make_phase(
            id="bridge",
            cluster_sizes=(Range(9, 10),),
            cluster_count=Range(1, 1),
            wild_behavior="bridge",
        )
        arc = _make_arc((bridge_phase,))
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)

        min_cs = default_config.board.min_cluster_size  # typically 5

        # Empties = min_cluster_size → feasible (bridge reuses wilds)
        board = _board_with_n_empties(min_cs, default_config)
        wild_pos = [Position(min_cs, 0)]  # a wild just outside the emptied area
        context = _make_context(board, default_config, active_wilds=wild_pos)

        est = _make_estimator(default_config)
        assert est.estimate(progress, context) is True

    def test_bridge_without_wild_falls_back_to_full_size(
        self, default_config: MasterConfig,
    ) -> None:
        """Bridge phase with no active wilds → uses full cluster size."""
        bridge_phase = _make_phase(
            id="bridge",
            cluster_sizes=(Range(9, 10),),
            cluster_count=Range(1, 1),
            wild_behavior="bridge",
        )
        arc = _make_arc((bridge_phase,))
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)

        # 7 empties, no wilds → needs 9 (full size), infeasible
        board = _board_with_n_empties(7, default_config)
        context = _make_context(board, default_config, active_wilds=[])

        est = _make_estimator(default_config)
        assert est.estimate(progress, context) is False

    def test_future_phases_not_summed(
        self, default_config: MasterConfig,
    ) -> None:
        """Only the next phase's requirement is checked — later phases ignored.

        4-phase arc at step 1. Next phase needs 5 cells, later phases
        would sum to 20+. With 6 empties, must report feasible.
        """
        phase_0 = _make_phase(id="done", cluster_sizes=(Range(7, 7),))
        phase_1 = _make_phase(id="next", cluster_sizes=(Range(5, 6),))
        phase_2 = _make_phase(id="later_a", cluster_sizes=(Range(10, 12),))
        phase_3 = _make_phase(id="later_b", cluster_sizes=(Range(10, 12),))
        arc = _make_arc((phase_0, phase_1, phase_2, phase_3))
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        progress.current_phase_index = 1  # at phase_1

        board = _board_with_n_empties(6, default_config)
        context = _make_context(board, default_config)

        est = _make_estimator(default_config)
        # Old bug: 5 + 10 + 10 = 25 > 6 → False. Fix: 5 ≤ 6 → True.
        assert est.estimate(progress, context) is True

    def test_full_arc_walkthrough_wild_enable_rocket(
        self, default_config: MasterConfig,
    ) -> None:
        """Simulate wild_enable_rocket arc — feasible at each step.

        Phase 0: spawn_wild — size 7, empties ~49 (full board)
        Phase 1: bridge_rocket — size 9 (bridge), empties ~7-8 after explosion
        Phase 2: arm_rocket — size 5, empties ~9-10 after bridge explosion
        Phase 3: fire_rocket — no cluster, empties don't matter
        """
        phase_0 = _make_phase(id="spawn_wild", cluster_sizes=(Range(7, 7),), wild_behavior="spawn")
        phase_1 = _make_phase(id="bridge_rocket", cluster_sizes=(Range(9, 10),), wild_behavior="bridge")
        phase_2 = _make_phase(id="arm_rocket", cluster_sizes=(Range(5, 6),))
        phase_3 = _make_phase(id="fire_rocket", cluster_sizes=())
        arc = _make_arc((phase_0, phase_1, phase_2, phase_3))
        sig = _make_signature(narrative_arc=arc)

        est = _make_estimator(default_config)
        min_cs = default_config.board.min_cluster_size

        # Step 0: full empty board, checking phase_0 (size 7)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        progress.current_phase_index = 0
        board_0 = _board_with_n_empties(49, default_config)
        ctx_0 = _make_context(board_0, default_config)
        assert est.estimate(progress, ctx_0) is True

        # Step 1: ~8 empties from step 0 explosion, checking phase_1 (bridge, needs min_cs)
        progress.current_phase_index = 1
        board_1 = _board_with_n_empties(8, default_config)
        wild_pos = [Position(3, 3)]
        ctx_1 = _make_context(board_1, default_config, active_wilds=wild_pos)
        assert est.estimate(progress, ctx_1) is True
        # Verify: bridge requires min_cluster_size (5), not full size (9)
        assert 8 >= min_cs

        # Step 2: ~9 empties from bridge explosion, checking phase_2 (size 5)
        progress.current_phase_index = 2
        board_2 = _board_with_n_empties(9, default_config)
        ctx_2 = _make_context(board_2, default_config)
        assert est.estimate(progress, ctx_2) is True

        # Step 3: fire phase, no cluster needed
        progress.current_phase_index = 3
        board_3 = _board_with_n_empties(0, default_config)
        ctx_3 = _make_context(board_3, default_config)
        assert est.estimate(progress, ctx_3) is True


# ---------------------------------------------------------------------------
# Category-shaped regression tests
# ---------------------------------------------------------------------------

class TestCategoryRegression:
    """Verify feasibility verdicts for phase shapes matching known archetypes.

    Category C (passing today) must remain feasible.
    Category A/B (previously blocked by defects) must now report feasible.
    """

    @pytest.mark.parametrize("archetype_label,phases,empties,active_wilds", [
        # Category C — single-phase or small arcs that pass by luck
        ("wild_bridge_small", (
            {"id": "bridge_small", "cluster_sizes": (Range(5, 6),), "wild_behavior": "bridge"},
        ), 8, [Position(3, 3)]),
        ("rocket_fire_single", (
            {"id": "arm", "cluster_sizes": (Range(5, 6),)},
        ), 10, []),
        ("t1_single", (
            {"id": "cluster", "cluster_sizes": (Range(5, 8),)},
        ), 49, []),
        ("dead_empty", (
            {"id": "terminal", "cluster_sizes": ()},
        ), 0, []),
    ])
    def test_category_c_still_feasible(
        self,
        default_config: MasterConfig,
        archetype_label: str,
        phases: tuple[dict, ...],
        empties: int,
        active_wilds: list[Position],
    ) -> None:
        built_phases = tuple(_make_phase(**p) for p in phases)
        arc = _make_arc(built_phases)
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)

        board = _board_with_n_empties(empties, default_config)
        context = _make_context(board, default_config, active_wilds=active_wilds)

        est = _make_estimator(default_config)
        assert est.estimate(progress, context) is True, (
            f"Category C archetype '{archetype_label}' should remain feasible"
        )

    @pytest.mark.parametrize("archetype_label,phases,phase_index,empties,active_wilds", [
        # Category A — multi-phase arcs broken by cross-phase summing
        ("wild_enable_rocket_step1", (
            {"id": "spawn", "cluster_sizes": (Range(7, 7),), "wild_behavior": "spawn"},
            {"id": "bridge", "cluster_sizes": (Range(9, 10),), "wild_behavior": "bridge"},
            {"id": "arm", "cluster_sizes": (Range(5, 6),)},
            {"id": "fire", "cluster_sizes": ()},
        ), 1, 8, [Position(3, 3)]),
        ("wild_enable_bomb_step1", (
            {"id": "spawn", "cluster_sizes": (Range(7, 7),), "wild_behavior": "spawn"},
            {"id": "bridge", "cluster_sizes": (Range(11, 12),), "wild_behavior": "bridge"},
            {"id": "arm", "cluster_sizes": (Range(5, 6),)},
            {"id": "fire", "cluster_sizes": ()},
        ), 1, 8, [Position(3, 3)]),
        ("rocket_chain_bomb_step1", (
            {"id": "arm_rocket", "cluster_sizes": (Range(9, 10),)},
            {"id": "arm_bomb", "cluster_sizes": (Range(11, 12),)},
            {"id": "fire", "cluster_sizes": ()},
        ), 1, 12, []),
        ("booster_phase_multi_step1", (
            {"id": "phase_0", "cluster_sizes": (Range(5, 6),)},
            {"id": "phase_1", "cluster_sizes": (Range(11, 12),)},
            {"id": "phase_2", "cluster_sizes": (Range(5, 6),)},
            {"id": "fire", "cluster_sizes": ()},
        ), 1, 12, []),
        # Category B — bridge inflation (single remaining bridge phase)
        ("wild_bridge_large", (
            {"id": "spawn", "cluster_sizes": (Range(7, 7),), "wild_behavior": "spawn"},
            {"id": "bridge_large", "cluster_sizes": (Range(9, 10),), "wild_behavior": "bridge"},
        ), 1, 7, [Position(3, 3)]),
        ("wild_rocket_tease", (
            {"id": "spawn", "cluster_sizes": (Range(7, 7),), "wild_behavior": "spawn"},
            {"id": "bridge_tease", "cluster_sizes": (Range(9, 10),), "wild_behavior": "bridge"},
        ), 1, 7, [Position(3, 3)]),
    ])
    def test_previously_blocked_archetypes_now_feasible(
        self,
        default_config: MasterConfig,
        archetype_label: str,
        phases: tuple[dict, ...],
        phase_index: int,
        empties: int,
        active_wilds: list[Position],
    ) -> None:
        built_phases = tuple(_make_phase(**p) for p in phases)
        arc = _make_arc(built_phases)
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)
        progress.current_phase_index = phase_index

        board = _board_with_n_empties(empties, default_config)
        context = _make_context(board, default_config, active_wilds=active_wilds)

        est = _make_estimator(default_config)
        assert est.estimate(progress, context) is True, (
            f"Category A/B archetype '{archetype_label}' at phase {phase_index} "
            f"should now be feasible with {empties} empties"
        )


# ---------------------------------------------------------------------------
# Category D — A2 bottleneck-phase rejection
# ---------------------------------------------------------------------------

class TestBottleneckRejection:
    """A2: an arc where a *later* phase exceeds the steady-state floor must
    be rejected by estimate(), even when the immediately-next phase fits.
    """

    def test_a2_late_phase_bottleneck_rejected(
        self, default_config: MasterConfig,
    ) -> None:
        """Arc with a small phase-1 (fits current empties) followed by a
        phase requiring more cells than the steady-state floor (current +
        min_cluster_size) — must reject without summing requirements.
        """
        # min_cluster_size for royal_tumble is 5 → steady-state floor when
        # current empties is 8 = 8 + 5 = 13. A late phase needing 14 must
        # trip the bottleneck.
        small_phase = _make_phase(
            id="small", cluster_sizes=(Range(5, 6),), cluster_count=Range(1, 1),
        )
        # 14 cells: cluster_sizes 14 × count 1 — exceeds 13 floor
        bottleneck_phase = _make_phase(
            id="huge", cluster_sizes=(Range(14, 14),), cluster_count=Range(1, 1),
        )
        arc = _make_arc((small_phase, bottleneck_phase))
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)

        board = _board_with_n_empties(8, default_config)
        context = _make_context(board, default_config)

        est = _make_estimator(default_config)
        assert est.estimate(progress, context) is False, (
            "Late-phase bottleneck (needs 14 cells; floor is 13) must "
            "trigger A2 rejection, even though phase 1 fits the 8 empties"
        )

    def test_a2_within_floor_is_feasible(
        self, default_config: MasterConfig,
    ) -> None:
        """Same shape, but the late phase fits the steady-state floor —
        confirms the bottleneck check doesn't over-reject.
        """
        small_phase = _make_phase(
            id="small", cluster_sizes=(Range(5, 6),), cluster_count=Range(1, 1),
        )
        # 13 cells: exactly at the floor (8 empties + 5 min_cluster_size)
        edge_phase = _make_phase(
            id="edge", cluster_sizes=(Range(13, 13),), cluster_count=Range(1, 1),
        )
        arc = _make_arc((small_phase, edge_phase))
        sig = _make_signature(narrative_arc=arc)
        progress = ProgressTracker(signature=sig, centipayout_multiplier=100)

        board = _board_with_n_empties(8, default_config)
        context = _make_context(board, default_config)

        est = _make_estimator(default_config)
        assert est.estimate(progress, context) is True, (
            "Late phase at exactly the steady-state floor must remain "
            "feasible — bottleneck must not be off-by-one strict"
        )
