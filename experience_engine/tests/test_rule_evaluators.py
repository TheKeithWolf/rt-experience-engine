"""Tests for shared rule evaluators — SpawnEvaluator, ChainEvaluator,
PayoutEstimator, TerminalEvaluator.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ..archetypes.registry import TerminalNearMissSpec
from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol, SymbolTier
from ..step_reasoner.evaluators import (
    ChainEvaluator,
    PayoutEstimator,
    SpawnEvaluator,
    TerminalEvaluator,
)


# ---------------------------------------------------------------------------
# SpawnEvaluator
# ---------------------------------------------------------------------------

class TestSpawnEvaluator:
    """Spawn threshold lookups from config — size → booster mapping."""

    def test_size_7_spawns_wild(self, default_config: MasterConfig) -> None:
        """Cluster size 7 falls in Wild range (7-8)."""
        spawn = SpawnEvaluator(default_config.boosters)
        assert spawn.booster_for_size(7) == "W"
        assert spawn.booster_for_size(8) == "W"

    def test_size_9_spawns_rocket(self, default_config: MasterConfig) -> None:
        """Cluster size 9 falls in Rocket range (9-10)."""
        spawn = SpawnEvaluator(default_config.boosters)
        assert spawn.booster_for_size(9) == "R"
        assert spawn.booster_for_size(10) == "R"

    def test_size_11_spawns_bomb(self, default_config: MasterConfig) -> None:
        """Cluster size 11 falls in Bomb range (11-12)."""
        spawn = SpawnEvaluator(default_config.boosters)
        assert spawn.booster_for_size(11) == "B"

    def test_size_13_spawns_lightball(self, default_config: MasterConfig) -> None:
        """Cluster size 13 falls in LB range (13-14)."""
        spawn = SpawnEvaluator(default_config.boosters)
        assert spawn.booster_for_size(13) == "LB"

    def test_size_15_spawns_superlightball(self, default_config: MasterConfig) -> None:
        """Cluster size 15 falls in SLB range (15-49)."""
        spawn = SpawnEvaluator(default_config.boosters)
        assert spawn.booster_for_size(15) == "SLB"
        assert spawn.booster_for_size(49) == "SLB"

    def test_size_4_no_booster(self, default_config: MasterConfig) -> None:
        """Cluster size 4 is below all thresholds — no booster."""
        spawn = SpawnEvaluator(default_config.boosters)
        assert spawn.booster_for_size(4) is None

    def test_size_range_for_rocket(self, default_config: MasterConfig) -> None:
        """size_range_for_booster returns (min, max) from config."""
        spawn = SpawnEvaluator(default_config.boosters)
        assert spawn.size_range_for_booster("R") == (9, 10)

    def test_size_range_for_unknown(self, default_config: MasterConfig) -> None:
        """Unknown booster returns None."""
        spawn = SpawnEvaluator(default_config.boosters)
        assert spawn.size_range_for_booster("X") is None

    def test_all_thresholds_matches_config(self, default_config: MasterConfig) -> None:
        """all_thresholds returns the same tuple as config."""
        spawn = SpawnEvaluator(default_config.boosters)
        assert spawn.all_thresholds() is default_config.boosters.spawn_thresholds


# ---------------------------------------------------------------------------
# ChainEvaluator
# ---------------------------------------------------------------------------

class TestChainEvaluator:
    """Chain initiator and valid pair queries from config."""

    def test_rocket_can_initiate(self, default_config: MasterConfig) -> None:
        chain = ChainEvaluator(default_config.boosters)
        assert chain.can_initiate_chain("R") is True

    def test_bomb_can_initiate(self, default_config: MasterConfig) -> None:
        chain = ChainEvaluator(default_config.boosters)
        assert chain.can_initiate_chain("B") is True

    def test_lb_cannot_initiate(self, default_config: MasterConfig) -> None:
        chain = ChainEvaluator(default_config.boosters)
        assert chain.can_initiate_chain("LB") is False

    def test_slb_cannot_initiate(self, default_config: MasterConfig) -> None:
        chain = ChainEvaluator(default_config.boosters)
        assert chain.can_initiate_chain("SLB") is False

    def test_wild_cannot_initiate(self, default_config: MasterConfig) -> None:
        chain = ChainEvaluator(default_config.boosters)
        assert chain.can_initiate_chain("W") is False

    def test_no_self_pairs(self, default_config: MasterConfig) -> None:
        """No chain pair where source == target."""
        chain = ChainEvaluator(default_config.boosters)
        for source, target in chain.valid_chain_pairs():
            assert source != target, f"Self-pair: ({source}, {target})"

    def test_all_pairs_have_initiator_source(self, default_config: MasterConfig) -> None:
        """Every pair source must be a chain initiator."""
        chain = ChainEvaluator(default_config.boosters)
        for source, _ in chain.valid_chain_pairs():
            assert chain.can_initiate_chain(source), f"{source} not an initiator"

    def test_all_booster_types_excludes_wild(self, default_config: MasterConfig) -> None:
        """Chainable booster types derived from spawn_order, without W."""
        chain = ChainEvaluator(default_config.boosters)
        assert "W" not in chain.all_booster_types


# ---------------------------------------------------------------------------
# PayoutEstimator
# ---------------------------------------------------------------------------

class TestPayoutEstimator:
    """Tier-median payout estimation from paytable config."""

    def test_low_tier_size_5_positive(self, default_config: MasterConfig) -> None:
        """LOW tier at size 5 should produce a positive centipayout."""
        est = PayoutEstimator(
            default_config.paytable, default_config.centipayout,
            default_config.win_levels, default_config.symbols,
        )
        payout = est.estimate_step_payout(5, SymbolTier.LOW)
        assert payout > 0, "Expected positive payout for LOW tier size 5"

    def test_high_tier_higher_than_low(self, default_config: MasterConfig) -> None:
        """HIGH tier should yield >= LOW tier payout at the same size."""
        est = PayoutEstimator(
            default_config.paytable, default_config.centipayout,
            default_config.win_levels, default_config.symbols,
        )
        low = est.estimate_step_payout(5, SymbolTier.LOW)
        high = est.estimate_step_payout(5, SymbolTier.HIGH)
        assert high >= low

    def test_any_tier_uses_low_conservative(self, default_config: MasterConfig) -> None:
        """ANY tier should return the same as LOW (conservative estimate)."""
        est = PayoutEstimator(
            default_config.paytable, default_config.centipayout,
            default_config.win_levels, default_config.symbols,
        )
        low = est.estimate_step_payout(5, SymbolTier.LOW)
        any_payout = est.estimate_step_payout(5, SymbolTier.ANY)
        assert any_payout == low

    def test_tier_payout_facts_not_empty(self, default_config: MasterConfig) -> None:
        """tier_payout_facts produces non-empty ASP fact string."""
        est = PayoutEstimator(
            default_config.paytable, default_config.centipayout,
            default_config.win_levels, default_config.symbols,
        )
        facts = est.tier_payout_facts()
        assert len(facts) > 0
        assert "tier_size_payout(" in facts


# ---------------------------------------------------------------------------
# Helpers — board builders for TerminalEvaluator tests
# ---------------------------------------------------------------------------

def _make_checkerboard(config: MasterConfig) -> Board:
    """7×7 board with alternating L1/L2 — no group reaches min_cluster_size."""
    board = Board.empty(config.board)
    symbols = (Symbol.L1, Symbol.L2)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            board.set(Position(reel, row), symbols[(reel + row) % 2])
    return board


def _make_board_with_cluster(config: MasterConfig) -> Board:
    """Board with a 5-connected L1 cluster, rest filled with alternating L2/L3."""
    board = Board.empty(config.board)
    # L1 cluster: (0,0)-(0,1)-(0,2)-(1,0)-(1,1) — size 5
    cluster_positions = {
        Position(0, 0), Position(0, 1), Position(0, 2),
        Position(1, 0), Position(1, 1),
    }
    for pos in cluster_positions:
        board.set(pos, Symbol.L1)
    # Fill remaining with alternating L2/L3 to avoid accidental clusters
    filler = (Symbol.L2, Symbol.L3)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            pos = Position(reel, row)
            if pos not in cluster_positions and board.get(pos) is None:
                board.set(pos, filler[(reel + row) % 2])
    return board


def _make_near_miss_board(config: MasterConfig) -> Board:
    """Board with two HIGH-tier near-miss groups (size 4 each), no clusters.

    H1 column at reel 0 rows 0-3, H2 column at reel 3 rows 0-3.
    Remaining cells alternate L1/L2 to prevent clusters and extra near-misses.
    """
    board = Board.empty(config.board)
    # Near-miss group 1: H1 vertical strip of 4
    for row in range(4):
        board.set(Position(0, row), Symbol.H1)
    # Near-miss group 2: H2 vertical strip of 4
    for row in range(4):
        board.set(Position(3, row), Symbol.H2)
    # Fill rest with alternating LOW symbols — checkerboard avoids clusters
    filler = (Symbol.L1, Symbol.L2)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            pos = Position(reel, row)
            if board.get(pos) is None:
                board.set(pos, filler[(reel + row) % 2])
    return board


# ---------------------------------------------------------------------------
# TerminalEvaluator
# ---------------------------------------------------------------------------

class TestTerminalEvaluator:
    """Terminal state checks — dead board, near-misses, dormant boosters."""

    # -- TEST-R2-007 --------------------------------------------------------

    def test_is_dead_on_dead_board(self, default_config: MasterConfig) -> None:
        """Checkerboard board has no clusters — is_dead returns True."""
        te = TerminalEvaluator(default_config)
        board = _make_checkerboard(default_config)
        assert te.is_dead(board) is True

    # -- TEST-R2-008 --------------------------------------------------------

    def test_is_dead_with_cluster(self, default_config: MasterConfig) -> None:
        """Board with a size-5 L1 cluster — is_dead returns False."""
        te = TerminalEvaluator(default_config)
        board = _make_board_with_cluster(default_config)
        assert te.is_dead(board) is False

    # -- TEST-R2-009 --------------------------------------------------------

    def test_satisfies_near_misses_two_high(self, default_config: MasterConfig) -> None:
        """Two HIGH-tier near-miss groups satisfy Range(2, 2) spec."""
        te = TerminalEvaluator(default_config)
        board = _make_near_miss_board(default_config)
        spec = TerminalNearMissSpec(count=Range(2, 2), symbol_tier=SymbolTier.HIGH)
        assert te.satisfies_terminal_near_misses(board, spec) is True

    def test_near_misses_too_few_for_spec(self, default_config: MasterConfig) -> None:
        """Two near-miss groups fail Range(3, 3) — not enough."""
        te = TerminalEvaluator(default_config)
        board = _make_near_miss_board(default_config)
        spec = TerminalNearMissSpec(count=Range(3, 3), symbol_tier=SymbolTier.HIGH)
        assert te.satisfies_terminal_near_misses(board, spec) is False

    def test_near_misses_any_tier(self, default_config: MasterConfig) -> None:
        """ANY tier counts both LOW and HIGH near-misses."""
        te = TerminalEvaluator(default_config)
        board = _make_near_miss_board(default_config)
        # Board has 2 HIGH NMs — ANY tier should find at least those 2
        spec = TerminalNearMissSpec(count=Range(2, 10), symbol_tier=None)
        assert te.satisfies_terminal_near_misses(board, spec) is True

    # -- TEST-R2-010 --------------------------------------------------------

    def test_has_dormant_rocket(self, default_config: MasterConfig) -> None:
        """Dormant rocket present — has_dormant_boosters returns True."""
        te = TerminalEvaluator(default_config)
        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.R, Position(3, 3), orientation="H")
        assert te.has_dormant_boosters(("R",), tracker) is True

    def test_missing_dormant_bomb(self, default_config: MasterConfig) -> None:
        """Only R is dormant — requiring both R and B returns False."""
        te = TerminalEvaluator(default_config)
        tracker = BoosterTracker(default_config.board)
        tracker.add(Symbol.R, Position(3, 3), orientation="H")
        assert te.has_dormant_boosters(("R", "B"), tracker) is False

    def test_has_dormant_empty_required(self, default_config: MasterConfig) -> None:
        """Empty required tuple — trivially True (no requirements to satisfy)."""
        te = TerminalEvaluator(default_config)
        tracker = BoosterTracker(default_config.board)
        assert te.has_dormant_boosters((), tracker) is True


# ---------------------------------------------------------------------------
# TEST-R2-011: No remaining imports from sequence_planner
# ---------------------------------------------------------------------------

_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+.*sequence_planner", re.MULTILINE,
)

_ENGINE_ROOT = Path(__file__).resolve().parent.parent


def test_no_sequence_planner_imports() -> None:
    """No .py file under experience_engine/ imports from the deleted sequence_planner."""
    violations: list[str] = []
    for py_file in _ENGINE_ROOT.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if _IMPORT_PATTERN.search(text):
            violations.append(str(py_file.relative_to(_ENGINE_ROOT)))
    assert violations == [], f"sequence_planner imports found in: {violations}"


