"""Tests for narrative data types and transition rules (NRA-001 through NRA-008).

Validates frozen behavior, field storage, transition rule presence,
and the _any_wild_can_bridge helper function.
"""

from __future__ import annotations

import pytest

from ..config.schema import MasterConfig
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..narrative.transitions import (
    ALLOWED_TRANSITION_KEYS,
    _any_wild_can_bridge,
    build_transition_rules,
)
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol, SymbolTier
from ..step_reasoner.context import BoardContext
from ..primitives.grid_multipliers import GridMultiplierGrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase(**overrides) -> NarrativePhase:
    """Construct a NarrativePhase with sensible defaults, overridable per-field."""
    defaults = dict(
        id="test_phase",
        intent="A test phase.",
        repetitions=Range(1, 1),
        cluster_count=Range(1, 2),
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
    """Construct a NarrativeArc with sensible defaults, overridable per-field."""
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


def _make_board_context(
    board: Board,
    config: MasterConfig,
    active_wilds: list[Position] | None = None,
) -> BoardContext:
    """Lightweight BoardContext for testing transition predicates."""
    grid = GridMultiplierGrid(config.grid_multiplier, config.board)
    return BoardContext(
        board=board,
        grid_multipliers=grid,
        dormant_boosters=[],
        active_wilds=active_wilds or [],
        _board_config=config.board,
    )


# ---------------------------------------------------------------------------
# NRA-001: NarrativePhase is frozen
# ---------------------------------------------------------------------------

class TestNRA001:
    def test_phase_is_frozen(self):
        phase = _make_phase()
        with pytest.raises(AttributeError):
            phase.id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NRA-002: NarrativeArc is frozen
# ---------------------------------------------------------------------------

class TestNRA002:
    def test_arc_is_frozen(self):
        arc = _make_arc(phases=(_make_phase(),))
        with pytest.raises(AttributeError):
            arc.payout = RangeFloat(0.0, 0.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NRA-003: Phase with repetitions=Range(0,0) is valid (skippable)
# ---------------------------------------------------------------------------

class TestNRA003:
    def test_skippable_phase(self):
        phase = _make_phase(repetitions=Range(0, 0))
        assert phase.repetitions.min_val == 0
        assert phase.repetitions.max_val == 0


# ---------------------------------------------------------------------------
# NRA-004: Phase spawns stores tuple correctly
# ---------------------------------------------------------------------------

class TestNRA004:
    def test_spawns_tuple(self):
        phase = _make_phase(spawns=("R", "B"))
        assert phase.spawns == ("R", "B")
        assert isinstance(phase.spawns, tuple)

    def test_spawns_none(self):
        phase = _make_phase(spawns=None)
        assert phase.spawns is None


# ---------------------------------------------------------------------------
# NRA-005: ALLOWED_TRANSITION_KEYS contains expected keys
# ---------------------------------------------------------------------------

class TestNRA005:
    def test_transition_keys_present(self):
        expected = {"always", "no_clusters", "no_bridges", "booster_fired"}
        assert expected == ALLOWED_TRANSITION_KEYS

    def test_build_transition_rules_keys_match(self, default_config: MasterConfig):
        rules = build_transition_rules(default_config.board, default_config.symbols)
        assert set(rules.keys()) == ALLOWED_TRANSITION_KEYS


# ---------------------------------------------------------------------------
# NRA-006: _any_wild_can_bridge returns True when wild has >= 2 same-symbol neighbors
# ---------------------------------------------------------------------------

class TestNRA006:
    def test_wild_can_bridge(self, default_config: MasterConfig):
        """Wild at (3,3) with L1 at (2,3) and (4,3) — two same-symbol neighbors."""
        board = Board.empty(default_config.board)
        wild_pos = Position(3, 3)
        board.set(wild_pos, Symbol.W)
        board.set(Position(2, 3), Symbol.L1)
        board.set(Position(4, 3), Symbol.L1)

        result = _any_wild_can_bridge(
            _make_board_context(board, default_config, active_wilds=[wild_pos]),
            default_config.board,
            default_config.symbols,
        )
        assert result is True


# ---------------------------------------------------------------------------
# NRA-007: _any_wild_can_bridge returns False when wild has < 2 same-symbol neighbors
# ---------------------------------------------------------------------------

class TestNRA007:
    def test_wild_cannot_bridge_different_symbols(self, default_config: MasterConfig):
        """Wild at (3,3) with L1 at (2,3) and H1 at (4,3) — no pair."""
        board = Board.empty(default_config.board)
        wild_pos = Position(3, 3)
        board.set(wild_pos, Symbol.W)
        board.set(Position(2, 3), Symbol.L1)
        board.set(Position(4, 3), Symbol.H1)

        result = _any_wild_can_bridge(
            _make_board_context(board, default_config, active_wilds=[wild_pos]),
            default_config.board,
            default_config.symbols,
        )
        assert result is False

    def test_wild_cannot_bridge_single_neighbor(self, default_config: MasterConfig):
        """Wild at (0,0) corner with only one neighbor occupied."""
        board = Board.empty(default_config.board)
        wild_pos = Position(0, 0)
        board.set(wild_pos, Symbol.W)
        board.set(Position(1, 0), Symbol.L1)

        result = _any_wild_can_bridge(
            _make_board_context(board, default_config, active_wilds=[wild_pos]),
            default_config.board,
            default_config.symbols,
        )
        assert result is False


# ---------------------------------------------------------------------------
# NRA-008: _any_wild_can_bridge ignores non-standard symbols
# ---------------------------------------------------------------------------

class TestNRA008:
    def test_wild_ignores_special_neighbors(self, default_config: MasterConfig):
        """Wild at (3,3) with Scatter at (2,3) and Scatter at (4,3) — specials ignored."""
        board = Board.empty(default_config.board)
        wild_pos = Position(3, 3)
        board.set(wild_pos, Symbol.W)
        board.set(Position(2, 3), Symbol.S)
        board.set(Position(4, 3), Symbol.S)

        result = _any_wild_can_bridge(
            _make_board_context(board, default_config, active_wilds=[wild_pos]),
            default_config.board,
            default_config.symbols,
        )
        assert result is False

    def test_wild_ignores_booster_neighbors(self, default_config: MasterConfig):
        """Wild with two Rocket neighbors — not standard, should not bridge."""
        board = Board.empty(default_config.board)
        wild_pos = Position(3, 3)
        board.set(wild_pos, Symbol.W)
        board.set(Position(2, 3), Symbol.R)
        board.set(Position(4, 3), Symbol.R)

        result = _any_wild_can_bridge(
            _make_board_context(board, default_config, active_wilds=[wild_pos]),
            default_config.board,
            default_config.symbols,
        )
        assert result is False
