"""Tests for B1 — FamilyValidator protocol & registry (partial).

Verifies the dead/t1 family rule (no wilds or boosters on initial /
terminal boards) produces the same errors via the new registry as it did
via the inline if-chain.
"""

from __future__ import annotations

import pytest

from ..config.schema import MasterConfig
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol, SymbolTier
from ..validation.family_validators import (
    DEFAULT_FAMILY_VALIDATOR,
    FAMILY_VALIDATORS,
)
from ..archetypes.registry import ArchetypeSignature


def _make_signature(family: str) -> ArchetypeSignature:
    return ArchetypeSignature(
        id="test_sig",
        family=family,
        criteria="basegame",
        required_cluster_count=Range(1, 2),
        required_cluster_sizes=(Range(5, 8),),
        required_cluster_symbols=SymbolTier.LOW,
        required_scatter_count=Range(0, 0),
        required_near_miss_count=Range(0, 0),
        required_near_miss_symbol_tier=None,
        required_cascade_depth=Range(0, 0),
        cascade_steps=None,
        symbol_tier_per_step=None,
        payout_range=RangeFloat(0.0, 5.0),
        required_booster_spawns={},
        required_booster_fires={},
        required_chain_depth=Range(0, 0),
        rocket_orientation=None,
        lb_target_tier=None,
        terminal_near_misses=None,
        dormant_boosters_on_terminal=None,
        max_component_size=None,
        triggers_freespin=False,
        reaches_wincap=False,
    )


def _board_with(default_config: MasterConfig, sym_at: dict[Position, Symbol]) -> Board:
    """Build a board with given symbols; the rest stays empty (None)."""
    board = Board(default_config.board.num_reels, default_config.board.num_rows)
    for pos, sym in sym_at.items():
        board.set(pos, sym)
    return board


def test_b1_dead_validator_rejects_wild_on_terminal(
    default_config: MasterConfig,
) -> None:
    sig = _make_signature("dead")
    initial = _board_with(default_config, {})
    terminal = _board_with(default_config, {Position(3, 3): Symbol.W})
    errors = FAMILY_VALIDATORS["dead"].validate(
        sig, initial, terminal, is_cascade=False,
    )
    assert any("W" in err for err in errors)


def test_b1_t1_validator_rejects_booster_on_initial(
    default_config: MasterConfig,
) -> None:
    sig = _make_signature("t1")
    initial = _board_with(default_config, {Position(0, 0): Symbol.R})
    terminal = _board_with(default_config, {})
    errors = FAMILY_VALIDATORS["t1"].validate(
        sig, initial, terminal, is_cascade=True,
    )
    assert any("R" in err for err in errors)


def test_b1_default_validator_returns_no_errors(
    default_config: MasterConfig,
) -> None:
    """Wild family doesn't have a registered validator yet — falls
    through to the no-op default. (The wild-specific rules remain inline
    in InstanceValidator until the surrounding state extraction lands.)
    """
    sig = _make_signature("wild")
    terminal = _board_with(default_config, {Position(3, 3): Symbol.W})
    fv = FAMILY_VALIDATORS.get("wild", DEFAULT_FAMILY_VALIDATOR)
    assert fv.validate(sig, terminal, terminal, is_cascade=False) == ()


def test_b1_unrelated_family_uses_default(
    default_config: MasterConfig,
) -> None:
    sig = _make_signature("rocket")
    terminal = _board_with(default_config, {})
    fv = FAMILY_VALIDATORS.get("rocket", DEFAULT_FAMILY_VALIDATOR)
    assert fv.validate(sig, terminal, terminal, is_cascade=False) == ()
