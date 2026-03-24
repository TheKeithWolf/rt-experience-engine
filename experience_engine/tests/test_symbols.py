"""TEST-P1-006 through TEST-P1-009: Symbol enum and classification functions."""

from ..config.schema import MasterConfig
from ..primitives.symbols import (
    Symbol,
    SymbolTier,
    is_booster,
    is_scatter,
    is_standard,
    is_wild,
    symbols_in_tier,
    tier_of,
)


def test_p1_006_symbol_enum_has_13_members() -> None:
    """TEST-P1-006: Symbol enum has exactly 13 members."""
    assert len(Symbol) == 13


def test_p1_007_is_standard_classification(default_config: MasterConfig) -> None:
    """TEST-P1-007: is_standard() returns True for L1-H3, False for specials."""
    sym_config = default_config.symbols
    for sym in (Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4,
                Symbol.H1, Symbol.H2, Symbol.H3):
        assert is_standard(sym, sym_config), f"{sym.name} should be standard"

    for sym in (Symbol.S, Symbol.W, Symbol.R, Symbol.B, Symbol.LB, Symbol.SLB):
        assert not is_standard(sym, sym_config), f"{sym.name} should not be standard"


def test_p1_008_is_booster_classification() -> None:
    """TEST-P1-008: is_booster() returns True for R/B/LB/SLB only."""
    for sym in (Symbol.R, Symbol.B, Symbol.LB, Symbol.SLB):
        assert is_booster(sym), f"{sym.name} should be a booster"

    for sym in (Symbol.L1, Symbol.H3, Symbol.S, Symbol.W):
        assert not is_booster(sym), f"{sym.name} should not be a booster"

    assert is_wild(Symbol.W)
    assert not is_wild(Symbol.R)
    assert is_scatter(Symbol.S)
    assert not is_scatter(Symbol.W)


def test_p1_009_symbol_integer_values() -> None:
    """TEST-P1-009: Symbol integer values match event stream reference."""
    assert Symbol.L1.value == 1
    assert Symbol.L2.value == 2
    assert Symbol.L3.value == 3
    assert Symbol.L4.value == 4
    assert Symbol.H1.value == 5
    assert Symbol.H2.value == 6
    assert Symbol.H3.value == 7
    assert Symbol.S.value == 100
    assert Symbol.W.value == 101
    assert Symbol.R.value == 102
    assert Symbol.B.value == 103
    assert Symbol.LB.value == 104
    assert Symbol.SLB.value == 105
