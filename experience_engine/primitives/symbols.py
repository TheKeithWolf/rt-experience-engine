"""Symbol definitions and classification functions for the Experience Engine.

Symbol enum values match the event stream specification — L1-H3 are standard
symbols (1-7), specials start at 100. Classification functions receive config
via parameter (dependency injection) rather than reading globals.
"""

from __future__ import annotations

from enum import Enum, IntEnum

from ..config.schema import SymbolConfig


class Symbol(IntEnum):
    """Game symbols with integer values matching the event stream protocol."""

    # Standard symbols — used in clusters and paytable lookups
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    H1 = 5
    H2 = 6
    H3 = 7
    # Special symbols — not part of standard cluster formation
    S = 100    # Scatter — triggers freespins
    W = 101    # Wild — bridges standard symbol groups
    R = 102    # Rocket — clears row or column
    B = 103    # Bomb — clears 3x3 area
    LB = 104   # Lightball — clears all of one symbol
    SLB = 105  # Super Lightball — clears two symbols + increments multipliers


class SymbolTier(Enum):
    """Symbol value tier for archetype constraints."""

    LOW = "low"
    HIGH = "high"
    ANY = "any"


# Mapping from string name to Symbol enum — built once, reused for lookups
_NAME_TO_SYMBOL: dict[str, Symbol] = {s.name: s for s in Symbol}


def symbol_from_name(name: str) -> Symbol:
    """Convert a string symbol name to the Symbol enum member."""
    try:
        return _NAME_TO_SYMBOL[name]
    except KeyError:
        raise ValueError(f"Unknown symbol name: '{name}'") from None


def is_standard(sym: Symbol, config: SymbolConfig) -> bool:
    """True if the symbol is a standard (cluster-forming) symbol."""
    return sym.name in config.standard


def is_special(sym: Symbol, config: SymbolConfig) -> bool:
    """True if the symbol is not a standard symbol (scatter, wild, booster)."""
    return not is_standard(sym, config)


def is_booster(sym: Symbol) -> bool:
    """True for booster symbols that spawn from large clusters (R, B, LB, SLB)."""
    return sym in (Symbol.R, Symbol.B, Symbol.LB, Symbol.SLB)


def is_wild(sym: Symbol) -> bool:
    """True for the wild symbol that bridges standard symbol groups."""
    return sym is Symbol.W


def is_scatter(sym: Symbol) -> bool:
    """True for the scatter symbol that triggers freespins."""
    return sym is Symbol.S


def tier_of(sym: Symbol, config: SymbolConfig) -> SymbolTier:
    """Determine the value tier of a standard symbol. Raises for non-standard."""
    if sym.name in config.low_tier:
        return SymbolTier.LOW
    if sym.name in config.high_tier:
        return SymbolTier.HIGH
    raise ValueError(f"Symbol {sym.name} has no tier classification")


def symbols_in_tier(tier: SymbolTier, config: SymbolConfig) -> tuple[Symbol, ...]:
    """Return all standard symbols belonging to the given tier."""
    if tier is SymbolTier.LOW:
        return tuple(symbol_from_name(n) for n in config.low_tier)
    if tier is SymbolTier.HIGH:
        return tuple(symbol_from_name(n) for n in config.high_tier)
    # ANY returns all standard symbols
    return tuple(symbol_from_name(n) for n in config.standard)


def get_payout_rank(config: SymbolConfig) -> dict[Symbol, int]:
    """Build a payout rank lookup from config — higher rank = more valuable."""
    return {symbol_from_name(name): rank for name, rank in config.payout_rank}
