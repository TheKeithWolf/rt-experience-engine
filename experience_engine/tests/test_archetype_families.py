"""Tests for B8 — ArchetypeFamily StrEnum.

Verifies the enum's identifier set matches the documented 11 families
and that StrEnum string-equality preserves backward compatibility for
callers still using string literals.
"""

from __future__ import annotations

from ..archetypes.families import ArchetypeFamily
from ..archetypes.registry import FAMILY_CRITERIA, REGISTERED_FAMILIES


def test_b8_eleven_families_defined() -> None:
    """All 11 archetype families documented in the doc must be present."""
    expected = {
        "dead", "t1", "wild", "rocket", "bomb",
        "lb", "slb", "chain", "trigger", "wincap", "reel",
    }
    assert {f.value for f in ArchetypeFamily} == expected


def test_b8_str_enum_equals_string() -> None:
    """StrEnum must compare equal to its string value — preserves
    backward compat for callers using literal strings.
    """
    assert ArchetypeFamily.WILD == "wild"
    assert ArchetypeFamily.ROCKET == "rocket"
    assert str(ArchetypeFamily.DEAD) == "dead"


def test_b8_dict_lookup_works_with_either_form() -> None:
    """A dict keyed by ArchetypeFamily members must accept string lookup
    (and vice versa) — the basis for incremental migration.
    """
    assert FAMILY_CRITERIA[ArchetypeFamily.DEAD] == "0"
    assert FAMILY_CRITERIA["wild"] == "basegame"
    assert FAMILY_CRITERIA[ArchetypeFamily.TRIGGER] == "freegame"


def test_b8_registered_families_matches_enum() -> None:
    """REGISTERED_FAMILIES must derive from the enum so they cannot drift."""
    assert REGISTERED_FAMILIES == frozenset(ArchetypeFamily)
