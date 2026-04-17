"""Tests for B5 — WinSplitter dispatch in book_record.

Verifies the three criteria paths produce identical (base, free) tuples
to the pre-refactor if/elif chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ..output.book_record import book_record_from_instance


@dataclass(frozen=True, slots=True)
class _StubInstance:
    """Minimal stand-in for GeneratedInstance — only the fields the
    splitter and BookRecord constructor read.
    """

    sim_id: int
    centipayout: int
    criteria: str
    payout: float


@pytest.mark.parametrize(
    ("criteria", "expected_base", "expected_free"),
    [
        ("0", 0.0, 0.0),
        ("freegame", 0.0, 5.0),
        ("basegame", 5.0, 0.0),
        # wincap and any other unrecognized criteria fall to BaseGameSplitter
        ("wincap", 5.0, 0.0),
        ("unknown_future_value", 5.0, 0.0),
    ],
)
def test_b5_splitter_dispatches_per_criteria(
    criteria: str, expected_base: float, expected_free: float,
) -> None:
    instance = _StubInstance(
        sim_id=1, centipayout=500, criteria=criteria, payout=5.0,
    )
    record = book_record_from_instance(instance, events=[])
    assert record.baseGameWins == expected_base
    assert record.freeGameWins == expected_free
