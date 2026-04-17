"""BookRecord — RGS-compatible output record matching the main SDK's Book.to_json() format.

Each generated instance is converted to a BookRecord for JSONL output. The
criteria field determines how total payout is split between base and free wins.
B5: criteria-based split is dispatched through a small WinSplitter Protocol
registry instead of an if/elif chain so adding a new criteria split rule
requires only registering a new splitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..pipeline.data_types import GeneratedInstance


@dataclass(frozen=True, slots=True)
class BookRecord:
    """Single book for RGS output — one per generated instance.

    Field names use camelCase to match the SDK's Book JSON format exactly.
    payoutMultiplier is centipayout (integer), not the raw float multiplier.
    """

    id: int
    payoutMultiplier: int
    events: tuple[dict, ...]
    criteria: str
    baseGameWins: float
    freeGameWins: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict matching the RGS book format."""
        return {
            "id": self.id,
            "payoutMultiplier": self.payoutMultiplier,
            "events": list(self.events),
            "criteria": self.criteria,
            "baseGameWins": self.baseGameWins,
            "freeGameWins": self.freeGameWins,
        }


# ---------------------------------------------------------------------------
# B5 — WinSplitter protocol + per-criteria splitters
# ---------------------------------------------------------------------------


class WinSplitter(Protocol):
    """Splits a payout into (base_wins, free_wins) for a given criteria."""

    def split(self, payout: float) -> tuple[float, float]: ...


class _ZeroSplitter:
    """Dead spins (criteria == '0') — neither base nor free wins."""

    def split(self, payout: float) -> tuple[float, float]:
        return (0.0, 0.0)


class _FreeGameSplitter:
    """Free-game spins — entire payout attributed to free wins."""

    def split(self, payout: float) -> tuple[float, float]:
        return (0.0, payout)


class _BaseGameSplitter:
    """Default (basegame, wincap, …) — entire payout in base wins."""

    def split(self, payout: float) -> tuple[float, float]:
        return (payout, 0.0)


# Registry — adding a new criteria split rule requires only an entry here.
_SPLITTERS: dict[str, WinSplitter] = {
    "0": _ZeroSplitter(),
    "freegame": _FreeGameSplitter(),
}
_DEFAULT_SPLITTER: WinSplitter = _BaseGameSplitter()


def book_record_from_instance(
    instance: GeneratedInstance,
    events: list[dict],
) -> BookRecord:
    """Build a BookRecord from a generated instance and its event stream.

    Base/free split is dispatched via the WinSplitter registry (B5):
    - "0" (dead spin)  → ZeroSplitter      → (0, 0)
    - "freegame"       → FreeGameSplitter  → (0, payout)
    - any other        → BaseGameSplitter  → (payout, 0)  [basegame, wincap, …]
    """
    splitter = _SPLITTERS.get(instance.criteria, _DEFAULT_SPLITTER)
    base_wins, free_wins = splitter.split(instance.payout)

    return BookRecord(
        id=instance.sim_id,
        payoutMultiplier=instance.centipayout,
        events=tuple(events),
        criteria=instance.criteria,
        baseGameWins=base_wins,
        freeGameWins=free_wins,
    )
