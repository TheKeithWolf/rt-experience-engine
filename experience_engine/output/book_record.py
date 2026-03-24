"""BookRecord — RGS-compatible output record matching the main SDK's Book.to_json() format.

Each generated instance is converted to a BookRecord for JSONL output. The
criteria field determines how total payout is split between base and free wins.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def book_record_from_instance(
    instance: GeneratedInstance,
    events: list[dict],
) -> BookRecord:
    """Build a BookRecord from a generated instance and its event stream.

    Base/free split logic by criteria:
    - "0" (dead spin): both baseGameWins and freeGameWins are 0
    - "freegame": entire payout attributed to freeGameWins
    - "basegame", "wincap", or any other: entire payout to baseGameWins
    """
    if instance.criteria == "0":
        base_wins = 0.0
        free_wins = 0.0
    elif instance.criteria == "freegame":
        base_wins = 0.0
        free_wins = instance.payout
    else:
        # basegame, wincap — all value in base game
        base_wins = instance.payout
        free_wins = 0.0

    return BookRecord(
        id=instance.sim_id,
        payoutMultiplier=instance.centipayout,
        events=tuple(events),
        criteria=instance.criteria,
        baseGameWins=base_wins,
        freeGameWins=free_wins,
    )
