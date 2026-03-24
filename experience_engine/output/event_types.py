"""Event type constants and anticipation computation for RGS event streams.

All 18 event types matching RoyalTumble_EventStream.md. Constants are used by
both the EventStreamGenerator and the EventTracer to avoid string literals.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Event type string constants — frontend reads these to dispatch playback
# ---------------------------------------------------------------------------

REVEAL: str = "reveal"
UPDATE_GRID: str = "updateGrid"
WIN_INFO: str = "winInfo"
UPDATE_TUMBLE_WIN: str = "updateTumbleWin"
SET_WIN: str = "setWin"
SET_TOTAL_WIN: str = "setTotalWin"
FINAL_WIN: str = "finalWin"

# Spawn events — emitted BEFORE gravitySettle so frontend animates spawn first
WILD_SPAWN: str = "wildSpawn"
ROCKET_SPAWN: str = "rocketSpawn"
BOMB_SPAWN: str = "bombSpawn"
LIGHTBALL_SPAWN: str = "lightballSpawn"
SUPERLIGHTBALL_SPAWN: str = "superlightballSpawn"

# Cascade mechanics
GRAVITY_SETTLE: str = "gravitySettle"
BOOSTER_PHASE: str = "boosterPhase"

# Freespin lifecycle
FREE_SPIN_TRIGGER: str = "freeSpinTrigger"
UPDATE_FREE_SPIN: str = "updateFreeSpin"
FREE_SPIN_END: str = "freeSpinEnd"

# Win cap — halts cascade when cumulative payout reaches config.wincap.max_payout
WINCAP: str = "wincap"

# Dict dispatch: booster symbol name → spawn event type string
# Eliminates if/elif chains when emitting spawn events (py-developer pattern)
SPAWN_EVENT_TYPE: dict[str, str] = {
    "W": WILD_SPAWN,
    "R": ROCKET_SPAWN,
    "B": BOMB_SPAWN,
    "LB": LIGHTBALL_SPAWN,
    "SLB": SUPERLIGHTBALL_SPAWN,
}


def compute_anticipation(
    scatter_reels: list[int],
    num_reels: int,
    trigger_threshold: int,
) -> list[int]:
    """Build per-reel anticipation array from scatter positions.

    Anticipation activates on the reel AFTER the Nth scatter (N=trigger_threshold).
    Values increment from 1 upward on each subsequent reel. Reels before the
    threshold scatter have anticipation 0.

    Args:
        scatter_reels: sorted list of reel indices where scatters landed
        num_reels: total number of reels (from config.board.num_reels)
        trigger_threshold: number of scatters before anticipation starts
            (from config.anticipation.trigger_threshold)

    Returns:
        List of length num_reels where 0 = no anticipation, 1+ = anticipation level
    """
    anticipation = [0] * num_reels

    if len(scatter_reels) < trigger_threshold:
        return anticipation

    # Reel after the Nth scatter is where anticipation starts
    # scatter_reels must be sorted by reel index for left-to-right reveal
    sorted_reels = sorted(scatter_reels)
    # The Nth scatter (0-indexed: threshold-1) triggers anticipation on the next reel
    start_reel = sorted_reels[trigger_threshold - 1] + 1

    level = 1
    for reel in range(start_reel, num_reels):
        anticipation[reel] = level
        level += 1

    return anticipation
