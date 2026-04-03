"""Event type constants for RGS event streams.

Constants matching RoyalTumble_EventStream.md. Used by both the
EventStreamGenerator and the EventTracer to avoid string literals.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Event type string constants — frontend reads these to dispatch playback
# ---------------------------------------------------------------------------

REVEAL: str = "reveal"
WIN_INFO: str = "winInfo"
UPDATE_TUMBLE_WIN: str = "updateTumbleWin"
UPDATE_BOARD_MULTIPLIERS: str = "updateBoardMultipliers"
SET_WIN: str = "setWin"
SET_TOTAL_WIN: str = "setTotalWin"
FINAL_WIN: str = "finalWin"

# Booster lifecycle
BOOSTER_SPAWN_INFO: str = "boosterSpawnInfo"
BOOSTER_ARM_INFO: str = "boosterArmInfo"
BOOSTER_FIRE_INFO: str = "boosterFireInfo"

# Cascade mechanics
GRAVITY_SETTLE: str = "gravitySettle"

# Freespin lifecycle
FREE_SPIN_TRIGGER: str = "freeSpinTrigger"
UPDATE_FREE_SPIN: str = "updateFreeSpin"
FREE_SPIN_END: str = "freeSpinEnd"

# Win cap — halts cascade when cumulative payout reaches config.wincap.max_payout
WINCAP: str = "wincap"
