"""ASCII visual tracer — replays a book's event stream for manual verification.

Thin orchestrator: maintains board/grid/booster state across events, dispatches
to stateless renderers in renderers.py. Zero formatting logic — all ASCII
rendering delegates to the formatting/ package.

Output format matches RoyalTumble_ExperienceEngine_Tracer.md.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from ..config.schema import MasterConfig
from ..formatting.cells import CellStyle
from ..formatting.constants import MAJOR_SEP
from ..output.book_record import BookRecord
from ..output.event_types import (
    BOOSTER_ARM_INFO,
    BOOSTER_FIRE_INFO,
    BOOSTER_SPAWN_INFO,
    FINAL_WIN,
    FREE_SPIN_END,
    FREE_SPIN_TRIGGER,
    GRAVITY_SETTLE,
    REVEAL,
    SET_TOTAL_WIN,
    SET_WIN,
    UPDATE_BOARD_MULTIPLIERS,
    UPDATE_FREE_SPIN,
    UPDATE_TUMBLE_WIN,
    WINCAP,
    WIN_INFO,
)
from . import renderers


class EventTracer:
    """ASCII replay of an event stream. Reads list[dict], no generator coupling.

    Responsibilities: state management + dispatch only. All rendering logic
    lives in renderers.py; all formatting in formatting/.
    """

    __slots__ = (
        "_config",
        "_board_state",
        "_grid_state",
        "_booster_styles",
        "_renderers",
    )

    def __init__(self, config: MasterConfig) -> None:
        self._config = config
        self._board_state: list[list[dict]] = []
        self._grid_state: list[list[int]] = []
        # Booster visual lifecycle — position → current display style
        self._booster_styles: dict[tuple[int, int], CellStyle] = {}
        # Dict dispatch: event type → handler wrapper
        self._renderers: dict[str, Callable[[dict], list[str]]] = {
            REVEAL: self._handle_reveal,
            WIN_INFO: self._handle_win_info,
            UPDATE_TUMBLE_WIN: self._handle_update_tumble_win,
            UPDATE_BOARD_MULTIPLIERS: self._handle_update_board_multipliers,
            SET_WIN: self._handle_set_win,
            SET_TOTAL_WIN: self._handle_set_total_win,
            FINAL_WIN: self._handle_final_win,
            BOOSTER_SPAWN_INFO: self._handle_booster_spawn_info,
            BOOSTER_ARM_INFO: self._handle_booster_arm_info,
            BOOSTER_FIRE_INFO: self._handle_booster_fire_info,
            GRAVITY_SETTLE: self._handle_gravity_settle,
            FREE_SPIN_TRIGGER: self._handle_free_spin_trigger,
            UPDATE_FREE_SPIN: self._handle_update_free_spin,
            FREE_SPIN_END: self._handle_free_spin_end,
            WINCAP: self._handle_wincap,
        }

    def trace(self, book: BookRecord, output: TextIO = sys.stdout) -> None:
        """Walk the event list, dispatch each type to a renderer, write to output."""
        self._board_state = []
        self._grid_state = []
        self._booster_styles = {}

        lines: list[str] = []

        # Spec header: PLAYTEST TRACE -- Sim #N
        payout_mult = book.payoutMultiplier / 100.0 if book.payoutMultiplier else 0.0
        base_wins = book.baseGameWins if book.baseGameWins else 0.0
        free_wins = book.freeGameWins if book.freeGameWins else 0.0

        lines.append(MAJOR_SEP)
        lines.append(f"  PLAYTEST TRACE -- Sim #{book.id}")
        lines.append(f"  Criteria: {book.criteria}")
        lines.append(f"  Payout: {payout_mult:.2f}x")
        lines.append(f"  Base Wins: {base_wins:.2f}x")
        lines.append(f"  Free Wins: {free_wins:.2f}x")
        lines.append(MAJOR_SEP)
        lines.append("")

        for event in book.events:
            event_type = event.get("type", "")
            handler = self._renderers.get(event_type)
            if handler is not None:
                lines.extend(handler(event))

        output.write("\n".join(lines))
        output.write("\n")

    def trace_to_string(self, book: BookRecord) -> str:
        """Render trace to a string instead of a stream."""
        import io
        buf = io.StringIO()
        self.trace(book, output=buf)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Handler wrappers — thin bridges between dispatch and stateless renderers
    # ------------------------------------------------------------------

    def _handle_reveal(self, event: dict) -> list[str]:
        lines, self._board_state, self._grid_state = renderers.render_reveal(
            event,
            self._config.board.num_reels,
            self._config.board.num_rows,
        )
        # Fresh board = fresh booster state
        self._booster_styles = {}
        return lines

    def _handle_win_info(self, event: dict) -> list[str]:
        lines, winners = renderers.render_win_info(
            event,
            self._board_state,
            self._grid_state,
            self._booster_styles,
            self._config.board.num_reels,
            self._config.board.num_rows,
        )
        # Winning cluster symbols are removed by the game engine before gravity
        for reel, row in winners:
            if reel < len(self._board_state) and row < len(self._board_state[reel]):
                self._board_state[reel][row] = {"name": ""}
        return lines

    def _handle_update_tumble_win(self, event: dict) -> list[str]:
        return renderers.render_update_tumble_win(event)

    def _handle_update_board_multipliers(self, event: dict) -> list[str]:
        lines, self._grid_state = renderers.render_update_board_multipliers(
            event,
            self._grid_state,
            self._config.board.num_reels,
            self._config.board.num_rows,
        )
        return lines

    def _handle_set_win(self, event: dict) -> list[str]:
        return renderers.render_set_win(event)

    def _handle_set_total_win(self, event: dict) -> list[str]:
        return renderers.render_set_total_win(event)

    def _handle_final_win(self, event: dict) -> list[str]:
        return renderers.render_final_win(event)

    def _handle_booster_spawn_info(self, event: dict) -> list[str]:
        lines, self._board_state, spawn_styles = renderers.render_booster_spawn_info(
            event,
            self._board_state,
            self._config.board.num_reels,
            self._config.board.num_rows,
        )
        # Merge spawn styles into booster lifecycle tracking
        self._booster_styles.update(spawn_styles)
        return lines

    def _handle_booster_arm_info(self, event: dict) -> list[str]:
        lines, self._booster_styles = renderers.render_booster_arm_info(
            event,
            self._board_state,
            self._booster_styles,
            self._config.board.num_reels,
            self._config.board.num_rows,
        )
        return lines

    def _handle_booster_fire_info(self, event: dict) -> list[str]:
        lines, self._booster_styles = renderers.render_booster_fire_info(
            event,
            self._board_state,
            self._booster_styles,
            self._config.board.num_reels,
            self._config.board.num_rows,
        )
        return lines

    def _handle_gravity_settle(self, event: dict) -> list[str]:
        lines, self._board_state = renderers.render_gravity_settle(
            event,
            self._board_state,
            self._booster_styles,
            self._config.board.num_reels,
            self._config.board.num_rows,
        )
        # Remap booster styles for gravity-moved positions
        self._remap_booster_styles_after_gravity(event)
        return lines

    def _handle_free_spin_trigger(self, event: dict) -> list[str]:
        return renderers.render_free_spin_trigger(
            event,
            self._board_state,
            self._config.board.num_reels,
            self._config.board.num_rows,
        )

    def _handle_update_free_spin(self, event: dict) -> list[str]:
        return renderers.render_update_free_spin(event)

    def _handle_free_spin_end(self, event: dict) -> list[str]:
        return renderers.render_free_spin_end(event)

    def _handle_wincap(self, event: dict) -> list[str]:
        return renderers.render_wincap(event)

    # ------------------------------------------------------------------
    # Booster style lifecycle
    # ------------------------------------------------------------------

    def _remap_booster_styles_after_gravity(self, event: dict) -> None:
        """Remap booster style positions using gravity move steps.

        Delegates to renderers.remap_booster_styles for the two-phase remap
        logic, then replaces self._booster_styles with the remapped result.
        """
        self._booster_styles = renderers.remap_booster_styles(
            self._booster_styles, event.get("moveSteps", []),
        )
