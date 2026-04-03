"""ASCII visual tracer — replays a book's event stream for manual verification.

Reads from the event list (list[dict]) only — zero dependency on generator
internals. Can trace books from live generation or loaded from stored JSONL.

Rendering conventions match RoyalTumble_ExperienceEngine_Tracer.md:
- Board: | H2 | normal, |*H2*| winner, |[  ]| vacancy
- Grid mults: [N] touched this step, N from prior
- Gravity: ↓ ↙ ↘ direction arrows
- Sections: ======== major, -------- minor
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from typing import TextIO

from ..config.schema import MasterConfig
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

# Width of each cell in the board display — 4 chars for symbol name
_CELL_WIDTH: int = 4
_MAJOR_SEP: str = "=" * 60
_MINOR_SEP: str = "-" * 60


class EventTracer:
    """ASCII replay of an event stream. Reads list[dict], no generator coupling.

    Uses dict dispatch to route each event type to its renderer. Board dimensions
    come from config — no hardcoded grid sizes.
    """

    __slots__ = ("_config", "_lines", "_renderers", "_board_state", "_grid_state", "_spawn_positions")

    def __init__(self, config: MasterConfig) -> None:
        self._config = config
        self._lines: list[str] = []
        # Mutable board/grid state — cached from REVEAL for downstream renderers
        self._board_state: list[list[dict]] = []
        self._grid_state: list[list[int]] = []
        # Dict dispatch: event type → render method (py-developer pattern)
        self._renderers: dict[str, Callable[[dict], None]] = {
            REVEAL: self._render_reveal,
            WIN_INFO: self._render_win_info,
            UPDATE_TUMBLE_WIN: self._render_update_tumble_win,
            UPDATE_BOARD_MULTIPLIERS: self._render_update_board_multipliers,
            SET_WIN: self._render_set_win,
            SET_TOTAL_WIN: self._render_set_total_win,
            FINAL_WIN: self._render_final_win,
            BOOSTER_SPAWN_INFO: self._render_booster_spawn_info,
            BOOSTER_ARM_INFO: self._render_booster_arm_info,
            BOOSTER_FIRE_INFO: self._render_booster_fire_info,
            GRAVITY_SETTLE: self._render_gravity_settle,
            FREE_SPIN_TRIGGER: self._render_free_spin_trigger,
            UPDATE_FREE_SPIN: self._render_update_free_spin,
            FREE_SPIN_END: self._render_free_spin_end,
            WINCAP: self._render_wincap,
        }

    def trace(self, book: BookRecord, output: TextIO = sys.stdout) -> None:
        """Walk the event list, dispatch each type to a renderer, write to output."""
        self._lines = []
        self._board_state = []
        self._grid_state = []
        self._spawn_positions: set[tuple[int, int]] = set()

        # Header
        payout_mult = book.payoutMultiplier / 100.0 if book.payoutMultiplier else 0.0
        self._lines.append(_MAJOR_SEP)
        self._lines.append(
            f"  BOOK #{book.id} | criteria: {book.criteria} "
            f"| payout: {payout_mult:.1f}x"
        )
        self._lines.append(
            f"  baseGameWins: {book.baseGameWins:.2f} "
            f"| freeGameWins: {book.freeGameWins:.2f}"
        )
        self._lines.append(_MAJOR_SEP)
        self._lines.append("")

        for event in book.events:
            event_type = event.get("type", "")
            renderer = self._renderers.get(event_type)
            if renderer is not None:
                renderer(event)
            else:
                self._lines.append(f"[unknown event: {event_type}]")
                self._lines.append("")

        output.write("\n".join(self._lines))
        output.write("\n")

    def trace_to_string(self, book: BookRecord) -> str:
        """Render trace to a string instead of a stream."""
        import io
        buf = io.StringIO()
        self.trace(book, output=buf)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Per-event renderers
    # ------------------------------------------------------------------

    def _render_reveal(self, event: dict) -> None:
        """Render initial board display. Initializes grid state from boardMultipliers."""
        idx = event.get("index", "?")
        game_type = event.get("gameType", "basegame")

        self._lines.append(f"--- REVEAL [{idx}] ({game_type}) ---")
        board_grid = event.get("board", [])
        # Cache board — WIN_INFO needs it for side-by-side view, GRAVITY_SETTLE mutates it
        self._board_state = copy.deepcopy(board_grid)
        self._lines.extend(self._format_board(board_grid))

        # Initialize grid state from reveal's boardMultipliers
        board_mults = event.get("boardMultipliers", [])
        if board_mults:
            self._grid_state = copy.deepcopy(board_mults)

        self._lines.append("")

    def _render_update_board_multipliers(self, event: dict) -> None:
        """Render sparse board multiplier delta. Applies changes to cached grid state."""
        idx = event.get("index", "?")
        changes = event.get("boardMultipliers", [])

        if not changes:
            return

        # Apply sparse delta to cached grid state
        for entry in changes:
            mult = entry.get("multiplier", 0)
            pos = entry.get("position", {})
            reel = pos.get("reel", 0)
            row = pos.get("row", 0)
            if reel < len(self._grid_state) and row < len(self._grid_state[reel]):
                self._grid_state[reel][row] = mult

        self._lines.append(f"--- BOARD MULTIPLIERS [{idx}] ({len(changes)} changed) ---")
        self._lines.extend(self._format_grid_mults(self._grid_state))
        self._lines.append("")

    def _render_win_info(self, event: dict) -> None:
        """Render cluster win breakdown with payout math."""
        idx = event.get("index", "?")
        total_win = event.get("totalWin", 0)
        wins = event.get("wins", [])

        self._lines.append(f"--- WIN INFO [{idx}] | totalWin: {total_win} ---")
        for i, win in enumerate(wins):
            size = win.get("clusterSize", 0)
            base_payout = win.get("basePayout", 0)
            cluster_payout = win.get("clusterPayout", 0)
            cluster_mult = win.get("clusterMultiplier", 1)
            overlay = win.get("overlay", {})

            # Derive symbol from first cell in cluster
            cluster_data = win.get("cluster", {})
            cells = cluster_data.get("cells", [])
            symbol = cells[0].get("symbol", "?") if cells else "?"

            self._lines.append(
                f"  cluster {i}: {symbol} x{size} "
                f"= {base_payout} * {cluster_mult} = {cluster_payout}"
            )
            self._lines.append(
                f"    overlay: ({overlay.get('reel', '?')}, {overlay.get('row', '?')})"
            )

        # Side-by-side board + grid view
        # Collect ALL winner positions from cluster cells
        all_winners: set[tuple[int, int]] = set()
        for win in wins:
            cluster_data = win.get("cluster", {})
            for cell in cluster_data.get("cells", []):
                all_winners.add((cell.get("reel", 0), cell.get("row", 0)))

        if self._board_state and all_winners:
            board_lines = self._format_board(self._board_state, winners=all_winners)
            if self._grid_state:
                grid_lines = self._format_grid_mults(
                    self._grid_state, touched=all_winners,
                )
                self._lines.extend(self._format_side_by_side(
                    left_lines=board_lines,
                    right_lines=grid_lines,
                    left_header="Board with winners [*]:",
                    right_header="Grid multipliers touched [x]:",
                ))
            else:
                self._lines.append("  Board with winners [*]:")
                self._lines.extend(board_lines)
        self._lines.append("")

    def _render_update_tumble_win(self, event: dict) -> None:
        """Render running cascade payout."""
        idx = event.get("index", "?")
        amount = event.get("amount", 0)
        self._lines.append(f"  TUMBLE WIN [{idx}]: {amount}")

    def _render_booster_spawn_info(self, event: dict) -> None:
        """Render booster spawn event — single event with all spawned boosters."""
        idx = event.get("index", "?")
        boosters = event.get("boosters", [])

        self._lines.append(f"--- BOOSTER SPAWN [{idx}] ---")
        for b in boosters:
            symbol = b.get("symbol", "?")
            pos = b.get("position", {})
            self._lines.append(
                f"  {symbol} at ({pos.get('reel', '?')}, {pos.get('row', '?')})"
            )

            # Place spawned symbol on cached board so downstream gravity sees it
            if self._board_state:
                reel = pos.get("reel", 0)
                row = pos.get("row", 0)
                if reel < len(self._board_state) and row < len(self._board_state[reel]):
                    self._board_state[reel][row] = {"name": symbol}
                    self._spawn_positions.add((reel, row))

        self._lines.append("")

    def _render_booster_arm_info(self, event: dict) -> None:
        """Render booster arm event — boosters that transitioned to ARMED."""
        idx = event.get("index", "?")
        boosters = event.get("boosters", [])

        self._lines.append(f"--- BOOSTER ARM [{idx}] ---")
        for b in boosters:
            symbol = b.get("symbol", "?")
            pos = b.get("position", {})
            self._lines.append(
                f"  {symbol} ARMED at ({pos.get('reel', '?')}, {pos.get('row', '?')})"
            )
        self._lines.append("")

    def _render_booster_fire_info(self, event: dict) -> None:
        """Render booster fire event — per-booster cleared cells with symbols."""
        idx = event.get("index", "?")
        boosters = event.get("boosters", [])

        self._lines.append(f"--- BOOSTER FIRE [{idx}] ---")
        for b in boosters:
            symbol = b.get("symbol", "?")
            cleared = b.get("clearedCells", [])
            self._lines.append(f"  FIRE: {symbol} cleared {len(cleared)} cells")

        self._lines.append("")

    def _render_gravity_settle(self, event: dict) -> None:
        """Render gravity cascade: gravity passes → refill → settle.

        Tracks board state through each sub-step to render intermediate grids.
        Updates _board_state to the settled result for subsequent cascade steps.
        """
        idx = event.get("index", "?")
        move_steps = event.get("moveSteps", [])
        new_symbols = event.get("newSymbols", [])

        self._lines.append(f"--- GRAVITY SETTLE [{idx}] ---")

        # Work on a mutable copy — _board_state updates only at SETTLE
        board = copy.deepcopy(self._board_state) if self._board_state else []

        # Gravity passes — move symbols, render board with moved positions highlighted
        for pass_idx, pass_moves in enumerate(move_steps):
            if not pass_moves:
                continue
            self._lines.append(
                f"  Pass {pass_idx + 1}: GRAVITY "
                f"({len(pass_moves)} move(s))"
            )

            # Text description with direction arrows and symbol names
            moved_destinations: set[tuple[int, int]] = set()
            for move in pass_moves:
                from_cell = move.get("fromCell", {})
                to_cell = move.get("toCell", {})
                fr, fc = from_cell.get("reel", 0), from_cell.get("row", 0)
                tr, tc = to_cell.get("reel", 0), to_cell.get("row", 0)

                # Resolve symbol name from working board
                sym_name = ""
                if board and fr < len(board) and fc < len(board[fr]):
                    sym_name = board[fr][fc].get("name", "")

                # Direction arrow based on reel movement
                if fr == tr:
                    arrow = "↓"
                elif fr > tr:
                    arrow = "↙"
                else:
                    arrow = "↘"

                sym_prefix = f"{sym_name:>3s} " if sym_name else "    "
                self._lines.append(
                    f"    {sym_prefix}({fr},{fc}) {arrow} ({tr},{tc})"
                )
                moved_destinations.add((tr, tc))

            # Apply moves — two-phase to avoid overwrites when moves cross paths
            if board:
                collected: list[tuple[int, int, int, int, dict]] = []
                for move in pass_moves:
                    from_cell = move.get("fromCell", {})
                    to_cell = move.get("toCell", {})
                    fr, fc = from_cell.get("reel", 0), from_cell.get("row", 0)
                    tr, tc = to_cell.get("reel", 0), to_cell.get("row", 0)
                    sym = (
                        board[fr][fc]
                        if fr < len(board) and fc < len(board[fr])
                        else {"name": ""}
                    )
                    collected.append((fr, fc, tr, tc, sym))

                for fr, fc, _tr, _tc, _sym in collected:
                    if fr < len(board) and fc < len(board[fr]):
                        board[fr][fc] = {"name": ""}
                for _fr, _fc, tr, tc, sym in collected:
                    if tr < len(board) and tc < len(board[tr]):
                        board[tr][tc] = sym

                self._lines.extend(
                    self._format_board(board, winners=moved_destinations)
                )
                self._lines.append("")

        # REFILL — fill new symbols into vacancies
        refill_count = sum(len(reel) for reel in new_symbols)
        if refill_count > 0:
            self._lines.append(
                f"  REFILL ({refill_count} new symbol(s) from reel strip)"
            )
            for reel_entries in new_symbols:
                for entry in reel_entries:
                    pos = entry.get("position", {})
                    reel = pos.get("reel", 0)
                    row = pos.get("row", 0)
                    name = entry.get("symbol", "?")
                    self._lines.append(f"    {name} → (R{reel},row{row})")
                    # Apply refill to working board
                    if board and reel < len(board) and row < len(board[reel]):
                        board[reel][row] = {"name": name}
            self._lines.append("")

        # SETTLE — render final board, update cached state for next cascade step
        if board:
            self._lines.append("  SETTLE")
            self._lines.extend(self._format_board(board))
            # Transfer ownership — this becomes the board for the next cascade step
            self._board_state = board
        self._lines.append("")

    def _render_set_win(self, event: dict) -> None:
        """Render spin win result."""
        idx = event.get("index", "?")
        amount = event.get("amount", 0)
        win_level = event.get("winLevel", 0)
        self._lines.append(_MINOR_SEP)
        self._lines.append(
            f"  SPIN WIN [{idx}]: {amount} (level {win_level})"
        )

    def _render_set_total_win(self, event: dict) -> None:
        """Render cumulative total win."""
        idx = event.get("index", "?")
        amount = event.get("amount", 0)
        self._lines.append(f"  TOTAL WIN [{idx}]: {amount}")

    def _render_wincap(self, event: dict) -> None:
        """Render wincap indicator."""
        idx = event.get("index", "?")
        amount = event.get("amount", 0)
        self._lines.append(_MAJOR_SEP)
        self._lines.append(f"  *** WIN CAP [{idx}]: {amount} ***")
        self._lines.append(_MAJOR_SEP)

    def _render_free_spin_trigger(self, event: dict) -> None:
        """Render freespin trigger event."""
        idx = event.get("index", "?")
        total_fs = event.get("totalFs", 0)
        positions = event.get("positions", [])

        self._lines.append(_MAJOR_SEP)
        self._lines.append(f"  FREESPIN TRIGGER [{idx}]: {total_fs} spins")
        pos_str = ", ".join(
            f"({p.get('reel', '?')},{p.get('row', '?')})" for p in positions
        )
        self._lines.append(f"  scatters at: {pos_str}")
        self._lines.append(_MAJOR_SEP)
        self._lines.append("")

    def _render_update_free_spin(self, event: dict) -> None:
        """Render freespin counter."""
        idx = event.get("index", "?")
        current = event.get("amount", 0)
        total = event.get("total", 0)
        self._lines.append(f"{'=' * 40}")
        self._lines.append(f"  FREESPIN {current} / {total}  [{idx}]")
        self._lines.append(f"{'=' * 40}")
        self._lines.append("")

    def _render_free_spin_end(self, event: dict) -> None:
        """Render freespin completion summary."""
        idx = event.get("index", "?")
        amount = event.get("amount", 0)
        win_level = event.get("winLevel", 0)
        self._lines.append(_MAJOR_SEP)
        self._lines.append(
            f"  FREESPIN COMPLETE [{idx}]: {amount} (level {win_level})"
        )
        self._lines.append(_MAJOR_SEP)
        self._lines.append("")

    def _render_final_win(self, event: dict) -> None:
        """Render final result."""
        idx = event.get("index", "?")
        amount = event.get("amount", 0)
        self._lines.append(_MAJOR_SEP)
        self._lines.append(f"  FINAL WIN [{idx}]: {amount}")
        self._lines.append(_MAJOR_SEP)

    # ------------------------------------------------------------------
    # Board formatting helpers
    # ------------------------------------------------------------------

    def _format_board(
        self,
        board_grid: list[list[dict]],
        winners: set[tuple[int, int]] | None = None,
    ) -> list[str]:
        """Format board grid as ASCII table.

        Transposes [reel][row] to display rows top-to-bottom, reels left-to-right.
        Winners marked with *XX* notation.
        """
        if not board_grid:
            return ["  (empty board)"]
        if winners is None:
            winners = set()

        num_reels = len(board_grid)
        num_rows = len(board_grid[0]) if board_grid else 0
        lines: list[str] = []

        # Column headers
        header = "     " + "".join(f"  R{r:<3}" for r in range(num_reels))
        lines.append(header)

        for row in range(num_rows):
            cells: list[str] = []
            for reel in range(num_reels):
                sym_dict = board_grid[reel][row] if row < len(board_grid[reel]) else {}
                name = sym_dict.get("name", "")

                if not name:
                    cell = "[  ]"
                elif (reel, row) in winners:
                    # Highlight winner: *XX* with padding to 4 chars
                    cell = f"*{name:>2s}*"
                else:
                    cell = f" {name:>3s}"

                cells.append(f"|{cell}")
            line = f"  {row:>2d} " + "".join(cells) + "|"
            lines.append(line)

        return lines

    def _format_grid_mults(
        self,
        grid: list[list[int]],
        touched: set[tuple[int, int]] | None = None,
    ) -> list[str]:
        """Format grid multiplier matrix as ASCII table.

        When touched is provided, distinguishes positions incremented THIS step
        ([N] bracket notation) from positions active from prior steps (plain N).
        When touched is None, all non-zero values use bracket notation.
        """
        if not grid:
            return ["  (empty grid)"]

        num_reels = len(grid)
        num_rows = len(grid[0]) if grid else 0
        lines: list[str] = []

        header = "     " + "".join(f"  R{r:<3}" for r in range(num_reels))
        lines.append(header)

        for row in range(num_rows):
            cells: list[str] = []
            for reel in range(num_reels):
                val = grid[reel][row] if row < len(grid[reel]) else 0
                if val > 0 and (touched is None or (reel, row) in touched):
                    # Bracketed — either no distinction requested, or incremented this step
                    cell = f"[{val:>2d}]"
                elif val > 0:
                    # Plain — active from a prior step, not touched this step
                    cell = f" {val:>2d} "
                else:
                    cell = "  . "
                cells.append(f"|{cell}")
            line = f"  {row:>2d} " + "".join(cells) + "|"
            lines.append(line)

        return lines

    def _format_side_by_side(
        self,
        left_lines: list[str],
        right_lines: list[str],
        left_header: str,
        right_header: str,
        gap: int = 4,
    ) -> list[str]:
        """Paste two column blocks side by side with headers and a gap.

        Used by WIN_INFO to show board (with winners) next to grid (with touched markers).
        Pads shorter block with blank lines to match the taller block's height.
        """
        left_width = max((len(line) for line in left_lines), default=0)
        left_width = max(left_width, len(left_header))
        spacer = " " * gap

        lines: list[str] = []
        lines.append(f"  {left_header:<{left_width}}{spacer}{right_header}")

        max_rows = max(len(left_lines), len(right_lines))
        for i in range(max_rows):
            l_line = left_lines[i] if i < len(left_lines) else ""
            r_line = right_lines[i] if i < len(right_lines) else ""
            lines.append(f"{l_line:<{left_width}}{spacer}{r_line}")

        return lines
