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
    BOOSTER_PHASE,
    EVENT_TYPE_TO_SPAWN_SYMBOL,
    FINAL_WIN,
    FREE_SPIN_END,
    FREE_SPIN_TRIGGER,
    GRAVITY_SETTLE,
    REVEAL,
    SET_TOTAL_WIN,
    SET_WIN,
    SPAWN_EVENT_TYPE,
    UPDATE_FREE_SPIN,
    UPDATE_GRID,
    UPDATE_TUMBLE_WIN,
    WINCAP,
    WIN_INFO,
    WILD_SPAWN,
    ROCKET_SPAWN,
    BOMB_SPAWN,
    LIGHTBALL_SPAWN,
    SUPERLIGHTBALL_SPAWN,
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
        # Mutable board/grid state — cached from REVEAL/UPDATE_GRID for downstream renderers
        self._board_state: list[list[dict]] = []
        self._grid_state: list[list[int]] = []
        # Dict dispatch: event type → render method (py-developer pattern)
        self._renderers: dict[str, Callable[[dict], None]] = {
            REVEAL: self._render_reveal,
            UPDATE_GRID: self._render_update_grid,
            WIN_INFO: self._render_win_info,
            UPDATE_TUMBLE_WIN: self._render_update_tumble_win,
            SET_WIN: self._render_set_win,
            SET_TOTAL_WIN: self._render_set_total_win,
            FINAL_WIN: self._render_final_win,
            WILD_SPAWN: self._render_spawn,
            ROCKET_SPAWN: self._render_spawn,
            BOMB_SPAWN: self._render_spawn,
            LIGHTBALL_SPAWN: self._render_spawn,
            SUPERLIGHTBALL_SPAWN: self._render_spawn,
            GRAVITY_SETTLE: self._render_gravity_settle,
            BOOSTER_PHASE: self._render_booster_phase,
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
        """Render initial board display."""
        idx = event.get("index", "?")
        game_type = event.get("gameType", "basegame")
        anticipation = event.get("anticipation", [])

        self._lines.append(f"--- REVEAL [{idx}] ({game_type}) ---")
        board_grid = event.get("board", [])
        # Cache board — WIN_INFO needs it for side-by-side view, GRAVITY_SETTLE mutates it
        self._board_state = copy.deepcopy(board_grid)
        self._lines.extend(self._format_board(board_grid))

        if any(a > 0 for a in anticipation):
            self._lines.append(f"  anticipation: {anticipation}")
        self._lines.append("")

    def _render_update_grid(self, event: dict) -> None:
        """Render grid multiplier state."""
        idx = event.get("index", "?")
        grid = event.get("gridMultipliers", [])
        # Cache grid — WIN_INFO reads this for side-by-side touched-vs-prior rendering
        self._grid_state = copy.deepcopy(grid)

        # Skip rendering if all zeros (less noise)
        all_zero = all(
            val == 0
            for reel_vals in grid
            for val in reel_vals
        )
        if all_zero:
            return

        self._lines.append(f"--- GRID MULTIPLIERS [{idx}] ---")
        self._lines.extend(self._format_grid_mults(grid))
        self._lines.append("")

    def _render_win_info(self, event: dict) -> None:
        """Render cluster win breakdown with payout math."""
        idx = event.get("index", "?")
        total_win = event.get("totalWin", 0)
        wins = event.get("wins", [])

        self._lines.append(f"--- WIN INFO [{idx}] | totalWin: {total_win} ---")
        for i, win in enumerate(wins):
            symbol = win.get("symbol", "?")
            size = win.get("clusterSize", 0)
            payout = win.get("win", 0)
            meta = win.get("meta", {})
            cluster_mult = meta.get("clusterMult", 1)
            base_payout = meta.get("winWithoutMult", 0)
            overlay = meta.get("overlay", {})
            wild_positions = meta.get("wildPositions", [])

            self._lines.append(
                f"  cluster {i}: {symbol} x{size} "
                f"= {base_payout} * {cluster_mult} = {payout}"
            )
            self._lines.append(
                f"    overlay: ({overlay.get('reel', '?')}, {overlay.get('row', '?')})"
            )
            if wild_positions:
                wild_str = ", ".join(
                    f"({wp.get('reel', '?')},{wp.get('row', '?')})"
                    for wp in wild_positions
                )
                self._lines.append(f"    wilds: {wild_str}")

        # Side-by-side board + grid view (spec section 4)
        # Collect ALL winner positions across all clusters for the combined view
        all_winners: set[tuple[int, int]] = set()
        for win in wins:
            for p in win.get("positions", []):
                all_winners.add((p["reel"], p["row"]))

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

    def _render_spawn(self, event: dict) -> None:
        """Render booster spawn event."""
        idx = event.get("index", "?")
        event_type = event.get("type", "?")
        positions = event.get("positions", [])
        clusters = event.get("clusters", [])

        self._lines.append(f"--- {event_type.upper()} [{idx}] ---")
        for pos in positions:
            orient = pos.get("orientation", "")
            orient_str = f" ({orient})" if orient else ""
            self._lines.append(
                f"  at ({pos.get('reel', '?')}, {pos.get('row', '?')}){orient_str}"
            )
        for cl in clusters:
            self._lines.append(
                f"  from: {cl.get('symbol', '?')} x{cl.get('size', 0)} "
                f"centroid ({cl.get('centroid', {}).get('reel', '?')}, "
                f"{cl.get('centroid', {}).get('row', '?')})"
            )

        # Place spawned symbol on cached board so downstream gravity sees it.
        # Spawn events fire BEFORE gravitySettle — board must be current.
        sym_name = EVENT_TYPE_TO_SPAWN_SYMBOL.get(event_type, "")
        if sym_name and self._board_state:
            for pos in positions:
                reel = pos.get("reel", 0)
                row = pos.get("row", 0)
                if reel < len(self._board_state) and row < len(self._board_state[reel]):
                    self._board_state[reel][row] = {"name": sym_name}
                    # Track freshly spawned positions — only these survive explosion
                    self._spawn_positions.add((reel, row))

        self._lines.append("")

    def _render_gravity_settle(self, event: dict) -> None:
        """Render gravity cascade: explode → gravity passes → refill → settle.

        Tracks board state through each sub-step to render intermediate grids.
        Updates _board_state to the settled result for subsequent cascade steps.
        """
        idx = event.get("index", "?")
        exploding = event.get("explodingSymbols", [])
        move_steps = event.get("moveSteps", [])
        new_symbols = event.get("newSymbols", [])

        self._lines.append(f"--- GRAVITY SETTLE [{idx}] ---")

        # Work on a mutable copy — _board_state updates only at SETTLE
        board = copy.deepcopy(self._board_state) if self._board_state else []

        # Step 1: EXPLODE — clear winning positions, render vacancies as [  ]
        if exploding:
            if board:
                for e in exploding:
                    reel, row = e.get("reel", 0), e.get("row", 0)
                    if reel < len(board) and row < len(board[reel]):
                        # Only protect positions where a booster was freshly spawned
                        # this step — existing wilds/boosters that participated in a
                        # cluster are consumed and must be cleared
                        if (reel, row) not in self._spawn_positions:
                            board[reel][row] = {"name": ""}
                self._lines.append(
                    f"  Step 1: EXPLODE ({len(exploding)} symbols removed)"
                )
                self._lines.extend(self._format_board(board))
                self._lines.append("")
                # Fresh spawns have been accounted for — clear before gravity passes
                self._spawn_positions.clear()
            else:
                expl_str = ", ".join(
                    f"({e.get('reel', '?')},{e.get('row', '?')})" for e in exploding
                )
                self._lines.append(f"  EXPLODE: {expl_str}")

        # Step 2.N: GRAVITY — move symbols, render board with moved positions highlighted
        for pass_idx, pass_moves in enumerate(move_steps):
            if not pass_moves:
                continue
            self._lines.append(
                f"  Step 2.{pass_idx + 1}: GRAVITY "
                f"({len(pass_moves)} move(s) in pass {pass_idx + 1})"
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

        # Step 3: REFILL — fill new symbols into vacancies at top of each reel
        refill_count = sum(len(reel) for reel in new_symbols)
        if refill_count > 0:
            self._lines.append(
                f"  Step 3: REFILL ({refill_count} new symbol(s) from reel strip)"
            )
            for reel_entries in new_symbols:
                for entry in reel_entries:
                    reel = entry.get("reel", 0)
                    row = entry.get("row", 0)
                    name = entry.get("name", "?")
                    self._lines.append(f"    {name} → (R{reel},row{row})")
                    # Apply refill to working board
                    if board and reel < len(board) and row < len(board[reel]):
                        board[reel][row] = {"name": name}
            self._lines.append("")

        # Step 4: SETTLE — render final board, update cached state for next cascade step
        if board:
            self._lines.append("  Step 4: SETTLE")
            self._lines.extend(self._format_board(board))
            # Transfer ownership — this becomes the board for the next cascade step
            self._board_state = board
        self._lines.append("")

    def _render_booster_phase(self, event: dict) -> None:
        """Render booster phase — fired boosters and cleared cells."""
        idx = event.get("index", "?")
        fired = event.get("firedBoosters", [])
        cleared = event.get("clearedCells", [])

        self._lines.append(f"--- BOOSTER PHASE [{idx}] ---")
        for b in fired:
            btype = b.get("type", "?")
            reel = b.get("reel", "?")
            row = b.get("row", "?")
            extra = ""
            if btype == "rocket":
                extra = f" orientation={b.get('orientation', '?')}"
            elif btype == "lightball":
                extra = f" target={b.get('targetSymbol', '?')}"
            elif btype == "superlightball":
                targets = b.get("targetSymbols", [])
                extra = f" targets={targets}"
            self._lines.append(f"  FIRE: {btype} at ({reel},{row}){extra}")

        if cleared:
            self._lines.append(f"  CLEARED: {len(cleared)} cells")

        # Post-booster gravity if present
        move_steps = event.get("moveSteps", [])
        new_symbols = event.get("newSymbols", [])
        if move_steps:
            for pass_idx, pass_moves in enumerate(move_steps):
                if not pass_moves:
                    continue
                self._lines.append(f"  POST-BOOSTER GRAVITY PASS {pass_idx + 1}:")
                for move in pass_moves:
                    from_cell = move.get("fromCell", {})
                    to_cell = move.get("toCell", {})
                    self._lines.append(
                        f"    ({from_cell.get('reel', 0)},{from_cell.get('row', 0)}) → "
                        f"({to_cell.get('reel', 0)},{to_cell.get('row', 0)})"
                    )

        refill_count = sum(len(reel) for reel in new_symbols) if new_symbols else 0
        if refill_count > 0:
            self._lines.append(f"  REFILL: {refill_count} symbols")
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
