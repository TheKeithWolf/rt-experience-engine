"""Stateless event renderers — one function per event type.

Each function receives the event dict and any needed state as explicit
arguments, returns list[str] of output lines (and optionally updated state
as a tuple). No self, no mutation of external state.

Output format matches RoyalTumble_ExperienceEngine_Tracer.md spec.
"""

from __future__ import annotations

import copy

from ..formatting.board_formatter import CellResolver, format_board_grid
from ..formatting.cells import CellStyle
from ..formatting.constants import CELL_NAME_WIDTH, MAJOR_SEP
from ..formatting.grid_formatter import format_grid_mults
from ..formatting.layout import format_side_by_side


# ---------------------------------------------------------------------------
# Section header helper — produces "======== TITLE ========" per spec
# ---------------------------------------------------------------------------

def _section_header(title: str) -> str:
    """Format a major section header matching spec visual vocabulary."""
    return f"{'=' * 40}\n{title}\n{'=' * 40}"


# ---------------------------------------------------------------------------
# CellResolver factories — build resolve_cell closures for board rendering
# ---------------------------------------------------------------------------

def _make_board_resolver(
    board_state: list[list[dict]],
    *,
    winners: set[tuple[int, int]] | None = None,
    booster_styles: dict[tuple[int, int], CellStyle] | None = None,
    vacancies: set[tuple[int, int]] | None = None,
    scatter_positions: set[tuple[int, int]] | None = None,
) -> CellResolver:
    """Build a CellResolver for dict-based board_state[reel][row].

    Priority: vacancy > scatter > booster_styles > winner > regular/empty.
    """
    winners = winners or set()
    booster_styles = booster_styles or {}
    vacancies = vacancies or set()
    scatter_positions = scatter_positions or set()

    def resolve(reel: int, row: int) -> tuple[str, CellStyle]:
        # Vacancy — exploded/cleared cell
        if (reel, row) in vacancies:
            return "  ", CellStyle.VACANCY

        # Scatter highlight — freespin trigger positions
        if (reel, row) in scatter_positions:
            sym_dict = board_state[reel][row] if reel < len(board_state) and row < len(board_state[reel]) else {}
            name = sym_dict.get("name", "")
            return name, CellStyle.ARMED

        # Booster lifecycle state — spawned or armed visual
        if (reel, row) in booster_styles:
            sym_dict = board_state[reel][row] if reel < len(board_state) and row < len(board_state[reel]) else {}
            name = sym_dict.get("name", "")
            return name, booster_styles[(reel, row)]

        sym_dict = board_state[reel][row] if reel < len(board_state) and row < len(board_state[reel]) else {}
        name = sym_dict.get("name", "")

        if not name:
            return "  ", CellStyle.EMPTY

        if (reel, row) in winners:
            return name, CellStyle.WINNER

        return name, CellStyle.REGULAR

    return resolve


def _num_reels_rows(board_state: list[list[dict]]) -> tuple[int, int]:
    """Extract dimensions from dict-based board state."""
    num_reels = len(board_state)
    num_rows = len(board_state[0]) if board_state else 0
    return num_reels, num_rows


# ---------------------------------------------------------------------------
# Direction arrows for gravity moves
# ---------------------------------------------------------------------------

def remap_booster_styles(
    styles: dict[tuple[int, int], CellStyle],
    move_steps: list,
) -> dict[tuple[int, int], CellStyle]:
    """Return a new dict with booster positions updated per gravity moves.

    Two-phase per step to avoid key collisions when moves cross paths.
    Used by both render_gravity_settle (pure) and EventTracer (self-mutation wrapper).
    """
    if not styles:
        return {}

    result = dict(styles)
    for step_moves in move_steps:
        remap: dict[tuple[int, int], tuple[int, int]] = {}
        for move in step_moves:
            from_cell = move.get("fromCell", {})
            to_cell = move.get("toCell", {})
            fr = (from_cell.get("reel", 0), from_cell.get("row", 0))
            to = (to_cell.get("reel", 0), to_cell.get("row", 0))
            remap[fr] = to

        # Collect then apply — avoids overwrites when moves cross paths
        updates: list[tuple[tuple[int, int], CellStyle]] = []
        for pos in list(result):
            if pos in remap:
                style = result.pop(pos)
                updates.append((remap[pos], style))
        for new_pos, style in updates:
            result[new_pos] = style

    return result


def _gravity_direction(from_reel: int, to_reel: int) -> tuple[str, str]:
    """Determine direction arrow and label based on reel movement."""
    if from_reel == to_reel:
        return "↓", "down"
    elif from_reel > to_reel:
        return "↙", "down-left"
    else:
        return "↘", "down-right"


# ---------------------------------------------------------------------------
# Event renderers
# ---------------------------------------------------------------------------

def render_reveal(
    event: dict,
    num_reels: int,
    num_rows: int,
) -> tuple[list[str], list[list[dict]], list[list[int]]]:
    """Render REVEAL event. Returns (lines, new_board_state, new_grid_state).

    Spec format: ======== REVEAL ======== with side-by-side board + multipliers.
    """
    lines: list[str] = []
    lines.append(_section_header("REVEAL"))

    board_grid = event.get("board", [])
    board_state = copy.deepcopy(board_grid)

    board_mults = event.get("boardMultipliers", [])
    grid_state = copy.deepcopy(board_mults) if board_mults else []

    # Render board
    resolver = _make_board_resolver(board_state)
    br, bc = _num_reels_rows(board_state)
    board_lines = format_board_grid(br, bc, resolver)

    # Side-by-side with multipliers if grid exists
    if grid_state:
        grid_lines = format_grid_mults(grid_state, br, bc)
        lines.append("")
        lines.extend(format_side_by_side(
            board_lines, grid_lines, "Board", "Multipliers",
        ))
    else:
        lines.append("")
        lines.extend(board_lines)

    lines.append("")
    return lines, board_state, grid_state


def render_win_info(
    event: dict,
    board_state: list[list[dict]],
    grid_state: list[list[int]],
    booster_styles: dict[tuple[int, int], CellStyle],
    num_reels: int,
    num_rows: int,
) -> tuple[list[str], set[tuple[int, int]]]:
    """Render WIN INFO with multi-line cluster breakdown per spec.

    Returns (lines, all_winner_positions). The handler clears winner positions
    from board state — the game engine removes cluster symbols before gravity.
    """
    lines: list[str] = []
    total_win = event.get("totalWin", 0)
    wins = event.get("wins", [])

    lines.append(_section_header("WIN INFO"))
    lines.append(f"Total Win: {total_win}")

    # Collect all winner positions for board highlighting
    all_winners: set[tuple[int, int]] = set()

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

        lines.append("")
        lines.append(f"Cluster {i + 1}:")
        lines.append(f"  Base Payout: {symbol} x {size} = {base_payout}")
        lines.append(f"  Cluster Payout: {base_payout} x {cluster_mult} = {cluster_payout}")
        lines.append(f"  Cluster Size: {size}")
        lines.append(f"  Cluster Multiplier: {cluster_mult}")
        lines.append(f"  Overlay: ({overlay.get('reel', '?')}, {overlay.get('row', '?')})")

        # Cell list
        cell_strs = []
        for cell in cells:
            c_sym = cell.get("symbol", "?")
            c_reel = cell.get("reel", 0)
            c_row = cell.get("row", 0)
            c_mult = cell.get("multiplier", 0)
            cell_strs.append(f"[{c_sym} ({c_reel},{c_row}) {c_mult}]")
            all_winners.add((c_reel, c_row))
        lines.append(f"  Cluster: {', '.join(cell_strs)}")

    # Side-by-side board + grid view with winners/touched highlighted
    if board_state and all_winners:
        resolver = _make_board_resolver(
            board_state, winners=all_winners, booster_styles=booster_styles,
        )
        br, bc = _num_reels_rows(board_state)
        board_lines = format_board_grid(br, bc, resolver)

        if grid_state:
            grid_lines = format_grid_mults(grid_state, br, bc, touched=all_winners)
            lines.append("")
            lines.extend(format_side_by_side(
                board_lines, grid_lines, "Board", "Multipliers",
            ))
        else:
            lines.append("")
            lines.extend(board_lines)

    lines.append("")
    return lines, all_winners


def render_update_tumble_win(event: dict) -> list[str]:
    """Render running cascade payout per spec."""
    amount = event.get("amount", 0)
    lines: list[str] = []
    lines.append(_section_header("UPDATE TUMBLE WIN"))
    lines.append(f"Amount: {amount}")
    lines.append("")
    return lines


def render_update_board_multipliers(
    event: dict,
    grid_state: list[list[int]],
    num_reels: int,
    num_rows: int,
) -> tuple[list[str], list[list[int]]]:
    """Render board multiplier delta with (reel,row) old → new lines per spec.

    Returns (lines, updated_grid_state). Applies changes to a copy of grid_state.
    """
    changes = event.get("boardMultipliers", [])
    if not changes:
        return [], grid_state

    lines: list[str] = []
    lines.append(_section_header("UPDATE BOARD MULTIPLIERS"))

    # Copy grid and apply changes, tracking deltas
    new_grid = copy.deepcopy(grid_state) if grid_state else []
    updated_positions: set[tuple[int, int]] = set()

    for entry in changes:
        mult = entry.get("multiplier", 0)
        pos = entry.get("position", {})
        reel = pos.get("reel", 0)
        row = pos.get("row", 0)

        # Record old value for delta display
        old_val = 0
        if new_grid and reel < len(new_grid) and row < len(new_grid[reel]):
            old_val = new_grid[reel][row]
            new_grid[reel][row] = mult

        lines.append(f"({reel},{row}) {old_val} → {mult}")
        updated_positions.add((reel, row))

    # Render grid with updated positions highlighted as *N*
    if new_grid:
        lines.append("")
        grid_lines = format_grid_mults(
            new_grid, num_reels, num_rows, updated=updated_positions,
        )
        lines.extend(grid_lines)

    lines.append("")
    return lines, new_grid


def render_booster_spawn_info(
    event: dict,
    board_state: list[list[dict]],
    num_reels: int,
    num_rows: int,
) -> tuple[list[str], list[list[dict]], dict[tuple[int, int], CellStyle]]:
    """Render booster spawn with full board grid showing *RH* at spawn positions.

    Returns (lines, updated_board_state, new_spawn_styles).
    """
    boosters = event.get("boosters", [])
    lines: list[str] = []
    lines.append(_section_header("BOOSTER SPAWN INFO"))

    # Track new booster styles for this spawn event
    spawn_styles: dict[tuple[int, int], CellStyle] = {}
    board = copy.deepcopy(board_state) if board_state else []

    for b in boosters:
        symbol = b.get("symbol", "?")
        pos = b.get("position", {})
        reel = pos.get("reel", 0)
        row = pos.get("row", 0)
        lines.append(f"{symbol} at ({reel}, {row})")

        # Update board state with spawned booster
        if board and reel < len(board) and row < len(board[reel]):
            board[reel][row] = {"name": symbol}
        spawn_styles[(reel, row)] = CellStyle.SPAWNED

    # Render board with spawn positions highlighted
    if board:
        resolver = _make_board_resolver(board, booster_styles=spawn_styles)
        lines.append("")
        lines.extend(format_board_grid(num_reels, num_rows, resolver))

    lines.append("")
    return lines, board, spawn_styles


def render_booster_arm_info(
    event: dict,
    board_state: list[list[dict]],
    booster_styles: dict[tuple[int, int], CellStyle],
    num_reels: int,
    num_rows: int,
) -> tuple[list[str], dict[tuple[int, int], CellStyle]]:
    """Render booster arm with full board grid showing [RH] at armed positions.

    Returns (lines, updated_booster_styles).
    """
    boosters = event.get("boosters", [])
    lines: list[str] = []
    lines.append(_section_header("BOOSTER ARM INFO"))

    # Update styles: spawned → armed
    new_styles = dict(booster_styles)
    for b in boosters:
        symbol = b.get("symbol", "?")
        pos = b.get("position", {})
        reel = pos.get("reel", 0)
        row = pos.get("row", 0)
        lines.append(f"{symbol} ARMED at ({reel}, {row})")
        new_styles[(reel, row)] = CellStyle.ARMED

    # Render board with booster styles
    if board_state:
        resolver = _make_board_resolver(board_state, booster_styles=new_styles)
        lines.extend(format_board_grid(num_reels, num_rows, resolver))

    lines.append("")
    return lines, new_styles


def render_booster_fire_info(
    event: dict,
    board_state: list[list[dict]],
    booster_styles: dict[tuple[int, int], CellStyle],
    num_reels: int,
    num_rows: int,
) -> tuple[list[str], dict[tuple[int, int], CellStyle]]:
    """Render booster fire with board grid showing [  ] vacancies at cleared cells.

    Returns (lines, updated_booster_styles).
    """
    boosters = event.get("boosters", [])
    lines: list[str] = []
    lines.append(_section_header("BOOSTER FIRE INFO"))

    # Collect all cleared positions as vacancies
    vacancies: set[tuple[int, int]] = set()
    new_styles = dict(booster_styles)

    for i, b in enumerate(boosters):
        symbol = b.get("symbol", "?")
        cleared = b.get("clearedCells", [])
        targets = ", ".join(
            f"({c.get('position', {}).get('reel', '?')},{c.get('position', {}).get('row', '?')})"
            if "position" in c else f"({c.get('reel', '?')},{c.get('row', '?')})"
            for c in cleared
        )
        lines.append(f"{i}: {symbol} ({targets})")

        for c in cleared:
            if "position" in c:
                pos = c["position"]
                reel, row = pos.get("reel", 0), pos.get("row", 0)
            else:
                reel, row = c.get("reel", 0), c.get("row", 0)
            vacancies.add((reel, row))

        # Remove fired booster from style tracking
        pos_data = b.get("position", {})
        fire_pos = (pos_data.get("reel", -1), pos_data.get("row", -1))
        new_styles.pop(fire_pos, None)

    lines.append("---")

    # Render board with vacancies
    if board_state:
        resolver = _make_board_resolver(
            board_state, booster_styles=new_styles, vacancies=vacancies,
        )
        lines.append("")
        lines.extend(format_board_grid(num_reels, num_rows, resolver))

    lines.append("")
    return lines, new_styles


def render_gravity_settle(
    event: dict,
    board_state: list[list[dict]],
    booster_styles: dict[tuple[int, int], CellStyle],
    num_reels: int,
    num_rows: int,
) -> tuple[list[str], list[list[dict]]]:
    """Render gravity cascade: move steps → new symbols → final board per spec.

    Returns (lines, updated_board_state). Tracks board through each sub-step.
    """
    move_steps = event.get("moveSteps", [])
    new_symbols = event.get("newSymbols", [])

    lines: list[str] = []
    lines.append(_section_header("GRAVITY SETTLE"))

    # Work on mutable copy — board_state updates only at final settle
    board = copy.deepcopy(board_state) if board_state else []

    # Move steps — per-step symbol moves with direction arrows
    for step_idx, step_moves in enumerate(move_steps):
        if not step_moves:
            continue

        lines.append(f"Move Step {step_idx}:")

        moved_destinations: set[tuple[int, int]] = set()
        for move in step_moves:
            from_cell = move.get("fromCell", {})
            to_cell = move.get("toCell", {})
            fr, fc = from_cell.get("reel", 0), from_cell.get("row", 0)
            tr, tc = to_cell.get("reel", 0), to_cell.get("row", 0)

            # Resolve symbol name from working board
            sym_name = ""
            if board and fr < len(board) and fc < len(board[fr]):
                sym_name = board[fr][fc].get("name", "")

            arrow, direction = _gravity_direction(fr, tr)

            # Spec format: sym  (Rn,rowN) → (Rn,rowN)  ↓ direction-name
            sym_prefix = f"{sym_name:<{CELL_NAME_WIDTH}}" if sym_name else " " * CELL_NAME_WIDTH
            lines.append(
                f"{sym_prefix}  (R{fr},row{fc}) → (R{tr},row{tc})  {arrow} {direction}"
            )
            moved_destinations.add((tr, tc))

        # Apply moves — two-phase to avoid overwrites
        if board:
            collected: list[tuple[int, int, int, int, dict]] = []
            for move in step_moves:
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

            # Phase 1: clear source positions
            for fr, fc, _tr, _tc, _sym in collected:
                if fr < len(board) and fc < len(board[fr]):
                    board[fr][fc] = {"name": ""}
            # Phase 2: place at destinations
            for _fr, _fc, tr, tc, sym in collected:
                if tr < len(board) and tc < len(board[tr]):
                    board[tr][tc] = sym

        lines.append("")

    # Remap booster styles to match post-gravity positions — prevents ghost
    # SPAWNED markers (*  *) at vacated source cells
    settled_styles = remap_booster_styles(booster_styles, move_steps)

    # Intermediate board after gravity moves, before refill
    if board and move_steps:
        resolver = _make_board_resolver(board, booster_styles=settled_styles)
        br, bc = _num_reels_rows(board)
        lines.extend(format_board_grid(br, bc, resolver))
        lines.append("")

    # New symbols — refill vacancies from reel strip
    refill_count = sum(len(reel) for reel in new_symbols)
    if refill_count > 0:
        lines.append("New Symbols:")
        for reel_entries in new_symbols:
            for entry in reel_entries:
                pos = entry.get("position", {})
                reel = pos.get("reel", 0)
                row = pos.get("row", 0)
                name = entry.get("symbol", "?")
                lines.append(f"{name} → ({reel}, {row})")
                # Apply refill to working board
                if board and reel < len(board) and row < len(board[reel]):
                    board[reel][row] = {"name": name}
        lines.append("")

    # Final settled board
    if board:
        resolver = _make_board_resolver(board, booster_styles=settled_styles)
        br, bc = _num_reels_rows(board)
        lines.extend(format_board_grid(br, bc, resolver))

    lines.append("")
    return lines, board


def render_set_win(event: dict) -> list[str]:
    """Render spin win result per spec: ======== SET WIN -- N - Level M ========"""
    amount = event.get("amount", 0)
    win_level = event.get("winLevel", 0)
    return [
        _section_header(f"SET WIN -- {amount} - Level {win_level}"),
        "",
    ]


def render_set_total_win(event: dict) -> list[str]:
    """Render cumulative total per spec: ======== SET TOTAL WIN -- N ========"""
    amount = event.get("amount", 0)
    return [
        _section_header(f"SET TOTAL WIN -- {amount}"),
        "",
    ]


def render_final_win(event: dict) -> list[str]:
    """Render final result per spec."""
    amount = event.get("amount", 0)
    lines: list[str] = []
    lines.append(_section_header("FINAL WIN"))
    lines.append(f"Amount: {amount}")
    lines.append("")
    return lines


def render_wincap(event: dict) -> list[str]:
    """Render wincap per spec: ======== WIN CAP -- N ========"""
    amount = event.get("amount", 0)
    return [
        _section_header(f"WIN CAP -- {amount}"),
        "",
    ]


def render_free_spin_trigger(
    event: dict,
    board_state: list[list[dict]],
    num_reels: int,
    num_rows: int,
) -> list[str]:
    """Render freespin trigger with scatter positions highlighted as [S] per spec."""
    total_fs = event.get("totalFs", 0)
    positions = event.get("positions", [])

    lines: list[str] = []
    lines.append(_section_header("FREE SPIN TRIGGER"))
    lines.append(f"Total FS: {total_fs}")

    # Format positions list
    pos_strs = [f"({p.get('reel', '?')},{p.get('row', '?')})" for p in positions]
    lines.append(f"Positions: [{', '.join(pos_strs)}]")

    # Board with scatter positions highlighted as [S]
    if board_state:
        scatter_set = {
            (p.get("reel", 0), p.get("row", 0)) for p in positions
        }
        resolver = _make_board_resolver(board_state, scatter_positions=scatter_set)
        lines.append("")
        lines.extend(format_board_grid(num_reels, num_rows, resolver))

    lines.append("")
    return lines


def render_update_free_spin(event: dict) -> list[str]:
    """Render freespin counter per spec: ======== UPDATE FREE SPIN -- N / M ========"""
    current = event.get("amount", 0)
    total = event.get("total", 0)
    return [
        _section_header(f"UPDATE FREE SPIN -- {current} / {total}"),
        "",
    ]


def render_free_spin_end(event: dict) -> list[str]:
    """Render freespin end per spec: ======== FREE SPIN END -- N - Level M ========"""
    amount = event.get("amount", 0)
    win_level = event.get("winLevel", 0)
    return [
        _section_header(f"FREE SPIN END -- {amount} - Level {win_level}"),
        "",
    ]
