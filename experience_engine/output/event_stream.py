"""Event stream generator — converts GeneratedInstance to RGS-compatible event list.

Produces the exact event sequence matching RoyalTumble_EventStream.md. Each event
is a plain dict (JSON-destined), indexed sequentially starting at 0. The generator
tracks running state (event counter, cumulative payout) as it walks cascade steps.

All monetary values use centipayout (integer). All thresholds from MasterConfig.
"""

from __future__ import annotations

from ..config.schema import MasterConfig
from ..pipeline.data_types import (
    BoosterArmRecord,
    BoosterFireRecord,
    CascadeStepRecord,
    GeneratedInstance,
    GravityRecord,
)
from ..primitives.board import Board, Position


def _qualified_symbol(base: str, orientation: str | None) -> str:
    """Append rocket orientation suffix when present (e.g. 'R' + 'H' → 'RH')."""
    return base + orientation if orientation else base
from ..primitives.paytable import Paytable
from ..primitives.symbols import Symbol, is_booster, is_scatter, is_wild
from ..spatial_solver.data_types import ClusterAssignment
from .event_types import (
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


class EventStreamGenerator:
    """Converts a GeneratedInstance into an ordered list of event dicts.

    Stateful per generate() call — tracks monotonic event index and running
    cumulative centipayout. Thread-safe only if each call uses its own instance
    (reset happens at the start of generate()).
    """

    __slots__ = ("_config", "_paytable", "_event_index", "_cumulative_centipayout")

    def __init__(self, config: MasterConfig, paytable: Paytable) -> None:
        self._config = config
        self._paytable = paytable
        self._event_index: int = 0
        self._cumulative_centipayout: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self, instance: GeneratedInstance) -> list[dict]:
        """Produce the complete event stream for one instance.

        Dispatches to the appropriate sequence based on instance shape:
        dead spin, static win, cascade, freegame trigger, or wincap.
        """
        self._event_index = 0
        self._cumulative_centipayout = 0

        if instance.criteria == "0" or (
            instance.payout == 0.0 and instance.cascade_steps is None
        ):
            return self._generate_dead(instance)

        if instance.cascade_steps is None or len(instance.cascade_steps) == 0:
            return self._generate_static(instance)

        # Cascade-based instances (basegame, wincap, freegame)
        return self._generate_cascade(instance)

    # ------------------------------------------------------------------
    # Sequence generators
    # ------------------------------------------------------------------

    def _generate_dead(self, instance: GeneratedInstance) -> list[dict]:
        """Dead spin: reveal(boardMultipliers) → setTotalWin(0) → finalWin(0)."""
        events: list[dict] = []
        events.append(self._make_reveal(
            instance.board, "basegame", self._initial_grid_snapshot(),
        ))
        events.append(self._make_set_total_win(0))
        events.append(self._make_final_win(0))
        return events

    def _generate_static(self, instance: GeneratedInstance) -> list[dict]:
        """Static win: reveal → winInfo → updateTumbleWin → updateBoardMultipliers → gravitySettle → setWin → setTotalWin → finalWin."""
        events: list[dict] = []
        grid_state = self._initial_grid_snapshot()

        events.append(self._make_reveal(instance.board, "basegame", grid_state))

        # Win info from spatial_step clusters — evaluated against current grid state
        clusters = instance.spatial_step.clusters
        events.append(self._make_win_info(clusters, grid_state, instance.board))

        self._cumulative_centipayout = instance.centipayout
        events.append(self._make_update_tumble_win(self._cumulative_centipayout))

        # Clusters updated grid multipliers — emit sparse delta
        new_grid = self._increment_grid_snapshot(grid_state, clusters)
        events.append(self._make_update_board_multipliers(grid_state, new_grid))

        # Gravity settle — cluster positions explode, remaining symbols fall
        if instance.gravity_record is not None:
            events.append(self._make_gravity_settle(instance.gravity_record))

        events.append(self._make_set_win(instance.centipayout, instance.win_level))
        events.append(self._make_set_total_win(instance.centipayout))
        events.append(self._make_final_win(instance.centipayout))
        return events

    def _generate_cascade(self, instance: GeneratedInstance) -> list[dict]:
        """Cascade win — per-step events followed by spin resolution.

        Per-step emission order (spec steps 3-14):
        winInfo → updateTumbleWin → updateBoardMultipliers → [wincap] →
        [boosterSpawnInfo] → [boosterArmInfo] → gravitySettle →
        [boosterFireInfo] → [gravitySettle (post-fire)]
        """
        events: list[dict] = []
        steps = instance.cascade_steps
        assert steps is not None

        is_freegame = instance.criteria == "freegame"
        is_wincap = instance.criteria == "wincap"
        wincap_centipayout = self._paytable.to_centipayout(
            self._config.wincap.max_payout,
        )
        wincap_hit = False

        # Running grid state — starts at initial zeros before any wins
        grid_state = self._initial_grid_snapshot()

        # Reveal shows the filled initial board (step 0's board_after)
        events.append(self._make_reveal(
            steps[0].board_after, "basegame", grid_state,
        ))

        for step in steps:
            # Cluster-dependent events — only for steps that produced wins
            if step.clusters:
                # winInfo (spec step 3)
                events.append(self._make_win_info(
                    step.clusters, grid_state, step.board_before,
                ))

                # Accumulate payout (spec step 4)
                step_centipayout = self._paytable.to_centipayout(step.step_payout)
                self._cumulative_centipayout += step_centipayout
                events.append(self._make_update_tumble_win(
                    self._cumulative_centipayout,
                ))

                # updateBoardMultipliers (spec step 5) — sparse delta from snapshot
                new_grid = step.grid_multipliers_snapshot
                events.append(self._make_update_board_multipliers(grid_state, new_grid))
                grid_state = new_grid

                # Wincap check (spec step 6) — halt cascade if cap crossed
                if (is_wincap and self._config.wincap.halt_cascade
                        and self._cumulative_centipayout >= wincap_centipayout):
                    self._cumulative_centipayout = wincap_centipayout
                    events.append(self._make_wincap(wincap_centipayout))
                    wincap_hit = True
                    break

                # boosterSpawnInfo (spec step 7)
                if step.booster_spawn_positions:
                    events.append(self._make_booster_spawn_info(step))

                # boosterArmInfo (spec step 8)
                if step.booster_arm_records:
                    events.append(self._make_booster_arm_info(step.booster_arm_records))

                # gravitySettle — cluster explosion (spec step 9)
                if step.gravity_record is not None:
                    events.append(self._make_gravity_settle(step.gravity_record))

            # boosterFireInfo — post-terminal fires live on the dead terminal
            # step; mid-cascade fires on cluster steps (spec step 12)
            if step.booster_fire_records:
                events.append(self._make_booster_fire_info(step.booster_fire_records))

                # Post-fire gravitySettle (spec step 13)
                if step.booster_gravity_record is not None:
                    events.append(self._make_gravity_settle(
                        step.booster_gravity_record,
                    ))

        # --- Spin resolution (spec steps 15-28) ---
        final_centipayout = (
            wincap_centipayout if wincap_hit else instance.centipayout
        )

        if is_freegame:
            self._emit_freegame_tail(events, instance, final_centipayout)
        else:
            events.append(self._make_set_win(
                final_centipayout, instance.win_level,
            ))
            events.append(self._make_set_total_win(final_centipayout))

        events.append(self._make_final_win(final_centipayout))
        return events

    def _emit_freegame_tail(
        self,
        events: list[dict],
        instance: GeneratedInstance,
        final_centipayout: int,
    ) -> None:
        """Emit basegame setWin + setTotalWin, then freeSpinTrigger/End.

        Spec steps 15 → 17 → 19 → 27 → 28.
        """
        # Basegame portion — the cascade may have produced wins before scatter trigger
        basegame_centipayout = self._cumulative_centipayout
        if basegame_centipayout > 0:
            basegame_payout = sum(s.step_payout for s in instance.cascade_steps)
            basegame_win_level = self._paytable.get_win_level(basegame_payout)
            events.append(self._make_set_win(basegame_centipayout, basegame_win_level))
        events.append(self._make_set_total_win(basegame_centipayout))

        # freeSpinTrigger
        scatter_count = len(instance.spatial_step.scatter_positions)
        awards_lookup = dict(self._config.freespin.awards)
        total_fs = awards_lookup.get(
            scatter_count,
            # Fallback to max award if exact count not in config
            max(fs for _, fs in self._config.freespin.awards)
            if self._config.freespin.awards else 0,
        )
        events.append(self._make_free_spin_trigger(
            instance.spatial_step.scatter_positions, total_fs,
        ))

        # Freespin loop — stub. Per-spin events reuse the cascade step loop
        # when the freespin loop is implemented.
        events.append(self._make_free_spin_end(
            final_centipayout, instance.win_level,
        ))

    # ------------------------------------------------------------------
    # Individual event builders
    # ------------------------------------------------------------------

    def _make_reveal(
        self,
        board: Board,
        game_type: str,
        grid_snapshot: tuple[int, ...],
    ) -> dict:
        """Build reveal event — initial board display with board multipliers."""
        return {
            "index": self._next_index(),
            "type": REVEAL,
            "board": self._board_to_grid(board),
            "gameType": game_type,
            "boardMultipliers": self._snapshot_to_matrix(grid_snapshot),
        }

    def _make_win_info(
        self,
        clusters: tuple[ClusterAssignment, ...],
        grid_snapshot: tuple[int, ...],
        board: Board,
    ) -> dict:
        """Build winInfo event — per-cluster breakdown matching spec shape.

        Each win entry: basePayout, clusterPayout, clusterMultiplier, clusterSize,
        overlay, and cluster.cells[] with per-cell symbol/position/multiplier.
        """
        num_rows = self._config.board.num_rows
        wins: list[dict] = []
        total_centipayout = 0

        for cluster in clusters:
            base_payout = self._paytable.get_payout(cluster.size, cluster.symbol)

            # Sum grid multipliers at all cluster positions (standard + wild)
            all_positions = cluster.positions | cluster.wild_positions
            mult_sum = 0
            for pos in all_positions:
                flat_idx = pos.reel * num_rows + pos.row
                if flat_idx < len(grid_snapshot):
                    mult_sum += grid_snapshot[flat_idx]
            # Minimum contribution of 1 — ensures base payout always applies
            mult_sum = max(mult_sum, self._config.grid_multiplier.minimum_contribution)

            cluster_payout = base_payout * mult_sum
            cluster_centipayout = self._paytable.to_centipayout(cluster_payout)
            base_centipayout = self._paytable.to_centipayout(base_payout)
            total_centipayout += cluster_centipayout

            # Centroid — visual center for win amount overlay
            centroid = self._compute_centroid(cluster.positions)

            # Per-cell breakdown — all cells use the cluster's owning symbol
            # (wilds substitute for the cluster symbol)
            cells = [
                {
                    "symbol": cluster.symbol.name,
                    "reel": pos.reel,
                    "row": pos.row,
                    "multiplier": (
                        grid_snapshot[pos.reel * num_rows + pos.row]
                        if (pos.reel * num_rows + pos.row) < len(grid_snapshot)
                        else 0
                    ),
                }
                for pos in sorted(all_positions, key=lambda p: (p.reel, p.row))
            ]

            wins.append({
                "basePayout": base_centipayout,
                "clusterPayout": cluster_centipayout,
                "clusterMultiplier": mult_sum,
                "clusterSize": cluster.size,
                "overlay": centroid,
                "cluster": {"cells": cells},
            })

        return {
            "index": self._next_index(),
            "type": WIN_INFO,
            "totalWin": total_centipayout,
            "wins": wins,
        }

    def _make_update_board_multipliers(
        self,
        prev_snapshot: tuple[int, ...],
        curr_snapshot: tuple[int, ...],
    ) -> dict:
        """Build updateBoardMultipliers event — sparse delta of changed positions."""
        num_rows = self._config.board.num_rows
        changed: list[dict] = []
        for flat_idx, (old, new) in enumerate(zip(prev_snapshot, curr_snapshot)):
            if old != new:
                reel, row = divmod(flat_idx, num_rows)
                changed.append({
                    "multiplier": new,
                    "position": {"reel": reel, "row": row},
                })
        return {
            "index": self._next_index(),
            "type": UPDATE_BOARD_MULTIPLIERS,
            "boardMultipliers": changed,
        }

    def _make_booster_spawn_info(self, step: CascadeStepRecord) -> dict:
        """Build boosterSpawnInfo event — flat list of spawned boosters."""
        boosters = [
            {
                "symbol": _qualified_symbol(btype, orient),
                "position": {"reel": reel, "row": row},
            }
            for btype, reel, row, orient in step.booster_spawn_positions
        ]
        return {
            "index": self._next_index(),
            "type": BOOSTER_SPAWN_INFO,
            "boosters": boosters,
        }

    def _make_booster_arm_info(
        self, arm_records: tuple[BoosterArmRecord, ...],
    ) -> dict:
        """Build boosterArmInfo event — boosters that transitioned to ARMED."""
        boosters = [
            {
                "symbol": _qualified_symbol(rec.booster_type, rec.orientation),
                "position": {"reel": rec.position_reel, "row": rec.position_row},
            }
            for rec in arm_records
        ]
        return {
            "index": self._next_index(),
            "type": BOOSTER_ARM_INFO,
            "boosters": boosters,
        }

    def _make_booster_fire_info(
        self, fire_records: tuple[BoosterFireRecord, ...],
    ) -> dict:
        """Build boosterFireInfo event — per-booster clearedCells with symbol."""
        boosters: list[dict] = []
        for record in fire_records:
            boosters.append({
                "symbol": _qualified_symbol(record.booster_type, record.orientation),
                "clearedCells": [
                    {
                        "symbol": sym_name,
                        "position": {"reel": r, "row": c},
                    }
                    for r, c, sym_name in record.affected_positions_list
                ],
            })
        return {
            "index": self._next_index(),
            "type": BOOSTER_FIRE_INFO,
            "boosters": boosters,
        }

    def _make_gravity_settle(self, gravity_record: GravityRecord) -> dict:
        """Build gravitySettle event from captured gravity data.

        Contains per-pass movement data and refill symbols. No explodingSymbols.
        """
        # Format per-pass move steps: each move is {symbol, fromCell, toCell}
        move_steps: list[list[dict]] = []
        for pass_moves in gravity_record.move_steps:
            pass_list: list[dict] = []
            for sym, (src_r, src_c), (dst_r, dst_c) in pass_moves:
                pass_list.append({
                    "symbol": sym,
                    "fromCell": {"reel": src_r, "row": src_c},
                    "toCell": {"reel": dst_r, "row": dst_c},
                })
            move_steps.append(pass_list)

        # Format refill symbols — spec shape: {symbol, position: {reel, row}}
        num_reels = self._config.board.num_reels
        new_symbols: list[list[dict]] = [[] for _ in range(num_reels)]
        for reel, row, sym_name in gravity_record.refill_entries:
            new_symbols[reel].append({
                "symbol": sym_name,
                "position": {"reel": reel, "row": row},
            })

        return {
            "index": self._next_index(),
            "type": GRAVITY_SETTLE,
            "moveSteps": move_steps,
            "newSymbols": new_symbols,
        }

    def _make_update_tumble_win(self, cumulative_centipayout: int) -> dict:
        """Build updateTumbleWin event — running cascade accumulation."""
        return {
            "index": self._next_index(),
            "type": UPDATE_TUMBLE_WIN,
            "amount": cumulative_centipayout,
        }

    def _make_set_win(self, centipayout: int, win_level: int) -> dict:
        """Build setWin event — final spin payout with celebration level."""
        return {
            "index": self._next_index(),
            "type": SET_WIN,
            "amount": centipayout,
            "winLevel": win_level,
        }

    def _make_set_total_win(self, centipayout: int) -> dict:
        """Build setTotalWin event — cumulative round payout."""
        return {
            "index": self._next_index(),
            "type": SET_TOTAL_WIN,
            "amount": centipayout,
        }

    def _make_free_spin_trigger(
        self,
        scatter_positions: frozenset[Position],
        total_fs: int,
    ) -> dict:
        """Build freeSpinTrigger event — scatter feature trigger."""
        return {
            "index": self._next_index(),
            "type": FREE_SPIN_TRIGGER,
            "totalFs": total_fs,
            "positions": self._positions_to_json(scatter_positions),
        }

    def _make_update_free_spin(self, current: int, total: int) -> dict:
        """Build updateFreeSpin event — freespin counter."""
        return {
            "index": self._next_index(),
            "type": UPDATE_FREE_SPIN,
            "amount": current,
            "total": total,
        }

    def _make_free_spin_end(self, centipayout: int, win_level: int) -> dict:
        """Build freeSpinEnd event — feature completion summary."""
        return {
            "index": self._next_index(),
            "type": FREE_SPIN_END,
            "amount": centipayout,
            "winLevel": win_level,
        }

    def _make_wincap(self, centipayout: int) -> dict:
        """Build wincap event — payout reached config.wincap.max_payout."""
        return {
            "index": self._next_index(),
            "type": WINCAP,
            "amount": centipayout,
        }

    def _make_final_win(self, centipayout: int) -> dict:
        """Build finalWin event — always the last event in a book."""
        return {
            "index": self._next_index(),
            "type": FINAL_WIN,
            "amount": centipayout,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_index(self) -> int:
        """Return current event index and increment for the next call."""
        idx = self._event_index
        self._event_index += 1
        return idx

    def _board_to_grid(self, board: Board) -> list[list[dict]]:
        """Serialize Board to nested list of symbol dicts for JSON.

        Matches SDK json_ready_sym format: {"name": "H2"} for standard,
        with special attributes added for scatter/wild.
        """
        num_reels = self._config.board.num_reels
        num_rows = self._config.board.num_rows
        grid: list[list[dict]] = []

        for reel in range(num_reels):
            reel_list: list[dict] = []
            for row in range(num_rows):
                pos = Position(reel, row)
                sym = board.get(pos)
                if sym is None:
                    reel_list.append({"name": ""})
                else:
                    entry: dict = {"name": sym.name}
                    if is_scatter(sym):
                        entry["scatter"] = True
                    if is_wild(sym):
                        entry["wild"] = True
                    reel_list.append(entry)
            grid.append(reel_list)

        return grid

    def _snapshot_to_matrix(self, grid_snapshot: tuple[int, ...]) -> list[list[int]]:
        """Convert flat grid snapshot to 2D matrix [reel][row]."""
        num_rows = self._config.board.num_rows
        num_reels = self._config.board.num_reels
        matrix: list[list[int]] = []
        for reel in range(num_reels):
            start = reel * num_rows
            matrix.append(list(grid_snapshot[start:start + num_rows]))
        return matrix

    def _initial_grid_snapshot(self) -> tuple[int, ...]:
        """All-zeros grid snapshot for the initial state before any wins."""
        total = self._config.board.num_reels * self._config.board.num_rows
        return (self._config.grid_multiplier.initial_value,) * total

    def _increment_grid_snapshot(
        self,
        grid_snapshot: tuple[int, ...],
        clusters: tuple[ClusterAssignment, ...],
    ) -> tuple[int, ...]:
        """Advance grid snapshot by incrementing at all cluster positions.

        Each time clusters update the grid, this produces the new running state.
        Mirrors GridMultiplierGrid.increment() on the flat-tuple representation.
        """
        num_rows = self._config.board.num_rows
        cfg = self._config.grid_multiplier
        updated = list(grid_snapshot)
        for cluster in clusters:
            for pos in cluster.positions | cluster.wild_positions:
                idx = pos.reel * num_rows + pos.row
                if updated[idx] == cfg.initial_value:
                    updated[idx] = cfg.first_hit_value
                else:
                    updated[idx] = min(updated[idx] + cfg.increment, cfg.cap)
        return tuple(updated)

    def _positions_to_json(
        self, positions: frozenset[Position],
    ) -> list[dict[str, int]]:
        """Convert Position frozenset to sorted list of {reel, row} dicts."""
        return sorted(
            [{"reel": p.reel, "row": p.row} for p in positions],
            key=lambda d: (d["reel"], d["row"]),
        )

    def _compute_centroid(
        self, positions: frozenset[Position],
    ) -> dict[str, int]:
        """Compute arithmetic mean of positions, snapped to nearest member.

        Used for win amount overlay positioning — frontend renders the payout
        number at this position.
        """
        if not positions:
            return {"reel": 0, "row": 0}

        pos_list = list(positions)
        avg_reel = sum(p.reel for p in pos_list) / len(pos_list)
        avg_row = sum(p.row for p in pos_list) / len(pos_list)

        # Snap to nearest member position (Euclidean distance)
        closest = min(
            pos_list,
            key=lambda p: (p.reel - avg_reel) ** 2 + (p.row - avg_row) ** 2,
        )
        return {"reel": closest.reel, "row": closest.row}
