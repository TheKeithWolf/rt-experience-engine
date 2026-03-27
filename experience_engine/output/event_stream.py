"""Event stream generator — converts GeneratedInstance to RGS-compatible event list.

Produces the exact event sequence matching RoyalTumble_EventStream.md. Each event
is a plain dict (JSON-destined), indexed sequentially starting at 0. The generator
tracks running state (event counter, cumulative payout) as it walks cascade steps.

All monetary values use centipayout (integer). All thresholds from MasterConfig.
"""

from __future__ import annotations

from ..config.schema import MasterConfig
from ..pipeline.data_types import (
    BoosterFireRecord,
    CascadeStepRecord,
    GeneratedInstance,
    GravityRecord,
)
from ..primitives.board import Board, Position
from ..primitives.paytable import Paytable
from ..primitives.symbols import Symbol, is_booster, is_scatter, is_wild
from ..spatial_solver.data_types import ClusterAssignment
from .event_types import (
    BOOSTER_PHASE,
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
    compute_anticipation,
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
        """Dead spin: reveal → updateGrid → setTotalWin(0) → finalWin."""
        events: list[dict] = []
        events.append(self._make_reveal(
            instance.board, "basegame", instance.spatial_step.scatter_positions,
        ))
        events.append(self._make_update_grid(
            self._initial_grid_snapshot(),
        ))
        events.append(self._make_set_total_win(0))
        events.append(self._make_final_win(0))
        return events

    def _generate_static(self, instance: GeneratedInstance) -> list[dict]:
        """Static win: reveal → updateGrid → winInfo → updateTumbleWin → updateGrid → gravitySettle → setWin → setTotalWin → finalWin."""
        events: list[dict] = []
        events.append(self._make_reveal(
            instance.board, "basegame", instance.spatial_step.scatter_positions,
        ))

        # Running grid state — starts at initial zeros
        grid_state = self._initial_grid_snapshot()
        events.append(self._make_update_grid(grid_state))

        # Win info from spatial_step clusters — evaluated against current grid state
        clusters = instance.spatial_step.clusters
        events.append(self._make_win_info(
            clusters, grid_state, instance.board,
        ))

        self._cumulative_centipayout = instance.centipayout
        events.append(self._make_update_tumble_win(self._cumulative_centipayout))

        # Clusters updated grid multipliers — capture new running state
        grid_state = self._increment_grid_snapshot(grid_state, clusters)
        events.append(self._make_update_grid(grid_state))

        # Gravity settle — cluster positions explode, remaining symbols fall
        if instance.gravity_record is not None:
            events.append(self._make_gravity_settle(instance.gravity_record))

        events.append(self._make_set_win(instance.centipayout, instance.win_level))
        events.append(self._make_set_total_win(instance.centipayout))
        events.append(self._make_final_win(instance.centipayout))
        return events

    def _generate_cascade(self, instance: GeneratedInstance) -> list[dict]:
        """Cascade win: reveal → updateGrid → per-step events → setWin → setTotalWin → finalWin.

        Handles basegame cascades, wincap (halts on cap), and freegame trigger
        sequences. Freegame instances emit freeSpinTrigger after basegame reveal.
        """
        events: list[dict] = []
        steps = instance.cascade_steps
        assert steps is not None

        game_type = "basegame"
        is_freegame = instance.criteria == "freegame"
        is_wincap = instance.criteria == "wincap"
        wincap_centipayout = self._paytable.to_centipayout(
            self._config.wincap.max_payout,
        )
        wincap_hit = False

        # Reveal shows the filled initial board — board_after has symbols placed
        # by the executor, board_before at step 0 is empty (pre-fill snapshot).
        events.append(self._make_reveal(
            steps[0].board_after, game_type,
            instance.spatial_step.scatter_positions,
        ))
        # Running grid state — starts at initial zeros before any wins
        grid_state = self._initial_grid_snapshot()
        events.append(self._make_update_grid(grid_state))

        for step in steps:
            # Win info for this step's clusters — evaluated against current grid state
            if step.clusters:
                events.append(self._make_win_info(
                    step.clusters, grid_state,
                    step.board_before,
                ))

                # Running cumulative centipayout
                step_centipayout = self._paytable.to_centipayout(step.step_payout)
                self._cumulative_centipayout += step_centipayout
                events.append(self._make_update_tumble_win(
                    self._cumulative_centipayout,
                ))

                # Clusters updated grid multipliers — capture new running state
                grid_state = self._increment_grid_snapshot(grid_state, step.clusters)
                events.append(self._make_update_grid(grid_state))

            # Wincap check — halt cascade if cumulative crosses the cap
            if (is_wincap and self._config.wincap.halt_cascade
                    and self._cumulative_centipayout >= wincap_centipayout):
                self._cumulative_centipayout = wincap_centipayout
                events.append(self._make_wincap(wincap_centipayout))
                wincap_hit = True
                break

            # Spawn events — emitted BEFORE gravitySettle per EventStream spec
            if step.booster_spawn_types:
                events.extend(self._make_spawn_events(step))

            # Gravity settle — only for steps 1+ that have gravity data
            if step.gravity_record is not None:
                events.append(self._make_gravity_settle(step.gravity_record))

            # Booster phase — if boosters fired this step
            if step.booster_fire_records:
                events.append(self._make_booster_phase(
                    step.booster_fire_records, step.booster_gravity_record,
                ))

        # Final payout events
        final_centipayout = (
            wincap_centipayout if wincap_hit else instance.centipayout
        )

        # Freegame trigger — basegame scatter count triggers freespins
        if is_freegame:
            scatter_count = len(instance.spatial_step.scatter_positions)
            # awards is tuple of (scatter_count, freespins) pairs
            awards_lookup = dict(self._config.freespin.awards)
            total_fs = awards_lookup.get(
                scatter_count,
                # Fallback to max award if exact count not in config
                max(fs for _, fs in self._config.freespin.awards)
                if self._config.freespin.awards else 0,
            )
            events.append(self._make_set_total_win(0))
            events.append(self._make_free_spin_trigger(
                instance.spatial_step.scatter_positions, total_fs,
            ))
            # Freespin spins would be generated here in production —
            # for now the instance carries the total freegame payout
            events.append(self._make_free_spin_end(
                final_centipayout, instance.win_level,
            ))
        else:
            events.append(self._make_set_win(final_centipayout, instance.win_level))
            events.append(self._make_set_total_win(final_centipayout))

        events.append(self._make_final_win(final_centipayout))
        return events

    # ------------------------------------------------------------------
    # Individual event builders
    # ------------------------------------------------------------------

    def _make_reveal(
        self,
        board: Board,
        game_type: str,
        scatter_positions: frozenset[Position],
    ) -> dict:
        """Build reveal event — initial board display with anticipation array."""
        scatter_reels = sorted({pos.reel for pos in scatter_positions})
        anticipation = compute_anticipation(
            scatter_reels,
            self._config.board.num_reels,
            self._config.anticipation.trigger_threshold,
        )
        return {
            "index": self._next_index(),
            "type": REVEAL,
            "board": self._board_to_grid(board),
            "gameType": game_type,
            "anticipation": anticipation,
        }

    def _make_update_grid(self, grid_snapshot: tuple[int, ...]) -> dict:
        """Build updateGrid event — position multiplier matrix."""
        return {
            "index": self._next_index(),
            "type": UPDATE_GRID,
            "gridMultipliers": self._snapshot_to_matrix(grid_snapshot),
        }

    def _make_win_info(
        self,
        clusters: tuple[ClusterAssignment, ...],
        grid_snapshot: tuple[int, ...],
        board: Board,
    ) -> dict:
        """Build winInfo event — per-cluster breakdown with meta.

        Matches the SDK's win_data structure: totalWin + per-cluster wins with
        symbol, clusterSize, win (centipayout), positions, and meta dict.
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
            total_centipayout += cluster_centipayout

            # Centroid — visual center for win amount overlay
            centroid = self._compute_centroid(cluster.positions)

            # Wild positions for diagnostics and frontend highlighting
            wild_pos_list = self._positions_to_json(cluster.wild_positions)

            wins.append({
                "symbol": cluster.symbol.name,
                "clusterSize": cluster.size,
                "win": cluster_centipayout,
                "positions": self._positions_to_json(
                    cluster.positions | cluster.wild_positions,
                ),
                "meta": {
                    "globalMult": 1,
                    "clusterMult": mult_sum,
                    "winWithoutMult": self._paytable.to_centipayout(base_payout),
                    "overlay": centroid,
                    "wildPositions": wild_pos_list,
                },
            })

        return {
            "index": self._next_index(),
            "type": WIN_INFO,
            "totalWin": total_centipayout,
            "wins": wins,
        }

    def _make_update_tumble_win(self, cumulative_centipayout: int) -> dict:
        """Build updateTumbleWin event — running cascade accumulation."""
        return {
            "index": self._next_index(),
            "type": UPDATE_TUMBLE_WIN,
            "amount": cumulative_centipayout,
        }

    def _make_spawn_events(self, step: CascadeStepRecord) -> list[dict]:
        """Build spawn events for boosters created this step.

        Groups spawn types and emits one event per booster type that spawned.
        Uses SPAWN_EVENT_TYPE dict dispatch to map symbol name → event type.
        Positions come from TransitionResult.spawns (post-collision-resolution).
        """
        from collections import defaultdict

        events: list[dict] = []

        # Build type → list[position_dict] index from resolved spawn data.
        # Multiple spawns of the same type (e.g. two wilds) group naturally.
        positions_by_type: dict[str, list[dict]] = defaultdict(list)
        for btype, reel, row, orient in step.booster_spawn_positions:
            pos_dict: dict[str, object] = {"reel": reel, "row": row}
            if orient is not None:
                pos_dict["orientation"] = orient
            positions_by_type[btype].append(pos_dict)

        for spawn_type in step.booster_spawn_types:
            event_type = SPAWN_EVENT_TYPE.get(spawn_type)
            if event_type is None:
                continue

            spawn_clusters: list[dict] = []
            for cluster in step.clusters:
                centroid = self._compute_centroid(cluster.positions)
                spawn_clusters.append({
                    "symbol": cluster.symbol.name,
                    "size": cluster.size,
                    "centroid": centroid,
                })

            event: dict = {
                "index": self._next_index(),
                "type": event_type,
                "positions": positions_by_type.get(spawn_type, []),
                "clusters": spawn_clusters,
            }
            events.append(event)
        return events

    def _make_gravity_settle(self, gravity_record: GravityRecord) -> dict:
        """Build gravitySettle event from captured gravity data.

        Contains exploding positions, per-pass movement data, and refill symbols.
        """
        # Format exploding positions as {reel, row} dicts
        exploding = [
            {"reel": r, "row": c} for r, c in gravity_record.exploded_positions
        ]

        # Format per-pass move steps: each move is {fromCell, toCell} with direction
        move_steps: list[list[dict]] = []
        for pass_moves in gravity_record.move_steps:
            pass_list: list[dict] = []
            for (src_r, src_c), (dst_r, dst_c) in pass_moves:
                pass_list.append({
                    "fromCell": {"reel": src_r, "row": src_c},
                    "toCell": {"reel": dst_r, "row": dst_c},
                })
            move_steps.append(pass_list)

        # Format refill symbols grouped by reel
        num_reels = self._config.board.num_reels
        new_symbols: list[list[dict]] = [[] for _ in range(num_reels)]
        for reel, row, sym_name in gravity_record.refill_entries:
            new_symbols[reel].append({
                "name": sym_name,
                "reel": reel,
                "row": row,
            })

        return {
            "index": self._next_index(),
            "type": GRAVITY_SETTLE,
            "explodingSymbols": exploding,
            "moveSteps": move_steps,
            "newSymbols": new_symbols,
        }

    def _make_booster_phase(
        self,
        fire_records: tuple[BoosterFireRecord, ...],
        booster_gravity: GravityRecord | None,
    ) -> dict:
        """Build boosterPhase event — fired boosters + post-fire gravity."""
        fired: list[dict] = []
        all_cleared: list[dict] = []

        for record in fire_records:
            entry: dict = {
                "type": record.booster_type.lower(),
                "reel": record.position_reel,
                "row": record.position_row,
            }
            # Type-specific fields matching EventStream spec
            if record.booster_type == "R" and record.orientation is not None:
                entry["orientation"] = record.orientation
            elif record.booster_type == "LB" and record.target_symbols:
                entry["targetSymbol"] = record.target_symbols[0]
            elif record.booster_type == "SLB" and record.target_symbols:
                entry["targetSymbols"] = list(record.target_symbols)

            fired.append(entry)

            # Collect cleared cells from all fires
            for r, c in record.affected_positions_list:
                all_cleared.append({"reel": r, "row": c})

        # Deduplicate and sort cleared cells
        seen: set[tuple[int, int]] = set()
        unique_cleared: list[dict] = []
        for cell in sorted(all_cleared, key=lambda p: (p["reel"], p["row"])):
            key = (cell["reel"], cell["row"])
            if key not in seen:
                seen.add(key)
                unique_cleared.append(cell)

        event: dict = {
            "index": self._next_index(),
            "type": BOOSTER_PHASE,
            "firedBoosters": fired,
            "clearedCells": unique_cleared,
        }

        # Post-booster gravity if available
        if booster_gravity is not None:
            move_steps: list[list[dict]] = []
            for pass_moves in booster_gravity.move_steps:
                pass_list: list[dict] = []
                for (src_r, src_c), (dst_r, dst_c) in pass_moves:
                    pass_list.append({
                        "fromCell": {"reel": src_r, "row": src_c},
                        "toCell": {"reel": dst_r, "row": dst_c},
                    })
                move_steps.append(pass_list)

            num_reels = self._config.board.num_reels
            new_symbols: list[list[dict]] = [[] for _ in range(num_reels)]
            for reel, row, sym_name in booster_gravity.refill_entries:
                new_symbols[reel].append({
                    "name": sym_name,
                    "reel": reel,
                    "row": row,
                })

            event["moveSteps"] = move_steps
            event["newSymbols"] = new_symbols

        return event

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
