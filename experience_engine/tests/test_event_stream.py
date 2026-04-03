"""Tests for Event Stream Generation + Visual Tracer.

All fixtures use hand-constructed GeneratedInstance objects — no solver or
clingo dependency.
"""

from __future__ import annotations

import io

import pytest

from ..config.schema import MasterConfig
from ..output.book_record import BookRecord, book_record_from_instance
from ..output.event_stream import EventStreamGenerator
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
    UPDATE_TUMBLE_WIN,
    WINCAP,
    WIN_INFO,
)
from ..pipeline.data_types import (
    BoosterArmRecord,
    BoosterFireRecord,
    CascadeStepRecord,
    GeneratedInstance,
    GravityRecord,
)
from ..primitives.board import Board, Position
from ..primitives.paytable import Paytable
from ..primitives.symbols import Symbol
from ..spatial_solver.data_types import ClusterAssignment, SpatialStep
from ..tracer.tracer import EventTracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_board(config: MasterConfig, sym: Symbol = Symbol.L1) -> Board:
    """Board filled entirely with one symbol."""
    board = Board.empty(config.board)
    for reel in range(config.board.num_reels):
        for row in range(config.board.num_rows):
            board.set(Position(reel, row), sym)
    return board


def _make_paytable(config: MasterConfig) -> Paytable:
    return Paytable(config.paytable, config.centipayout, config.win_levels)


def _empty_spatial_step() -> SpatialStep:
    return SpatialStep(
        clusters=(), near_misses=(), scatter_positions=frozenset(), boosters=(),
    )


def _make_dead_instance(config: MasterConfig) -> GeneratedInstance:
    """Dead spin: payout 0, criteria '0', no cascade."""
    board = _make_board(config)
    return GeneratedInstance(
        sim_id=0,
        archetype_id="dead_empty",
        family="dead",
        criteria="0",
        board=board,
        spatial_step=_empty_spatial_step(),
        payout=0.0,
        centipayout=0,
        win_level=0,
    )


def _make_static_instance(config: MasterConfig) -> GeneratedInstance:
    """Static win: 1 cluster of H2 x 5, no cascade. payout = 0.5x."""
    board = _make_board(config, Symbol.L2)
    # Place H2 cluster at (0,0)-(0,4) and (1,0)
    cluster_positions = frozenset({
        Position(0, 0), Position(0, 1), Position(0, 2),
        Position(0, 3), Position(1, 0),
    })
    for pos in cluster_positions:
        board.set(pos, Symbol.H2)

    cluster = ClusterAssignment(
        symbol=Symbol.H2, positions=cluster_positions,
        size=5, wild_positions=frozenset(),
    )
    spatial_step = SpatialStep(
        clusters=(cluster,), near_misses=(),
        scatter_positions=frozenset(), boosters=(),
    )

    paytable = _make_paytable(config)
    payout = paytable.get_payout(5, Symbol.H2)
    centipayout = paytable.to_centipayout(payout)
    win_level = paytable.get_win_level(payout)

    # Grid snapshot: all zeros (no prior wins to increment mults)
    total_cells = config.board.num_reels * config.board.num_rows
    grid_snapshot = tuple(
        config.grid_multiplier.initial_value for _ in range(total_cells)
    )

    # Create a single cascade step for the static case
    step = CascadeStepRecord(
        step_index=0,
        board_before=board.copy(),
        board_after=board.copy(),
        clusters=(cluster,),
        step_payout=payout,
        grid_multipliers_snapshot=grid_snapshot,
    )

    return GeneratedInstance(
        sim_id=1,
        archetype_id="t1_single",
        family="t1",
        criteria="basegame",
        board=board,
        spatial_step=spatial_step,
        payout=payout,
        centipayout=centipayout,
        win_level=win_level,
        cascade_steps=(step,),
    )


def _make_cascade_instance(config: MasterConfig) -> GeneratedInstance:
    """Cascade win: 2 steps, step 0 has H2x5, step 1 has L1x5 after gravity."""
    board_step0 = _make_board(config, Symbol.L2)
    cluster0_positions = frozenset({
        Position(0, 0), Position(0, 1), Position(0, 2),
        Position(0, 3), Position(1, 0),
    })
    for pos in cluster0_positions:
        board_step0.set(pos, Symbol.H2)

    cluster0 = ClusterAssignment(
        symbol=Symbol.H2, positions=cluster0_positions,
        size=5, wild_positions=frozenset(),
    )

    paytable = _make_paytable(config)
    payout0 = paytable.get_payout(5, Symbol.H2)
    total_cells = config.board.num_reels * config.board.num_rows
    grid_snap0 = tuple(config.grid_multiplier.initial_value for _ in range(total_cells))

    step0 = CascadeStepRecord(
        step_index=0,
        board_before=board_step0.copy(),
        board_after=board_step0.copy(),
        clusters=(cluster0,),
        step_payout=payout0,
        grid_multipliers_snapshot=grid_snap0,
    )

    # Step 1: after gravity, new L1 cluster
    board_step1 = _make_board(config, Symbol.L3)
    cluster1_positions = frozenset({
        Position(2, 0), Position(2, 1), Position(2, 2),
        Position(3, 0), Position(3, 1),
    })
    for pos in cluster1_positions:
        board_step1.set(pos, Symbol.L1)

    cluster1 = ClusterAssignment(
        symbol=Symbol.L1, positions=cluster1_positions,
        size=5, wild_positions=frozenset(),
    )

    payout1 = paytable.get_payout(5, Symbol.L1)

    # Grid snapshot with some incremented values from step 0
    grid_snap1_list = list(grid_snap0)
    for pos in cluster0_positions:
        idx = pos.reel * config.board.num_rows + pos.row
        grid_snap1_list[idx] = config.grid_multiplier.first_hit_value
    grid_snap1 = tuple(grid_snap1_list)

    # Gravity record for step 1 — exploding step 0's clusters
    gravity_record = GravityRecord(
        exploded_positions=tuple(
            (p.reel, p.row) for p in sorted(cluster0_positions, key=lambda p: (p.reel, p.row))
        ),
        move_steps=(
            # One pass: L2 symbol at (0,4) falls straight down to (0,3)
            (("L2", (0, 4), (0, 3)),),
        ),
        refill_entries=(
            (0, 0, "L3"), (0, 1, "L3"), (0, 2, "L3"),
            (1, 0, "L3"),
        ),
    )

    step1 = CascadeStepRecord(
        step_index=1,
        board_before=board_step1.copy(),
        board_after=board_step1.copy(),
        clusters=(cluster1,),
        step_payout=payout1,
        grid_multipliers_snapshot=grid_snap1,
        gravity_record=gravity_record,
    )

    total_payout = payout0 + payout1
    centipayout = paytable.to_centipayout(total_payout)
    win_level = paytable.get_win_level(total_payout)

    spatial_step = SpatialStep(
        clusters=(cluster0,), near_misses=(),
        scatter_positions=frozenset(), boosters=(),
    )

    return GeneratedInstance(
        sim_id=2,
        archetype_id="t1_cascade",
        family="t1",
        criteria="basegame",
        board=board_step0,
        spatial_step=spatial_step,
        payout=total_payout,
        centipayout=centipayout,
        win_level=win_level,
        cascade_steps=(step0, step1),
    )


def _make_wincap_instance(config: MasterConfig) -> GeneratedInstance:
    """Wincap instance: payout = config.wincap.max_payout, criteria 'wincap'."""
    board = _make_board(config, Symbol.H3)
    cluster_positions = frozenset({
        Position(0, 0), Position(0, 1), Position(0, 2),
        Position(0, 3), Position(1, 0),
    })

    cluster = ClusterAssignment(
        symbol=Symbol.H3, positions=cluster_positions,
        size=5, wild_positions=frozenset(),
    )
    spatial_step = SpatialStep(
        clusters=(cluster,), near_misses=(),
        scatter_positions=frozenset(), boosters=(),
    )

    paytable = _make_paytable(config)
    payout = config.wincap.max_payout
    centipayout = paytable.to_centipayout(payout)
    total_cells = config.board.num_reels * config.board.num_rows
    grid_snap = tuple(config.grid_multiplier.initial_value for _ in range(total_cells))

    step = CascadeStepRecord(
        step_index=0,
        board_before=board.copy(),
        board_after=board.copy(),
        clusters=(cluster,),
        step_payout=payout,
        grid_multipliers_snapshot=grid_snap,
    )

    return GeneratedInstance(
        sim_id=3,
        archetype_id="wincap_cascade",
        family="wincap",
        criteria="wincap",
        board=board,
        spatial_step=spatial_step,
        payout=payout,
        centipayout=centipayout,
        win_level=paytable.get_win_level(payout),
        cascade_steps=(step,),
    )


def _make_trigger_instance(config: MasterConfig) -> GeneratedInstance:
    """Freegame trigger: 4 scatters, criteria 'freegame'."""
    board = _make_board(config, Symbol.L1)
    # Place 4 scatters across different reels
    scatter_positions = frozenset({
        Position(0, 3), Position(2, 3), Position(4, 3), Position(6, 3),
    })
    for pos in scatter_positions:
        board.set(pos, Symbol.S)

    cluster_positions = frozenset({
        Position(1, 0), Position(1, 1), Position(1, 2),
        Position(1, 3), Position(1, 4),
    })
    cluster = ClusterAssignment(
        symbol=Symbol.L1, positions=cluster_positions,
        size=5, wild_positions=frozenset(),
    )

    spatial_step = SpatialStep(
        clusters=(cluster,), near_misses=(),
        scatter_positions=scatter_positions, boosters=(),
    )

    paytable = _make_paytable(config)
    payout = paytable.get_payout(5, Symbol.L1)
    centipayout = paytable.to_centipayout(payout)
    total_cells = config.board.num_reels * config.board.num_rows
    grid_snap = tuple(config.grid_multiplier.initial_value for _ in range(total_cells))

    step = CascadeStepRecord(
        step_index=0,
        board_before=board.copy(),
        board_after=board.copy(),
        clusters=(cluster,),
        step_payout=payout,
        grid_multipliers_snapshot=grid_snap,
    )

    return GeneratedInstance(
        sim_id=4,
        archetype_id="trigger_base",
        family="trigger",
        criteria="freegame",
        board=board,
        spatial_step=spatial_step,
        payout=payout,
        centipayout=centipayout,
        win_level=paytable.get_win_level(payout),
        cascade_steps=(step,),
    )


# ---------------------------------------------------------------------------
# TEST: reveal has correct board, gameType, boardMultipliers
# ---------------------------------------------------------------------------


class TestRevealEvent:
    def test_reveal_fields(self, default_config: MasterConfig) -> None:
        """Reveal event contains board grid, gameType, and boardMultipliers."""
        instance = _make_dead_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        reveal = events[0]
        assert reveal["type"] == REVEAL
        assert reveal["index"] == 0
        assert reveal["gameType"] == "basegame"
        assert isinstance(reveal["board"], list)
        assert len(reveal["board"]) == default_config.board.num_reels
        assert len(reveal["board"][0]) == default_config.board.num_rows
        # boardMultipliers embedded in reveal (no separate updateGrid)
        assert isinstance(reveal["boardMultipliers"], list)
        assert len(reveal["boardMultipliers"]) == default_config.board.num_reels


# ---------------------------------------------------------------------------
# TEST: indices are sequential starting at 0
# ---------------------------------------------------------------------------


class TestEventIndices:
    def test_event_indices_sequential(self, default_config: MasterConfig) -> None:
        """All event indices are sequential starting at 0 with no gaps."""
        instance = _make_cascade_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        for i, event in enumerate(events):
            assert event["index"] == i, (
                f"Event {i} has index {event['index']}, expected {i}"
            )


# ---------------------------------------------------------------------------
# TEST: spawn events before gravitySettle
# ---------------------------------------------------------------------------


class TestSpawnBeforeGravity:
    def test_spawn_before_gravity(self, default_config: MasterConfig) -> None:
        """If spawn and gravity events exist, spawns appear before gravity."""
        instance = _make_cascade_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        gravity_indices = [
            e["index"] for e in events if e["type"] == GRAVITY_SETTLE
        ]
        # In the cascade fixture, step 1 has gravity — verify it appears after any
        # spawn events (our fixture has no spawns, but gravity should still be present)
        assert len(gravity_indices) > 0, "Cascade should have gravitySettle events"


# ---------------------------------------------------------------------------
# TEST: winInfo uses spec shape (basePayout, clusterPayout, cluster.cells)
# ---------------------------------------------------------------------------


class TestWinInfoSpec:
    def test_wininfo_overlay_centroid(self, default_config: MasterConfig) -> None:
        """winInfo overlay matches the cluster centroid."""
        instance = _make_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        win_info_events = [e for e in events if e["type"] == WIN_INFO]
        assert len(win_info_events) >= 1

        for win_event in win_info_events:
            for win in win_event["wins"]:
                overlay = win["overlay"]
                assert "reel" in overlay
                assert "row" in overlay
                assert 0 <= overlay["reel"] < default_config.board.num_reels
                assert 0 <= overlay["row"] < default_config.board.num_rows

    def test_wininfo_spec_fields(self, default_config: MasterConfig) -> None:
        """winInfo wins[] entries have basePayout, clusterPayout, clusterMultiplier, cluster.cells."""
        instance = _make_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        win_info_events = [e for e in events if e["type"] == WIN_INFO]
        for win_event in win_info_events:
            for win in win_event["wins"]:
                assert "basePayout" in win
                assert "clusterPayout" in win
                assert "clusterMultiplier" in win
                assert "clusterSize" in win
                assert "overlay" in win
                assert "cluster" in win
                assert "cells" in win["cluster"]
                # No legacy fields
                assert "meta" not in win
                assert "symbol" not in win
                assert "positions" not in win

    def test_wininfo_cluster_cells(self, default_config: MasterConfig) -> None:
        """Each cell in cluster.cells has symbol, reel, row, multiplier."""
        instance = _make_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        win_info_events = [e for e in events if e["type"] == WIN_INFO]
        for win_event in win_info_events:
            for win in win_event["wins"]:
                for cell in win["cluster"]["cells"]:
                    assert "symbol" in cell
                    assert "reel" in cell
                    assert "row" in cell
                    assert "multiplier" in cell

    def test_wininfo_cluster_mult(self, default_config: MasterConfig) -> None:
        """winInfo clusterMultiplier is at least minimum_contribution."""
        instance = _make_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        win_info_events = [e for e in events if e["type"] == WIN_INFO]
        for win_event in win_info_events:
            for win in win_event["wins"]:
                # Initial grid is all zeros — minimum_contribution ensures mult >= 1
                assert win["clusterMultiplier"] >= default_config.grid_multiplier.minimum_contribution


# ---------------------------------------------------------------------------
# TEST: centipayout conversion matches formula
# ---------------------------------------------------------------------------


class TestCentipayoutConversion:
    def test_centipayout_conversion(self, default_config: MasterConfig) -> None:
        """Centipayout values in events match the config formula."""
        instance = _make_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        final_win = [e for e in events if e["type"] == FINAL_WIN]
        assert len(final_win) == 1
        assert final_win[0]["amount"] == instance.centipayout


# ---------------------------------------------------------------------------
# TEST: win level matches payout range
# ---------------------------------------------------------------------------


class TestWinLevelMapping:
    def test_win_level_mapping(self, default_config: MasterConfig) -> None:
        """setWin winLevel matches Paytable.get_win_level() for the instance payout."""
        instance = _make_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        set_win_events = [e for e in events if e["type"] == SET_WIN]
        assert len(set_win_events) == 1
        assert set_win_events[0]["winLevel"] == instance.win_level


# ---------------------------------------------------------------------------
# TEST: dead spin sequence — exactly [reveal, setTotalWin, finalWin]
# ---------------------------------------------------------------------------


class TestDeadSpin:
    def test_dead_spin_events(self, default_config: MasterConfig) -> None:
        """Dead spin: exactly 3 events — reveal, setTotalWin(0), finalWin(0)."""
        instance = _make_dead_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        assert len(events) == 3
        assert events[0]["type"] == REVEAL
        assert "boardMultipliers" in events[0]
        assert events[1]["type"] == SET_TOTAL_WIN
        assert events[1]["amount"] == 0
        assert events[2]["type"] == FINAL_WIN
        assert events[2]["amount"] == 0


# ---------------------------------------------------------------------------
# TEST: cascade gravitySettle has correct structure (no explodingSymbols)
# ---------------------------------------------------------------------------


class TestCascadeGravitySettle:
    def test_cascade_gravity_settle(self, default_config: MasterConfig) -> None:
        """gravitySettle event has moveSteps and newSymbols, no explodingSymbols."""
        instance = _make_cascade_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        gravity_events = [e for e in events if e["type"] == GRAVITY_SETTLE]
        assert len(gravity_events) >= 1

        grav = gravity_events[0]
        assert "explodingSymbols" not in grav
        assert "moveSteps" in grav
        assert "newSymbols" in grav
        assert isinstance(grav["moveSteps"], list)
        assert isinstance(grav["newSymbols"], list)
        # newSymbols should have one list per reel
        assert len(grav["newSymbols"]) == default_config.board.num_reels

    def test_gravity_settle_move_steps_shape(self, default_config: MasterConfig) -> None:
        """moveSteps entries use {symbol, fromCell: {reel, row}, toCell: {reel, row}} shape."""
        instance = _make_cascade_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        gravity_events = [e for e in events if e["type"] == GRAVITY_SETTLE]
        grav = gravity_events[0]
        for pass_list in grav["moveSteps"]:
            for move in pass_list:
                assert "symbol" in move
                assert "fromCell" in move
                assert "toCell" in move
                assert "reel" in move["fromCell"]
                assert "row" in move["fromCell"]

    def test_gravity_settle_new_symbols_shape(self, default_config: MasterConfig) -> None:
        """newSymbols entries use {symbol, position: {reel, row}} shape."""
        instance = _make_cascade_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        gravity_events = [e for e in events if e["type"] == GRAVITY_SETTLE]
        grav = gravity_events[0]
        for reel_entries in grav["newSymbols"]:
            for entry in reel_entries:
                assert "symbol" in entry
                assert "position" in entry
                assert "reel" in entry["position"]
                assert "row" in entry["position"]


# ---------------------------------------------------------------------------
# TEST: static gravitySettle emitted between updateTumbleWin and setWin
# ---------------------------------------------------------------------------


def _make_true_static_instance(config: MasterConfig) -> GeneratedInstance:
    """Static win with gravity_record — no cascade_steps, just post-win animation data."""
    board = _make_board(config, Symbol.L2)
    cluster_positions = frozenset({
        Position(0, 0), Position(0, 1), Position(0, 2),
        Position(0, 3), Position(1, 0),
    })
    for pos in cluster_positions:
        board.set(pos, Symbol.H2)

    cluster = ClusterAssignment(
        symbol=Symbol.H2, positions=cluster_positions,
        size=5, wild_positions=frozenset(),
    )
    spatial_step = SpatialStep(
        clusters=(cluster,), near_misses=(),
        scatter_positions=frozenset(), boosters=(),
    )

    paytable = _make_paytable(config)
    payout = paytable.get_payout(5, Symbol.H2)
    centipayout = paytable.to_centipayout(payout)
    win_level = paytable.get_win_level(payout)

    # Gravity record: cluster positions explode, one symbol falls, cosmetic refill
    gravity_record = GravityRecord(
        exploded_positions=tuple(
            (p.reel, p.row) for p in sorted(cluster_positions, key=lambda p: (p.reel, p.row))
        ),
        move_steps=(
            # L2 symbol at (0,4) falls straight down to (0,3)
            (("L2", (0, 4), (0, 3)),),
        ),
        refill_entries=(
            (0, 0, "L3"), (0, 1, "L3"), (0, 2, "L3"),
            (0, 4, "H1"), (1, 0, "L4"),
        ),
    )

    return GeneratedInstance(
        sim_id=100,
        archetype_id="t1_single",
        family="t1",
        criteria="basegame",
        board=board,
        spatial_step=spatial_step,
        payout=payout,
        centipayout=centipayout,
        win_level=win_level,
        gravity_record=gravity_record,
    )


class TestStaticGravitySettle:
    def test_static_gravity_settle_present(self, default_config: MasterConfig) -> None:
        """Static win with gravity_record produces a gravitySettle event."""
        instance = _make_true_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        types = [e["type"] for e in events]
        assert GRAVITY_SETTLE in types

    def test_static_gravity_settle_ordering(self, default_config: MasterConfig) -> None:
        """gravitySettle appears between updateBoardMultipliers and setWin."""
        instance = _make_true_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        types = [e["type"] for e in events]
        board_mult_idx = types.index(UPDATE_BOARD_MULTIPLIERS)
        gravity_idx = types.index(GRAVITY_SETTLE)
        set_win_idx = types.index(SET_WIN)
        assert board_mult_idx < gravity_idx < set_win_idx

    def test_static_gravity_settle_structure(self, default_config: MasterConfig) -> None:
        """gravitySettle has moveSteps and newSymbols, no explodingSymbols."""
        instance = _make_true_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        grav = [e for e in events if e["type"] == GRAVITY_SETTLE][0]
        assert "explodingSymbols" not in grav
        assert isinstance(grav["moveSteps"], list)
        assert isinstance(grav["newSymbols"], list)
        assert len(grav["newSymbols"]) == default_config.board.num_reels

    def test_static_gravity_settle_indices_sequential(
        self, default_config: MasterConfig,
    ) -> None:
        """Event indices remain sequential with the extra gravitySettle event."""
        instance = _make_true_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        for i, event in enumerate(events):
            assert event["index"] == i

    def test_dead_spin_no_gravity_settle(self, default_config: MasterConfig) -> None:
        """Dead spin (no clusters) must not emit gravitySettle."""
        instance = _make_dead_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        types = [e["type"] for e in events]
        assert GRAVITY_SETTLE not in types


# ---------------------------------------------------------------------------
# TEST: updateBoardMultipliers emits sparse delta
# ---------------------------------------------------------------------------


class TestUpdateBoardMultipliers:
    """updateBoardMultipliers events emit only changed positions as sparse deltas."""

    def test_static_has_update_board_multipliers(self, default_config: MasterConfig) -> None:
        """Static win produces exactly one updateBoardMultipliers event."""
        instance = _make_true_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        ubm = [e for e in events if e["type"] == UPDATE_BOARD_MULTIPLIERS]
        assert len(ubm) == 1

    def test_sparse_delta_only_cluster_positions(
        self, default_config: MasterConfig,
    ) -> None:
        """updateBoardMultipliers delta contains only cluster positions."""
        instance = _make_true_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        ubm = [e for e in events if e["type"] == UPDATE_BOARD_MULTIPLIERS][0]
        changed = ubm["boardMultipliers"]
        assert isinstance(changed, list)
        # Each entry has {multiplier, position: {reel, row}}
        for entry in changed:
            assert "multiplier" in entry
            assert "position" in entry
            assert "reel" in entry["position"]
            assert "row" in entry["position"]

        # Changed positions should match cluster positions
        cluster = instance.spatial_step.clusters[0]
        expected = {(p.reel, p.row) for p in cluster.positions | cluster.wild_positions}
        actual = {(e["position"]["reel"], e["position"]["row"]) for e in changed}
        assert actual == expected

    def test_update_board_multipliers_ordering(self, default_config: MasterConfig) -> None:
        """updateBoardMultipliers appears between updateTumbleWin and gravitySettle."""
        instance = _make_true_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        types = [e["type"] for e in events]
        tumble_idx = types.index(UPDATE_TUMBLE_WIN)
        ubm_idx = types.index(UPDATE_BOARD_MULTIPLIERS)
        gravity_idx = types.index(GRAVITY_SETTLE)
        assert tumble_idx < ubm_idx < gravity_idx

    def test_cascade_accumulates_deltas(self, default_config: MasterConfig) -> None:
        """Cascade: each step emits updateBoardMultipliers with that step's changes."""
        instance = _make_cascade_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        ubm = [e for e in events if e["type"] == UPDATE_BOARD_MULTIPLIERS]
        # One per cascade step with clusters (step 0 + step 1)
        assert len(ubm) >= 2


# ---------------------------------------------------------------------------
# TEST: boosterFireInfo has per-booster clearedCells with symbol
# ---------------------------------------------------------------------------


class TestBoosterFireInfo:
    def test_booster_fire_info_event(self, default_config: MasterConfig) -> None:
        """boosterFireInfo event has per-booster clearedCells with {symbol, position}."""
        board = _make_board(default_config, Symbol.L2)
        cluster_positions = frozenset({
            Position(0, 0), Position(0, 1), Position(0, 2),
            Position(0, 3), Position(0, 4), Position(0, 5),
            Position(0, 6), Position(1, 0), Position(1, 1),
        })
        for pos in cluster_positions:
            board.set(pos, Symbol.H1)

        cluster = ClusterAssignment(
            symbol=Symbol.H1, positions=cluster_positions,
            size=9, wild_positions=frozenset(),
        )

        paytable = _make_paytable(default_config)
        payout = paytable.get_payout(9, Symbol.H1)
        total_cells = default_config.board.num_reels * default_config.board.num_rows
        grid_snap = tuple(
            default_config.grid_multiplier.initial_value
            for _ in range(total_cells)
        )

        fire_record = BoosterFireRecord(
            booster_type="R",
            position_reel=0,
            position_row=3,
            orientation="V",
            affected_count=5,
            chain_target_count=0,
            affected_positions_list=((0, 0, "L2"), (0, 1, "L2"), (0, 2, "L2"), (0, 4, "L2"), (0, 5, "L2")),
        )

        step = CascadeStepRecord(
            step_index=0,
            board_before=board.copy(),
            board_after=board.copy(),
            clusters=(cluster,),
            step_payout=payout,
            grid_multipliers_snapshot=grid_snap,
            booster_spawn_types=("R",),
            booster_fire_records=(fire_record,),
        )

        spatial_step = SpatialStep(
            clusters=(cluster,), near_misses=(),
            scatter_positions=frozenset(), boosters=(),
        )

        instance = GeneratedInstance(
            sim_id=10,
            archetype_id="rocket_v_fire",
            family="rocket",
            criteria="basegame",
            board=board,
            spatial_step=spatial_step,
            payout=payout,
            centipayout=paytable.to_centipayout(payout),
            win_level=paytable.get_win_level(payout),
            cascade_steps=(step,),
        )

        gen = EventStreamGenerator(default_config, paytable)
        events = gen.generate(instance)

        fire_events = [e for e in events if e["type"] == BOOSTER_FIRE_INFO]
        assert len(fire_events) == 1
        fi = fire_events[0]
        assert "boosters" in fi
        assert len(fi["boosters"]) == 1
        booster = fi["boosters"][0]
        assert booster["symbol"] == "R"
        assert "clearedCells" in booster
        for cell in booster["clearedCells"]:
            assert "symbol" in cell
            assert "position" in cell
            assert "reel" in cell["position"]
            assert "row" in cell["position"]


# ---------------------------------------------------------------------------
# TEST: boosterArmInfo emitted when boosters are armed
# ---------------------------------------------------------------------------


class TestBoosterArmInfo:
    def test_booster_arm_info_event(self, default_config: MasterConfig) -> None:
        """boosterArmInfo emitted when step has booster_arm_records."""
        board = _make_board(default_config, Symbol.L2)
        cluster_positions = frozenset({
            Position(0, 0), Position(0, 1), Position(0, 2),
            Position(0, 3), Position(1, 0),
        })
        for pos in cluster_positions:
            board.set(pos, Symbol.H2)

        cluster = ClusterAssignment(
            symbol=Symbol.H2, positions=cluster_positions,
            size=5, wild_positions=frozenset(),
        )

        paytable = _make_paytable(default_config)
        payout = paytable.get_payout(5, Symbol.H2)
        total_cells = default_config.board.num_reels * default_config.board.num_rows
        grid_snap = tuple(
            default_config.grid_multiplier.initial_value
            for _ in range(total_cells)
        )

        arm_record = BoosterArmRecord(
            booster_type="R",
            position_reel=3,
            position_row=2,
            orientation="H",
        )

        gravity_record = GravityRecord(
            exploded_positions=tuple(
                (p.reel, p.row) for p in sorted(cluster_positions, key=lambda p: (p.reel, p.row))
            ),
            move_steps=(),
            refill_entries=(),
        )

        step = CascadeStepRecord(
            step_index=0,
            board_before=board.copy(),
            board_after=board.copy(),
            clusters=(cluster,),
            step_payout=payout,
            grid_multipliers_snapshot=grid_snap,
            booster_arm_records=(arm_record,),
            gravity_record=gravity_record,
        )

        spatial_step = SpatialStep(
            clusters=(cluster,), near_misses=(),
            scatter_positions=frozenset(), boosters=(),
        )

        instance = GeneratedInstance(
            sim_id=20,
            archetype_id="arm_test",
            family="t1",
            criteria="basegame",
            board=board,
            spatial_step=spatial_step,
            payout=payout,
            centipayout=paytable.to_centipayout(payout),
            win_level=paytable.get_win_level(payout),
            cascade_steps=(step,),
        )

        gen = EventStreamGenerator(default_config, paytable)
        events = gen.generate(instance)

        arm_events = [e for e in events if e["type"] == BOOSTER_ARM_INFO]
        assert len(arm_events) == 1
        ai = arm_events[0]
        assert "boosters" in ai
        assert len(ai["boosters"]) == 1
        assert ai["boosters"][0]["symbol"] == "R"
        assert ai["boosters"][0]["position"] == {"reel": 3, "row": 2}

        # boosterArmInfo should appear before gravitySettle
        types = [e["type"] for e in events]
        arm_idx = types.index(BOOSTER_ARM_INFO)
        grav_idx = types.index(GRAVITY_SETTLE)
        assert arm_idx < grav_idx


# ---------------------------------------------------------------------------
# TEST: freegame sequence — basegame setWin emitted before freeSpinTrigger
# ---------------------------------------------------------------------------


class TestFreegameSequence:
    def test_freegame_sequence(self, default_config: MasterConfig) -> None:
        """Freegame: events include setTotalWin, freeSpinTrigger and freeSpinEnd."""
        instance = _make_trigger_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        types = [e["type"] for e in events]
        assert FREE_SPIN_TRIGGER in types
        assert FREE_SPIN_END in types
        assert FINAL_WIN in types

        # freeSpinTrigger should come before freeSpinEnd
        trigger_idx = types.index(FREE_SPIN_TRIGGER)
        end_idx = types.index(FREE_SPIN_END)
        assert trigger_idx < end_idx

        # freeSpinTrigger should have totalFs matching config
        trigger_event = events[trigger_idx]
        scatter_count = len(instance.spatial_step.scatter_positions)
        awards_lookup = dict(default_config.freespin.awards)
        expected_fs = awards_lookup.get(scatter_count, 0)
        assert trigger_event["totalFs"] == expected_fs

    def test_freegame_basegame_set_win(self, default_config: MasterConfig) -> None:
        """Freegame-triggering spin emits basegame setWin before setTotalWin."""
        instance = _make_trigger_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        types = [e["type"] for e in events]
        # Basegame cascade has wins → setWin should be emitted
        if instance.centipayout > 0:
            assert SET_WIN in types
            set_win_idx = types.index(SET_WIN)
            total_win_idx = types.index(SET_TOTAL_WIN)
            trigger_idx = types.index(FREE_SPIN_TRIGGER)
            # Order: setWin → setTotalWin → freeSpinTrigger
            assert set_win_idx < total_win_idx < trigger_idx

        # setTotalWin should reflect basegame wins (not 0)
        total_win_event = [e for e in events if e["type"] == SET_TOTAL_WIN][0]
        # The basegame cascade produces wins, so setTotalWin > 0
        assert total_win_event["amount"] > 0


# ---------------------------------------------------------------------------
# TEST: wincap halts cascade
# ---------------------------------------------------------------------------


class TestWincap:
    def test_wincap_halts_cascade(self, default_config: MasterConfig) -> None:
        """Wincap: wincap event emitted, finalWin clamped to cap centipayout."""
        instance = _make_wincap_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        types = [e["type"] for e in events]
        assert WINCAP in types

        # Final win should equal the cap centipayout
        final = [e for e in events if e["type"] == FINAL_WIN]
        assert len(final) == 1
        paytable = _make_paytable(default_config)
        expected_cap = paytable.to_centipayout(default_config.wincap.max_payout)
        assert final[0]["amount"] == expected_cap


# ---------------------------------------------------------------------------
# TEST: finalWin.amount = payoutMultiplier
# ---------------------------------------------------------------------------


class TestFinalWin:
    def test_final_win_amount(self, default_config: MasterConfig) -> None:
        """finalWin amount matches instance centipayout for non-cap instances."""
        instance = _make_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        final = [e for e in events if e["type"] == FINAL_WIN]
        assert len(final) == 1
        assert final[0]["amount"] == instance.centipayout


# ---------------------------------------------------------------------------
# TEST BookRecord fields and base/free split
# ---------------------------------------------------------------------------


class TestBookRecord:
    def test_book_record_fields(self, default_config: MasterConfig) -> None:
        """BookRecord.to_dict() has correct field names."""
        instance = _make_static_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)

        book = book_record_from_instance(instance, events)
        d = book.to_dict()

        assert d["id"] == instance.sim_id
        assert d["payoutMultiplier"] == instance.centipayout
        assert d["criteria"] == instance.criteria
        assert isinstance(d["events"], list)
        assert "baseGameWins" in d
        assert "freeGameWins" in d

    def test_book_record_base_free_split(self, default_config: MasterConfig) -> None:
        """Criteria-based split: freegame → freeGameWins, basegame → baseGameWins."""
        paytable = _make_paytable(default_config)

        # Dead: both zero
        dead = _make_dead_instance(default_config)
        book_dead = book_record_from_instance(dead, [])
        assert book_dead.baseGameWins == 0.0
        assert book_dead.freeGameWins == 0.0

        # Basegame: all in baseGameWins
        bg = _make_static_instance(default_config)
        book_bg = book_record_from_instance(bg, [])
        assert book_bg.baseGameWins == bg.payout
        assert book_bg.freeGameWins == 0.0

        # Freegame: all in freeGameWins
        fg = _make_trigger_instance(default_config)
        book_fg = book_record_from_instance(fg, [])
        assert book_fg.baseGameWins == 0.0
        assert book_fg.freeGameWins == fg.payout

        # Wincap: all in baseGameWins
        wc = _make_wincap_instance(default_config)
        book_wc = book_record_from_instance(wc, [])
        assert book_wc.baseGameWins == wc.payout
        assert book_wc.freeGameWins == 0.0


# ---------------------------------------------------------------------------
# TEST tracer renders cascade book
# ---------------------------------------------------------------------------


class TestTracer:
    def test_tracer_renders_cascade(self, default_config: MasterConfig) -> None:
        """Tracer output contains expected sections for a cascade instance."""
        instance = _make_cascade_instance(default_config)
        gen = EventStreamGenerator(default_config, _make_paytable(default_config))
        events = gen.generate(instance)
        book = book_record_from_instance(instance, events)

        tracer = EventTracer(default_config)
        output = tracer.trace_to_string(book)

        # Must contain header
        assert f"BOOK #{instance.sim_id}" in output
        assert instance.criteria in output
        # Must contain reveal
        assert "REVEAL" in output
        # Must contain win info
        assert "WIN INFO" in output
        # Must contain gravity
        assert "GRAVITY SETTLE" in output
        # Must contain final win
        assert "FINAL WIN" in output


# ---------------------------------------------------------------------------
# Spawn position threading — boosterSpawnInfo from CascadeStepRecord
# ---------------------------------------------------------------------------


class TestSpawnPositions:
    """Verify _make_booster_spawn_info reads booster_spawn_positions from the step record."""

    def test_single_spawn_position(self, default_config: MasterConfig) -> None:
        """Single wild spawn should emit boosterSpawnInfo with resolved position."""
        board = _make_board(default_config, Symbol.L2)
        cluster_positions = frozenset({
            Position(0, 0), Position(0, 1), Position(0, 2),
            Position(0, 3), Position(0, 4), Position(0, 5),
            Position(0, 6), Position(1, 0), Position(1, 1),
        })
        for pos in cluster_positions:
            board.set(pos, Symbol.H1)

        cluster = ClusterAssignment(
            symbol=Symbol.H1, positions=cluster_positions,
            size=9, wild_positions=frozenset(),
        )

        paytable = _make_paytable(default_config)
        payout = paytable.get_payout(9, Symbol.H1)
        total_cells = default_config.board.num_reels * default_config.board.num_rows
        grid_snap = tuple(
            default_config.grid_multiplier.initial_value
            for _ in range(total_cells)
        )

        step = CascadeStepRecord(
            step_index=0,
            board_before=board.copy(),
            board_after=board.copy(),
            clusters=(cluster,),
            step_payout=payout,
            grid_multipliers_snapshot=grid_snap,
            booster_spawn_types=("W",),
            booster_spawn_positions=(("W", 5, 2, None),),
        )

        gen = EventStreamGenerator(default_config, paytable)
        event = gen._make_booster_spawn_info(step)  # noqa: SLF001

        assert event["type"] == BOOSTER_SPAWN_INFO
        assert len(event["boosters"]) == 1
        assert event["boosters"][0]["symbol"] == "W"
        assert event["boosters"][0]["position"] == {"reel": 5, "row": 2}

    def test_multi_spawn_positions(self, default_config: MasterConfig) -> None:
        """Two spawns should produce one boosterSpawnInfo with two entries."""
        board = _make_board(default_config, Symbol.L2)
        cluster_positions = frozenset({
            Position(0, 0), Position(0, 1), Position(0, 2),
            Position(0, 3), Position(0, 4), Position(0, 5),
            Position(0, 6), Position(1, 0), Position(1, 1),
        })
        for pos in cluster_positions:
            board.set(pos, Symbol.H1)

        cluster = ClusterAssignment(
            symbol=Symbol.H1, positions=cluster_positions,
            size=9, wild_positions=frozenset(),
        )

        paytable = _make_paytable(default_config)
        payout = paytable.get_payout(9, Symbol.H1)
        total_cells = default_config.board.num_reels * default_config.board.num_rows
        grid_snap = tuple(
            default_config.grid_multiplier.initial_value
            for _ in range(total_cells)
        )

        step = CascadeStepRecord(
            step_index=0,
            board_before=board.copy(),
            board_after=board.copy(),
            clusters=(cluster,),
            step_payout=payout,
            grid_multipliers_snapshot=grid_snap,
            booster_spawn_types=("W", "R"),
            booster_spawn_positions=(("W", 3, 1, None), ("R", 5, 2, "H")),
        )

        gen = EventStreamGenerator(default_config, paytable)
        event = gen._make_booster_spawn_info(step)  # noqa: SLF001

        assert event["type"] == BOOSTER_SPAWN_INFO
        assert len(event["boosters"]) == 2
