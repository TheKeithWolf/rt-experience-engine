"""Shared booster-spawn loop tests (TEST-REEL-025 through 029).

These cover the pure extraction from cascade_generator/simulator. Regression
coverage for both original call sites lives in their existing test files
(test_cascade_pipeline.py, test_rocket_bomb.py, etc.) — passing those after
the refactor is the proof that the extraction preserves behavior.
"""

from __future__ import annotations

import pytest

from ..boosters.tracker import BoosterTracker
from ..config.schema import MasterConfig
from ..pipeline.booster_spawning import (
    SpawnEvent,
    spawn_boosters_from_clusters,
)
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.cluster_detection import Cluster
from ..primitives.symbols import Symbol


def _make_cluster(positions: list[tuple[int, int]], symbol: Symbol) -> Cluster:
    pos_set = frozenset(Position(r, c) for r, c in positions)
    return Cluster(
        symbol=symbol,
        positions=pos_set,
        wild_positions=frozenset(),
        size=len(pos_set),
    )


def _fresh_fixtures(
    default_config: MasterConfig,
) -> tuple[Board, BoosterTracker, BoosterRules]:
    board = Board.empty(default_config.board)
    # Fill the board with a non-clustering pattern so booster placement has
    # room to resolve collisions when the centroid is occupied.
    for reel in range(default_config.board.num_reels):
        for row in range(default_config.board.num_rows):
            board.set(Position(reel, row), Symbol.L1)
    tracker = BoosterTracker(default_config.board)
    rules = BoosterRules(
        default_config.boosters, default_config.board, default_config.symbols,
    )
    return board, tracker, rules


class TestSpawnBoostersFromClusters:

    def test_reel_025_size_9_cluster_spawns_rocket(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-025: cluster of size 9 → spawns Rocket (per default.yaml)."""
        board, tracker, rules = _fresh_fixtures(default_config)
        # Size 9 — falls in Rocket range (9-10) per default.yaml spawn_thresholds
        cluster = _make_cluster(
            [(0, r) for r in range(5)] + [(1, r) for r in range(4)],
            Symbol.H1,
        )

        events = spawn_boosters_from_clusters(
            [cluster], board, tracker, rules,
            default_config.boosters.spawn_order,
        )

        assert len(events) == 1
        assert events[0].booster_type is Symbol.R
        assert events[0].orientation in ("H", "V")

    def test_reel_026_size_11_cluster_spawns_bomb(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-026: cluster of size 11 → spawns Bomb."""
        board, tracker, rules = _fresh_fixtures(default_config)
        positions = [(r, c) for r in range(3) for c in range(4)][:11]
        cluster = _make_cluster(positions, Symbol.H2)

        events = spawn_boosters_from_clusters(
            [cluster], board, tracker, rules,
            default_config.boosters.spawn_order,
        )

        assert len(events) == 1
        assert events[0].booster_type is Symbol.B
        # Bombs don't carry orientation.
        assert events[0].orientation is None

    def test_reel_027_spawn_order_respected(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-027: spawn events emitted in configured spawn_order.

        Spawn order per default.yaml: W, R, B, LB, SLB. Given clusters whose
        sizes match W (7) and B (11), the W event must precede the B event.
        """
        board, tracker, rules = _fresh_fixtures(default_config)
        w_cluster = _make_cluster(
            [(0, r) for r in range(5)] + [(1, 0), (1, 1)],  # size 7 → W
            Symbol.H1,
        )
        b_cluster = _make_cluster(
            [(r, c) for r in range(3, 6) for c in range(4)][:11],  # size 11 → B
            Symbol.H2,
        )

        events = spawn_boosters_from_clusters(
            [b_cluster, w_cluster], board, tracker, rules,
            default_config.boosters.spawn_order,
        )

        assert [e.booster_type for e in events] == [Symbol.W, Symbol.B]

    def test_reel_028_no_position_conflicts(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-028: multiple spawns occupy distinct positions."""
        board, tracker, rules = _fresh_fixtures(default_config)
        # Two cluster patterns far enough apart that independent centroids
        # cannot collide — but collision tracking must still prevent reuse
        # within the spawn loop when it runs on overlapping clusters.
        c1 = _make_cluster(
            [(0, r) for r in range(5)] + [(1, 0), (1, 1)], Symbol.H1,
        )
        c2 = _make_cluster(
            [(5, r) for r in range(5)] + [(6, 0), (6, 1)], Symbol.H2,
        )

        events = spawn_boosters_from_clusters(
            [c1, c2], board, tracker, rules,
            default_config.boosters.spawn_order,
        )

        positions = [e.position for e in events]
        assert len(positions) == len(set(positions))

    def test_reel_029_empty_clusters_produce_no_events(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-029: no qualifying clusters → zero events, no tracker churn."""
        board, tracker, rules = _fresh_fixtures(default_config)
        events = spawn_boosters_from_clusters(
            [], board, tracker, rules,
            default_config.boosters.spawn_order,
        )
        assert events == ()
        assert list(tracker.all_boosters()) == []


class TestSpawnEventShape:

    def test_spawn_event_is_frozen(self) -> None:
        """SpawnEvent is immutable — generators cannot mutate post-return."""
        event = SpawnEvent(
            booster_type=Symbol.R,
            position=Position(0, 0),
            source_cluster_index=0,
            orientation="H",
        )
        with pytest.raises(Exception):
            event.position = Position(1, 1)  # type: ignore[misc]
