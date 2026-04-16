"""ReelStripGenerator tests (TEST-REEL-030 through 048).

Covers the full cascade loop: initial population, cluster detection, gravity
settling, reel-strip refill, cascade termination, and determinism.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from ..archetypes.registry import ArchetypeRegistry
from ..archetypes.reel import register_reel_archetypes
from ..config.schema import MasterConfig
from ..pipeline.reel_generator import ReelStripGenerator
from ..primitives.board import Board, Position
from ..primitives.cluster_detection import detect_clusters
from ..primitives.gravity import GravityDAG
from ..primitives.reel_strip import ReelStrip, load_reel_strip
from ..primitives.symbols import Symbol
from ..variance.hints import VarianceHints

REFERENCE_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "reel_strip.csv"
)


def _hints() -> VarianceHints:
    return VarianceHints(
        spatial_bias={}, symbol_weights={},
        near_miss_symbol_preference=(), cluster_size_preference=(),
    )


def _make_gen(
    default_config: MasterConfig,
    strip: ReelStrip | None = None,
) -> ReelStripGenerator:
    registry = ArchetypeRegistry(default_config)
    register_reel_archetypes(registry)
    dag = GravityDAG(default_config.board, default_config.gravity)
    strip = strip or load_reel_strip(REFERENCE_CSV, default_config.board)
    return ReelStripGenerator(default_config, registry, dag, strip)


def _uniform_strip(
    default_config: MasterConfig, symbol: Symbol, length: int = 9,
) -> ReelStrip:
    """Strip where every cell is the same symbol — guarantees a full-board cluster."""
    column = tuple(symbol for _ in range(length))
    return ReelStrip(
        columns=tuple(column for _ in range(default_config.board.num_reels)),
        num_reels=default_config.board.num_reels,
        strip_length=length,
    )


def _standard_shuffled_strip(default_config: MasterConfig) -> ReelStrip:
    """Diagonal 9x7 standard-symbol strip — same shape as reference CSV."""
    return load_reel_strip(REFERENCE_CSV, default_config.board)


# ---------------------------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------------------------

class TestBasicGeneration:

    def test_reel_030_success_true(self, default_config: MasterConfig) -> None:
        """TEST-REEL-030: Generator produces GenerationResult with success=True."""
        gen = _make_gen(default_config)
        result = gen.generate("reel_base", 1, _hints(), random.Random(42))
        assert result.success is True
        assert result.instance is not None
        assert result.attempts == 1
        assert result.failure_reason is None

    def test_reel_031_board_fully_populated(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-031: Final board has no empty cells."""
        gen = _make_gen(default_config)
        result = gen.generate("reel_base", 1, _hints(), random.Random(42))
        assert result.instance.board.empty_positions() == []

    def test_reel_032_initial_symbols_all_standard(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-032: No wilds/scatters/boosters from the strip (reference CSV)."""
        strip = _standard_shuffled_strip(default_config)
        for column in strip.columns:
            for sym in column:
                assert sym.name in default_config.symbols.standard

    def test_reel_037_identity_fields_match_signature(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-037: family, criteria, archetype_id on instance match signature."""
        gen = _make_gen(default_config)
        result = gen.generate("reel_base", 7, _hints(), random.Random(0))
        inst = result.instance
        assert inst.archetype_id == "reel_base"
        assert inst.family == "reel"
        assert inst.criteria == "basegame"
        assert inst.sim_id == 7


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_reel_034_same_seed_same_output(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-034: Identical seed → identical board + payout."""
        gen = _make_gen(default_config)
        r1 = gen.generate("reel_base", 1, _hints(), random.Random(123))
        r2 = gen.generate("reel_base", 1, _hints(), random.Random(123))
        assert r1.instance.payout == r2.instance.payout
        assert r1.instance.centipayout == r2.instance.centipayout
        assert r1.instance.board.board_hash() == r2.instance.board.board_hash()

    def test_reel_035_different_seeds_differ(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-035: Different seeds explore different stops → different outcomes."""
        gen = _make_gen(default_config)
        boards = {
            gen.generate(
                "reel_base", 1, _hints(), random.Random(seed),
            ).instance.board.board_hash()
            for seed in range(1, 30)
        }
        # At least 2 distinct boards across 29 seeds — the strip has enough
        # variety that random stops must produce at least one differing board.
        assert len(boards) >= 2


# ---------------------------------------------------------------------------
# No-cluster path (payout == 0, no cascade)
# ---------------------------------------------------------------------------

class TestNoClusterPath:

    def test_reel_039_no_clusters_means_no_payout(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-039: Board with no initial clusters → payout=0, no cascade steps.

        The reference 9x7 strip uses a diagonal pattern — for many stop vectors
        the initial board contains no 5+ same-symbol cluster. Seed 42 produces
        such a board (confirmed by running the generator).
        """
        gen = _make_gen(default_config)
        result = gen.generate("reel_base", 1, _hints(), random.Random(42))
        if not detect_clusters(result.instance.board, default_config):
            assert result.instance.payout == 0.0
            assert result.instance.centipayout == 0
            assert result.instance.cascade_steps is None


# ---------------------------------------------------------------------------
# Forced-cluster path (uniform strip)
# ---------------------------------------------------------------------------

class TestForcedClusterPath:

    def test_reel_041_multi_cascade_triggers(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-041: Uniform strip → board is all one symbol → mega-cluster
        explodes, refill is the same symbol (since strip is uniform), so the next
        step re-clusters. At least one cascade step must be recorded.
        """
        strip = _uniform_strip(default_config, Symbol.L1)
        gen = _make_gen(default_config, strip)
        result = gen.generate("reel_base", 1, _hints(), random.Random(0))
        # Uniform strip = guaranteed full-board 49-cell cluster → cascade runs.
        assert result.instance.cascade_steps is not None
        assert len(result.instance.cascade_steps) >= 1

    def test_reel_038_cascade_terminates_within_budget(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-038: Cascade step count bounded by (cells / min_cluster) + 1.

        The generator's own budget is the termination guard; this test
        asserts the generator actually respects it under a pathological
        strip that would otherwise cascade forever.
        """
        strip = _uniform_strip(default_config, Symbol.L1)
        gen = _make_gen(default_config, strip)
        result = gen.generate("reel_base", 1, _hints(), random.Random(1))
        b = default_config.board
        max_steps = (b.num_reels * b.num_rows) // b.min_cluster_size + 1
        assert result.instance.cascade_steps is not None
        assert len(result.instance.cascade_steps) <= max_steps


# ---------------------------------------------------------------------------
# Strip-sourced refill
# ---------------------------------------------------------------------------

class TestStripSourcedRefill:

    def test_reel_040_refill_entries_come_from_strip(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-040: Refill entries in the cascade record are from the strip.

        With a uniform strip, every refill entry's symbol must equal the
        uniform symbol — proving the refill is strip-sourced, not random.
        """
        strip = _uniform_strip(default_config, Symbol.H1)
        gen = _make_gen(default_config, strip)
        result = gen.generate("reel_base", 1, _hints(), random.Random(0))
        for step in result.instance.cascade_steps or ():
            assert step.gravity_record is not None
            for _reel, _row, sym_name in step.gravity_record.refill_entries:
                assert sym_name == Symbol.H1.name


# ---------------------------------------------------------------------------
# Booster spawning within the cascade loop
# ---------------------------------------------------------------------------

class TestBoosterInteractions:

    def test_reel_042_large_cluster_spawns_booster(
        self, default_config: MasterConfig,
    ) -> None:
        """TEST-REEL-042: A full-board cluster (size 49) spawns the SLB booster.

        Default.yaml spawn_thresholds map size 15-49 → SLB. With a uniform
        strip, the first tumble step produces a size-49 cluster, which must
        be recorded as a spawned booster on that cascade step.
        """
        strip = _uniform_strip(default_config, Symbol.H1)
        gen = _make_gen(default_config, strip)
        result = gen.generate("reel_base", 1, _hints(), random.Random(0))
        # First cascade step must have a spawned booster matching the SLB bucket.
        first_step = result.instance.cascade_steps[0]
        assert "SLB" in first_step.booster_spawn_types

    def test_reel_043_booster_arm_records_populated(
        self, default_config: MasterConfig,
    ) -> None:
        """Arming adjacent dormant boosters must produce booster_arm_records
        on the step where arming occurred — event_stream.py needs these to
        emit boosterArmInfo (RoyalTumble_EventStream.md §5, §8).

        Uniform strip: step 1 spawns SLB from the full-board cluster. Step 2's
        fresh cluster lands adjacent to that dormant SLB → must be armed, and
        step 2's record must carry the arm snapshot.
        """
        strip = _uniform_strip(default_config, Symbol.H1)
        gen = _make_gen(default_config, strip)
        result = gen.generate("reel_base", 1, _hints(), random.Random(0))

        steps = result.instance.cascade_steps or ()
        # Parallel arrays: types always mirrors records one-to-one.
        for step in steps:
            assert len(step.booster_arm_types) == len(step.booster_arm_records)
            for rec, name in zip(step.booster_arm_records, step.booster_arm_types):
                assert rec.booster_type == name
                assert 0 <= rec.position_reel < default_config.board.num_reels
                assert 0 <= rec.position_row < default_config.board.num_rows

        # At least one step across the cascade must have armed a booster —
        # the uniform-strip scenario guarantees spawn-then-adjacent-cluster.
        assert any(step.booster_arm_records for step in steps)

    def test_reel_044_cascade_steps_carry_cluster_assignments(
        self, default_config: MasterConfig,
    ) -> None:
        """Event stream gates cascade emission on step.clusters truthiness —
        reel records must carry the detected clusters so winInfo,
        updateTumbleWin, updateBoardMultipliers, boosterSpawnInfo,
        boosterArmInfo, and gravitySettle all emit for reel books
        (RoyalTumble_EventStream.md §5, §8).
        """
        strip = _uniform_strip(default_config, Symbol.H1)
        gen = _make_gen(default_config, strip)
        result = gen.generate("reel_base", 1, _hints(), random.Random(0))
        steps = result.instance.cascade_steps or ()

        # Every winning step must expose at least one cluster with a valid
        # symbol, non-empty positions, and size accounting for wilds.
        winning_steps = [s for s in steps if s.step_payout > 0]
        assert winning_steps, "uniform H1 strip must produce at least one win"
        for step in winning_steps:
            assert step.clusters, "winning step must carry ClusterAssignment records"
            for cluster in step.clusters:
                assert cluster.symbol == Symbol.H1
                assert len(cluster.positions) > 0
                assert cluster.size == len(cluster.positions) + len(cluster.wild_positions)

    def test_reel_045_step_0_board_after_is_pre_explosion(
        self, default_config: MasterConfig,
    ) -> None:
        """event_stream.py and validator.py treat cascade_steps[0].board_after
        as the initial filled board shown to the player. With a uniform strip,
        every cell of that board must be the uniform symbol (no spawns or
        gravity have applied yet).
        """
        strip = _uniform_strip(default_config, Symbol.H1)
        gen = _make_gen(default_config, strip)
        result = gen.generate("reel_base", 1, _hints(), random.Random(0))

        steps = result.instance.cascade_steps or ()
        assert steps, "uniform strip must produce at least one cascade step"

        board_after = steps[0].board_after
        for reel in range(default_config.board.num_reels):
            for row in range(default_config.board.num_rows):
                sym = board_after.get(Position(reel, row))
                assert sym == Symbol.H1, (
                    f"steps[0].board_after must be the pre-explosion initial "
                    f"board (uniform H1), got {sym} at ({reel},{row})"
                )

    def test_reel_046_board_after_matches_cluster_positions(
        self, default_config: MasterConfig,
    ) -> None:
        """The symbols at cluster positions on step.board_after must match
        cluster.symbol — this is the board the player sees when winInfo
        highlights these cells.
        """
        strip = _uniform_strip(default_config, Symbol.H1)
        gen = _make_gen(default_config, strip)
        result = gen.generate("reel_base", 1, _hints(), random.Random(0))

        for step in result.instance.cascade_steps or ():
            for cluster in step.clusters:
                for pos in cluster.positions:
                    assert step.board_after.get(pos) == cluster.symbol, (
                        f"cluster position {pos} should carry {cluster.symbol} "
                        f"on board_after, got {step.board_after.get(pos)}"
                    )
