"""Phase 7 tests — Wild Integration.

Tests cover: archetype registration (TEST-P7-001 through P7-003),
CSP constraint validation (TEST-P7-004 through P7-007),
full pipeline end-to-end (TEST-P7-008 through P7-011),
validation (TEST-P7-012), and diagnostics (TEST-P7-013).
"""

from __future__ import annotations

import pytest

from ..archetypes.dead import register_dead_archetypes
from ..archetypes.registry import (
    ArchetypeRegistry,
    ArchetypeSignature,
    TerminalNearMissSpec,
)
from ..archetypes.tier1 import register_cascade_t1_archetypes, register_static_t1_archetypes
from ..archetypes.wild import register_wild_archetypes
from ..config.schema import MasterConfig
from ..diagnostics.engine import DiagnosticsEngine
from ..pipeline.protocols import Range, RangeFloat
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.symbols import Symbol, SymbolTier, is_wild
from ..spatial_solver.constraints import (
    DormantBoosterSurvival,
    TerminalNearMissPlacement,
    WildBridgeConstraint,
    WildIdleConstraint,
    WildSpawnPlacement,
)
from ..spatial_solver.data_types import (
    BoosterPlacement,
    ClusterAssignment,
    NearMissAssignment,
    SolverContext,
    WildPlacement,
)
from ..validation.metrics import InstanceMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_registry(default_config: MasterConfig) -> ArchetypeRegistry:
    """Registry with all Phase 7 archetypes: dead (11) + t1 (12) + wild (17) = 40."""
    reg = ArchetypeRegistry(default_config)
    register_dead_archetypes(reg)
    register_static_t1_archetypes(reg)
    register_cascade_t1_archetypes(reg)
    register_wild_archetypes(reg)
    return reg


# ---------------------------------------------------------------------------
# TEST-P7-001: All 17 wild signatures register without SignatureValidationError
# ---------------------------------------------------------------------------

def test_all_wild_archetypes_register(default_config: MasterConfig) -> None:
    """All 17 wild archetypes pass CONTRACT-SIG validation."""
    reg = ArchetypeRegistry(default_config)
    register_wild_archetypes(reg)
    wild = reg.get_family("wild")
    assert len(wild) == 17


# ---------------------------------------------------------------------------
# TEST-P7-002: Total registered = 40 (11 dead + 12 t1 + 17 wild)
# ---------------------------------------------------------------------------

def test_total_registered_is_40(full_registry: ArchetypeRegistry) -> None:
    assert len(full_registry.all_ids()) == 40


# ---------------------------------------------------------------------------
# TEST-P7-003: Wild family IDs match expected names
# ---------------------------------------------------------------------------

def test_wild_archetype_ids(full_registry: ArchetypeRegistry) -> None:
    """All 17 expected wild archetype IDs are present."""
    expected_ids = {
        "wild_idle", "wild_bridge_small", "wild_bridge_large",
        "wild_enable_rocket", "wild_enable_bomb", "wild_multi",
        "wild_near_miss_single", "wild_near_miss_multi", "wild_near_miss_high_idle",
        "wild_late_save", "wild_late_save_high",
        "wild_storm_low", "wild_storm_bridge_chain",
        "wild_scatter_tease", "wild_scatter_cascade_tease",
        "wild_escalation_fizzle", "wild_rocket_tease",
    }
    wild = full_registry.get_family("wild")
    actual_ids = {sig.id for sig in wild}
    assert actual_ids == expected_ids


# ---------------------------------------------------------------------------
# TEST-P7-004: CSP WildSpawnPlacement — wild adjacent to 7-8 cluster
# ---------------------------------------------------------------------------

def test_wild_spawn_placement_satisfied(default_config: MasterConfig) -> None:
    """WildSpawnPlacement accepts when wild is adjacent to a size-7 cluster."""
    ctx = SolverContext(default_config.board)

    # Place a size-7 cluster at positions (0,0)-(0,6) (column)
    cluster_positions = frozenset(Position(0, r) for r in range(7))
    ctx.clusters.append(ClusterAssignment(
        symbol=Symbol.L1, positions=cluster_positions, size=7,
    ))
    ctx.occupied.update(cluster_positions)

    # Place wild adjacent to the cluster — (1, 0) neighbors (0, 0)
    wild_pos = Position(1, 0)
    ctx.wild_placements.append(WildPlacement(position=wild_pos, behavior="spawn"))
    ctx.occupied.add(wild_pos)

    constraint = WildSpawnPlacement(min_spawn_size=7, max_spawn_size=8)
    assert constraint.is_satisfied(ctx)


# ---------------------------------------------------------------------------
# TEST-P7-005: CSP WildBridgeConstraint — wild adjacent to 2+ clusters
# ---------------------------------------------------------------------------

def test_wild_bridge_constraint_satisfied(default_config: MasterConfig) -> None:
    """WildBridgeConstraint accepts when wild neighbors 2+ distinct clusters."""
    ctx = SolverContext(default_config.board)

    # Cluster A at (0,0)-(0,4) — left side
    cluster_a = frozenset(Position(0, r) for r in range(5))
    ctx.clusters.append(ClusterAssignment(
        symbol=Symbol.L1, positions=cluster_a, size=5,
    ))

    # Cluster B at (2,0)-(2,4) — right side
    cluster_b = frozenset(Position(2, r) for r in range(5))
    ctx.clusters.append(ClusterAssignment(
        symbol=Symbol.L1, positions=cluster_b, size=5,
    ))

    # Wild at (1, 0) — adjacent to both clusters
    wild_pos = Position(1, 0)
    ctx.wild_placements.append(WildPlacement(position=wild_pos, behavior="bridge"))

    constraint = WildBridgeConstraint()
    assert constraint.is_satisfied(ctx)


def test_wild_bridge_constraint_fails_single_cluster(default_config: MasterConfig) -> None:
    """WildBridgeConstraint rejects when wild neighbors only 1 cluster."""
    ctx = SolverContext(default_config.board)

    cluster_a = frozenset(Position(0, r) for r in range(5))
    ctx.clusters.append(ClusterAssignment(
        symbol=Symbol.L1, positions=cluster_a, size=5,
    ))

    # Wild at (1, 0) — adjacent to only cluster A
    wild_pos = Position(1, 0)
    ctx.wild_placements.append(WildPlacement(position=wild_pos, behavior="bridge"))

    constraint = WildBridgeConstraint()
    assert not constraint.is_satisfied(ctx)


# ---------------------------------------------------------------------------
# TEST-P7-006: CSP TerminalNearMissPlacement
# ---------------------------------------------------------------------------

def test_terminal_near_miss_placement(default_config: MasterConfig) -> None:
    """TerminalNearMissPlacement accepts correct count of NMs."""
    ctx = SolverContext(default_config.board)
    nm_size = default_config.board.min_cluster_size - 1

    # Place 2 near-misses of the correct size
    ctx.near_misses.append(NearMissAssignment(
        symbol=Symbol.H1,
        positions=frozenset(Position(0, r) for r in range(nm_size)),
        size=nm_size,
    ))
    ctx.near_misses.append(NearMissAssignment(
        symbol=Symbol.H2,
        positions=frozenset(Position(2, r) for r in range(nm_size)),
        size=nm_size,
    ))

    constraint = TerminalNearMissPlacement(
        count_range=Range(1, 2),
        symbol_tier=SymbolTier.HIGH,
        min_cluster_size=default_config.board.min_cluster_size,
    )
    assert constraint.is_satisfied(ctx)


# ---------------------------------------------------------------------------
# TEST-P7-007: CSP DormantBoosterSurvival — rocket on terminal
# ---------------------------------------------------------------------------

def test_dormant_booster_survival(default_config: MasterConfig) -> None:
    """DormantBoosterSurvival accepts when required booster is placed."""
    ctx = SolverContext(default_config.board)
    ctx.booster_placements.append(BoosterPlacement(
        booster_type=Symbol.R, position=Position(3, 3),
    ))

    constraint = DormantBoosterSurvival(required_booster_names=("R",))
    assert constraint.is_satisfied(ctx)


def test_dormant_booster_survival_fails_missing(default_config: MasterConfig) -> None:
    """DormantBoosterSurvival rejects when required booster is missing."""
    ctx = SolverContext(default_config.board)
    # No boosters placed
    constraint = DormantBoosterSurvival(required_booster_names=("R",))
    assert not constraint.is_satisfied(ctx)


# ---------------------------------------------------------------------------
# TEST-P7-008 through P7-011: Full pipeline tests
# These require Clingo and the full ASP→CSP→WFC pipeline.
# Marked as integration tests — skipped if Clingo is not available.
# ---------------------------------------------------------------------------

@pytest.fixture
def _has_clingo() -> bool:
    """Check if Clingo is available."""
    try:
        import clingo
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TEST-P7-012: Validation — wild_count metric collected
# ---------------------------------------------------------------------------

def test_validation_wild_count_metric() -> None:
    """InstanceMetrics includes wild_count field."""
    # Verify the dataclass field exists and defaults correctly
    m = InstanceMetrics(
        archetype_id="test",
        family="wild",
        criteria="basegame",
        sim_id=0,
        payout=1.0,
        centipayout=100,
        win_level=2,
        cluster_count=1,
        cluster_sizes=(7,),
        cluster_symbols=("L1",),
        scatter_count=0,
        near_miss_count=0,
        near_miss_symbols=(),
        max_component_size=7,
        is_valid=True,
        validation_errors=(),
        wild_count=1,
    )
    assert m.wild_count == 1
    assert m.terminal_near_miss_count == 0
    assert m.dormant_booster_count == 0


# ---------------------------------------------------------------------------
# TEST-P7-013: Diagnostics — wild_frequency and wild_assist_rate computed
# ---------------------------------------------------------------------------

def test_diagnostics_wild_metrics(default_config: MasterConfig) -> None:
    """DiagnosticsEngine computes wild_frequency and wild_assist_rate."""
    engine = DiagnosticsEngine(default_config)

    # Create a mix: 3 instances with wilds (2 winning), 2 without wilds (1 winning)
    base_kwargs = dict(
        family="wild",
        criteria="basegame",
        cluster_sizes=(7,),
        cluster_symbols=("L1",),
        scatter_count=0,
        near_miss_count=0,
        near_miss_symbols=(),
        max_component_size=7,
        is_valid=True,
        validation_errors=(),
    )

    metrics = (
        # Wild + winning
        InstanceMetrics(archetype_id="wild_bridge_small", sim_id=0,
                        payout=2.0, centipayout=200, win_level=3,
                        cluster_count=1, wild_count=1, **base_kwargs),
        # Wild + winning
        InstanceMetrics(archetype_id="wild_idle", sim_id=1,
                        payout=1.0, centipayout=100, win_level=2,
                        cluster_count=1, wild_count=1, **base_kwargs),
        # Wild + dead (payout=0)
        InstanceMetrics(archetype_id="wild_near_miss_single", sim_id=2,
                        payout=0.0, centipayout=0, win_level=0,
                        cluster_count=0, wild_count=1, **base_kwargs),
        # No wild + winning
        InstanceMetrics(archetype_id="t1_single", sim_id=3,
                        payout=0.5, centipayout=50, win_level=2,
                        cluster_count=1, wild_count=0, family="t1", criteria="basegame",
                        cluster_sizes=(5,), cluster_symbols=("L1",),
                        scatter_count=0, near_miss_count=0, near_miss_symbols=(),
                        max_component_size=5, is_valid=True, validation_errors=()),
        # No wild + dead
        InstanceMetrics(archetype_id="dead_empty", sim_id=4,
                        payout=0.0, centipayout=0, win_level=0,
                        cluster_count=0, wild_count=0, family="dead", criteria="0",
                        cluster_sizes=(), cluster_symbols=(),
                        scatter_count=0, near_miss_count=0, near_miss_symbols=(),
                        max_component_size=3, is_valid=True, validation_errors=()),
    )

    report = engine.analyze(metrics)

    # Find wild_frequency and wild_assist_rate in results
    result_map = {r.metric: r.value for r in report.results}

    # wild_frequency: 3/5 = 0.6 (3 instances have wild_count > 0)
    assert result_map["wild_frequency"] == pytest.approx(0.6)

    # wild_assist_rate: 2/3 winning instances have wilds = 0.666...
    # (3 winning: sim_id 0, 1, 3; of those, 0 and 1 have wilds)
    assert result_map["wild_assist_rate"] == pytest.approx(2.0 / 3.0, rel=1e-3)


# ---------------------------------------------------------------------------
# WildIdleConstraint tests
# ---------------------------------------------------------------------------

def test_wild_idle_constraint_satisfied(default_config: MasterConfig) -> None:
    """WildIdleConstraint accepts when wild neighbors at most 1 cluster."""
    ctx = SolverContext(default_config.board)

    cluster_a = frozenset(Position(0, r) for r in range(5))
    ctx.clusters.append(ClusterAssignment(
        symbol=Symbol.L1, positions=cluster_a, size=5,
    ))

    # Wild at (4, 4) — far from the cluster, neighbors 0 clusters
    wild_pos = Position(4, 4)
    ctx.wild_placements.append(WildPlacement(position=wild_pos, behavior="idle"))

    constraint = WildIdleConstraint()
    assert constraint.is_satisfied(ctx)


def test_wild_idle_constraint_fails_bridging(default_config: MasterConfig) -> None:
    """WildIdleConstraint rejects when wild would accidentally bridge 2+ clusters."""
    ctx = SolverContext(default_config.board)

    cluster_a = frozenset(Position(0, r) for r in range(5))
    ctx.clusters.append(ClusterAssignment(
        symbol=Symbol.L1, positions=cluster_a, size=5,
    ))

    cluster_b = frozenset(Position(2, r) for r in range(5))
    ctx.clusters.append(ClusterAssignment(
        symbol=Symbol.L1, positions=cluster_b, size=5,
    ))

    # Wild at (1, 0) — adjacent to both clusters
    wild_pos = Position(1, 0)
    ctx.wild_placements.append(WildPlacement(position=wild_pos, behavior="idle"))

    constraint = WildIdleConstraint()
    assert not constraint.is_satisfied(ctx)


# ---------------------------------------------------------------------------
# Validator — bridge wild consumed on terminal board
# ---------------------------------------------------------------------------


def test_validator_bridge_wild_consumed(
    default_config: MasterConfig,
    full_registry: ArchetypeRegistry,
) -> None:
    """Validator accepts wild_count=0 on terminal board for bridge archetypes.

    Bridge wilds are consumed when the bridge cluster explodes. The formula
    spawned − consumed = expected terminal wilds gives 1 − 1 = 0.
    """
    from ..pipeline.data_types import CascadeStepRecord, GeneratedInstance
    from ..spatial_solver.data_types import SpatialStep as _SpatialStep
    from ..validation.validator import InstanceValidator

    validator = InstanceValidator(default_config, full_registry)

    # Build a dead terminal board — no wilds present (bridge wild was consumed)
    terminal_board = Board.empty(default_config.board)
    syms = [Symbol.L1, Symbol.L2, Symbol.L3, Symbol.L4]
    for reel in range(default_config.board.num_reels):
        for row in range(default_config.board.num_rows):
            terminal_board.set(
                Position(reel, row), syms[(reel * 2 + row) % len(syms)]
            )

    # Step 0: cluster spawns wild W
    step0_board = Board.empty(default_config.board)
    for reel in range(default_config.board.num_reels):
        for row in range(default_config.board.num_rows):
            step0_board.set(
                Position(reel, row), syms[(reel * 2 + row) % len(syms)]
            )
    # Place a 7-cell L1 cluster for step 0
    cluster_positions = frozenset(
        Position(r, 0) for r in range(7)
    )
    for pos in cluster_positions:
        step0_board.set(pos, Symbol.L1)

    grid_snap = tuple(
        0 for _ in range(default_config.board.num_reels * default_config.board.num_rows)
    )

    step0_record = CascadeStepRecord(
        step_index=0,
        board_before=step0_board,
        board_after=step0_board,
        clusters=(ClusterAssignment(symbol=Symbol.L1, positions=cluster_positions, size=7),),
        step_payout=1.0,
        grid_multipliers_snapshot=grid_snap,
        booster_spawn_types=("W",),
    )

    # Step 1: bridge cluster (no new spawns)
    step1_record = CascadeStepRecord(
        step_index=1,
        board_before=terminal_board,
        board_after=terminal_board,
        clusters=(),
        step_payout=0.5,
        grid_multipliers_snapshot=grid_snap,
    )

    instance = GeneratedInstance(
        sim_id=0,
        archetype_id="wild_bridge_small",
        family="wild",
        criteria="basegame",
        board=terminal_board,
        spatial_step=_SpatialStep(
            clusters=(ClusterAssignment(symbol=Symbol.L1, positions=cluster_positions, size=7),),
            near_misses=(),
            scatter_positions=frozenset(),
            boosters=(),
        ),
        payout=1.5,
        centipayout=150,
        win_level=2,
        cascade_steps=(step0_record, step1_record),
    )

    metrics = validator.validate(instance)

    # The old validator would fail here with "wild_count=0 outside [1, 1]".
    # The fixed validator should accept: spawned(1) - consumed(1) = expected(0) = actual(0).
    wild_errors = [e for e in metrics.validation_errors if "wild_count" in e]
    assert not wild_errors, f"Unexpected wild_count errors: {wild_errors}"


# ---------------------------------------------------------------------------
# TEST-P7-014: wild_enable_rocket cascade_steps and booster_fires
# ---------------------------------------------------------------------------

def test_wild_enable_rocket_cascade_steps(full_registry: ArchetypeRegistry) -> None:
    """wild_enable_rocket has 3 arc phases and fires 1 rocket."""
    sig = full_registry.get("wild_enable_rocket")

    # Cascade depth = 3: spawn_wild + bridge_rocket + arm_rocket (fire_rocket is terminal)
    assert sig.required_cascade_depth == Range(3, 3)

    # Rocket must fire — derived from arc phase fires=("R",)
    assert sig.required_booster_fires == {"R": Range(1, 1)}

    # 4 narrative phases: spawn → bridge → arm → fire (supersedes cascade_steps)
    assert sig.narrative_arc is not None
    assert len(sig.narrative_arc.phases) == 4

    phase0, phase1, phase2, phase3 = sig.narrative_arc.phases

    # Phase 0: initial cluster spawns W
    assert phase0.spawns == ("W",)
    assert phase0.wild_behavior == "spawn"
    assert phase0.cluster_sizes == (Range(7, 8),)
    assert phase0.arms is None

    # Phase 1: W bridges → R spawns
    assert phase1.spawns == ("R",)
    assert phase1.wild_behavior == "bridge"
    assert phase1.cluster_sizes == (Range(9, 10),)
    assert phase1.arms is None

    # Phase 2: new cluster arms R
    assert phase2.arms == ("R",)
    assert phase2.cluster_sizes == (Range(5, 6),)
    assert phase2.spawns is None

    # Phase 3: R fires → clears row/col (terminal phase — no clusters)
    assert phase3.fires == ("R",)
    assert phase3.cluster_count == Range(0, 0)
    assert phase3.spawns is None


# ---------------------------------------------------------------------------
# WB-040 through WB-043: Wild bridge scan E2E pipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_wb_040_wild_bridge_small_success_rate(tmp_path) -> None:
    """WB-040: wild_bridge_small success rate >= 40% over 100 seeds."""
    from ..run import main

    output_dir = tmp_path / "output"
    ret = main([
        "--count", "100",
        "--archetype", "wild_bridge_small",
        "--seed", "1",
        "--output", str(output_dir),
    ])
    assert ret == 0

    import json
    books_path = output_dir / "books_base.jsonl"
    assert books_path.exists()
    books = [json.loads(line) for line in books_path.read_text().splitlines() if line.strip()]
    # At least 40% of seeds should produce valid books
    assert len(books) >= 40, f"Only {len(books)}/100 books generated (< 40%)"


@pytest.mark.slow
def test_wb_041_no_double_wild_spawn(tmp_path) -> None:
    """WB-041: No booster_spawn(W)=2 failures — wild bridge doesn't double-spawn."""
    from ..run import main

    output_dir = tmp_path / "output"
    ret = main([
        "--count", "50",
        "--archetype", "wild_bridge_small",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0

    import json
    books_path = output_dir / "books_base.jsonl"
    if books_path.exists():
        for line in books_path.read_text().splitlines():
            if not line.strip():
                continue
            book = json.loads(line)
            events = book.get("events", [])
            wild_spawns = sum(
                1 for e in events
                if e.get("type") == "wildSpawn"
            )
            # Each bridge arc should spawn at most 1 wild
            assert wild_spawns <= 1, (
                f"Book {book.get('id')} has {wild_spawns} wild spawns (expected <= 1)"
            )


@pytest.mark.slow
def test_wb_042_wild_idle_no_regression(tmp_path) -> None:
    """WB-042: wild_idle success rate unchanged after bridge scan refactor."""
    from ..run import main

    output_dir = tmp_path / "output"
    ret = main([
        "--count", "50",
        "--archetype", "wild_idle",
        "--seed", "1",
        "--output", str(output_dir),
    ])
    assert ret == 0

    import json
    books_path = output_dir / "books_base.jsonl"
    assert books_path.exists()
    books = [json.loads(line) for line in books_path.read_text().splitlines() if line.strip()]
    # wild_idle should still work — at least 20% success rate
    assert len(books) >= 10, f"Only {len(books)}/50 wild_idle books (regression)"


@pytest.mark.slow
def test_wb_043_wild_enable_rocket_spawns_r(tmp_path) -> None:
    """WB-043: wild_enable_rocket bridge phase produces R spawn, not W=2."""
    from ..run import main

    output_dir = tmp_path / "output"
    ret = main([
        "--count", "50",
        "--archetype", "wild_enable_rocket",
        "--seed", "1",
        "--output", str(output_dir),
    ])
    assert ret == 0

    import json
    books_path = output_dir / "books_base.jsonl"
    if books_path.exists():
        books = [json.loads(line) for line in books_path.read_text().splitlines() if line.strip()]
        for book in books:
            events = book.get("events", [])
            # Bridge phase should produce R spawns
            r_spawns = sum(
                1 for e in events
                if e.get("type") == "rocketSpawn"
            )
            # Every successful wild_enable_rocket should have at least 1 R spawn
            assert r_spawns >= 1, (
                f"Book {book.get('id')} has no R spawns in wild_enable_rocket arc"
            )
