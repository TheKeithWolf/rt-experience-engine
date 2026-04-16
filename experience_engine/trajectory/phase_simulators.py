"""Phase simulators — one per narrative phase shape.

Each simulator consumes the phase's constraints and the evolving SketchState,
produces a TrajectoryWaypoint, and advances the state (board, dormants,
reserve zones). Dispatch uses a dict registry keyed on a phase shape tuple —
no if/elif chains over NarrativePhase fields.

Keeping the simulators behind the PhaseSimulator protocol means the planner
stays Open/Closed: adding a new phase shape is one new class + one registry
entry, nothing else in the package changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..config.schema import MasterConfig
from ..narrative.arc import NarrativePhase
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.gravity import settle
from ..primitives.gravity import GravityDAG
from ..primitives.symbols import Symbol, SymbolTier, symbols_in_tier
from ..step_reasoner.evaluators import ChainEvaluator
from ..step_reasoner.services.forward_simulator import ForwardSimulator
from ..step_reasoner.services.landing_evaluator import BoosterLandingEvaluator
from .data_types import TrajectoryWaypoint
from .sketch_fill import neutral_fill


# ---------------------------------------------------------------------------
# Shared state + dependencies
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SketchDependencies:
    """Bundle of services every simulator needs.

    Groups the injection target so PhaseSimulator implementations can share
    the same constructor contract — no per-class dependency list drift.
    """

    config: MasterConfig
    gravity_dag: GravityDAG
    forward_sim: ForwardSimulator
    landing_eval: BoosterLandingEvaluator
    booster_rules: BoosterRules
    chain_eval: ChainEvaluator
    standard_symbols: tuple[Symbol, ...]


@dataclass(slots=True)
class SketchState:
    """Mutable per-sketch state threaded between phases.

    board — current board snapshot; simulators mutate it through settle +
      neutral_fill to maintain continuity between phases.
    dormant_positions — list of landed booster positions that must survive
      subsequent explosions; simulators update this via trace-through-settle.
    reserve_zone — accumulated cells the previous phases reserved for later
      arming; OR-unioned between phases.
    phase_index — 0-based index into the arc.phases sequence.
    """

    board: Board
    dormant_positions: list[Position] = field(default_factory=list)
    reserve_zone: frozenset[Position] = field(default_factory=frozenset)
    phase_index: int = 0


@runtime_checkable
class PhaseSimulator(Protocol):
    """The one entry point every simulator implements."""

    def simulate(
        self,
        phase: NarrativePhase,
        state: SketchState,
        deps: SketchDependencies,
        rng: random.Random,
    ) -> TrajectoryWaypoint | None:
        ...


# ---------------------------------------------------------------------------
# Shared placement helpers (rule-of-three extraction)
# ---------------------------------------------------------------------------


def _pick_cluster_size(phase: NarrativePhase, rng: random.Random) -> int:
    """Sample a cluster size from the phase's first range.

    The sketch doesn't need to honor every range — it needs *a* representative
    size whose booster maps cleanly into the spawn thresholds. The first
    range is the primary constraint, matching AtlasQuery's midpoint logic.
    """
    first = phase.cluster_sizes[0]
    return rng.randint(first.min_val, first.max_val)


def _pick_cluster_symbol(
    tier: SymbolTier | None,
    deps: SketchDependencies,
    rng: random.Random,
) -> Symbol:
    """Random choice from the standard symbol pool within the phase's tier.

    Tier-ANY and missing tiers both fall back to the full standard symbol
    set; sketches don't optimize payout, they only need a symbol identity
    so settle() has something concrete to move around.
    """
    resolved = tier or SymbolTier.ANY
    candidates = tuple(symbols_in_tier(resolved, deps.config.symbols))
    if not candidates:
        candidates = deps.standard_symbols
    return rng.choice(candidates)


def _pick_cluster_positions(
    size: int,
    board: Board,
    rng: random.Random,
) -> frozenset[Position]:
    """Grab `size` empty cells to represent a cluster shape.

    Sketches don't need orthogonal connectivity — they need the settle to
    reflect the right explosion count in each column. We pick the topmost
    available cells column by column to emulate a "typical" cluster footprint;
    the atlas tier already proved the column profile is viable.
    """
    picks: list[Position] = []
    # Distribute picks across columns for realism, then fall back to any.
    remaining = size
    columns = list(range(board.num_reels))
    rng.shuffle(columns)
    for column in columns:
        if remaining <= 0:
            break
        for row in range(board.num_rows):
            pos = Position(column, row)
            if board.get(pos) is not None:
                picks.append(pos)
                remaining -= 1
                if remaining <= 0:
                    break
    return frozenset(picks)


def _simulate_explosion(
    cluster: frozenset[Position],
    state: SketchState,
    deps: SketchDependencies,
    rng: random.Random,
):
    """Run settle() on the current board, then neutral-fill empties.

    Post-fill the board stays fully populated so the next phase can pick a
    cluster without re-deriving an empty set. Returns the SettleResult so
    callers can record it on the waypoint.
    """
    result = settle(
        deps.gravity_dag, state.board, cluster, deps.config.gravity
    )
    state.board = result.board
    neutral_fill(
        state.board,
        result.empty_positions,
        deps.standard_symbols,
        {s: 1.0 for s in deps.standard_symbols},
        rng,
    )
    return result


def _trace_dormants(
    state: SketchState,
    settle_result,
) -> None:
    """Follow every dormant booster through the settle's moves.

    Boosters destroyed by the explosion (their position appears as a source
    in the move stream followed by their pre-position in the exploded set)
    are dropped from the tracking list.
    """
    updated: list[Position] = []
    for dormant in state.dormant_positions:
        current = dormant
        destroyed = False
        for pass_moves in settle_result.move_steps:
            for _, source, dest in pass_moves:
                if source == current:
                    current = dest
                    break
        # If the dormant lies inside the original explosion set it's destroyed
        # (settle marks those cells empty, so they don't get traced).
        if state.board.get(current) is None:
            destroyed = True
        if not destroyed:
            updated.append(current)
    state.dormant_positions = updated


def _build_waypoint(
    phase_index: int,
    cluster: frozenset[Position],
    cluster_symbol: Symbol,
    settle_result,
    booster_type: str | None,
    booster_spawn_pos: Position | None,
    booster_landing_pos: Position | None,
    landing_context,
    landing_score: float,
    reserve_zone: frozenset[Position],
    chain_target_pos: Position | None,
) -> TrajectoryWaypoint:
    """Factory — collapses the waypoint construction boilerplate."""
    return TrajectoryWaypoint(
        phase_index=phase_index,
        cluster_region=cluster,
        cluster_symbol=cluster_symbol,
        booster_type=booster_type,
        booster_spawn_pos=booster_spawn_pos,
        booster_landing_pos=booster_landing_pos,
        settle_result=settle_result,
        landing_context=landing_context,
        landing_score=landing_score,
        reserve_zone=reserve_zone,
        chain_target_pos=chain_target_pos,
        seed_hints={},
    )


# ---------------------------------------------------------------------------
# Concrete simulators
# ---------------------------------------------------------------------------


class CascadePhaseSimulator:
    """Default simulator — a phase that only grows a cluster (no booster).

    Registered as the fallback so every phase resolves to *some* simulator,
    preserving Open/Closed: unknown shapes still generate a waypoint.
    """

    def simulate(
        self,
        phase: NarrativePhase,
        state: SketchState,
        deps: SketchDependencies,
        rng: random.Random,
    ) -> TrajectoryWaypoint | None:
        size = _pick_cluster_size(phase, rng)
        cluster = _pick_cluster_positions(size, state.board, rng)
        if len(cluster) != size:
            return None
        symbol = _pick_cluster_symbol(phase.cluster_symbol_tier, deps, rng)
        result = _simulate_explosion(cluster, state, deps, rng)
        _trace_dormants(state, result)
        return _build_waypoint(
            phase_index=state.phase_index,
            cluster=cluster,
            cluster_symbol=symbol,
            settle_result=result,
            booster_type=None,
            booster_spawn_pos=None,
            booster_landing_pos=None,
            landing_context=None,
            landing_score=1.0,  # No booster → no landing penalty.
            reserve_zone=state.reserve_zone,
            chain_target_pos=None,
        )


class SpawnPhaseSimulator:
    """Simulator for phases that spawn a booster via cluster_size threshold."""

    def simulate(
        self,
        phase: NarrativePhase,
        state: SketchState,
        deps: SketchDependencies,
        rng: random.Random,
    ) -> TrajectoryWaypoint | None:
        size = _pick_cluster_size(phase, rng)
        cluster = _pick_cluster_positions(size, state.board, rng)
        if len(cluster) != size:
            return None
        symbol = _pick_cluster_symbol(phase.cluster_symbol_tier, deps, rng)

        booster_symbol = deps.booster_rules.booster_type_for_size(size)
        if booster_symbol is None:
            return None
        booster_type = booster_symbol.name
        centroid = deps.booster_rules.compute_centroid(cluster)
        ctx = deps.landing_eval.evaluate(cluster, state.board, booster_type)
        score = deps.landing_eval.score(ctx)

        result = _simulate_explosion(cluster, state, deps, rng)
        _trace_dormants(state, result)
        # The spawned booster becomes a dormant tracked through later phases.
        state.dormant_positions.append(ctx.landing_position)
        return _build_waypoint(
            phase_index=state.phase_index,
            cluster=cluster,
            cluster_symbol=symbol,
            settle_result=result,
            booster_type=booster_type,
            booster_spawn_pos=centroid,
            booster_landing_pos=ctx.landing_position,
            landing_context=ctx,
            landing_score=score,
            reserve_zone=state.reserve_zone,
            chain_target_pos=None,
        )


class ArmPhaseSimulator:
    """Simulator for phases whose cluster must land adjacent to a dormant."""

    def simulate(
        self,
        phase: NarrativePhase,
        state: SketchState,
        deps: SketchDependencies,
        rng: random.Random,
    ) -> TrajectoryWaypoint | None:
        if not state.dormant_positions:
            return None
        size = _pick_cluster_size(phase, rng)
        cluster = _pick_cluster_positions(size, state.board, rng)
        if len(cluster) != size:
            return None
        symbol = _pick_cluster_symbol(phase.cluster_symbol_tier, deps, rng)

        target_dormant = state.dormant_positions[-1]
        # Score the armed cluster by landing adjacency to the dormant.
        booster_symbol = deps.booster_rules.booster_type_for_size(size)
        landing_context = None
        score = 0.5  # Neutral default; arming without a new booster is feasible.
        if booster_symbol is not None:
            booster_type = booster_symbol.name
            landing_context = deps.landing_eval.evaluate(
                cluster, state.board, booster_type
            )
            score = deps.landing_eval.score(landing_context)

        result = _simulate_explosion(cluster, state, deps, rng)
        _trace_dormants(state, result)
        return _build_waypoint(
            phase_index=state.phase_index,
            cluster=cluster,
            cluster_symbol=symbol,
            settle_result=result,
            booster_type=booster_symbol.name if booster_symbol else None,
            booster_spawn_pos=None,
            booster_landing_pos=landing_context.landing_position if landing_context else None,
            landing_context=landing_context,
            landing_score=score,
            reserve_zone=state.reserve_zone,
            chain_target_pos=target_dormant,
        )


class FirePhaseSimulator:
    """Simulator for phases that fire a chain-initiator at a chain target."""

    def simulate(
        self,
        phase: NarrativePhase,
        state: SketchState,
        deps: SketchDependencies,
        rng: random.Random,
    ) -> TrajectoryWaypoint | None:
        if not state.dormant_positions:
            return None
        size = _pick_cluster_size(phase, rng)
        cluster = _pick_cluster_positions(size, state.board, rng)
        if len(cluster) != size:
            return None
        symbol = _pick_cluster_symbol(phase.cluster_symbol_tier, deps, rng)

        # The chain target is any still-dormant booster other than the most
        # recent one (which is presumably the firing initiator).
        if len(state.dormant_positions) >= 2:
            target = state.dormant_positions[-2]
        else:
            target = state.dormant_positions[-1]

        booster_symbol = deps.booster_rules.booster_type_for_size(size)
        landing_context = None
        score = 0.5
        if booster_symbol is not None:
            booster_type = booster_symbol.name
            landing_context = deps.landing_eval.evaluate(
                cluster, state.board, booster_type
            )
            score = deps.landing_eval.score(landing_context)

        result = _simulate_explosion(cluster, state, deps, rng)
        _trace_dormants(state, result)
        return _build_waypoint(
            phase_index=state.phase_index,
            cluster=cluster,
            cluster_symbol=symbol,
            settle_result=result,
            booster_type=booster_symbol.name if booster_symbol else None,
            booster_spawn_pos=None,
            booster_landing_pos=landing_context.landing_position if landing_context else None,
            landing_context=landing_context,
            landing_score=score,
            reserve_zone=state.reserve_zone,
            chain_target_pos=target,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _phase_shape(phase: NarrativePhase) -> tuple[bool, bool, bool]:
    """(spawns, arms, fires) presence flags — the registry lookup key."""
    return (
        phase.spawns is not None,
        phase.arms is not None,
        phase.fires is not None,
    )


def build_default_phase_registry() -> dict[tuple[bool, bool, bool], PhaseSimulator]:
    """Default mapping from phase shape to simulator.

    Every shape falls through to CascadePhaseSimulator if no specialized
    handler matches — this preserves the design doc's "graceful degradation":
    even un-recognized shapes produce a waypoint.
    """
    spawn = SpawnPhaseSimulator()
    arm = ArmPhaseSimulator()
    fire = FirePhaseSimulator()
    cascade = CascadePhaseSimulator()
    return {
        (True, False, False): spawn,
        (True, True, False): spawn,
        (False, True, False): arm,
        (False, True, True): fire,
        (False, False, True): fire,
        (True, False, True): fire,
        (True, True, True): fire,
        (False, False, False): cascade,
    }


def resolve_simulator(
    phase: NarrativePhase,
    registry: dict[tuple[bool, bool, bool], PhaseSimulator],
) -> PhaseSimulator:
    """Dict dispatch — no if/elif chain over phase shape.

    Falls back to CascadePhaseSimulator when the shape isn't registered
    (robustness over strictness: the planner can still emit a waypoint).
    """
    key = _phase_shape(phase)
    if key in registry:
        return registry[key]
    return registry[(False, False, False)]
