"""Booster landing evaluator — shared service for post-gravity landing prediction and scoring.

Forward-simulates where a booster will land after cluster explosion + gravity,
then delegates scoring to per-booster-type criteria via dict dispatch.
Strategies call this to evaluate candidate cluster shapes before committing.

Physics reuse: centroid from BoosterRules, gravity from ForwardSimulator.
No duplicated computation across strategies.

Used by: InitialCluster, BoosterSetup, BoosterArm (injected via registry).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ...config.schema import BoardConfig
from ...primitives.board import Position, orthogonal_neighbors
from ...primitives.booster_rules import BoosterRules
from ...primitives.gravity import SettleResult
from ...primitives.symbols import Symbol
from ...variance.hints import VarianceHints
from .forward_simulator import ForwardSimulator
from .landing_criteria import LandingCriterion

# Re-export Board for type usage in evaluate()
from ...primitives.board import Board


# ---------------------------------------------------------------------------
# Data classes — immutable snapshots produced by evaluate()
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ShapeStats:
    """Precomputed cluster shape properties — avoids redundant computation in criteria.

    Orientation delegates to BoosterRules.compute_rocket_orientation() so
    rockets inherit the config-driven tie-breaking rule.
    """

    col_span: int                       # max(reels) - min(reels) + 1
    row_span: int                       # max(rows) - min(rows) + 1
    orientation: str                    # "H" or "V" (from BoosterRules)
    centroid: Position                  # BoosterRules.compute_centroid()
    columns_used: frozenset[int]        # unique reel indices in the cluster
    max_depth_per_column: dict[int, int]  # reel → deepest (highest row index) in cluster


@dataclass(frozen=True, slots=True)
class LandingContext:
    """Everything a criterion needs to score a booster's post-gravity position.

    Produced once per evaluate() call, consumed by criterion.score().
    Immutable — strategies can safely store and compare contexts.
    """

    booster_type: str
    cluster_positions: frozenset[Position]
    landing_position: Position
    adjacent_refill: tuple[Position, ...]   # refill-zone cells adjacent to landing
    all_refill: tuple[Position, ...]        # full refill zone after settle
    settle_result: SettleResult
    cluster_shape_stats: ShapeStats


# ---------------------------------------------------------------------------
# Evaluator service — one instance, shared across strategies
# ---------------------------------------------------------------------------

class BoosterLandingEvaluator:
    """Predicts booster landing positions and scores them for next-step viability.

    Shared service injected into strategies that spawn boosters. Constructed
    once at registry build time.

    Uses ForwardSimulator for gravity prediction and BoosterRules for centroid/
    orientation — no duplicated physics.
    """

    __slots__ = ("_forward_sim", "_booster_rules", "_board_config", "_criteria")

    def __init__(
        self,
        forward_sim: ForwardSimulator,
        booster_rules: BoosterRules,
        board_config: BoardConfig,
        criteria: dict[str, LandingCriterion],
    ) -> None:
        self._forward_sim = forward_sim
        self._booster_rules = booster_rules
        self._board_config = board_config
        self._criteria = criteria

    @property
    def board_config(self) -> BoardConfig:
        return self._board_config

    def evaluate(
        self,
        cluster_positions: frozenset[Position],
        board: Board,
        booster_type: str,
    ) -> LandingContext:
        """Forward-simulate and build the landing context.

        Pure computation — no side effects. Strategies call this in a
        loop to score candidate cluster shapes.
        """
        centroid = self._booster_rules.compute_centroid(cluster_positions)
        landing = self._forward_sim.predict_booster_landing(
            centroid, board, cluster_positions,
        )
        settle = self._forward_sim.simulate_explosion(
            board, cluster_positions,
        )

        # Refill cells orthogonally adjacent to the booster's landing position
        landing_neighbors = set(
            orthogonal_neighbors(landing, self._board_config)
        )
        adjacent_refill = tuple(
            pos for pos in settle.empty_positions if pos in landing_neighbors
        )

        shape_stats = self._compute_shape_stats(cluster_positions, centroid)

        return LandingContext(
            booster_type=booster_type,
            cluster_positions=cluster_positions,
            landing_position=landing,
            adjacent_refill=adjacent_refill,
            all_refill=tuple(settle.empty_positions),
            settle_result=settle,
            cluster_shape_stats=shape_stats,
        )

    def score(self, ctx: LandingContext) -> float:
        """Score the landing using the registered criterion for this booster type.

        Returns 1.0 for unknown types — no criterion means no constraint.
        """
        criterion = self._criteria.get(ctx.booster_type)
        if criterion is None:
            return 1.0
        return criterion.score(ctx)

    def evaluate_and_score(
        self,
        cluster_positions: frozenset[Position],
        board: Board,
        booster_type: str,
    ) -> tuple[LandingContext, float]:
        """Convenience: evaluate + score in one call."""
        ctx = self.evaluate(cluster_positions, board, booster_type)
        return ctx, self.score(ctx)

    def find_best_shape(
        self,
        candidates: Iterable[tuple[frozenset[Position], Board]],
        booster_type: str,
        threshold: float = 0.8,
    ) -> tuple[LandingContext, float] | None:
        """Score multiple candidate shapes, return the best above threshold.

        candidates: iterable of (cluster_positions, hypothetical_board) pairs.
        Strategies generate these however they like — BFS with varying bias,
        multiple rng seeds, etc.

        Returns None if no candidates were provided.
        Early-exits when a candidate meets the threshold.
        """
        best_ctx: LandingContext | None = None
        best_score = -1.0

        for positions, board in candidates:
            ctx, score = self.evaluate_and_score(positions, board, booster_type)
            if score > best_score:
                best_ctx = ctx
                best_score = score
            if score >= threshold:
                break

        return (best_ctx, best_score) if best_ctx is not None else None

    # -- Private helpers ---------------------------------------------------

    def _compute_shape_stats(
        self,
        positions: frozenset[Position],
        centroid: Position,
    ) -> ShapeStats:
        """Derive shape properties from cluster positions.

        Computed once per evaluate() call — criteria read from the frozen result.
        """
        reels = [p.reel for p in positions]
        rows = [p.row for p in positions]

        col_span = max(reels) - min(reels) + 1
        row_span = max(rows) - min(rows) + 1

        columns_used = frozenset(reels)

        # Deepest row per column — useful for predicting how far below the
        # centroid the refill zone extends after explosion
        depth_map: dict[int, int] = {}
        for p in positions:
            if p.reel not in depth_map or p.row > depth_map[p.reel]:
                depth_map[p.reel] = p.row

        orientation = self._booster_rules.compute_rocket_orientation(positions)

        return ShapeStats(
            col_span=col_span,
            row_span=row_span,
            orientation=orientation,
            centroid=centroid,
            columns_used=columns_used,
            max_depth_per_column=depth_map,
        )


# ---------------------------------------------------------------------------
# Reshape bias — progressive vertical concentration (shared, not per-strategy)
# ---------------------------------------------------------------------------

def compute_reshape_bias(
    variance: VarianceHints,
    board_config: BoardConfig,
    attempt: int,
) -> VarianceHints:
    """Progressively bias toward upper-board, vertically concentrated shapes.

    attempt 0: no modification (natural placement)
    attempt 1: mild upper-row preference
    attempt 2+: strong vertical concentration

    Physics rationale: upper centroids stay near the refill zone (row 0+).
    Vertical concentration means more cluster cells per column, creating
    deeper refill that reaches the booster's landing position.

    Returns a new VarianceHints — immutable, no mutation of the input.
    """
    if attempt == 0:
        return variance

    # Strength scales with attempt index — higher attempts push harder toward
    # upper rows. 1.5 per attempt gives mild→strong progression.
    strength = attempt * 1.5
    num_rows = board_config.num_rows

    new_bias: dict[Position, float] = {}
    for pos, weight in variance.spatial_bias.items():
        # Upper rows (low row index) get exponentially higher weight
        # row 0 → factor ~1.0, row (num_rows-1) → factor approaches 0
        row_factor = ((num_rows - pos.row) / num_rows) ** strength
        new_bias[pos] = weight * max(0.01, row_factor)

    return VarianceHints(
        spatial_bias=new_bias,
        symbol_weights=variance.symbol_weights,
        near_miss_symbol_preference=variance.near_miss_symbol_preference,
        cluster_size_preference=variance.cluster_size_preference,
    )
