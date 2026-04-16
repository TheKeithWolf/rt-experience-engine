"""Multi-objective position scorer with pluggable factors.

Integrates spatial signals (influence map, gravity alignment, booster
proximity, merge risk) into a single 0.0–1.0 utility score per candidate
position. The UtilityScorer is Open/Closed: new factors are added by
implementing ScoringFactor and registering — no scorer code changes.

Factors are standalone classes sharing only the ScoringFactor protocol.
No inheritance hierarchy, no if/else chains on factor type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

from ...planning.region_constraint import RegionConstraint
from ...primitives.board import Position, orthogonal_neighbors

if TYPE_CHECKING:
    from ...config.schema import BoardConfig
    from .gravity_field import GravityFieldService
    from .influence_map import DemandSpec


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """Read-only snapshot of spatial state for utility evaluation.

    Assembled once per seed-planning call. Factors read from this;
    none of them mutate it.

    region — optional Tier-1/Tier-2 planning guidance. RegionPreferenceFactor
      consumes it to steer seed placement toward the pre-validated columns;
      all other factors ignore it. Leaving it None preserves the historic
      unconstrained behavior for call sites that don't inject guidance.
    """

    influence: dict[Position, float]
    gravity_field: GravityFieldService
    demand: DemandSpec
    cluster_positions: frozenset[Position]
    board_config: BoardConfig
    booster_landing: Position | None
    region: RegionConstraint | None = None


@runtime_checkable
class ScoringFactor(Protocol):
    """One component of the utility score.

    Each factor computes a 0.0–1.0 value for a single candidate position.
    Factors are registered by name in the scorer's factor registry —
    no if/else chains on factor type.
    """

    @property
    def name(self) -> str: ...

    def evaluate(self, pos: Position, context: ScoringContext) -> float: ...


# ---------------------------------------------------------------------------
# Concrete factors — standalone classes, no inheritance
# ---------------------------------------------------------------------------


class InfluenceFactor:
    """Score = influence map value at this position.

    Higher influence means the position is closer to the demand centroid
    where the next step needs cluster formation space.
    """

    name = "influence"

    def evaluate(self, pos: Position, ctx: ScoringContext) -> float:
        return ctx.influence.get(pos, 0.0)


class GravityAlignmentFactor:
    """Score = how well gravity at pos flows toward the demand centroid.

    Replaces column-matching: instead of binary column membership, this
    gives a continuous 0.0–1.0 alignment score that accounts for diagonal
    gravity paths on edge columns.
    """

    name = "gravity_alignment"

    def evaluate(self, pos: Position, ctx: ScoringContext) -> float:
        return ctx.gravity_field.alignment_score(pos, ctx.demand.centroid)


class BoosterAdjacencyFactor:
    """Score = proximity to predicted booster landing (inverse manhattan, normalized).

    Replaces column-adjacency heuristic in plan_arm_seeds: instead of
    binary column membership, this gives continuous scoring where adjacent
    cells score near 1.0 and distant cells fade toward 0.0.
    """

    name = "booster_adjacency"

    def evaluate(self, pos: Position, ctx: ScoringContext) -> float:
        if ctx.booster_landing is None:
            return 0.0
        dist = (
            abs(pos.reel - ctx.booster_landing.reel)
            + abs(pos.row - ctx.booster_landing.row)
        )
        if dist == 0:
            return 1.0
        # Normalize: max_dist derived from board dimensions (not hardcoded)
        max_dist = ctx.board_config.num_reels + ctx.board_config.num_rows - 2
        return max(0.0, 1.0 - dist / max_dist)


class MergeRiskFactor:
    """Score = 1.0 if orthogonally adjacent to a cluster cell, else 0.0.

    This factor is SUBTRACTED (negative weight in config), penalizing
    positions that risk merging seeds into existing clusters. Prevents
    successive clusters from accidentally joining into one oversized cluster.
    """

    name = "merge_risk"

    def evaluate(self, pos: Position, ctx: ScoringContext) -> float:
        for neighbor in orthogonal_neighbors(pos, ctx.board_config):
            if neighbor in ctx.cluster_positions:
                return 1.0
        return 0.0


class RegionPreferenceFactor:
    """Score = membership in the atlas/trajectory region, with column falloff.

    Cells inside region.viable_columns score 1.0. Cells outside score a
    configurable per-column falloff (falloff ** column_distance), smoothly
    biasing seed placement toward the pre-validated columns without forcing
    exclusion. When the scoring context has no region attached the factor
    returns 1.0 — neutral weight, preserving un-guided behavior.
    """

    name = "region"

    __slots__ = ("_falloff",)

    def __init__(self, falloff: float) -> None:
        if not (0.0 < falloff <= 1.0):
            raise ValueError("falloff must be in (0.0, 1.0]")
        self._falloff = falloff

    def evaluate(self, pos: Position, ctx: ScoringContext) -> float:
        region = ctx.region
        if region is None:
            return 1.0
        if pos.reel in region.viable_columns:
            return 1.0
        # Distance measured as column gap to the nearest viable column — the
        # atlas only constrains columns, so row distance doesn't factor in.
        distance = min(abs(pos.reel - c) for c in region.viable_columns)
        return self._falloff ** distance


# ---------------------------------------------------------------------------
# Scorer — weighted aggregation of pluggable factors
# ---------------------------------------------------------------------------


class UtilityScorer:
    """Multi-objective position scorer with pluggable factors.

    Factors are registered by name. Weights come from config. Adding a
    new factor means: (1) implement ScoringFactor, (2) register it,
    (3) add its weight to YAML. No scorer code changes.
    """

    __slots__ = ("_factors", "_weights")

    def __init__(
        self,
        factors: Iterable[ScoringFactor],
        weights: dict[str, float],
    ) -> None:
        self._factors = {f.name: f for f in factors}
        self._weights = weights

    def score(self, pos: Position, ctx: ScoringContext) -> float:
        """Compute composite utility for a candidate position.

        Positive weights are additive, negative weights are subtractive.
        Result clamped to [0.0, 1.0].
        """
        total = 0.0
        for name, factor in self._factors.items():
            weight = self._weights.get(name, 0.0)
            total += weight * factor.evaluate(pos, ctx)
        return max(0.0, min(1.0, total))

    def score_candidates(
        self,
        candidates: Iterable[Position],
        ctx: ScoringContext,
    ) -> dict[Position, float]:
        """Score all candidates — returns Position -> utility for weighted selection."""
        return {pos: self.score(pos, ctx) for pos in candidates}
