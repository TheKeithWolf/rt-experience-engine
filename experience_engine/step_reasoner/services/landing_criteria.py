"""Landing criteria — per-booster scoring for post-gravity landing viability.

Each criterion answers: "How viable is this booster's landing position for the
next cascade step?" Criteria are registered in a dict keyed by booster type —
no if/else chains. Adding a new booster means adding one entry.

Used by: BoosterLandingEvaluator (dict dispatch to criterion.score()).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...config.schema import BoardConfig, LandingCriteriaConfig
from ...primitives.board import Position, orthogonal_neighbors
from ...primitives.booster_rules import BoosterRules

if TYPE_CHECKING:
    from .landing_evaluator import LandingContext


# ---------------------------------------------------------------------------
# Protocol — one method, no booster-type switching
# ---------------------------------------------------------------------------

@runtime_checkable
class LandingCriterion(Protocol):
    """Scores how viable a booster's post-gravity landing is for the next step.

    0.0 = impossible (no adjacent refill, clipped blast, etc.)
    1.0 = ideal (plenty of adjacent refill, central position, etc.)
    """

    def score(self, ctx: LandingContext) -> float: ...


# ---------------------------------------------------------------------------
# Chain constraint — propagated from Step N to Step N+1 for chain archetypes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ChainConstraint:
    """The next booster must land within effect_zone for the chain to work.

    Produced by Step N's strategy after evaluating its booster's landing.
    Consumed by Step N+1's seed planner to filter seed columns.
    """

    source_type: str                    # booster that creates the zone (e.g. "R")
    source_position: Position           # where that booster landed
    source_orientation: str | None      # "H"/"V" for rockets, None for bombs
    effect_zone: frozenset[Position]    # from rocket_path() or bomb_blast()


# ---------------------------------------------------------------------------
# Wild Bridge — needs adjacent refill cells to form a bridge cluster
# ---------------------------------------------------------------------------

class WildBridgeCriterion:
    """Scores whether refill cells can form a bridge cluster through the Wild.

    A bridge needs (min_cluster_size - 1) symbols adjacent to the Wild.
    Bonus when adjacent cells span multiple columns — increases the chance
    of a genuine two-sided bridge rather than a one-sided appendage.
    """

    __slots__ = ("_needed", "_side_bonus")

    def __init__(
        self,
        board_config: BoardConfig,
        landing_criteria_config: LandingCriteriaConfig,
    ) -> None:
        # Wild counts as 1 member, so bridge needs (min_size - 1) more adjacents
        self._needed = max(1, board_config.min_cluster_size - 1)
        self._side_bonus = landing_criteria_config.wild_bridge_multi_column_bonus

    def score(self, ctx: LandingContext) -> float:
        adjacent = len(ctx.adjacent_refill)
        if adjacent == 0:
            return 0.0

        base = min(1.0, adjacent / self._needed)

        # Cells in multiple columns → bridge more likely to span both sides of the Wild
        cols = {p.reel for p in ctx.adjacent_refill}
        side_bonus = self._side_bonus if len(cols) >= 2 else 0.0

        return min(1.0, base + side_bonus)


# ---------------------------------------------------------------------------
# Rocket Arm — needs adjacent refill + central position for path coverage
# ---------------------------------------------------------------------------

class RocketArmCriterion:
    """Scores arming feasibility plus rocket path coverage.

    Arm component (60%): enough adjacent refill cells to form an arming cluster.
    Chain component (40%): central landings maximize path length (full row or column).
    Optional orientation penalty when cluster shape produces the wrong firing direction.
    """

    __slots__ = (
        "_booster_rules", "_needed", "_max_manhattan",
        "_desired_orientation", "_arm_weight", "_chain_weight",
        "_orientation_penalty",
    )

    def __init__(
        self,
        booster_rules: BoosterRules,
        board_config: BoardConfig,
        landing_criteria_config: LandingCriteriaConfig,
        desired_orientation: str | None = None,
    ) -> None:
        self._booster_rules = booster_rules
        self._needed = max(1, board_config.min_cluster_size - 1)
        # Maximum possible manhattan distance from board center — normalization denominator
        self._max_manhattan = max(
            1, (board_config.num_reels // 2) + (board_config.num_rows // 2),
        )
        self._desired_orientation = desired_orientation
        self._arm_weight = landing_criteria_config.rocket_arm_weight
        self._chain_weight = landing_criteria_config.rocket_chain_weight
        self._orientation_penalty = landing_criteria_config.rocket_orientation_penalty

    def score(self, ctx: LandingContext) -> float:
        adjacent = len(ctx.adjacent_refill)

        # Arm feasibility: enough adjacent cells to seed an arming cluster
        arm_score = min(1.0, adjacent / self._needed)

        # Chain geometry: central positions maximize the rocket's row/column path
        landing = ctx.landing_position
        mid_reel = self._max_manhattan - (self._max_manhattan // 2)
        mid_row = self._max_manhattan - mid_reel
        # Recompute from actual board center
        manhattan_from_center = abs(landing.reel - mid_reel) + abs(landing.row - mid_row)
        centrality = max(0.0, 1.0 - manhattan_from_center / self._max_manhattan)
        chain_score = max(0.2, centrality)

        combined = arm_score * self._arm_weight + chain_score * self._chain_weight

        # Orientation penalty — wrong firing direction makes the chain impossible
        if self._desired_orientation is not None:
            actual = ctx.cluster_shape_stats.orientation
            if actual != self._desired_orientation:
                combined *= self._orientation_penalty

        return combined


# ---------------------------------------------------------------------------
# Bomb Arm — needs adjacent refill + interior position for blast coverage
# ---------------------------------------------------------------------------

class BombArmCriterion:
    """Scores arming feasibility plus bomb blast coverage.

    Arm component (60%): enough adjacent refill cells for arming cluster.
    Blast component (40%): interior positions have full blast radius coverage,
    edge/corner positions lose cells to board clipping.
    """

    __slots__ = (
        "_booster_rules", "_board_config", "_needed", "_max_blast_area",
        "_arm_weight", "_blast_weight",
    )

    def __init__(
        self,
        booster_rules: BoosterRules,
        board_config: BoardConfig,
        landing_criteria_config: LandingCriteriaConfig,
    ) -> None:
        self._booster_rules = booster_rules
        self._board_config = board_config
        self._needed = max(1, board_config.min_cluster_size - 1)
        # Theoretical maximum blast area (no edge clipping) — from config radius
        radius = booster_rules._config.bomb_blast_radius
        self._max_blast_area = (2 * radius + 1) ** 2
        self._arm_weight = landing_criteria_config.bomb_arm_weight
        self._blast_weight = landing_criteria_config.bomb_blast_weight

    def score(self, ctx: LandingContext) -> float:
        adjacent = len(ctx.adjacent_refill)

        # Arm feasibility
        arm_score = min(1.0, adjacent / self._needed)

        # Blast coverage: interior positions cover more cells → better chain potential
        blast = self._booster_rules.bomb_blast(ctx.landing_position)
        blast_coverage = len(blast) / self._max_blast_area

        return arm_score * self._arm_weight + blast_coverage * self._blast_weight


# ---------------------------------------------------------------------------
# Lightball / Super Lightball — pure armability (board composition via WFC)
# ---------------------------------------------------------------------------

class LightballArmCriterion:
    """Scores pure arming feasibility — board composition is handled by WFC weights.

    LB/SLB target the most abundant standard symbol, which is a board-level
    property managed by WFC symbol weights. The landing criterion only needs
    to ensure enough adjacent refill cells exist for an arming cluster.
    """

    __slots__ = ("_needed",)

    def __init__(self, board_config: BoardConfig) -> None:
        self._needed = max(1, board_config.min_cluster_size - 1)

    def score(self, ctx: LandingContext) -> float:
        adjacent = len(ctx.adjacent_refill)
        return min(1.0, adjacent / self._needed)


# SLB reuses the same criterion — armability is the bottleneck, not board composition
SuperLightballArmCriterion = LightballArmCriterion


# ---------------------------------------------------------------------------
# Arm Feasibility — BFS flood-fill of post-settle empty cells near the landing
# ---------------------------------------------------------------------------

class ArmFeasibilityCriterion:
    """Scores whether the rocket's landing has enough connected empty space
    for an arming cluster of min_cluster_size.

    Starts BFS from the landing's orthogonal refill neighbors and expands
    through the full post-settle refill zone. Reports the connected component
    size as a fraction of the cluster threshold. Unlike RocketArmCriterion
    (which counts only immediate adjacencies and centrality), this criterion
    measures reachable post-gravity space — the true bottleneck for arming.
    """

    __slots__ = ("_board_config", "_required")

    def __init__(self, board_config: BoardConfig) -> None:
        self._board_config = board_config
        # An arming cluster must reach min_cluster_size connected cells in the
        # refill zone adjacent to the rocket. Derived from game rules, not magic.
        self._required = board_config.min_cluster_size

    def score(self, ctx: LandingContext) -> float:
        if not ctx.adjacent_refill:
            return 0.0

        refill_set = frozenset(ctx.all_refill)
        if not refill_set:
            return 0.0

        # BFS seeded from every landing-adjacent refill cell — the connected
        # component reachable through the refill zone is the pool the WFC
        # fill + gravity cascade has to form an arming cluster within.
        visited: set[Position] = set()
        queue: deque[Position] = deque()
        for seed in ctx.adjacent_refill:
            if seed in refill_set and seed not in visited:
                visited.add(seed)
                queue.append(seed)

        while queue:
            pos = queue.popleft()
            for neighbor in orthogonal_neighbors(pos, self._board_config):
                if neighbor in refill_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        reachable = len(visited)
        return min(1.0, reachable / self._required)


# ---------------------------------------------------------------------------
# Composite — combine multiple criteria via a configurable aggregator
# ---------------------------------------------------------------------------

class CompositeCriterion:
    """Composes N criteria via a configurable aggregation function.

    The aggregator is a callable taking an iterable of floats and returning
    one float — pass `min` for worst-of, `statistics.mean` for average,
    `math.prod` for multiplicative. No if/else dispatch on mode strings,
    so adding new aggregation strategies requires no edits here.
    """

    __slots__ = ("_criteria", "_aggregate")

    def __init__(
        self,
        criteria: tuple[LandingCriterion, ...],
        aggregate: Callable[[Iterable[float]], float],
    ) -> None:
        self._criteria = criteria
        self._aggregate = aggregate

    def score(self, ctx: LandingContext) -> float:
        return self._aggregate(c.score(ctx) for c in self._criteria)


# ---------------------------------------------------------------------------
# Chain Target — landing must fall inside the initiator's effect zone
# ---------------------------------------------------------------------------

class ChainTargetCriterion:
    """Binary criterion: 1.0 if the target booster lands inside the initiator's
    effect zone, 0.0 otherwise.

    Consumes a ChainConstraint describing the zone produced by the chain
    initiator (rocket path, bomb blast, …). Binary because near-misses aren't
    useful for chains — the target is either in the path or it isn't.

    The constraint is supplied at construction time by the strategy that
    resolves it at plan-time; this keeps the criterion free of cross-step
    state lookups.
    """

    __slots__ = ("_constraint",)

    def __init__(self, constraint: ChainConstraint) -> None:
        self._constraint = constraint

    def score(self, ctx: LandingContext) -> float:
        return 1.0 if ctx.landing_position in self._constraint.effect_zone else 0.0
