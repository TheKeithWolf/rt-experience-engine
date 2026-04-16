"""Atlas query — converts a NarrativeArc into an AtlasConfiguration.

For each phase the query intersects candidate topologies against the arc's
accumulated constraints (dormant survival, arming adjacency, chain fire
reach) and chooses the best-scoring survivor. Missing any phase — or
failing the min_composite_score threshold — yields None, signalling the
CascadeInstanceGenerator to fall through to Tier 2.

No simulation happens here; every lookup is a dict read or set intersection.
AtlasBuilder pre-paid the compute cost.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..config.schema import AtlasConfig, MasterConfig, SymbolConfig
from ..narrative.arc import NarrativeArc, NarrativePhase
from ..pipeline.protocols import Range
from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.symbols import is_special
from ..step_reasoner.evaluators import ChainEvaluator
from .data_types import (
    AtlasConfiguration,
    BoosterLandingEntry,
    ColumnProfile,
    DormantSurvivalEntry,
    PhaseGuidance,
    SettleTopology,
    SpatialAtlas,
)
from .storage import AtlasStorage


@dataclass(frozen=True, slots=True)
class _ResolvedPhase:
    """Working state threaded across the query loop.

    Carries per-phase choices so later phases can filter against earlier
    landings / dormant positions without re-deriving them. Private to query.py.
    """

    profile: ColumnProfile
    topology: SettleTopology
    landing: BoosterLandingEntry | None
    dormant: DormantSurvivalEntry | None
    chain_target_zone: frozenset[Position] | None
    booster_type: str | None
    guidance: PhaseGuidance
    score: float


class AtlasQuery:
    """Stateless query interface over a pre-built SpatialAtlas.

    The atlas, BoosterRules, ChainEvaluator, and AtlasConfig are injected
    once. query_arc() is safe to call concurrently for different arcs.
    """

    __slots__ = (
        "_atlas", "_booster_rules", "_chain_eval",
        "_config", "_symbols",
    )

    def __init__(
        self,
        atlas: SpatialAtlas,
        booster_rules: BoosterRules,
        chain_eval: ChainEvaluator,
        config: AtlasConfig,
        symbols: SymbolConfig,
    ) -> None:
        self._atlas = atlas
        self._booster_rules = booster_rules
        self._chain_eval = chain_eval
        self._config = config
        # Needed for is_special() used by the board-compatibility predicate —
        # a profile whose columns host wilds / scatters / boosters cannot
        # realize at generation time regardless of what the atlas indexed.
        self._symbols = symbols

    @classmethod
    def from_config(
        cls,
        config: MasterConfig,
        config_path: Path,
        chain_eval: ChainEvaluator,
    ) -> "AtlasQuery | None":
        # Single canonical builder: gates on config.atlas.enabled, resolves the
        # atlas file relative to the YAML, and returns None for any miss
        # (section absent, disabled, missing file, or hash mismatch). Callers
        # forward the None straight into CascadeInstanceGenerator, which treats
        # it as "no Tier-1 guidance" and falls through to Tier 2.
        if config.atlas is None or not config.atlas.enabled:
            return None
        storage = AtlasStorage()
        atlas = storage.load(config, AtlasStorage.default_path(config, config_path))
        if atlas is None:
            return None
        booster_rules = BoosterRules(config.boosters, config.board, config.symbols)
        return cls(atlas, booster_rules, chain_eval, config.atlas, config.symbols)

    def query_arc(
        self,
        arc: NarrativeArc,
        board: Board | None = None,
        dormants: Iterable[Position] | None = None,
    ) -> AtlasConfiguration | None:
        """Resolve every phase in the arc to PhaseGuidance, or return None.

        board / dormants — live generator state. When provided, profiles
          whose explosion columns conflict with immovable symbols or tracked
          dormants are filtered out. Both default to None so existing call
          sites that don't have live state still work (with the filter
          effectively disabled).

        Returning None is the normal miss path — the orchestrator falls
        through to the trajectory planner. Missing can come from any phase;
        the function short-circuits on the first unresolvable phase to keep
        the query cost proportional to the depth actually explored.
        """
        if not self._config.enabled:
            return None
        blocking = self._blocking_columns(board, dormants)
        resolved: list[_ResolvedPhase] = []
        composite = 1.0
        for phase in arc.phases:
            choice = self._resolve_phase(phase, resolved, blocking, arc)
            if choice is None:
                return None
            resolved.append(choice)
            composite *= max(choice.score, 1e-6)  # Prevents zero-ing the product.

        if composite < self._config.min_composite_score:
            return None
        return AtlasConfiguration(
            phases=tuple(r.guidance for r in resolved),
            composite_score=composite,
        )

    def _blocking_columns(
        self,
        board: Board | None,
        dormants: Iterable[Position] | None,
    ) -> frozenset[int]:
        """Columns where at least one cell is immovable or hosts a dormant.

        `immovable` is a single predicate (`is_special` on the symbol OR
        position membership in the dormant set). One pass over the board
        rather than per-profile repetition — the blocking set is invariant
        for the whole query.
        """
        if board is None and dormants is None:
            return frozenset()
        dormant_set: frozenset[Position] = frozenset(dormants or ())
        blocked: set[int] = {pos.reel for pos in dormant_set}
        if board is not None:
            for reel in range(board.num_reels):
                if reel in blocked:
                    continue
                for row in range(board.num_rows):
                    sym = board.get(Position(reel, row))
                    if sym is not None and is_special(sym, self._symbols):
                        blocked.add(reel)
                        break
        return frozenset(blocked)

    # ------------------------------------------------------------------
    # Per-phase resolution
    # ------------------------------------------------------------------

    def _resolve_phase(
        self,
        phase: NarrativePhase,
        prior: list[_ResolvedPhase],
        blocking_columns: frozenset[int],
        arc: NarrativeArc,
    ) -> _ResolvedPhase | None:
        """Intersect the atlas against the phase's constraints.

        Picks a representative cluster size from phase.cluster_sizes (the
        first range midpoint); the atlas stores one entry per composition,
        not per exact size, so a midpoint is sufficient for lookup.
        """
        size = _midpoint_size(phase.cluster_sizes)
        if size is None:
            return None
        booster_symbol = self._booster_rules.booster_type_for_size(size)
        booster_type = booster_symbol.name if booster_symbol else None

        dormant_column = _dormant_column_from_prior(prior)
        arming_target_column = _arming_target_from_prior(prior, phase)
        fire_target_column = _fire_target_from_prior(prior, phase)
        orientation = arc.rocket_orientation  # None for unconstrained arcs.

        best: _ResolvedPhase | None = None
        for profile in self._atlas.topologies_for_size(size):
            if _profile_hits_blocking_columns(profile, blocking_columns):
                continue
            topology = self._atlas.topologies[profile]

            dormant_entry = self._dormant_for(profile, dormant_column)
            if dormant_column is not None and (
                dormant_entry is None or not dormant_entry.survives
            ):
                continue

            landing = self._landing_for(profile, booster_type)
            if arming_target_column is not None and landing is not None:
                if landing.landing_position.reel != arming_target_column:
                    continue

            # When this phase spawns a booster and a later phase arms it,
            # reject landings whose refill zone can't host an arming cluster.
            # Looking ahead one phase is sufficient — the arming phase is
            # always immediate in current arcs; deeper lookups would need
            # the arc structure.
            if _next_phase_arms(arc, len(prior)) and landing is not None:
                arm = self._arm_for(profile, booster_type)
                if arm is None or not arm.sufficient:
                    continue

            chain_zone = self._fire_zone_for(landing, booster_type, orientation)
            if fire_target_column is not None and chain_zone is not None:
                if not any(p.reel == fire_target_column for p in chain_zone):
                    continue

            guidance = _phase_guidance(
                profile, topology, landing, dormant_entry, chain_zone
            )
            score = landing.landing_score if landing is not None else 1.0
            if best is None or score > best.score:
                best = _ResolvedPhase(
                    profile=profile,
                    topology=topology,
                    landing=landing,
                    dormant=dormant_entry,
                    chain_target_zone=chain_zone,
                    booster_type=booster_type,
                    guidance=guidance,
                    score=score,
                )
        return best

    def _dormant_for(
        self, profile: ColumnProfile, column: int | None
    ) -> DormantSurvivalEntry | None:
        if column is None:
            return None
        return self._atlas.dormant_survivals.get((profile, column))

    def _landing_for(
        self, profile: ColumnProfile, booster_type: str | None
    ) -> BoosterLandingEntry | None:
        if booster_type is None:
            return None
        for (p, b_type, _col), entry in self._atlas.booster_landings.items():
            if p is profile and b_type == booster_type:
                return entry
        return None

    def _fire_zone_for(
        self,
        landing: BoosterLandingEntry | None,
        booster_type: str | None,
        orientation: str | None,
    ) -> frozenset[Position] | None:
        """Fire zone lookup — one dict read per call.

        The atlas already indexed every (booster_type, landing, orientation)
        key at build time, including a None-orientation fallback pointing at
        the argmax-score orientation, so unconstrained arcs and orientation-
        pinned arcs share this single lookup path — no branching on whether
        the arc pins an orientation.
        """
        if landing is None or booster_type is None:
            return None
        if not self._chain_eval.can_initiate_chain(booster_type):
            return None
        return self._atlas.fire_zones.get(
            (booster_type, landing.landing_position, orientation)
        )

    def _arm_for(
        self, profile: ColumnProfile, booster_type: str | None,
    ) -> object | None:
        """Pick the ArmAdjacencyEntry for this profile's booster landing.

        Same lookup shape as `_landing_for`. Returns None when no arm
        adjacency was indexed (profile doesn't spawn a booster, or the
        booster type was skipped at build time).
        """
        if booster_type is None:
            return None
        for (p, b_type, _col), entry in self._atlas.arm_adjacencies.items():
            if p is profile and b_type == booster_type:
                return entry
        return None


# ----------------------------------------------------------------------
# Pure helpers — testable in isolation
# ----------------------------------------------------------------------


def _midpoint_size(cluster_sizes: tuple[Range, ...]) -> int | None:
    """Pick a representative cluster size from the phase's first range.

    Atlas profiles are keyed by exact size, so we need a single number per
    phase. The first range's midpoint is deterministic and central — if the
    arc specifies multiple ranges, later phases resolve the others.
    """
    if not cluster_sizes:
        return None
    first = cluster_sizes[0]
    return (first.min_val + first.max_val) // 2


def _dormant_column_from_prior(prior: list[_ResolvedPhase]) -> int | None:
    """The column of the most recently introduced dormant booster, if any.

    Prior phases flag a dormant booster via their landing (the landed booster
    is the new dormant). The accumulated tracking is column-level, matching
    how the atlas indexes survival.
    """
    for resolved in reversed(prior):
        if resolved.landing is not None:
            return resolved.landing.landing_position.reel
    return None


def _arming_target_from_prior(
    prior: list[_ResolvedPhase], phase: NarrativePhase
) -> int | None:
    """Column that this phase's cluster must arm (touch adjacently).

    Atlas landings are indexed by centroid column; aligning the cluster's
    centroid column with the target booster's landing column is a sufficient
    soft constraint at the atlas tier. Finer placement happens at
    ClusterBuilder time via RegionConstraint.
    """
    if phase.arms is None or not prior:
        return None
    last_landing = prior[-1].landing
    return last_landing.landing_position.reel if last_landing else None


def _profile_hits_blocking_columns(
    profile: ColumnProfile, blocking: frozenset[int]
) -> bool:
    """True when any of the profile's active columns is in the blocking set.

    An active column is one with a non-zero count — those cells must be
    cleared by the explosion, but an immovable symbol or dormant in the
    same column makes that impossible.
    """
    if not blocking:
        return False
    return any(
        count > 0 and reel in blocking
        for reel, count in enumerate(profile.counts)
    )


def _next_phase_arms(arc: NarrativeArc, current_index: int) -> bool:
    """True when the arc's phase immediately after `current_index` arms.

    Uses the arc directly rather than threading extra state through the
    resolver — the arc is already available at call time, so keeping this
    lookup stateless avoids adding another parameter to every helper.
    """
    next_index = current_index + 1
    if next_index >= len(arc.phases):
        return False
    return arc.phases[next_index].arms is not None


def _fire_target_from_prior(
    prior: list[_ResolvedPhase], phase: NarrativePhase
) -> int | None:
    """Column of the chain target booster a fires-phase must reach."""
    if phase.fires is None:
        return None
    # Chain target is the most recent still-dormant booster — previous phases
    # will have landed them into known columns.
    for resolved in reversed(prior[:-1]):
        if resolved.landing is not None:
            return resolved.landing.landing_position.reel
    return None


def _phase_guidance(
    profile: ColumnProfile,
    topology: SettleTopology,
    landing: BoosterLandingEntry | None,
    dormant: DormantSurvivalEntry | None,
    chain_zone: frozenset[Position] | None,
) -> PhaseGuidance:
    """Project atlas entries onto a PhaseGuidance the generator consumes.

    viable_columns are the columns the profile lives in; preferred_row_range
    spans the empty-position row band so ClusterBuilder aims cluster cells
    at the refill zone.
    """
    columns = frozenset(
        reel for reel, count in enumerate(profile.counts) if count > 0
    )
    if topology.empty_positions:
        rows = tuple(p.row for p in topology.empty_positions)
        preferred = (min(rows), max(rows))
    else:
        preferred = (0, 0)
    return PhaseGuidance(
        viable_columns=columns,
        preferred_row_range=preferred,
        column_profile=profile,
        booster_landing=landing,
        dormant_survival=dormant,
        chain_target_zone=chain_zone,
    )
