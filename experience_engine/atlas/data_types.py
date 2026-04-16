"""Atlas data types — keys and entries produced by the offline builder.

All types are immutable value objects. They carry no references to runtime
services, so they serialize cleanly via AtlasStorage and can be constructed
in unit tests without engine wiring.

Design reference: games/royal_tumble/docs/trajectory-planner-impl-plan.md §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..planning.region_constraint import BridgeHint, RegionConstraint
from ..primitives.board import Position


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Per-column explosion count signature — the atlas lookup key.

    Gravity is column-local (straight-down donor has priority over diagonals),
    so two clusters with the same (counts, depth_band) produce near-identical
    post-gravity topologies. The depth_band attribute buckets exact row
    positions into "low"/"mid"/"deep" to keep the key space tractable without
    discarding the diagonal-redistribution differences that depth introduces.

    counts — length equals board.num_reels; each entry is an int in [0, num_rows]
    depth_band — name from atlas.depth_bands config (e.g. "low", "mid", "deep")
    total — sum(counts); equals the cluster size the profile represents
    """

    counts: tuple[int, ...]
    depth_band: str
    total: int


@dataclass(frozen=True, slots=True)
class SettleTopology:
    """Post-gravity board shape for one (ColumnProfile, depth_band) key.

    Produced by running the engine's settle() on a representative synthetic
    board. Only structural results are captured — symbol identities are
    irrelevant at this layer.

    empty_positions — cells left empty after settle; the refill target set
    refill_per_column — count of new symbols needed per column (len == num_reels)
    has_diagonal_redistribution — True if any cell moved across columns during settle
    gravity_mapping — column index → tuple of pre-settle row indices that
      shifted within that column. Used to trace dormant-booster survival.
    """

    empty_positions: frozenset[Position]
    refill_per_column: tuple[int, ...]
    has_diagonal_redistribution: bool
    gravity_mapping: dict[int, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class BoosterLandingEntry:
    """Pre-computed booster landing for a (profile, booster_type, centroid_column) key.

    landing_position — where the booster settles after gravity, using the
      engine's existing ForwardSimulator.predict_booster_landing
    adjacent_refill — orthogonal neighbors of landing_position that fall inside
      the topology's empty_positions; the arming cluster must overlap this set
    landing_score — BoosterLandingEvaluator.score() of the landing, used when
      the query picks among multiple candidate profiles
    """

    landing_position: Position
    adjacent_refill: frozenset[Position]
    landing_score: float


@dataclass(frozen=True, slots=True)
class ArmAdjacencyEntry:
    """Whether a post-gravity refill zone can host an arming cluster.

    adjacent_refill — the landing's orthogonal neighbors that fall inside the
      topology's empty_positions. An arming cluster must overlap this set.
    adjacent_count — |adjacent_refill|; precomputed so the query filter is
      a single comparison.
    sufficient — True when adjacent_count meets the arming threshold derived
      from board.min_cluster_size. The threshold is config-driven: an arming
      cluster needs at least one cell to bridge, which scales with the
      minimum cluster size; we use `max(1, min_cluster_size - 1)` so the
      check degrades sensibly for small boards without hardcoding "4".
    """

    adjacent_refill: frozenset[Position]
    adjacent_count: int
    sufficient: bool


@dataclass(frozen=True, slots=True)
class DormantSurvivalEntry:
    """Whether a dormant booster at dormant_column survives a given profile.

    survives — False if the dormant position is in the explosion set, or if
      gravity would push it off-board in a degenerate mapping
    post_gravity_position — where the dormant ends up after settle, or None
      when survives=False
    column_shift — True if gravity moved the dormant across columns (signals
      the query's fire-zone filter must recompute reachability)
    """

    survives: bool
    post_gravity_position: Position | None
    column_shift: bool


@dataclass(frozen=True, slots=True)
class BridgeFeasibilityEntry:
    """Whether a column profile can support a wild-bridge placement.

    A bridge requires a zero-count gap column with non-zero counts on both
    sides. bridge_score gates on the weaker side's adjacency — min(left, right)
    normalized by profile total — so score == 0 means structurally unbridgeable.
    """

    gap_column: int
    left_columns: frozenset[int]
    right_columns: frozenset[int]
    left_adjacency_count: int
    right_adjacency_count: int
    bridge_score: float


@dataclass(frozen=True, slots=True)
class PhaseGuidance:
    """Placement guidance for a single arc phase, emitted by the atlas query.

    Every field is optional except viable_columns + preferred_row_range — a
    spawn-only phase has no booster_landing, a no-chain phase has no
    chain_target_zone, and early phases have no dormant_survival record.
    """

    viable_columns: frozenset[int]
    preferred_row_range: tuple[int, int]
    column_profile: ColumnProfile
    booster_landing: BoosterLandingEntry | None
    dormant_survival: DormantSurvivalEntry | None
    chain_target_zone: frozenset[Position] | None
    # Bridge-phase fields — None for non-bridge phases so existing callers
    # are unaffected. ClusterBuilder (future PR) uses these to split BFS
    # into two sub-growths connected by the wild at gap_column.
    bridge_gap_column: int | None = None
    left_group_columns: frozenset[int] | None = None
    right_group_columns: frozenset[int] | None = None

    def as_region(self) -> RegionConstraint:
        """Project onto the shared RegionConstraint consumed by ClusterBuilder."""
        return RegionConstraint(
            viable_columns=self.viable_columns,
            preferred_row_range=self.preferred_row_range,
            profile_hint=self.column_profile,
        )


@dataclass(frozen=True, slots=True)
class SpatialAtlas:
    """Container holding every pre-computed atlas entry.

    topologies — one SettleTopology per keyed ColumnProfile
    booster_landings — keyed by (ColumnProfile, booster_symbol, centroid_reel)
      because gravity depends on which column the centroid lives in
    arm_adjacencies — same key as booster_landings; separate type so the
      query's arming filter is a dict lookup on a dedicated entry, not a
      derived count from BoosterLandingEntry
    fire_zones — keyed by (booster_symbol, landing_position, orientation).
      Rockets emit three entries per landing — "H", "V", and a None entry
      pointing at the higher-scoring orientation — so the query lookup is
      orientation-agnostic when the arc leaves the field unset.
    dormant_survivals — keyed by (ColumnProfile, dormant_reel); only tracks
      column-level survival, since dormant boosters may sit in any row within
      their column and the trace through gravity is column-indexed

    The dicts are left mutable by dataclass convention (frozen applies to the
    field bindings, not the contents) — AtlasBuilder populates them once then
    hands the atlas off, and no runtime code is expected to mutate after.
    """

    topologies: dict[ColumnProfile, SettleTopology]
    booster_landings: dict[tuple[ColumnProfile, str, int], BoosterLandingEntry]
    arm_adjacencies: dict[tuple[ColumnProfile, str, int], ArmAdjacencyEntry]
    fire_zones: dict[tuple[str, Position, str | None], frozenset[Position]]
    dormant_survivals: dict[tuple[ColumnProfile, int], DormantSurvivalEntry]
    bridge_feasibilities: dict[tuple[ColumnProfile, int], BridgeFeasibilityEntry]

    def topologies_for_size(self, size: int) -> tuple[ColumnProfile, ...]:
        """Every keyed profile whose cluster size equals `size`.

        Callers (AtlasQuery) intersect this with per-phase filters — dormant
        survival, arming adjacency — so returning the unfiltered set keeps the
        atlas agnostic to query semantics.
        """
        return tuple(p for p in self.topologies if p.total == size)


@dataclass(frozen=True, slots=True)
class AtlasConfiguration:
    """Atlas-resolved per-phase guidance for a whole narrative arc.

    phases[i] corresponds to NarrativeArc.phases[i]. composite_score is the
    product of each phase's landing_score (when present), surfaced to the
    orchestrator so it can reject low-confidence configurations via the
    atlas.min_composite_score threshold.
    """

    phases: tuple[PhaseGuidance, ...]
    composite_score: float

    def region_at(self, step_index: int) -> RegionConstraint | None:
        """GuidanceSource implementation — delegates to phase.as_region()."""
        if not 0 <= step_index < len(self.phases):
            return None
        return self.phases[step_index].as_region()

    def bridge_hint_at(self, step_index: int) -> BridgeHint | None:
        """Return bridge placement hint for the given step, or None.

        Only bridge phases populate the three bridge fields on PhaseGuidance;
        non-bridge phases return None so callers need no type checks.
        """
        if not 0 <= step_index < len(self.phases):
            return None
        phase = self.phases[step_index]
        if phase.bridge_gap_column is None:
            return None
        return BridgeHint(
            gap_column=phase.bridge_gap_column,
            left_columns=phase.left_group_columns,
            right_columns=phase.right_group_columns,
        )
