"""Offline atlas builder — composes existing services, no new physics.

Iterates every (ColumnProfile, depth_band) the game config generates via
profiles.enumerate_column_profiles(), runs a synthetic explosion through the
engine's settle(), and indexes the resulting topology. For each topology it
also pre-computes booster landings per centroid column and dormant-booster
survival per column.

All physical behavior — gravity, landing, booster rules — delegates to the
existing services. The builder's only responsibility is *composition* and
*indexing*.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import time
from dataclasses import dataclass
from typing import Callable

from ..config.schema import MasterConfig
from ..primitives.board import Board, Position, orthogonal_neighbors
from ..primitives.booster_rules import BoosterRules
from ..primitives.gravity import GravityDAG, settle
from ..primitives.symbols import Symbol, SymbolTier, symbols_in_tier
from ..step_reasoner.evaluators import ChainEvaluator
from ..step_reasoner.services.forward_simulator import ForwardSimulator
from ..step_reasoner.services.landing_criteria import (
    BombArmCriterion,
    LightballArmCriterion,
    RocketArmCriterion,
    WildBridgeCriterion,
)
from ..step_reasoner.services.landing_evaluator import BoosterLandingEvaluator
from .data_types import (
    ArmAdjacencyEntry,
    BoosterLandingEntry,
    BridgeFeasibilityEntry,
    ColumnProfile,
    DormantSurvivalEntry,
    SettleTopology,
    SpatialAtlas,
)
from .profiles import (
    atlas_cluster_sizes,
    enumerate_column_profiles,
    profile_to_positions,
)


def _NO_REPORT(_line: str) -> None:
    """Silent progress sink — used when the caller passes no callback."""
    return None


@dataclass(frozen=True, slots=True)
class AtlasServices:
    """Bundle of wired services used by both the CLI and test fixtures.

    Groups the dependencies AtlasBuilder and AtlasQuery share so every caller
    — tests, CLI, future integrations — stays on one construction path. Each
    field is a live service instance; callers consume whichever subset they
    need.
    """

    builder: "AtlasBuilder"
    booster_rules: BoosterRules
    chain_eval: ChainEvaluator
    gravity_dag: GravityDAG


def build_atlas_services(config: MasterConfig) -> AtlasServices:
    """Wire every atlas service from a MasterConfig.

    Eliminates the repeated DAG + ForwardSimulator + criteria + evaluator
    dance that each test file (and now the CLI) would otherwise carry. The
    criteria dict is assembled from the existing LandingCriterion classes —
    no new scoring logic, just composition.
    """
    dag = GravityDAG(config.board, config.gravity)
    forward = ForwardSimulator(dag, config.board, config.gravity)
    booster_rules = BoosterRules(config.boosters, config.board, config.symbols)
    criteria = {
        "W": WildBridgeCriterion(config.board, config.landing_criteria),
        "R": RocketArmCriterion(
            booster_rules, config.board, config.landing_criteria,
        ),
        "B": BombArmCriterion(
            booster_rules, config.board, config.landing_criteria,
        ),
        "LB": LightballArmCriterion(config.board),
        "SLB": LightballArmCriterion(config.board),
    }
    landing_eval = BoosterLandingEvaluator(
        forward, booster_rules, config.board, criteria,
    )
    builder = AtlasBuilder(config, dag, forward, booster_rules, landing_eval)
    return AtlasServices(
        builder=builder,
        booster_rules=booster_rules,
        chain_eval=ChainEvaluator(config.boosters),
        gravity_dag=dag,
    )


class AtlasBuilder:
    """Builds a SpatialAtlas from the injected services and config.

    Dependencies are all supplied via __init__. No internal construction;
    the builder is a pure orchestration layer.
    """

    __slots__ = (
        "_config",
        "_gravity_dag",
        "_forward_sim",
        "_booster_rules",
        "_landing_eval",
    )

    def __init__(
        self,
        config: MasterConfig,
        gravity_dag: GravityDAG,
        forward_sim: ForwardSimulator,
        booster_rules: BoosterRules,
        landing_eval: BoosterLandingEvaluator,
    ) -> None:
        if config.atlas is None:
            raise ValueError(
                "AtlasBuilder requires config.atlas — enable the atlas section "
                "in default.yaml before invoking the builder."
            )
        self._config = config
        self._gravity_dag = gravity_dag
        self._forward_sim = forward_sim
        self._booster_rules = booster_rules
        self._landing_eval = landing_eval

    def build(
        self,
        sizes: tuple[int, ...] | None = None,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> SpatialAtlas:
        """Produce a SpatialAtlas for the current config.

        sizes — optional override for the cluster sizes the atlas covers.
          None uses atlas_cluster_sizes(), the single source of truth derived
          from board geometry and spawn_thresholds. Tests supply a narrower
          range to keep build time bounded.
        progress — optional callback receiving one formatted line per cluster
          size (during the loop) plus one summary line at the end. None keeps
          the build silent; the CLI passes `print`, tests pass a collecting
          list.append. Keeping the callback a plain Callable avoids adding a
          logging dependency just for developer-facing output.
        """
        topologies: dict[ColumnProfile, SettleTopology] = {}
        booster_landings: dict[
            tuple[ColumnProfile, str, int], BoosterLandingEntry
        ] = {}
        arm_adjacencies: dict[
            tuple[ColumnProfile, str, int], ArmAdjacencyEntry
        ] = {}
        fire_zones: dict[
            tuple[str, Position, str | None], frozenset[Position]
        ] = {}
        dormant_survivals: dict[
            tuple[ColumnProfile, int], DormantSurvivalEntry
        ] = {}
        bridge_feasibilities: dict[
            tuple[ColumnProfile, int], BridgeFeasibilityEntry
        ] = {}

        if sizes is None:
            sizes = atlas_cluster_sizes(
                self._config.board, self._config.boosters.spawn_thresholds
            )
        depth_bands = self._config.atlas.depth_bands  # type: ignore[union-attr]
        report = progress if progress is not None else _NO_REPORT

        overall_start = time.perf_counter()
        for size in sizes:
            size_start = time.perf_counter()
            topology_count = 0
            for profile in enumerate_column_profiles(
                self._config.board, depth_bands, (size,)
            ):
                exploded = profile_to_positions(
                    profile, depth_bands, self._config.board.num_rows
                )
                if exploded is None:
                    # Depth band too narrow for this composition's tallest
                    # column — skip rather than emit a degenerate topology.
                    continue
                topology = self._build_topology(exploded)
                topologies[profile] = topology
                topology_count += 1

                self._index_landing_and_arm(
                    profile, exploded, topology,
                    booster_landings, arm_adjacencies, fire_zones,
                )
                self._index_dormants(
                    profile, exploded, topology, dormant_survivals,
                )
                self._index_bridge_feasibility(
                    profile, topology, bridge_feasibilities,
                )
            elapsed = time.perf_counter() - size_start
            report(
                f"  Size {size:>2}: {topology_count:>6,} topologies "
                f"[{elapsed:5.2f}s]"
            )
        total_elapsed = time.perf_counter() - overall_start
        report(
            f"Total: {len(topologies):,} topologies, "
            f"{len(booster_landings):,} landings, "
            f"{len(arm_adjacencies):,} arms, "
            f"{len(fire_zones):,} fire zones, "
            f"{len(dormant_survivals):,} survivals, "
            f"{len(bridge_feasibilities):,} bridges "
            f"[{total_elapsed:5.2f}s]"
        )

        return SpatialAtlas(
            topologies=topologies,
            booster_landings=booster_landings,
            arm_adjacencies=arm_adjacencies,
            fire_zones=fire_zones,
            dormant_survivals=dormant_survivals,
            bridge_feasibilities=bridge_feasibilities,
        )

    # ------------------------------------------------------------------
    # Topology construction
    # ------------------------------------------------------------------

    def _build_topology(
        self, exploded: frozenset[Position]
    ) -> SettleTopology:
        """Run settle() on a fully-filled board, capture the structural result."""
        board = self._filled_board()
        result = settle(
            self._gravity_dag,
            board,
            exploded,
            self._config.gravity,
        )
        refill_per_column = _refill_per_column(
            result.empty_positions, self._config.board.num_reels
        )
        diagonal, mapping = _trace_moves(result.move_steps)
        return SettleTopology(
            empty_positions=frozenset(result.empty_positions),
            refill_per_column=refill_per_column,
            has_diagonal_redistribution=diagonal,
            gravity_mapping=mapping,
        )

    def _filled_board(self) -> Board:
        """A fully-populated, cluster-free board — the baseline for settle().

        Uses `syms[(reel + row) % len(syms)]` over `symbols_in_tier(ANY)`.
        This matches `test_gravity.py::_filled_board` (the reference pattern
        used across the settle test suite) and guarantees no pre-existing
        clusters that could interact with settle edge cases. Symbol identity
        never influences settle, but adjacent same-symbol regions would
        break downstream consumers that might peek at the board.
        """
        syms = symbols_in_tier(SymbolTier.ANY, self._config.symbols)
        board = Board.empty(self._config.board)
        for reel in range(self._config.board.num_reels):
            for row in range(self._config.board.num_rows):
                board.set(Position(reel, row), syms[(reel + row) % len(syms)])
        return board

    # ------------------------------------------------------------------
    # Booster landing + arm adjacency + fire zones
    # ------------------------------------------------------------------

    def _index_landing_and_arm(
        self,
        profile: ColumnProfile,
        exploded: frozenset[Position],
        topology: SettleTopology,
        landings: dict[tuple[ColumnProfile, str, int], BoosterLandingEntry],
        arm_adjacencies: dict[
            tuple[ColumnProfile, str, int], ArmAdjacencyEntry
        ],
        fire_zones: dict[tuple[str, Position, str | None], frozenset[Position]],
    ) -> None:
        """For each profile that spawns a booster, index landing, arm
        adjacency, and fire zones together.

        They share the (profile, booster_type, centroid_col) key because a
        single centroid drives all three derivations — splitting the work
        across methods would force a second centroid computation per profile.
        """
        booster_symbol = self._booster_rules.booster_type_for_size(profile.total)
        if booster_symbol is None:
            return
        booster_type = booster_symbol.name
        centroid = self._booster_rules.compute_centroid(exploded)
        ctx = self._landing_eval.evaluate(
            exploded, self._filled_board(), booster_type
        )
        landing_score = self._landing_eval.score(ctx)

        landings[(profile, booster_type, centroid.reel)] = BoosterLandingEntry(
            landing_position=ctx.landing_position,
            adjacent_refill=frozenset(ctx.adjacent_refill),
            landing_score=landing_score,
        )
        arm_adjacencies[(profile, booster_type, centroid.reel)] = (
            self._build_arm_entry(ctx.landing_position, topology)
        )

        # Fire zones — only chain initiators get them (R, B per default config).
        # Dispatch table keyed on booster type; no if/elif over types.
        fire_builders = self._fire_zone_builders()
        if booster_type in fire_builders:
            fire_builders[booster_type](
                booster_type, ctx.landing_position, landing_score, fire_zones,
            )

    def _build_arm_entry(
        self,
        landing: Position,
        topology: SettleTopology,
    ) -> ArmAdjacencyEntry:
        """Compose ArmAdjacencyEntry from existing geometry primitives.

        Reuses `orthogonal_neighbors` and the topology's `empty_positions` —
        no new adjacency logic. `sufficient` derives from board's minimum
        cluster size: an arming cluster needs at least min_cluster_size-1
        adjacent refill cells to bridge into the booster's landing zone.
        """
        adjacent = frozenset(
            pos for pos in orthogonal_neighbors(landing, self._config.board)
            if pos in topology.empty_positions
        )
        threshold = max(1, self._config.board.min_cluster_size - 1)
        return ArmAdjacencyEntry(
            adjacent_refill=adjacent,
            adjacent_count=len(adjacent),
            sufficient=len(adjacent) >= threshold,
        )

    def _fire_zone_builders(
        self,
    ) -> dict[
        str,
        Callable[
            [str, Position, float,
             dict[tuple[str, Position, str | None], frozenset[Position]]],
            None,
        ],
    ]:
        """Dict dispatch table for per-booster fire-zone indexing.

        Registers one handler per known initiator; the result is filtered
        against `config.boosters.chain_initiators` so the builder only
        indexes types the game actually uses. Adding a new initiator means
        adding one method and one row in the registry — no branches.
        """
        registry: dict[
            str,
            Callable[
                [str, Position, float,
                 dict[tuple[str, Position, str | None], frozenset[Position]]],
                None,
            ],
        ] = {
            "R": self._write_rocket_fire_zones,
            "B": self._write_bomb_fire_zones,
        }
        return {
            initiator: registry[initiator]
            for initiator in self._config.boosters.chain_initiators
            if initiator in registry
        }

    def _write_rocket_fire_zones(
        self,
        booster_type: str,
        landing: Position,
        landing_score: float,
        fire_zones: dict[tuple[str, Position, str | None], frozenset[Position]],
    ) -> None:
        """Emit H-, V-, and None-keyed entries for a rocket landing.

        The None entry points at whichever orientation scores higher via the
        landing score — so the query's unconstrained path (arc without a
        rocket_orientation pin) is a single dict lookup, not a branch.

        Orientation scores are approximated by fire-zone size: the larger
        zone covers more cells, which correlates with greater chain reach —
        a cheap proxy that avoids re-scoring through BoosterLandingEvaluator
        and stays consistent with the "structural guidance" role of the atlas.
        """
        h_zone = self._booster_rules.rocket_path(landing, "H")
        v_zone = self._booster_rules.rocket_path(landing, "V")
        fire_zones[(booster_type, landing, "H")] = h_zone
        fire_zones[(booster_type, landing, "V")] = v_zone
        # landing_score is included in the signature for symmetry with the
        # bomb handler; rocket uses zone size as the tiebreaker since both
        # orientations share the same landing score.
        _ = landing_score
        preferred = h_zone if len(h_zone) >= len(v_zone) else v_zone
        fire_zones[(booster_type, landing, None)] = preferred

    def _write_bomb_fire_zones(
        self,
        booster_type: str,
        landing: Position,
        landing_score: float,
        fire_zones: dict[tuple[str, Position, str | None], frozenset[Position]],
    ) -> None:
        """Emit a single None-keyed entry for a bomb landing.

        Bombs have no orientation — the None key is the only valid lookup.
        """
        _ = landing_score  # Signature parity with the rocket handler.
        fire_zones[(booster_type, landing, None)] = (
            self._booster_rules.bomb_blast(landing)
        )

    # ------------------------------------------------------------------
    # Dormant survival index
    # ------------------------------------------------------------------

    def _index_dormants(
        self,
        profile: ColumnProfile,
        exploded: frozenset[Position],
        topology: SettleTopology,
        survivals: dict[tuple[ColumnProfile, int], DormantSurvivalEntry],
    ) -> None:
        """For each column, determine whether a dormant booster at its lowest
        non-exploded cell survives the profile's explosion.

        Survival is a column-level answer — the atlas consumer knows the
        dormant's column at runtime; exact row is resolved by tracing through
        gravity_mapping.
        """
        num_reels = self._config.board.num_reels
        num_rows = self._config.board.num_rows
        for reel in range(num_reels):
            dormant = self._pick_dormant_slot(reel, exploded, num_rows)
            if dormant is None:
                survivals[(profile, reel)] = DormantSurvivalEntry(
                    survives=False,
                    post_gravity_position=None,
                    column_shift=False,
                )
                continue
            post = self._trace_dormant(dormant, exploded)
            survivals[(profile, reel)] = DormantSurvivalEntry(
                survives=post is not None,
                post_gravity_position=post,
                column_shift=post is not None and post.reel != dormant.reel,
            )

    def _pick_dormant_slot(
        self, reel: int, exploded: frozenset[Position], num_rows: int
    ) -> Position | None:
        """Topmost cell in the column not in the explosion set.

        Returns None if the whole column is exploded — no survivable slot.
        """
        for row in range(num_rows):
            candidate = Position(reel, row)
            if candidate not in exploded:
                return candidate
        return None

    def _trace_dormant(
        self,
        dormant: Position,
        exploded: frozenset[Position],
    ) -> Position | None:
        """Run settle() with the dormant's neighbors pruned, return its
        destination. None when gravity displaces the dormant off its column
        in a way that leaves no trace (explosion set includes it)."""
        if dormant in exploded:
            return None
        board = self._filled_board()
        result = settle(
            self._gravity_dag,
            board,
            exploded,
            self._config.gravity,
        )
        return _trace_through_settle(dormant, result.move_steps)

    # ------------------------------------------------------------------
    # Bridge feasibility index
    # ------------------------------------------------------------------

    def _index_bridge_feasibility(
        self,
        profile: ColumnProfile,
        topology: SettleTopology,
        bridge_feasibilities: dict[
            tuple[ColumnProfile, int], BridgeFeasibilityEntry
        ],
    ) -> None:
        """For profiles in the wild spawn range, record bridgeable gaps.

        A gap is a zero-count column with non-zero counts on both sides.
        Eligibility is gated on booster_type_for_size returning the wild
        symbol, so the set of bridge-eligible sizes derives from
        config.boosters.spawn_thresholds — no hardcoded size check.

        A3: Reachability is computed by BFS from the wild's candidate
        position through `topology.empty_positions`. Cells reachable from
        the wild are partitioned into left/right of the gap column. A
        wall in the post-settle empties (BFS reaches only one side, or
        neither) marks the gap as `structurally_unbridgeable` — no WFC
        refill can ever form a continuation cluster across the wild.
        """
        booster_symbol = self._booster_rules.booster_type_for_size(profile.total)
        if booster_symbol is None or booster_symbol.name != "W":
            return
        counts = profile.counts
        num_reels = len(counts)
        for col in range(num_reels):
            if counts[col] != 0:
                continue
            left = frozenset(c for c in range(col) if counts[c] > 0)
            right = frozenset(
                c for c in range(col + 1, num_reels) if counts[c] > 0
            )
            if not left or not right:
                continue
            # Adjacency counts from the columns immediately flanking the gap
            left_adj = counts[col - 1]
            right_adj = counts[col + 1] if col + 1 < num_reels else 0
            reachable_left, reachable_right = self._bfs_reachable_sides(
                topology, col,
            )
            structurally_unbridgeable = (
                reachable_left == 0 or reachable_right == 0
            )
            # Score normalized by min_cluster_size — the smallest cluster a
            # bridge can form on either side. Clipped to [0.0, 1.0].
            min_size = self._config.board.min_cluster_size
            score = min(1.0, min(reachable_left, reachable_right) / min_size)
            bridge_feasibilities[(profile, col)] = BridgeFeasibilityEntry(
                gap_column=col,
                left_columns=left,
                right_columns=right,
                left_adjacency_count=left_adj,
                right_adjacency_count=right_adj,
                bridge_score=score,
                reachable_left=reachable_left,
                reachable_right=reachable_right,
                structurally_unbridgeable=structurally_unbridgeable,
            )

    def _bfs_reachable_sides(
        self,
        topology: SettleTopology,
        gap_column: int,
    ) -> tuple[int, int]:
        """BFS through topology.empty_positions seeded at the gap column.

        Returns (reachable_left_count, reachable_right_count). The wild's
        own column is excluded from both counts — it acts as the bridge,
        not as a side member.
        """
        empty_set = topology.empty_positions
        # Seed from any empty cell in the gap column — the wild lands there
        # and bridges through to the reachable adjacents.
        seeds = [p for p in empty_set if p.reel == gap_column]
        if not seeds:
            return (0, 0)
        visited: set[Position] = set()
        queue: deque[Position] = deque()
        for seed in seeds:
            if seed not in visited:
                visited.add(seed)
                queue.append(seed)
        while queue:
            pos = queue.popleft()
            for nbr in orthogonal_neighbors(pos, self._config.board):
                if nbr in empty_set and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        # Partition reachable cells by column relative to the gap. The gap
        # column itself is excluded so we count only true bridge sides.
        reachable_left = sum(1 for p in visited if p.reel < gap_column)
        reachable_right = sum(1 for p in visited if p.reel > gap_column)
        return (reachable_left, reachable_right)


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def _refill_per_column(
    empty_positions: Iterable[Position], num_reels: int
) -> tuple[int, ...]:
    """Tally empty cells per column — the post-settle refill demand."""
    counts = [0] * num_reels
    for pos in empty_positions:
        counts[pos.reel] += 1
    return tuple(counts)


def _trace_moves(
    move_steps: tuple[tuple[tuple[str, Position, Position], ...], ...],
) -> tuple[bool, dict[int, tuple[int, ...]]]:
    """Summarize gravity moves for the topology's debug fields.

    Returns (has_diagonal_redistribution, per_column_source_rows) where the
    column key is the destination reel and the value is the tuple of source
    rows that fed that column (sorted, de-duplicated).
    """
    diagonal = False
    mapping: dict[int, set[int]] = {}
    for pass_moves in move_steps:
        for _, source, dest in pass_moves:
            if source.reel != dest.reel:
                diagonal = True
            mapping.setdefault(dest.reel, set()).add(source.row)
    return diagonal, {
        reel: tuple(sorted(rows)) for reel, rows in mapping.items()
    }


def _trace_through_settle(
    start: Position,
    move_steps: tuple[tuple[tuple[str, Position, Position], ...], ...],
) -> Position:
    """Follow a single cell's trajectory across settle passes.

    Matches the pattern used by ForwardSimulator's internal position tracing:
    whenever a move's source equals the current position, jump to the
    destination and continue.
    """
    current = start
    for pass_moves in move_steps:
        for _, source, dest in pass_moves:
            if source == current:
                current = dest
                break
    return current
