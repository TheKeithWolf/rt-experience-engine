"""Wild bridge strategy — forms a cluster through a wild symbol.

The wild is at a known position on the board. This strategy scans the
full board for same-symbol chains reachable through the wild via BFS,
ranks candidates by how few additional cells are needed, and places
only the shortfall. Phase constraints (tier, spawns) are read as data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ...board_filler.propagators import (
    NoClusterPropagator,
    NoSpecialSymbolPropagator,
    WildBridgePropagator,
)
from ...config.schema import BoardConfig, MasterConfig
from ...primitives.board import Board, Position, orthogonal_neighbors
from ...primitives.cluster_detection import detect_clusters, detect_components
from ...primitives.symbols import Symbol, SymbolTier, is_standard, tier_of
from ..context import BoardContext
from ..intent import StepIntent, StepType
from ..progress import ProgressTracker
from ..services.cluster_builder import ClusterBuilder
from ..services.forward_simulator import ForwardSimulator
from ..services.influence_map import DemandSpec
from ..services.seed_planner import SeedPlanner, build_cluster_exclusions
from ..services.spatial_context import StepSpatialContext
from ...archetypes.registry import ArchetypeSignature
from ...pipeline.protocols import Range
from ...variance.hints import VarianceHints


@dataclass(frozen=True, slots=True)
class BridgeCandidate:
    """One possible bridge through the wild.

    Produced by _scan_bridge_candidates. Ranked by needed ascending.
    """

    symbol: Symbol
    # Existing same-symbol cells reachable from the wild via orthogonal BFS
    reachable: frozenset[Position]
    # len(reachable) + 1 (C-COMPAT-1: wild counts toward cluster size)
    score: int
    # max(0, target_size - score) — cells still needed to complete the bridge
    needed: int
    # Empty cells adjacent to reachable | {wild} — valid expansion sites
    growth_sites: tuple[Position, ...]


class WildBridgeStrategy:
    """Plans a cluster that uses a wild as a bridge.

    Scans the board for same-symbol chains reachable through the wild,
    ranks candidates by fewest cells needed, and delegates placement to
    ClusterBuilder with an expanded adjacency zone.
    """

    __slots__ = (
        "_config", "_forward_sim", "_cluster_builder",
        "_seed_planner", "_rng", "_spatial",
    )

    def __init__(
        self,
        config: MasterConfig,
        forward_sim: ForwardSimulator,
        cluster_builder: ClusterBuilder,
        seed_planner: SeedPlanner,
        rng: random.Random,
        spatial: StepSpatialContext | None = None,
    ) -> None:
        self._config = config
        self._forward_sim = forward_sim
        self._cluster_builder = cluster_builder
        self._seed_planner = seed_planner
        self._rng = rng
        self._spatial = spatial

    # ------------------------------------------------------------------
    # Board scanning
    # ------------------------------------------------------------------

    @staticmethod
    def _reachable_through_wild(
        wild_pos: Position,
        symbol: Symbol,
        board: Board,
        board_config: BoardConfig,
    ) -> frozenset[Position]:
        """Same-symbol cells reachable from the wild via orthogonal adjacency.

        Delegates to detect_components (DRY) and filters to components
        that have at least one cell adjacent to the wild.
        """
        wild_neighbors = frozenset(orthogonal_neighbors(wild_pos, board_config))
        components = detect_components(board, symbol, board_config)
        reachable: set[Position] = set()
        for component in components:
            if component & wild_neighbors:
                reachable.update(component)
        return frozenset(reachable)

    def _scan_bridge_candidates(
        self,
        wild_pos: Position,
        context: BoardContext,
        target_size: int,
        required_tier: SymbolTier | None,
    ) -> list[BridgeCandidate]:
        """Scan the board for all viable bridge clusters through the wild.

        For each standard symbol adjacent to the wild (filtered by tier when
        the phase constrains it), discovers the full reachable component,
        computes how many cells are still needed, and identifies empty cells
        where those cells could be placed.

        Returns candidates sorted by needed ascending (best first).
        """
        board = context.board
        board_config = self._config.board

        # Unique standard symbols adjacent to the wild, filtered by tier
        seen_symbols: set[Symbol] = set()
        for pos in orthogonal_neighbors(wild_pos, board_config):
            sym = board.get(pos)
            if sym is None or not is_standard(sym, self._config.symbols):
                continue
            if (required_tier is not None
                    and required_tier is not SymbolTier.ANY
                    and tier_of(sym, self._config.symbols) is not required_tier):
                continue
            seen_symbols.add(sym)

        candidates: list[BridgeCandidate] = []
        for sym in seen_symbols:
            reachable = self._reachable_through_wild(
                wild_pos, sym, board, board_config,
            )
            # C-COMPAT-1: wild counts as 1 toward cluster size
            score = len(reachable) + 1
            needed = max(0, target_size - score)

            # Growth sites: empty cells adjacent to reachable set or wild
            expansion_zone = reachable | {wild_pos}
            seen_sites: set[Position] = set()
            growth_sites: list[Position] = []
            for cell in expansion_zone:
                for neighbor in orthogonal_neighbors(cell, board_config):
                    if (neighbor not in seen_sites
                            and neighbor not in expansion_zone
                            and board.get(neighbor) is None):
                        seen_sites.add(neighbor)
                        growth_sites.append(neighbor)

            candidates.append(BridgeCandidate(
                symbol=sym,
                reachable=reachable,
                score=score,
                needed=needed,
                growth_sites=tuple(growth_sites),
            ))

        candidates.sort(key=lambda c: c.needed)
        return candidates

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        wild_pos = context.active_wilds[0]
        target_size = self._select_target_size(signature, progress)
        phase = progress.current_phase()

        # Phase constraints — read as data, no per-archetype branching
        required_tier = (
            phase.cluster_symbol_tier if phase is not None else None
        )
        expected_spawns = (
            list(phase.spawns) if phase is not None and phase.spawns else []
        )

        # Scan the board — find what's already there
        candidates = self._scan_bridge_candidates(
            wild_pos, context, target_size, required_tier,
        )

        # Filter to candidates with enough growth room
        viable = [c for c in candidates if len(c.growth_sites) >= c.needed]

        if not viable:
            if candidates:
                viable = [candidates[0]]
            else:
                raise ValueError(
                    f"No viable bridge symbols adjacent to wild at {wild_pos}"
                )

        best = viable[0]
        bridge_symbol = best.symbol
        bridge_tier = tier_of(bridge_symbol, self._config.symbols)

        # Place only the cells needed to complete the bridge.
        # The scan already subtracted existing reachable cells from the target,
        # so bypass merge-policy reduction (which would double-count survivors).
        # ClusterBuilder's simple BFS growth places exactly best.needed cells.
        if best.needed == 0:
            bridge_positions: frozenset[Position] = frozenset()
        else:
            result = self._cluster_builder.find_positions(
                context, best.needed, self._rng, variance,
                # Expand from the full reachable set + wild, not just {wild_pos}
                must_be_adjacent_to=best.reachable | {wild_pos},
            )
            bridge_positions = result.planned_positions

        # Forward simulate: verify the bridge forms
        placements = {pos: bridge_symbol for pos in bridge_positions}
        hypothetical = self._forward_sim.build_hypothetical(context.board, placements)
        detected = detect_clusters(hypothetical, self._config)

        bridge_ok = any(
            c.symbol is bridge_symbol
            and wild_pos in c.wild_positions
            and c.size >= target_size
            for c in detected
        )
        if not bridge_ok:
            raise ValueError(
                f"Bridge verification failed: {bridge_symbol.name} "
                f"cluster through wild at {wild_pos} did not reach "
                f"target size {target_size}"
            )

        # Backward reasoning: seed the next step if not terminal
        strategic_cells: dict[Position, Symbol] = {}
        reserve_zone: frozenset[Position] | None = None
        if not progress.must_terminate_soon():
            # Explosion includes existing reachable cells + placed cells + wild
            explosion_zone = best.reachable | bridge_positions | {wild_pos}
            settle_result = self._forward_sim.simulate_explosion(
                hypothetical, frozenset(explosion_zone),
            )
            exclusions = build_cluster_exclusions(
                [(best.reachable | bridge_positions, bridge_symbol)],
                self._config.board,
            )

            if self._spatial and progress.dormant_boosters:
                target_booster = progress.dormant_boosters[0]
                demand = DemandSpec(
                    centroid=target_booster.position,
                    cluster_size=self._config.board.min_cluster_size,
                    booster_type=target_booster.booster_type,
                )
                influence = self._spatial.influence_map.compute(demand)
                reserve_zone = self._spatial.influence_map.reserve_zone(influence)

            strategic_cells = self._seed_planner.plan_generic_seeds(
                settle_result, progress, signature, variance, self._rng,
                exclusions=exclusions,
            )

        constrained = {pos: bridge_symbol for pos in bridge_positions}

        return StepIntent(
            step_type=StepType.CASCADE_CLUSTER,
            constrained_cells=constrained,
            strategic_cells=strategic_cells,
            expected_cluster_count=Range(1, 1),
            expected_cluster_sizes=[Range(target_size, target_size)],
            expected_cluster_tier=bridge_tier,
            expected_spawns=expected_spawns,
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=[
                NoSpecialSymbolPropagator(self._config.symbols),
                NoClusterPropagator(self._config.board.min_cluster_size),
                WildBridgePropagator(frozenset(context.active_wilds)),
            ],
            wfc_symbol_weights=variance.symbol_weights,
            predicted_post_gravity=None,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            planned_explosion=(
                frozenset(best.reachable | bridge_positions | {wild_pos})
                if bridge_positions or best.reachable else None
            ),
            is_terminal=False,
            reserve_zone=reserve_zone,
        )

    def _select_target_size(
        self,
        signature: ArchetypeSignature,
        progress: ProgressTracker,
    ) -> int:
        """Determine the target cluster size for the bridge.

        Reads from the current phase's cluster sizes so each narrative
        step honours its own size range (e.g. bridge_small → 5-6),
        rather than always using the signature-level union.
        """
        step_sizes = progress.current_step_size_ranges()
        if step_sizes:
            return step_sizes[0].min_val
        return self._config.board.min_cluster_size
