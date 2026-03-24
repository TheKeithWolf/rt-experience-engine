"""Static instance generator — CSP → WFC pipeline for cascade_depth=0.

Generates single-step board instances by placing clusters, near-misses,
and scatters via the CSP spatial solver, then filling remaining cells via
the WFC board filler. Payout computed from detected clusters using the paytable.

All thresholds from MasterConfig + ArchetypeSignature — zero hardcoded values.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ..archetypes.registry import ArchetypeRegistry, ArchetypeSignature
from ..board_filler.propagators import WildBridgePropagator
from ..board_filler.wfc_solver import FillFailed, WFCBoardFiller
from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.cluster_detection import Cluster, detect_clusters
from ..primitives.gravity import GravityDAG, settle
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.paytable import Paytable
from ..primitives.symbols import (
    Symbol,
    SymbolTier,
    symbol_from_name,
    symbols_in_tier,
)
from ..spatial_solver.solver import CSPSpatialSolver, SolveFailed
from ..variance.hints import VarianceHints
from .data_types import (
    GeneratedInstance,
    GenerationResult,
    GravityRecord,
    build_gravity_record,
    compute_refill_entries,
)

if TYPE_CHECKING:
    from ..spatial_solver.data_types import SpatialStep


class StaticInstanceGenerator:
    """Generates static (cascade_depth=0) board instances via CSP → WFC.

    For dead archetypes: CSP places near-misses and scatters, WFC fills with
    MaxComponentPropagator to prevent any component exceeding the dead threshold.
    For t1 static: CSP places clusters + NMs + scatters, WFC fills remaining
    cells with the default NoClusterPropagator.
    """

    __slots__ = ("_config", "_registry", "_paytable", "_gravity_dag")

    def __init__(
        self,
        config: MasterConfig,
        registry: ArchetypeRegistry,
        gravity_dag: GravityDAG,
    ) -> None:
        self._config = config
        self._registry = registry
        self._paytable = Paytable(
            config.paytable, config.centipayout, config.win_levels,
        )
        self._gravity_dag = gravity_dag

    def generate(
        self,
        archetype_id: str,
        sim_id: int,
        hints: VarianceHints,
        rng: random.Random,
    ) -> GenerationResult:
        """Generate a single static instance.

        Retries up to max_retries_per_instance on solver failures.
        Returns GenerationResult with success=False if all retries exhausted.
        """
        sig = self._registry.get(archetype_id)
        max_retries = self._config.solvers.max_retries_per_instance

        for attempt in range(1, max_retries + 1):
            try:
                instance = self._attempt_generation(sig, sim_id, hints, rng)
                return GenerationResult(
                    instance=instance,
                    success=True,
                    attempts=attempt,
                    failure_reason=None,
                )
            except (SolveFailed, FillFailed) as exc:
                last_error = str(exc)
                continue

        return GenerationResult(
            instance=None,
            success=False,
            attempts=max_retries,
            failure_reason=f"exhausted {max_retries} retries: {last_error}",
        )

    def _attempt_generation(
        self,
        sig: ArchetypeSignature,
        sim_id: int,
        hints: VarianceHints,
        rng: random.Random,
    ) -> GeneratedInstance:
        """Single attempt at generating an instance. Raises on failure."""
        # 1. Derive CSP specs from signature + variance hints
        cluster_specs, nm_specs, scatter_count = self._derive_csp_specs(
            sig, hints, rng,
        )

        # 2. Solve spatial placement via CSP
        csp_solver = CSPSpatialSolver(self._config)
        spatial_step = csp_solver.solve_step(
            cluster_specs=cluster_specs,
            near_miss_specs=nm_specs,
            scatter_count=scatter_count,
            booster_specs=[],
            spatial_bias=hints.spatial_bias,
            rng=rng,
        )

        # 3. Build board with CSP placements pinned
        board = Board.empty(self._config.board)
        pinned = self._apply_spatial_step(board, spatial_step)

        # 4. Fill remaining cells via WFC
        wfc_filler = self._build_wfc_filler(sig)
        # Prevent WFC from bridging clusters through wild positions
        wild_positions = frozenset(wp.position for wp in spatial_step.wild_placements)
        if wild_positions:
            wfc_filler.add_propagator(WildBridgePropagator(wild_positions))
        board = wfc_filler.fill(
            board=board,
            pinned=pinned,
            rng=rng,
            weights=hints.symbol_weights,
        )

        # 5. Detect clusters and compute payout
        clusters = detect_clusters(board, self._config)
        payout, centipayout, win_level = self._compute_payout(clusters)

        # 6. Compute gravity record for the post-win animation (explode → gravity → refill)
        gravity_record = (
            self._compute_gravity_record(board, clusters, rng)
            if clusters else None
        )

        return GeneratedInstance(
            sim_id=sim_id,
            archetype_id=sig.id,
            family=sig.family,
            criteria=sig.criteria,
            board=board,
            spatial_step=spatial_step,
            payout=payout,
            centipayout=centipayout,
            win_level=win_level,
            gravity_record=gravity_record,
        )

    def _derive_csp_specs(
        self,
        sig: ArchetypeSignature,
        hints: VarianceHints,
        rng: random.Random,
    ) -> tuple[list[tuple[Symbol, int]], list[tuple[Symbol, int]], int]:
        """Convert archetype signature into concrete CSP solve_step arguments.

        Randomly selects within signature ranges using variance preferences
        to steer toward underrepresented choices.
        """
        # Cluster count — pick within range
        cluster_count = rng.randint(
            sig.required_cluster_count.min_val,
            sig.required_cluster_count.max_val,
        )

        # Determine which symbols are allowed for clusters
        if sig.required_cluster_symbols is not None:
            allowed_symbols = list(symbols_in_tier(
                sig.required_cluster_symbols, self._config.symbols,
            ))
        else:
            allowed_symbols = [
                symbol_from_name(n) for n in self._config.symbols.standard
            ]

        # Build cluster specs: (symbol, size) for each cluster
        cluster_specs: list[tuple[Symbol, int]] = []
        for i in range(cluster_count):
            # Pick size from the applicable range
            if sig.required_cluster_sizes:
                # Use first range for all clusters (Phase 4 archetypes have one range)
                size_range = sig.required_cluster_sizes[
                    min(i, len(sig.required_cluster_sizes) - 1)
                ]
                size = rng.randint(size_range.min_val, size_range.max_val)
            else:
                size = self._config.board.min_cluster_size

            # Pick symbol — prefer least-used from variance hints
            sym = self._pick_symbol(allowed_symbols, hints, rng)
            cluster_specs.append((sym, size))

        # Near-miss count — pick within range
        nm_count = rng.randint(
            sig.required_near_miss_count.min_val,
            sig.required_near_miss_count.max_val,
        )

        # Determine allowed NM symbols
        if sig.required_near_miss_symbol_tier is not None:
            nm_allowed = list(symbols_in_tier(
                sig.required_near_miss_symbol_tier, self._config.symbols,
            ))
        else:
            nm_allowed = [
                symbol_from_name(n) for n in self._config.symbols.standard
            ]

        # Near-miss size is always min_cluster_size - 1
        nm_size = self._config.board.min_cluster_size - 1
        nm_specs: list[tuple[Symbol, int]] = []
        for _ in range(nm_count):
            sym = self._pick_nm_symbol(nm_allowed, hints, rng)
            nm_specs.append((sym, nm_size))

        # Scatter count — pick within range
        scatter_count = rng.randint(
            sig.required_scatter_count.min_val,
            sig.required_scatter_count.max_val,
        )

        return cluster_specs, nm_specs, scatter_count

    def _pick_symbol(
        self,
        allowed: list[Symbol],
        hints: VarianceHints,
        rng: random.Random,
    ) -> Symbol:
        """Pick a cluster symbol with preference toward underrepresented ones."""
        if not allowed:
            raise ValueError("no allowed symbols for cluster")
        # Weight by variance symbol_weights — higher weight = more preferred
        weights = [hints.symbol_weights.get(s, 1.0) for s in allowed]
        return rng.choices(allowed, weights=weights, k=1)[0]

    def _pick_nm_symbol(
        self,
        allowed: list[Symbol],
        hints: VarianceHints,
        rng: random.Random,
    ) -> Symbol:
        """Pick a near-miss symbol with preference toward underrepresented ones."""
        if not allowed:
            raise ValueError("no allowed symbols for near-miss")
        # Use near_miss_symbol_preference if available
        for sym in hints.near_miss_symbol_preference:
            if sym in allowed:
                # Probabilistic — 60% chance to pick the least-used, else random
                if rng.random() < 0.6:
                    return sym
                break
        return rng.choice(allowed)

    def _apply_spatial_step(
        self,
        board: Board,
        step: SpatialStep,
    ) -> frozenset[Position]:
        """Apply CSP spatial assignments to the board, return pinned positions."""
        pinned: set[Position] = set()

        # Place cluster symbols
        for cluster in step.clusters:
            for pos in cluster.positions:
                board.set(pos, cluster.symbol)
                pinned.add(pos)

        # Place near-miss symbols
        for nm in step.near_misses:
            for pos in nm.positions:
                board.set(pos, nm.symbol)
                pinned.add(pos)

        # Place scatters
        for pos in step.scatter_positions:
            board.set(pos, Symbol.S)
            pinned.add(pos)

        return frozenset(pinned)

    def _build_wfc_filler(self, sig: ArchetypeSignature) -> WFCBoardFiller:
        """WFC filler — default propagators prevent accidental NMs.

        WFCBoardFiller includes MaxComponentPropagator(nm_fill_cap) by default,
        so no manual propagator addition needed for any archetype.
        """
        return WFCBoardFiller(self._config)

    def _compute_payout(
        self, clusters: list[Cluster],
    ) -> tuple[float, int, int]:
        """Compute total payout from pre-detected clusters.

        Static boards have grid multipliers at initial value — minimum_contribution
        applies, so effective multiplier = 1 for each cluster.
        """
        grid_mults = GridMultiplierGrid(
            self._config.grid_multiplier, self._config.board,
        )

        total_payout = 0.0
        for cluster in clusters:
            total_payout += self._paytable.compute_cluster_payout(
                cluster, grid_mults,
            )

        centipayout = self._paytable.to_centipayout(total_payout)
        win_level = self._paytable.get_win_level(total_payout)

        return total_payout, centipayout, win_level

    def _compute_gravity_record(
        self,
        board: Board,
        clusters: list[Cluster],
        rng: random.Random,
    ) -> GravityRecord:
        """Compute gravity settle data for the post-win animation.

        Explodes cluster positions, runs gravity to settle remaining symbols,
        and generates cosmetic refill symbols for the empty spaces. The refill
        has no gameplay consequence — static wins have no follow-up cascade.
        """
        # Collect all cluster positions (standard + wild) for explosion
        exploded: frozenset[Position] = frozenset(
            pos
            for cluster in clusters
            for pos in cluster.positions | cluster.wild_positions
        )

        settle_result = settle(
            self._gravity_dag, board, exploded, self._config.gravity,
        )

        refill_entries = compute_refill_entries(
            settle_result.empty_positions,
            self._config.symbols.standard,
            rng,
        )
        return build_gravity_record(exploded, settle_result, refill_entries)
