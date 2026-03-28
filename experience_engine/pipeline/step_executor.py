"""Step executor — translates a StepIntent into a filled board.

Pins the reasoner's constrained and strategic cells on the board, then
delegates to WFC to fill remaining cells. The executor contains no game
logic — it mechanically applies the intent's cell placements and WFC
configuration.

WFCBoardFiller is instantiated fresh per execute() call because each
intent may specify different propagators and symbol weights.

When the intent has a planned_explosion and the config includes gravity_wfc,
the executor builds gravity-aware FillConstraints (spatial weights, post-gravity
propagation, gravity-group ordering) and uses fill_step() instead of fill().
"""

from __future__ import annotations

import random

from ..board_filler.fill_constraints import FillConstraints
from ..board_filler.gravity_adjacency import PostGravityAdjacency
from ..board_filler.gravity_ordering import (
    GravityAwareEntropySelector,
    GravityGroupComputer,
)
from ..board_filler.propagators import PostGravityPropagator
from ..board_filler.spatial_weights import (
    SpatialWeightMap, build_reserve_zone, build_weight_zones,
)
from ..board_filler.wfc_solver import WFCBoardFiller
from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.gravity import GravityDAG
from ..primitives.symbols import Symbol
from ..step_reasoner.intent import StepIntent


class StepExecutionFailed(Exception):
    """Raised when step execution fails after all internal retries."""


class StepExecutor:
    """Executes a StepIntent by pinning cells and WFC-filling the rest.

    The executor does NOT reason — strategies already computed which cells
    to place where. This class mechanically applies those decisions and
    fills unconstrained cells via WFC.

    When gravity_dag and config.gravity_wfc are both available and the intent
    has a planned_explosion, the executor builds gravity-aware constraints
    and uses WFC fill_step() for improved post-gravity correctness.
    """

    __slots__ = ("_config", "_gravity_dag")

    def __init__(
        self,
        config: MasterConfig,
        gravity_dag: GravityDAG | None = None,
    ) -> None:
        self._config = config
        # Precomputed gravity DAG — reused across all execute() calls.
        # None disables gravity-aware WFC (backward compat).
        self._gravity_dag = gravity_dag

    def execute(
        self,
        intent: StepIntent,
        board: Board,
        rng: random.Random,
    ) -> Board:
        """Pin constrained + strategic cells, then WFC-fill remaining cells.

        1. Copies the board (input is never mutated)
        2. Applies constrained cells (cluster cores, scatters — must-place)
        3. Applies strategic cells (gravity seeds — should-place)
        4. Dispatches to gravity-aware or baseline fill path
        5. Returns the fully-filled board

        Raises FillFailed (from WFC) if the fill is unsatisfiable — the
        caller (cascade generator) handles retry at the instance level.
        """
        filled = board.copy()

        # Pin cells the reasoner decided must be placed
        pinned: set[Position] = set()
        for pos, sym in intent.constrained_cells.items():
            filled.set(pos, sym)
            pinned.add(pos)

        # Cap strategic cells to prevent over-constraining future steps
        strategic = intent.strategic_cells
        max_strategic = self._config.reasoner.max_strategic_cells_per_step
        if len(strategic) > max_strategic:
            # Keep first N cells — insertion order reflects strategy priority
            strategic = dict(list(strategic.items())[:max_strategic])

        for pos, sym in strategic.items():
            # Strategic seeds model post-explosion board state but execute on the
            # pre-explosion board — skip pinned cells and positions that already hold
            # symbols (wilds, boosters) to prevent overwriting them (BP-EXEC-GUARD)
            if pos in pinned or filled.get(pos) is not None:
                continue
            filled.set(pos, sym)
            pinned.add(pos)

        # Dispatch: gravity-aware path when explosion is planned AND config supports it
        if (
            intent.planned_explosion is not None
            and self._gravity_dag is not None
            and self._config.gravity_wfc is not None
        ):
            noise_cells = [
                pos for pos in filled.all_positions()
                if pos not in pinned and filled.get(pos) is None
            ]
            constraints = self._build_gravity_aware_constraints(
                intent, filled, frozenset(pinned), noise_cells,
            )
            filler = WFCBoardFiller(self._config, use_defaults=False)
            filled = filler.fill_step(
                filled, frozenset(pinned), noise_cells, constraints, rng,
            )
        else:
            # Baseline path — existing behavior unchanged
            filler = WFCBoardFiller(self._config, use_defaults=False)
            for propagator in intent.wfc_propagators:
                filler.add_propagator(propagator)
            filled = filler.fill(
                board=filled,
                pinned=frozenset(pinned),
                rng=rng,
                weights=intent.wfc_symbol_weights or None,
            )

        return filled

    def _build_gravity_aware_constraints(
        self,
        intent: StepIntent,
        board: Board,
        pinned: frozenset[Position],
        noise_cells: list[Position],
    ) -> FillConstraints:
        """Assemble all three gravity-aware mechanisms into FillConstraints.

        1. Spatial weights: per-cell suppression zones from the cluster plan
        2. Post-gravity adjacency + propagator: virtual neighbor graph
        3. Gravity-group ordering: collapse largest virtual group first
        """
        gwfc = self._config.gravity_wfc

        # Mechanism 1: spatial weight map from intent's cluster/seed positions
        zones = build_weight_zones(intent, self._config)

        # Spatial intelligence reserve zone — suppresses standard symbols in
        # positions the next step needs clear for cluster formation
        if intent.reserve_zone and self._config.spatial_intelligence:
            zones.append(build_reserve_zone(
                intent.reserve_zone,
                self._config.spatial_intelligence.reserve_suppression_multiplier,
                self._config.symbols,
            ))

        base_weights = intent.wfc_symbol_weights or {}
        spatial_weights = SpatialWeightMap(
            base_weights, zones, gwfc.min_symbol_weight,
        ) if zones else None

        # Mechanism 2: post-gravity virtual adjacency + propagator
        adjacency = PostGravityAdjacency(
            board, intent.planned_explosion,
            self._gravity_dag, self._config.gravity, self._config.board,
        )
        gravity_propagator = PostGravityPropagator(
            adjacency.virtual_neighbors,
            self._config.board.min_cluster_size,
            wild_positions=intent.predicted_wild_positions,
        )

        # Mechanism 3: gravity-group collapse ordering
        group_computer = GravityGroupComputer(adjacency)
        groups = group_computer.compute_groups(set(noise_cells))
        selector = GravityAwareEntropySelector(groups)

        # Combine strategy propagators with gravity propagator
        propagators = list(intent.wfc_propagators) + [gravity_propagator]

        return FillConstraints(
            propagators=propagators,
            spatial_weights=spatial_weights,
            gravity_adjacency=adjacency,
            gravity_groups=selector,
            flat_symbol_weights=intent.wfc_symbol_weights or {},
        )

    @staticmethod
    def _build_baseline_constraints(intent: StepIntent) -> FillConstraints:
        """Build FillConstraints for baseline (non-gravity-aware) fills.

        All mechanism fields are None — WFC uses standard min-entropy
        selection with flat weights.
        """
        return FillConstraints(
            propagators=list(intent.wfc_propagators),
            spatial_weights=None,
            gravity_adjacency=None,
            gravity_groups=None,
            flat_symbol_weights=intent.wfc_symbol_weights or {},
        )
