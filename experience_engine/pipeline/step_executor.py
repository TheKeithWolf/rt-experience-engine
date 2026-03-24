"""Step executor — translates a StepIntent into a filled board.

Pins the reasoner's constrained and strategic cells on the board, then
delegates to WFC to fill remaining cells. The executor contains no game
logic — it mechanically applies the intent's cell placements and WFC
configuration.

WFCBoardFiller is instantiated fresh per execute() call because each
intent may specify different propagators and symbol weights.
"""

from __future__ import annotations

import random

from ..board_filler.wfc_solver import WFCBoardFiller
from ..config.schema import MasterConfig
from ..primitives.board import Board, Position
from ..primitives.symbols import Symbol
from ..step_reasoner.intent import StepIntent


class StepExecutionFailed(Exception):
    """Raised when step execution fails after all internal retries."""


class StepExecutor:
    """Executes a StepIntent by pinning cells and WFC-filling the rest.

    The executor does NOT reason — strategies already computed which cells
    to place where. This class mechanically applies those decisions and
    fills unconstrained cells via WFC.
    """

    __slots__ = ("_config",)

    def __init__(self, config: MasterConfig) -> None:
        self._config = config

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
        4. Instantiates WFCBoardFiller with per-intent propagators
        5. Fills remaining empty cells via WFC
        6. Returns the fully-filled board

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
            # Constrained cells are cluster cores — strategic seeds must not overwrite them
            if pos in pinned:
                continue
            filled.set(pos, sym)
            pinned.add(pos)

        # Fresh filler per call — intent propagators are the COMPLETE set for
        # this step. Default propagators are not used because strategies already
        # specify the appropriate constraints for their context (e.g., NoCluster
        # for cascade boards, MaxComponent for dead boards).
        filler = WFCBoardFiller(self._config, use_defaults=False)
        for propagator in intent.wfc_propagators:
            filler.add_propagator(propagator)

        # WFC fills all non-pinned cells with standard symbols
        filled = filler.fill(
            board=filled,
            pinned=frozenset(pinned),
            rng=rng,
            weights=intent.wfc_symbol_weights or None,
        )

        return filled
