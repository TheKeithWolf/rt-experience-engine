"""Forward gravity simulator — hypothetical board reasoning for the step reasoner.

Wraps GravityDAG and gravity.settle() into a service interface. Strategies use
this to ask "what happens if I place these symbols here and then explode this
cluster?" without importing gravity primitives directly.

Used by: InitialCluster, CascadeCluster, WildBridge, BoosterArm, BoosterSetup.
"""

from __future__ import annotations

from ...config.schema import BoardConfig, GravityConfig
from ...primitives.board import Board, Position
from ...primitives.gravity import GravityDAG, SettleResult, settle
from ...primitives.symbols import Symbol


class ForwardSimulator:
    """Simulates gravity consequences of hypothetical placements.

    Single entry point for all gravity-related reasoning. Constructed once
    at engine init, shared across all strategies via dependency injection.
    """

    __slots__ = ("_dag", "_board_config", "_gravity_config")

    def __init__(
        self,
        dag: GravityDAG,
        board_config: BoardConfig,
        gravity_config: GravityConfig,
    ) -> None:
        self._dag = dag
        self._board_config = board_config
        self._gravity_config = gravity_config

    @property
    def board_config(self) -> BoardConfig:
        return self._board_config

    def build_hypothetical(
        self,
        board: Board,
        placements: dict[Position, Symbol],
    ) -> Board:
        """Create a board copy with hypothetical placements applied.

        The original board is never mutated — strategies can safely
        explore "what-if" scenarios without side effects.
        """
        result = board.copy()
        for pos, sym in placements.items():
            result.set(pos, sym)
        return result

    def simulate_explosion(
        self,
        board: Board,
        exploded: frozenset[Position],
    ) -> SettleResult:
        """Simulate cluster explosion + gravity settle.

        Delegates to gravity.settle() — does NOT reimplement the gravity
        algorithm. The settle function copies the board internally, so
        the input board is not mutated.
        """
        return settle(self._dag, board, exploded, self._gravity_config)

    def predict_booster_landing(
        self,
        centroid: Position,
        board: Board,
        exploded: frozenset[Position],
    ) -> Position:
        """Predict where a booster spawned at centroid lands after gravity.

        The booster occupies centroid AFTER the cluster explodes but BEFORE
        gravity settles. It falls like any other symbol. We simulate this by
        excluding centroid from the exploded set so the existing symbol at
        centroid (or a placeholder) survives and falls with gravity.
        """
        # Ensure centroid is not in the exploded set — the booster sits there
        safe_exploded = exploded - {centroid}
        result = settle(self._dag, board, safe_exploded, self._gravity_config)
        return _trace_position(centroid, result.move_steps)

    def predict_symbol_landings(
        self,
        board: Board,
        tracked_positions: frozenset[Position],
        exploded: frozenset[Position],
    ) -> dict[Position, Position]:
        """Map pre-gravity positions to their post-gravity destinations.

        For each tracked position (which must NOT be in the exploded set),
        traces where gravity carries it after the explosion settles.
        Returns {original_position: final_position}.
        """
        result = settle(self._dag, board, exploded, self._gravity_config)
        return {
            pos: _trace_position(pos, result.move_steps)
            for pos in tracked_positions
        }


def _trace_position(
    pos: Position,
    move_steps: tuple[tuple[tuple[str, Position, Position], ...], ...],
) -> Position:
    """Walk through gravity move passes to find where a position ends up.

    Each pass is a tuple of (symbol_name, source, destination) moves. If our
    tracked position appears as a source, it moved — update to the destination
    and continue tracing through subsequent passes.
    """
    current = pos
    for pass_moves in move_steps:
        for _sym, src, dst in pass_moves:
            if src == current:
                current = dst
                # Only one move per position per pass — stop scanning this pass
                break
    return current
