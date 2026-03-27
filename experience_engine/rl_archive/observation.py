"""Observation representation for the cascade RL environment.

CascadeObservation is a frozen snapshot of the environment state that the
policy network uses to decide actions. ObservationBuilder translates from
the engine's internal BoardContext/ProgressTracker to this representation.
ObservationEncoder converts observations to numpy arrays for the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config.schema import BoardConfig, SymbolConfig
from ..primitives.board import Position
from ..primitives.symbols import Symbol

if TYPE_CHECKING:
    from ..narrative.arc import NarrativeArc
    from ..step_reasoner.context import BoardContext
    from ..step_reasoner.progress import ProgressTracker


@dataclass(frozen=True, slots=True)
class CascadeObservation:
    """Discrete behavioral snapshot of the cascade environment state.

    All fields are immutable and serializable. Board data uses integer indices
    (not Symbol enums) so the policy network can process them directly.
    """

    # Board state — reel-major tuples of tuples
    board_symbols: tuple[tuple[int, ...], ...]
    board_multipliers: tuple[tuple[int, ...], ...]
    wild_positions: tuple[tuple[int, int], ...]
    dormant_boosters: tuple[tuple[str, int, int], ...]

    # Step progress
    step_index: int
    cumulative_payout: int

    # Narrative phase tracking — from ProgressTracker
    current_phase_id: str
    current_phase_index: int
    phase_repetition: int
    phases_remaining: int

    # Budget remaining — from ProgressTracker query methods
    remaining_spawns: dict[str, int]
    remaining_fires: dict[str, int]
    payout_remaining_min: float
    payout_remaining_max: float

    # Identity
    archetype_id: str


class ObservationBuilder:
    """Builds CascadeObservation from engine state objects.

    Maintains a symbol-to-index mapping built from SymbolConfig at init.
    Empty cells, wilds, and boosters get distinct indices beyond the
    standard symbol range.
    """

    __slots__ = ("_board_config", "_symbol_to_index", "_empty_index", "_num_channels")

    # Index for empty (None) cells — after all standard symbols
    _EMPTY_OFFSET = 0
    # Index offsets for special symbols (wild, boosters) beyond standard + empty
    _SPECIAL_OFFSET = 1

    def __init__(self, symbol_config: SymbolConfig, board_config: BoardConfig) -> None:
        self._board_config = board_config
        # Standard symbols get indices 0..N-1 matching SymbolConfig.standard order
        self._symbol_to_index: dict[Symbol, int] = {}
        for i, name in enumerate(symbol_config.standard):
            self._symbol_to_index[Symbol[name]] = i
        # Empty cells get the next index
        self._empty_index = len(symbol_config.standard)
        # Total channel count for one-hot encoding
        self._num_channels = self._empty_index + 1

    def build(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        arc: NarrativeArc,
    ) -> CascadeObservation:
        """Build observation from current board context and progress state."""
        board = context.board
        num_reels = self._board_config.num_reels
        num_rows = self._board_config.num_rows

        # Board symbols as integer indices (reel-major)
        board_symbols: list[tuple[int, ...]] = []
        for reel in range(num_reels):
            row_indices: list[int] = []
            for row in range(num_rows):
                sym = board.get(Position(reel, row))
                if sym is None:
                    row_indices.append(self._empty_index)
                elif sym in self._symbol_to_index:
                    row_indices.append(self._symbol_to_index[sym])
                else:
                    # Special symbols (wild, boosters) — use empty index
                    # since they're tracked separately
                    row_indices.append(self._empty_index)
            board_symbols.append(tuple(row_indices))

        # Grid multipliers (reel-major)
        board_multipliers: list[tuple[int, ...]] = []
        for reel in range(num_reels):
            row_mults: list[int] = []
            for row in range(num_rows):
                row_mults.append(context.grid_multipliers.get(Position(reel, row)))
            board_multipliers.append(tuple(row_mults))

        # Wild positions
        wild_positions = tuple(
            (p.reel, p.row) for p in context.active_wilds
        )

        # Dormant boosters
        dormant_boosters = tuple(
            (db.booster_type, db.position.reel, db.position.row)
            for db in context.dormant_boosters
        )

        # Phase tracking
        current_phase = progress.current_phase()
        current_phase_id = current_phase.id if current_phase is not None else ""
        phases_remaining = max(
            0, len(arc.phases) - progress.current_phase_index - 1
        )

        # Budget remaining
        remaining_spawns_ranges = progress.remaining_booster_spawns()
        remaining_fires_ranges = progress.remaining_booster_fires()
        payout_budget = progress.remaining_payout_budget()

        return CascadeObservation(
            board_symbols=tuple(board_symbols),
            board_multipliers=tuple(board_multipliers),
            wild_positions=wild_positions,
            dormant_boosters=dormant_boosters,
            step_index=progress.steps_completed,
            cumulative_payout=progress.cumulative_payout,
            current_phase_id=current_phase_id,
            current_phase_index=progress.current_phase_index,
            phase_repetition=progress.current_phase_repetitions,
            phases_remaining=phases_remaining,
            # Use min_val as the "minimum still needed" for spawns/fires
            remaining_spawns={
                k: v.min_val for k, v in remaining_spawns_ranges.items()
            },
            remaining_fires={
                k: v.min_val for k, v in remaining_fires_ranges.items()
            },
            payout_remaining_min=payout_budget.min_val,
            payout_remaining_max=payout_budget.max_val,
            archetype_id=progress.signature.id,
        )


class ObservationEncoder:
    """Converts CascadeObservation to numpy arrays for the policy network.

    Board is encoded as multi-channel tensor: one-hot symbol channels +
    multiplier channel + empty mask + wild mask + booster mask.
    Scalar features are concatenated into a separate vector.
    No PyTorch dependency — outputs numpy arrays.
    """

    __slots__ = ("_num_symbols", "_num_reels", "_num_rows")

    def __init__(self, symbol_config: SymbolConfig, board_config: BoardConfig) -> None:
        self._num_symbols = len(symbol_config.standard)
        self._num_reels = board_config.num_reels
        self._num_rows = board_config.num_rows

    def encode_board(self, obs: CascadeObservation) -> np.ndarray:
        """Encode board state as (channels, num_reels, num_rows) float32 array.

        Channels: [symbol_0, ..., symbol_N-1, multiplier, empty, wild, booster]
        """
        num_reels = self._num_reels
        num_rows = self._num_rows
        # symbol one-hot + multiplier + empty + wild + booster = N+4 channels
        num_channels = self._num_symbols + 4
        board = np.zeros((num_channels, num_reels, num_rows), dtype=np.float32)

        # Symbol one-hot encoding
        for reel in range(num_reels):
            for row in range(num_rows):
                idx = obs.board_symbols[reel][row]
                if idx < self._num_symbols:
                    board[idx, reel, row] = 1.0
                else:
                    # Empty cell
                    board[self._num_symbols + 1, reel, row] = 1.0

        # Multiplier channel (normalized by a reasonable cap)
        for reel in range(num_reels):
            for row in range(num_rows):
                board[self._num_symbols, reel, row] = float(
                    obs.board_multipliers[reel][row]
                )

        # Wild mask
        for reel, row in obs.wild_positions:
            board[self._num_symbols + 2, reel, row] = 1.0

        # Booster mask
        for _, reel, row in obs.dormant_boosters:
            board[self._num_symbols + 3, reel, row] = 1.0

        return board

    def encode_scalars(self, obs: CascadeObservation) -> np.ndarray:
        """Encode scalar features as a 1D float32 array.

        Features: [phase_index, phase_repetition, phases_remaining,
                   cumulative_payout, payout_remaining_min, payout_remaining_max]
        """
        return np.array([
            float(obs.current_phase_index),
            float(obs.phase_repetition),
            float(obs.phases_remaining),
            float(obs.cumulative_payout),
            obs.payout_remaining_min,
            obs.payout_remaining_max,
        ], dtype=np.float32)
