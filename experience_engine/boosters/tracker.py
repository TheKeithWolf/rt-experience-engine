"""Booster tracker — manages all booster instances during one cascade sequence.

Provides O(1) position lookup, adjacency detection via shared orthogonal_neighbors,
row-major fire ordering, and position remapping after gravity settle.
"""

from __future__ import annotations

from ..config.schema import BoardConfig
from ..primitives.board import Position, orthogonal_neighbors
from ..primitives.symbols import Symbol
from .state_machine import (
    BoosterInstance,
    BoosterState,
    transition,
)


class BoosterTracker:
    """Manages all booster instances during one cascade sequence.

    Internal storage: dict[Position, BoosterInstance] for O(1) position lookup.
    Boosters are added as DORMANT and progress through the lifecycle via
    arm_adjacent, mark_fired, and mark_chain_triggered.
    """

    __slots__ = ("_boosters", "_board_config")

    def __init__(self, board_config: BoardConfig) -> None:
        self._boosters: dict[Position, BoosterInstance] = {}
        self._board_config = board_config

    def add(
        self,
        booster_type: Symbol,
        position: Position,
        orientation: str | None = None,
        source_cluster_index: int | None = None,
    ) -> None:
        """Add a new DORMANT booster at the given position.

        Raises ValueError if a booster already exists at this position.
        """
        if position in self._boosters:
            raise ValueError(
                f"Booster already exists at ({position.reel},{position.row})"
            )
        self._boosters[position] = BoosterInstance(
            booster_type=booster_type,
            position=position,
            state=BoosterState.DORMANT,
            orientation=orientation,
            source_cluster_index=source_cluster_index,
        )

    def check_adjacency(
        self, cluster_positions: frozenset[Position],
    ) -> list[BoosterInstance]:
        """Return DORMANT boosters orthogonally adjacent to any cluster position.

        Uses orthogonal_neighbors from primitives.board — single adjacency
        algorithm across the entire engine (DRY).
        """
        result: list[BoosterInstance] = []
        seen: set[Position] = set()

        for pos in cluster_positions:
            for neighbor in orthogonal_neighbors(pos, self._board_config):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                booster = self._boosters.get(neighbor)
                if booster is not None and booster.state is BoosterState.DORMANT:
                    result.append(booster)

        return result

    def arm_adjacent(
        self, cluster_positions: frozenset[Position],
    ) -> list[BoosterInstance]:
        """Transition DORMANT → ARMED for boosters adjacent to cluster_positions.

        Returns the list of newly armed booster instances.
        """
        dormant_adjacent = self.check_adjacency(cluster_positions)
        armed: list[BoosterInstance] = []

        for booster in dormant_adjacent:
            new_instance = transition(booster, BoosterState.ARMED)
            self._boosters[booster.position] = new_instance
            armed.append(new_instance)

        return armed

    def get_armed(self) -> list[BoosterInstance]:
        """Return all ARMED boosters sorted by (row, reel) — row-major order.

        Row-major means top-to-bottom, left-to-right. This deterministic
        ordering ensures reproducible fire sequences across runs.
        """
        armed = [
            b for b in self._boosters.values()
            if b.state is BoosterState.ARMED
        ]
        armed.sort(key=lambda b: (b.position.row, b.position.reel))
        return armed

    def mark_fired(self, position: Position) -> BoosterInstance:
        """Transition the booster at position from ARMED → FIRED.

        Returns the updated instance. Raises KeyError if no booster at position.
        """
        booster = self._boosters[position]
        new_instance = transition(booster, BoosterState.FIRED)
        self._boosters[position] = new_instance
        return new_instance

    def mark_chain_triggered(self, position: Position) -> BoosterInstance:
        """Transition the booster at position to CHAIN_TRIGGERED.

        Valid from DORMANT or ARMED — chain reactions bypass normal arming.
        Returns the updated instance. Raises KeyError if no booster at position.
        """
        booster = self._boosters[position]
        new_instance = transition(booster, BoosterState.CHAIN_TRIGGERED)
        self._boosters[position] = new_instance
        return new_instance

    def update_positions_after_gravity(
        self, position_map: dict[Position, Position],
    ) -> None:
        """Remap booster positions after gravity settle.

        position_map: old_position → new_position for boosters that moved.
        Boosters at positions not in the map are assumed stationary.
        Boosters fall with gravity (Rule 7 from GravityDesign.md).
        """
        new_boosters: dict[Position, BoosterInstance] = {}

        for old_pos, booster in self._boosters.items():
            new_pos = position_map.get(old_pos, old_pos)
            # Create new instance with updated position
            moved = BoosterInstance(
                booster_type=booster.booster_type,
                position=new_pos,
                state=booster.state,
                orientation=booster.orientation,
                source_cluster_index=booster.source_cluster_index,
            )
            new_boosters[new_pos] = moved

        self._boosters = new_boosters

    def get_at(self, position: Position) -> BoosterInstance | None:
        """Return the booster at a position, or None if empty."""
        return self._boosters.get(position)

    def all_boosters(self) -> list[BoosterInstance]:
        """All tracked boosters in arbitrary order."""
        return list(self._boosters.values())

    def unfired_count(self) -> int:
        """Count of boosters not yet in FIRED or CHAIN_TRIGGERED state."""
        terminal = {BoosterState.FIRED, BoosterState.CHAIN_TRIGGERED}
        return sum(
            1 for b in self._boosters.values()
            if b.state not in terminal
        )
