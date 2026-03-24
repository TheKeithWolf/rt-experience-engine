"""Booster lifecycle state machine — states, instances, and validated transitions.

A booster progresses through a strict lifecycle: DORMANT → ARMED → FIRED,
or DORMANT/ARMED → CHAIN_TRIGGERED (bypass arming via chain reaction).
FIRED and CHAIN_TRIGGERED are terminal states — no further transitions allowed.

BoosterInstance is frozen; transition() returns a new instance with updated state.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ..primitives.board import Position
from ..primitives.symbols import Symbol


class BoosterState(enum.Enum):
    """Lifecycle state of a booster on the board."""

    # Spawned at cluster centroid, waiting for adjacent cluster to arm it
    DORMANT = "dormant"
    # Adjacent cluster detected — ready to fire during booster phase
    ARMED = "armed"
    # Fired during booster phase (terminal)
    FIRED = "fired"
    # Fired via chain reaction from another booster (terminal)
    CHAIN_TRIGGERED = "chain_triggered"


# Valid state transitions — single source of truth for lifecycle rules
_VALID_TRANSITIONS: dict[BoosterState, frozenset[BoosterState]] = {
    BoosterState.DORMANT: frozenset({
        BoosterState.ARMED,
        BoosterState.CHAIN_TRIGGERED,
    }),
    BoosterState.ARMED: frozenset({
        BoosterState.FIRED,
        BoosterState.CHAIN_TRIGGERED,
    }),
    # Terminal states — no outgoing transitions
    BoosterState.FIRED: frozenset(),
    BoosterState.CHAIN_TRIGGERED: frozenset(),
}


class InvalidTransition(Exception):
    """Raised when attempting an invalid booster state transition."""

    __slots__ = ("booster_type", "position", "from_state", "to_state")

    def __init__(
        self,
        booster_type: Symbol,
        position: Position,
        from_state: BoosterState,
        to_state: BoosterState,
    ) -> None:
        self.booster_type = booster_type
        self.position = position
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"{booster_type.name} at ({position.reel},{position.row}): "
            f"cannot transition {from_state.value} → {to_state.value}"
        )


@dataclass(frozen=True, slots=True)
class BoosterInstance:
    """One booster on the board with its lifecycle state.

    Frozen: state transitions produce new instances via transition().
    Only non-wild boosters (R, B, LB, SLB) — wilds use the wild lifecycle.
    """

    booster_type: Symbol
    position: Position
    state: BoosterState
    # "H" or "V" for rockets, None for bombs/lightballs/superlightballs
    orientation: str | None
    # Index of the cluster that spawned this booster (for chain depth tracking)
    source_cluster_index: int | None


def transition(instance: BoosterInstance, new_state: BoosterState) -> BoosterInstance:
    """Return a new BoosterInstance with the updated state.

    Validates against _VALID_TRANSITIONS. Raises InvalidTransition if the
    transition is not allowed (e.g., FIRED → anything, DORMANT → FIRED).
    """
    allowed = _VALID_TRANSITIONS[instance.state]
    if new_state not in allowed:
        raise InvalidTransition(
            instance.booster_type,
            instance.position,
            instance.state,
            new_state,
        )

    return BoosterInstance(
        booster_type=instance.booster_type,
        position=instance.position,
        state=new_state,
        orientation=instance.orientation,
        source_cluster_index=instance.source_cluster_index,
    )
