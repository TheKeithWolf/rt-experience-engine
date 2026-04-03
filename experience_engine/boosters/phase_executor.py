"""Booster phase executor — fires armed boosters in row-major order with depth-first chains.

Phase 8 provides the framework with stub fire dispatchers. Phases 9-10 plug in real
fire behaviors (rocket paths, bomb blasts, LB/SLB targeting) via register_fire_handler()
without modifying this module.

Chain initiation is restricted to types in config.boosters.chain_initiators (R, B).
LB/SLB can be chain-triggered but cannot initiate chains themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.symbols import Symbol
from .state_machine import BoosterInstance, BoosterState
from .tracker import BoosterTracker


@dataclass(frozen=True, slots=True)
class BoosterFireResult:
    """Result of firing one booster — what positions were affected."""

    booster: BoosterInstance
    # Board positions cleared or affected by this fire
    affected_positions: frozenset[Position]
    # Positions of boosters hit by this fire that should chain-trigger
    chain_targets: tuple[Position, ...]
    # Symbols targeted by LB (1 symbol) or SLB (2 symbols) — empty for R/B
    target_symbols: tuple[str, ...] = ()
    # Position → symbol name captured before clearing — event stream needs
    # symbol identity for boosterFireInfo.clearedCells
    affected_symbols: tuple[tuple[Position, str], ...] = ()


# Type alias for the fire dispatch function signature
FireDispatch = Callable[[BoosterInstance, Board, BoosterRules], BoosterFireResult]


def _stub_fire(
    booster: BoosterInstance,
    board: Board,
    rules: BoosterRules,
) -> BoosterFireResult:
    """Placeholder fire — returns empty result. Real behaviors in Phase 9-10."""
    return BoosterFireResult(
        booster=booster,
        affected_positions=frozenset(),
        chain_targets=(),
    )


class BoosterPhaseExecutor:
    """Executes one complete booster phase: fire armed boosters, then chain.

    Fire order: row-major (top-left first) for deterministic reproducibility.
    Chain model: depth-first — after each fire, immediately chain into any
    unfired boosters hit by the blast before proceeding to the next armed booster.

    Chain initiation restricted to chain_initiators (from config). LB/SLB can
    receive chain triggers but cannot propagate chains further.
    """

    __slots__ = ("_tracker", "_rules", "_chain_initiators", "_fire_dispatch")

    def __init__(
        self,
        tracker: BoosterTracker,
        rules: BoosterRules,
        chain_initiators: frozenset[Symbol],
    ) -> None:
        self._tracker = tracker
        self._rules = rules
        # Only these types can initiate chains (R, B from config)
        self._chain_initiators = chain_initiators
        # Dict dispatch for type-specific fire behavior (py-developer pattern)
        # Phase 8: all stubs. Phase 9 replaces R/B, Phase 10 replaces LB/SLB.
        self._fire_dispatch: dict[Symbol, FireDispatch] = {
            Symbol.R: _stub_fire,
            Symbol.B: _stub_fire,
            Symbol.LB: _stub_fire,
            Symbol.SLB: _stub_fire,
        }

    def register_fire_handler(
        self, booster_type: Symbol, handler: FireDispatch,
    ) -> None:
        """Replace a stub with a real fire handler.

        Phase 9 registers rocket and bomb handlers. Phase 10 registers
        lightball and superlightball handlers. This keeps the framework
        independent of specific fire implementations.
        """
        self._fire_dispatch[booster_type] = handler

    def execute_booster_phase(self, board: Board) -> list[BoosterFireResult]:
        """Execute one complete booster phase.

        1. Get all ARMED boosters in row-major order
        2. Fire each in order, skipping already-fired (may have been chain-triggered)
        3. After each fire, depth-first chain: check affected_positions for
           unfired boosters — if the firing booster's type is a chain initiator,
           mark them CHAIN_TRIGGERED and fire immediately
        4. Return all fire results in execution order
        """
        armed = self._tracker.get_armed()
        if not armed:
            return []

        results: list[BoosterFireResult] = []
        # Visited set prevents infinite chain loops (a booster fires at most once)
        visited: set[Position] = set()

        for booster in armed:
            if booster.position in visited:
                # Already fired via chain from an earlier booster
                continue
            self._fire_and_chain(booster, board, visited, results)

        return results

    def _fire_and_chain(
        self,
        booster: BoosterInstance,
        board: Board,
        visited: set[Position],
        results: list[BoosterFireResult],
    ) -> None:
        """Fire one booster, then depth-first chain into affected boosters.

        visited tracks all positions that have already fired to prevent
        infinite loops (e.g., R→B→R at overlapping positions).
        """
        if booster.position in visited:
            return

        visited.add(booster.position)

        # Mark the booster as fired in the tracker
        if booster.state is BoosterState.ARMED:
            self._tracker.mark_fired(booster.position)
        # CHAIN_TRIGGERED boosters were already transitioned by the caller

        # Dispatch to type-specific fire handler
        fire_fn = self._fire_dispatch.get(booster.booster_type, _stub_fire)
        result = fire_fn(booster, board, self._rules)
        results.append(result)

        # Chain propagation: only chain_initiators can start chains
        if booster.booster_type not in self._chain_initiators:
            return

        # Check affected positions for unfired boosters to chain-trigger
        for target_pos in result.chain_targets:
            if target_pos in visited:
                continue

            target = self._tracker.get_at(target_pos)
            if target is None:
                continue

            # Only trigger unfired boosters (DORMANT or ARMED)
            if target.state in (BoosterState.FIRED, BoosterState.CHAIN_TRIGGERED):
                continue

            # Chain-trigger the target and fire it immediately (depth-first)
            chain_triggered = self._tracker.mark_chain_triggered(target_pos)
            self._fire_and_chain(chain_triggered, board, visited, results)
