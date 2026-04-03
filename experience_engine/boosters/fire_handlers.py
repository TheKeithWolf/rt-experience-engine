"""Booster fire handlers — rocket, bomb, lightball, superlightball.

Each function matches the FireDispatch signature and is registered on the
BoosterPhaseExecutor via register_fire_handler() in the cascade generator.
No modification to phase_executor.py is required.

All path/blast geometry, immunity sets, targeting logic, and board dimensions
come from BoosterRules (which reads from MasterConfig). Zero hardcoded values.
"""

from __future__ import annotations

from ..primitives.board import Board, Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.symbols import is_booster
from .phase_executor import BoosterFireResult
from .state_machine import BoosterInstance


def fire_rocket(
    booster: BoosterInstance,
    board: Board,
    rules: BoosterRules,
) -> BoosterFireResult:
    """Fire a rocket — clear all non-immune symbols along the row or column.

    Orientation determines direction: "H" clears the entire row,
    "V" clears the entire column. The rocket's own position is excluded
    from the affected set (it is consumed by the fire itself). Immune
    symbols (W, S from config) survive the blast.

    Chain targets are unfired boosters sitting in the rocket's path —
    the phase executor's chain logic handles their state transitions.
    """
    path = rules.rocket_path(booster.position, booster.orientation)

    affected: set[Position] = set()
    chain_targets: list[Position] = []
    symbol_captures: list[tuple[Position, str]] = []

    for pos in path:
        # Rocket's own position is consumed, not "cleared"
        if pos == booster.position:
            continue

        sym = board.get(pos)
        if sym is None:
            continue

        # Immune symbols survive rocket blasts (from config)
        if sym in rules.immune_to_rocket:
            continue

        affected.add(pos)
        symbol_captures.append((pos, sym.name))

        # Boosters in the path can be chain-triggered by the executor
        if is_booster(sym):
            chain_targets.append(pos)

    return BoosterFireResult(
        booster=booster,
        affected_positions=frozenset(affected),
        chain_targets=tuple(chain_targets),
        affected_symbols=tuple(symbol_captures),
    )


def fire_bomb(
    booster: BoosterInstance,
    board: Board,
    rules: BoosterRules,
) -> BoosterFireResult:
    """Fire a bomb — clear all non-immune symbols within the blast radius.

    Blast zone is a Manhattan-distance square centered on the bomb's position.
    Radius comes from config (default 1 = 3×3 area). The bomb's own position
    is excluded. Immune symbols survive. Boosters in the blast zone become
    chain targets.
    """
    blast = rules.bomb_blast(booster.position)

    affected: set[Position] = set()
    chain_targets: list[Position] = []
    symbol_captures: list[tuple[Position, str]] = []

    for pos in blast:
        # Bomb's own position is consumed, not "cleared"
        if pos == booster.position:
            continue

        sym = board.get(pos)
        if sym is None:
            continue

        # Immune symbols survive bomb blasts (from config)
        if sym in rules.immune_to_bomb:
            continue

        affected.add(pos)
        symbol_captures.append((pos, sym.name))

        # Boosters in the blast zone can be chain-triggered
        if is_booster(sym):
            chain_targets.append(pos)

    return BoosterFireResult(
        booster=booster,
        affected_positions=frozenset(affected),
        chain_targets=tuple(chain_targets),
        affected_symbols=tuple(symbol_captures),
    )


def fire_lightball(
    booster: BoosterInstance,
    board: Board,
    rules: BoosterRules,
) -> BoosterFireResult:
    """Fire a lightball — clear ALL instances of the most abundant standard symbol.

    Target selection: the standard symbol with the highest count on the board
    at fire time. On tie, the symbol with the higher payout_rank (from config)
    wins — this biases the player experience toward more valuable symbols.

    LB cannot initiate chains — chain_targets is always empty.
    """
    targets = rules.most_abundant_standard(board, count=1)
    target_sym = targets[0]

    # Collect every position containing the target symbol
    affected: set[Position] = set()
    symbol_captures: list[tuple[Position, str]] = []
    for reel in range(board.num_reels):
        for row in range(board.num_rows):
            pos = Position(reel, row)
            if board.get(pos) is target_sym:
                affected.add(pos)
                symbol_captures.append((pos, target_sym.name))

    return BoosterFireResult(
        booster=booster,
        affected_positions=frozenset(affected),
        # LB cannot initiate chains (not in config.boosters.chain_initiators)
        chain_targets=(),
        target_symbols=(target_sym.name,),
        affected_symbols=tuple(symbol_captures),
    )


def fire_superlightball(
    booster: BoosterInstance,
    board: Board,
    rules: BoosterRules,
) -> BoosterFireResult:
    """Fire a superlightball — clear ALL instances of the two most abundant standard symbols.

    Target selection: the two standard symbols with the highest counts.
    Tiebreaker: higher payout_rank (from config) wins.

    SLB cannot initiate chains — chain_targets is always empty.
    Grid multiplier increment at cleared positions is handled by the
    cascade generator after collecting fire results (not in this handler).
    """
    targets = rules.most_abundant_standard(board, count=2)

    # Collect every position containing either target symbol
    target_set = frozenset(targets)
    affected: set[Position] = set()
    symbol_captures: list[tuple[Position, str]] = []
    for reel in range(board.num_reels):
        for row in range(board.num_rows):
            pos = Position(reel, row)
            sym = board.get(pos)
            if sym in target_set:
                affected.add(pos)
                symbol_captures.append((pos, sym.name))

    return BoosterFireResult(
        booster=booster,
        affected_positions=frozenset(affected),
        # SLB cannot initiate chains (not in config.boosters.chain_initiators)
        chain_targets=(),
        target_symbols=tuple(s.name for s in targets),
        affected_symbols=tuple(symbol_captures),
    )
