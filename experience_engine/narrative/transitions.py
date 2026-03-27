"""Transition predicates — dict-dispatch registry for phase advancement.

Each transition rule is a callable that receives the step result and the
current board context, returning True when the phase should advance.

Adding a new transition rule is one dict entry. No validator changes needed.
Registration-time validation (CONTRACT-SIG-ARC) ensures every NarrativePhase
``ends_when`` value exists as a key in TRANSITION_RULES.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ..primitives.board import orthogonal_neighbors
from ..primitives.symbols import Symbol, is_standard

if TYPE_CHECKING:
    from ..config.schema import BoardConfig, SymbolConfig
    from ..step_reasoner.context import BoardContext
    from ..step_reasoner.progress import ProgressTracker
    from ..step_reasoner.results import StepResult


# (StepResult, BoardContext) → should this phase end?
TransitionPredicate = Callable[["StepResult", "BoardContext"], bool]


def _any_wild_can_bridge(
    ctx: BoardContext,
    board_config: BoardConfig,
    symbol_config: SymbolConfig,
) -> bool:
    """True if any active wild has >= 2 orthogonal neighbors of the same standard symbol.

    A wild that can bridge means it is adjacent to two or more cells sharing a
    standard symbol — it could extend or connect clusters if the cascade continued.
    """
    for wild_pos in ctx.active_wilds:
        # Collect standard symbols on orthogonal neighbors
        neighbor_symbols: list[Symbol] = []
        for npos in orthogonal_neighbors(wild_pos, board_config):
            sym = ctx.board.get(npos)
            if sym is not None and is_standard(sym, symbol_config):
                neighbor_symbols.append(sym)

        # Check if any standard symbol appears >= 2 times among neighbors
        if len(neighbor_symbols) >= 2:
            seen: set[Symbol] = set()
            for s in neighbor_symbols:
                if s in seen:
                    return True
                seen.add(s)

    return False


def build_transition_rules(
    board_config: BoardConfig,
    symbol_config: SymbolConfig,
) -> dict[str, TransitionPredicate]:
    """Build the transition rules dict with board/symbol config captured via closure.

    Returns a dict mapping rule names to predicates. This is the single source
    of truth for available transition rules — CONTRACT-SIG-ARC validates
    NarrativePhase.ends_when against these keys at registration time.
    """
    return {
        # Phase always advances after one repetition (or max reps)
        "always": lambda _result, _ctx: True,

        # Phase advances when no clusters were produced this step
        "no_clusters": lambda result, _ctx: len(result.clusters) == 0,

        # Phase advances when no wild can bridge standard symbol groups
        "no_bridges": lambda _result, ctx: not _any_wild_can_bridge(
            ctx, board_config, symbol_config,
        ),

        # Phase advances when at least one booster fired this step
        "booster_fired": lambda result, _ctx: len(result.fires) > 0,
    }


# Allowed transition rule keys — validated at registration time without
# needing the full rules dict (which requires config to build).
ALLOWED_TRANSITION_KEYS: frozenset[str] = frozenset({
    "always", "no_clusters", "no_bridges", "booster_fired",
})

# Transition rules that inspect BoardContext and can't evaluate when ctx is None
_CONTEXT_DEPENDENT_RULES: frozenset[str] = frozenset({"no_bridges"})


def try_advance_phase(
    progress: ProgressTracker,
    step_result: StepResult,
    transition_rules: dict[str, TransitionPredicate],
    context: BoardContext | None = None,
) -> bool:
    """Advance to the next narrative phase if the current phase is complete.

    Checks two conditions (same logic as NarrativeArcValidator):
    1. Max repetitions reached → advance unconditionally
    2. Transition predicate fires AND min repetitions met → advance

    Returns True if the phase advanced, False otherwise.

    context is optional — transition predicates that inspect board state
    (e.g., "no_bridges") require it. Context-independent predicates
    ("always", "no_clusters", "booster_fired") work with context=None.
    """
    phase = progress.current_phase()
    if phase is None:
        return False

    # Condition 1: max repetitions exhausted
    if progress.current_phase_repetitions >= phase.repetitions.max_val:
        progress.advance_phase()
        return True

    # Condition 2: transition predicate fires
    predicate = transition_rules.get(phase.ends_when)
    if predicate is not None:
        # Context-dependent predicates can't evaluate without context —
        # skip (conservative: don't advance without evidence)
        if context is None and phase.ends_when in _CONTEXT_DEPENDENT_RULES:
            return False
        if predicate(step_result, context):
            if progress.current_phase_repetitions >= phase.repetitions.min_val:
                progress.advance_phase()
                return True

    return False
