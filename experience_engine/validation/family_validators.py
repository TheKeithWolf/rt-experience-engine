"""Per-family validators (B1) — protocol + registry.

Lifts family-specific validation rules out of `InstanceValidator.validate`
so the main loop dispatches via dict lookup instead of growing if-chains
on `sig.family`. Adding a new family rule requires only registering a
FamilyValidator implementation; the validator's main loop is untouched.

Scope (Batch 7, B1, partial): the dead/t1 family rule (no wilds or
boosters on initial/terminal boards) is extracted here. The wild family's
rules remain inline in InstanceValidator because they share computed
state with the booster spawn/fire validation block — extracting them
without copying that state out would risk silent behavior drift. A
future PR can move them once the surrounding state is itself extracted
into a per-instance ValidationContext.
"""

from __future__ import annotations

from typing import Protocol

from ..archetypes.families import ArchetypeFamily
from ..archetypes.registry import ArchetypeSignature
from ..primitives.board import Board
from ..primitives.symbols import is_booster, is_wild


class FamilyValidator(Protocol):
    """Returns the list of validation errors for one archetype family."""

    def validate(
        self,
        sig: ArchetypeSignature,
        initial_board: Board,
        terminal_board: Board,
        is_cascade: bool,
    ) -> tuple[str, ...]: ...


class _DeadT1Validator:
    """Dead and t1 boards must contain no wilds or boosters.

    Wild family legitimately uses wilds; rocket/bomb/lb/slb spawn boosters.
    Only dead and t1 enforce the no-special-symbols invariant.
    """

    def validate(
        self,
        sig: ArchetypeSignature,
        initial_board: Board,
        terminal_board: Board,
        is_cascade: bool,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        boards = (
            (initial_board, terminal_board) if is_cascade else (terminal_board,)
        )
        for check_board in boards:
            for pos in check_board.all_positions():
                sym = check_board.get(pos)
                if sym is not None and (is_wild(sym) or is_booster(sym)):
                    errors.append(
                        f"unexpected {sym.name} at ({pos.reel}, {pos.row}) — "
                        f"{sig.family} boards must not contain wilds or boosters"
                    )
        return tuple(errors)


class _NoOpValidator:
    """Default for families without custom rules."""

    def validate(
        self,
        sig: ArchetypeSignature,
        initial_board: Board,
        terminal_board: Board,
        is_cascade: bool,
    ) -> tuple[str, ...]:
        return ()


# Registry — single dispatch table. dead and t1 share _DeadT1Validator
# because they share the no-special-symbols invariant. Keys use the typed
# ArchetypeFamily enum (B8); since StrEnum members compare equal to their
# string values, callers may look up with either form.
FAMILY_VALIDATORS: dict[str, FamilyValidator] = {
    ArchetypeFamily.DEAD: _DeadT1Validator(),
    ArchetypeFamily.T1: _DeadT1Validator(),
}
DEFAULT_FAMILY_VALIDATOR: FamilyValidator = _NoOpValidator()
