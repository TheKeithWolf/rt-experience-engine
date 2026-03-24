"""Per-cell WFC state — tracks remaining symbol possibilities.

Each free cell on the board gets a CellState that maintains which symbols
could still be placed there. Propagators narrow possibilities; the solver
collapses cells to single symbols. Snapshot/restore supports backtracking.
"""

from __future__ import annotations

from ..primitives.symbols import Symbol


class CellState:
    """Tracks possible symbols for a single uncollapsed board cell.

    Hot-path object — instantiated once per free cell (up to num_reels * num_rows).
    Uses __slots__ to eliminate per-instance __dict__ overhead.
    """

    __slots__ = ("_possibilities", "_collapsed")

    def __init__(self, possibilities: set[Symbol]) -> None:
        self._possibilities: set[Symbol] = set(possibilities)
        self._collapsed: bool = len(possibilities) == 1

    @property
    def collapsed(self) -> bool:
        """True when this cell has been fixed to a single symbol."""
        return self._collapsed

    @property
    def entropy(self) -> int:
        """Number of remaining possibilities — 0 means contradiction, 1 means collapsed."""
        return len(self._possibilities)

    @property
    def possibilities(self) -> frozenset[Symbol]:
        """Immutable view of remaining possibilities for external consumers."""
        return frozenset(self._possibilities)

    def remove(self, symbol: Symbol) -> bool:
        """Remove a symbol from possibilities.

        Returns True if the set changed (symbol was present).
        Caller checks entropy == 0 for contradiction detection.
        """
        if symbol not in self._possibilities:
            return False
        self._possibilities.discard(symbol)
        return True

    def collapse_to(self, symbol: Symbol) -> None:
        """Fix this cell to a single symbol. Called after weighted selection."""
        self._possibilities = {symbol}
        self._collapsed = True

    def snapshot(self) -> set[Symbol]:
        """Return a mutable copy of possibilities for backtrack state preservation."""
        return set(self._possibilities)

    def restore(self, possibilities: set[Symbol]) -> None:
        """Restore from a backtrack snapshot."""
        self._possibilities = set(possibilities)
        self._collapsed = len(possibilities) == 1
