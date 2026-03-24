"""WFC Board Filler — fills unconstrained cells while preventing unintended clusters.

The board filler is the third stage of the ASP → CSP → WFC pipeline. Given
pinned cells from the CSP spatial solver, it populates remaining cells with
standard symbols using Wave Function Collapse with constraint propagation.
"""

from .cell_state import CellState
from .propagators import (
    MaxComponentPropagator,
    NoClusterPropagator,
    NoSpecialSymbolPropagator,
    Propagator,
)
from .wfc_solver import FillFailed, WFCBoardFiller

__all__ = [
    "CellState",
    "FillFailed",
    "MaxComponentPropagator",
    "NoClusterPropagator",
    "NoSpecialSymbolPropagator",
    "Propagator",
    "WFCBoardFiller",
]
