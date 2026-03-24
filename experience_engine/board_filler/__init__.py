"""WFC Board Filler — fills unconstrained cells while preventing unintended clusters.

The board filler is the third stage of the ASP → CSP → WFC pipeline. Given
pinned cells from the CSP spatial solver, it populates remaining cells with
standard symbols using Wave Function Collapse with constraint propagation.

Gravity-aware mechanisms (spatial weights, post-gravity propagation, gravity-group
ordering) are bundled in FillConstraints and consumed by fill_step().
"""

from .cell_state import CellState
from .fill_constraints import FillConstraints
from .gravity_adjacency import PostGravityAdjacency
from .gravity_ordering import GravityAwareEntropySelector, GravityGroupComputer
from .propagators import (
    MaxComponentPropagator,
    NoClusterPropagator,
    NoSpecialSymbolPropagator,
    PostGravityPropagator,
    Propagator,
)
from .spatial_weights import SpatialWeightMap
from .wfc_solver import FillFailed, WFCBoardFiller

__all__ = [
    "CellState",
    "FillConstraints",
    "FillFailed",
    "GravityAwareEntropySelector",
    "GravityGroupComputer",
    "MaxComponentPropagator",
    "NoClusterPropagator",
    "NoSpecialSymbolPropagator",
    "PostGravityAdjacency",
    "PostGravityPropagator",
    "Propagator",
    "SpatialWeightMap",
    "WFCBoardFiller",
]
