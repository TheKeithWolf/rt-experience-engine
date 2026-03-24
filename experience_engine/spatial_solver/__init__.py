"""CSP Spatial Solver — resolves abstract plans into concrete board positions.

Pluggable constraint system: new constraints (booster placement, gravity-aware
mapping, chain relations) are added via add_constraint() without modifying the
solver. Phase 3 ships with four basic constraints: connectivity, non-overlap,
near-miss isolation, and scatter non-overlap.
"""

from .data_types import (
    BoosterPlacement,
    ClusterAssignment,
    NearMissAssignment,
    SpatialPlan,
    SpatialStep,
)
from .solver import CSPSpatialSolver, SolveFailed

__all__ = [
    "BoosterPlacement",
    "CSPSpatialSolver",
    "ClusterAssignment",
    "NearMissAssignment",
    "SolveFailed",
    "SpatialPlan",
    "SpatialStep",
]
