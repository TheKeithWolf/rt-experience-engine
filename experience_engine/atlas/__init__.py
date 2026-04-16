"""Spatial Atlas — offline pre-validated per-arc region guidance (Tier 1).

Public types are re-exported here; builder/query/storage live in dedicated
modules so the data layer can be imported without pulling in the build
pipeline's service dependencies.
"""

from .builder import AtlasBuilder, AtlasServices, build_atlas_services
from .query import AtlasQuery
from .data_types import (
    ArmAdjacencyEntry,
    AtlasConfiguration,
    BoosterLandingEntry,
    ColumnProfile,
    DormantSurvivalEntry,
    PhaseGuidance,
    SettleTopology,
    SpatialAtlas,
)
from .storage import AtlasStorage

__all__ = (
    "ArmAdjacencyEntry",
    "AtlasBuilder",
    "AtlasConfiguration",
    "AtlasQuery",
    "AtlasServices",
    "AtlasStorage",
    "build_atlas_services",
    "BoosterLandingEntry",
    "ColumnProfile",
    "DormantSurvivalEntry",
    "PhaseGuidance",
    "SettleTopology",
    "SpatialAtlas",
)
