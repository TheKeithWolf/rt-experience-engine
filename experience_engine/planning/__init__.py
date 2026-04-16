"""Shared planning primitives consumed by both spatial atlas and trajectory planner.

Both generators emit RegionConstraint objects; ClusterBuilder consumes them
without knowing which tier produced the guidance.
"""

from .region_constraint import (
    BridgeHint,
    GuidanceSource,
    RegionConstraint,
    bridge_hint_for_step,
    region_for_step,
)

__all__ = (
    "BridgeHint",
    "GuidanceSource",
    "RegionConstraint",
    "bridge_hint_for_step",
    "region_for_step",
)
