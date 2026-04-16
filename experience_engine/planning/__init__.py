"""Shared planning primitives consumed by both spatial atlas and trajectory planner.

Both generators emit RegionConstraint objects; ClusterBuilder consumes them
without knowing which tier produced the guidance.
"""

from .region_constraint import (
    GuidanceSource,
    RegionConstraint,
    region_for_step,
)

__all__ = ("GuidanceSource", "RegionConstraint", "region_for_step")
