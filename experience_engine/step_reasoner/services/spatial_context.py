"""Spatial intelligence context bundling the three foresight services.

Constructed once in build_default_registry() when SpatialIntelligenceConfig
is present, then injected into strategies that benefit from spatial awareness.
Terminal strategies receive None — no dead code for steps with no future demand.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gravity_field import GravityFieldService
from .influence_map import InfluenceMap
from .utility_scorer import UtilityScorer


@dataclass(frozen=True, slots=True)
class StepSpatialContext:
    """Bundle of spatial intelligence services for cascade seed placement.

    Strategies store this as self._spatial: StepSpatialContext | None.
    When None, existing column-matching logic runs unchanged.
    """

    gravity_field: GravityFieldService
    influence_map: InfluenceMap
    utility_scorer: UtilityScorer
