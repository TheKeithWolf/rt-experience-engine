"""FillConstraints — bundles all WFC configuration for a single fill_step() call.

Aggregates the three gravity-aware mechanisms (spatial weights, post-gravity
propagation, gravity-group ordering) into one immutable object. Each mechanism
field is None for baseline fills (dead/terminal) — the WFC core loop checks
at exactly three extension points, not via if/else chains.

Constructed by StepExecutor, consumed by WFCBoardFiller.fill_step().
"""

from __future__ import annotations

from dataclasses import dataclass

from ..primitives.symbols import Symbol
from .gravity_adjacency import PostGravityAdjacency
from .gravity_ordering import GravityAwareEntropySelector
from .propagators import Propagator
from .spatial_weights import SpatialWeightMap


@dataclass(frozen=True)
class FillConstraints:
    """All WFC configuration for a single fill_step() call.

    propagators: constraint propagators (always present — includes both
        strategy-specified and gravity-aware propagators)
    spatial_weights: per-cell weight suppression map (mechanism 1) —
        None for baseline fills, present for gravity-aware fills
    gravity_adjacency: post-gravity virtual adjacency graph (mechanism 2) —
        None for baseline, present when planned_explosion exists
    gravity_groups: gravity-group cell selector (mechanism 3) —
        None for baseline (uses global min-entropy), present for gravity-aware
    flat_symbol_weights: fallback weights when spatial_weights is None
    """

    propagators: list[Propagator]

    # Mechanism 1 — None = flat weights (dead/terminal only)
    spatial_weights: SpatialWeightMap | None

    # Mechanism 2 — None = pre-gravity propagation only
    # When present, PostGravityPropagator is already in propagators list
    gravity_adjacency: PostGravityAdjacency | None

    # Mechanism 3 — None = standard min-entropy selection
    gravity_groups: GravityAwareEntropySelector | None

    # Flat weights fallback (used when spatial_weights is None)
    flat_symbol_weights: dict[Symbol, float]
