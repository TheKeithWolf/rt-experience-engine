"""Step intent — what the reasoner wants a cascade step to achieve.

StepType classifies the purpose of a step. StepIntent captures the full
specification: which cells are constrained, what clusters are expected,
which boosters should spawn/arm/fire, and how WFC should be configured.

Consumers must not mutate container fields (dicts, lists) after construction —
frozen=True prevents reassignment of references but not in-place mutation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ..archetypes.registry import TerminalNearMissSpec
from ..board_filler.propagators import Propagator
from ..pipeline.protocols import Range
from ..primitives.board import Position
from ..primitives.gravity import SettleResult
from ..primitives.symbols import Symbol, SymbolTier


class StepType(enum.Enum):
    """Classification of what a cascade step accomplishes.

    Each value maps to a strategy in the StrategyRegistry (Step 5).
    """

    INITIAL = "initial"
    CASCADE_CLUSTER = "cascade_cluster"
    BOOSTER_ARM = "booster_arm"
    TERMINAL_DEAD = "terminal_dead"
    TERMINAL_NEAR_MISS = "terminal_near_miss"


@dataclass(frozen=True, slots=True)
class StepIntent:
    """What the reasoner wants a single cascade step to produce.

    Three cell categories:
    - constrained: MUST be this symbol at this position (cluster cores, scatters)
    - strategic: SHOULD be this symbol — gravity will carry it where a future
      step needs it (wild bridge seeds, booster arm setups)
    - noise: everything else — WFC fills freely with propagator constraints

    Frozen after construction. The execution layer reads these fields to
    configure CSP constraints and WFC propagators.
    """

    step_type: StepType

    # Cell classifications
    constrained_cells: dict[Position, Symbol]
    strategic_cells: dict[Position, Symbol]

    # Expected cluster outcomes
    expected_cluster_count: Range
    expected_cluster_sizes: list[Range]
    expected_cluster_tier: SymbolTier | None

    # Booster lifecycle expectations
    expected_spawns: list[str]
    expected_arms: list[str]
    expected_fires: list[str]

    # WFC configuration — propagators prevent unintended clusters,
    # weights steer symbol distribution for this step
    wfc_propagators: list[Propagator]
    wfc_symbol_weights: dict[Symbol, float]

    # Gravity simulation result the reasoner already computed (None if skipped)
    predicted_post_gravity: SettleResult | None

    # Terminal step constraints
    terminal_near_misses: TerminalNearMissSpec | None
    terminal_dormant_boosters: list[str] | None

    # Positions that will explode after this step's fill — gates gravity-aware
    # WFC mechanisms. None for terminal/dead steps (no explosion planned).
    planned_explosion: frozenset[Position] | None = None
    is_terminal: bool = False
    # Reserve zone for future-step WFC suppression — cells the next step
    # needs clear for cluster formation. None for terminal steps (no demand).
    reserve_zone: frozenset[Position] | None = None
    # Predicted wild landing positions — enables PostGravityPropagator to
    # count wilds as same-symbol during virtual BFS, preventing WFC from
    # building groups that merge through wilds into booster-spawning clusters.
    predicted_wild_positions: frozenset[Position] | None = None
    # Atlas-derived booster landing positions keyed by source_cluster_index.
    # Forwarded into BoosterRules.resolve_collision so post-gravity booster
    # placements match what AtlasQuery already validated as armable.
    # None falls through to centroid-distance placement (backward compatible).
    predicted_landings: dict[int, Position] | None = None

    @classmethod
    def passthrough(cls) -> StepIntent:
        """Non-terminal intent for re-cascade — validates existing board state.

        Used when clusters already exist on a fully-filled board (e.g. after
        post-terminal booster refill). Skips reasoning and execution — the
        validator detects whatever clusters are present and computes payout.
        """
        return cls(
            step_type=StepType.CASCADE_CLUSTER,
            constrained_cells={},
            strategic_cells={},
            expected_cluster_count=Range(0, 0),
            expected_cluster_sizes=[],
            expected_cluster_tier=None,
            expected_spawns=[],
            expected_arms=[],
            expected_fires=[],
            wfc_propagators=[],
            wfc_symbol_weights={},
            predicted_post_gravity=None,
            terminal_near_misses=None,
            terminal_dormant_boosters=None,
            is_terminal=False,
        )
