"""Strategy protocol — interface contract for all step strategies.

Defined here (Step 5) so StrategyRegistry can type-check registrations.
Concrete implementations are Step 6 deliverables.

Follows the SpatialSolver / BoardFiller pattern in pipeline.protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..context import BoardContext
from ..intent import StepIntent
from ..progress import ProgressTracker
from ...archetypes.registry import ArchetypeSignature
from ...variance.hints import VarianceHints


@runtime_checkable
class StepStrategy(Protocol):
    """Produces a StepIntent for one cascade step type.

    Each strategy encapsulates one approach to building a cascade step
    (e.g., cascade_cluster forms clusters, booster_arm positions dormant
    boosters for firing). The reasoner selects the strategy, then calls
    plan_step to get the intent for CSP/WFC execution.
    """

    def plan_step(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent: ...
