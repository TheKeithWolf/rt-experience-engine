"""Step reasoner — thin orchestration layer for per-step cascade decisions.

Delegates all work to three injected components:
1. StepAssessor: derives situation analysis from board state
2. StrategySelector: picks the right strategy from the assessment
3. StrategyRegistry: looks up the concrete strategy and calls plan_step

Zero game logic lives here — the reasoner is a pure coordinator.
"""

from __future__ import annotations

from ..archetypes.registry import ArchetypeSignature
from ..variance.hints import VarianceHints
from .assessor import StepAssessor
from .context import BoardContext
from .intent import StepIntent
from .progress import ProgressTracker
from .registry import StrategyRegistry
from .selector import StrategySelector


class StepReasoner:
    """Decides what to do at each cascade step via assess → select → delegate.

    Stateless — all mutable state lives in ProgressTracker. The reasoner
    reads observable state and produces a StepIntent describing what the
    next step should achieve.
    """

    __slots__ = ("_registry", "_selector", "_assessor")

    def __init__(
        self,
        registry: StrategyRegistry,
        selector: StrategySelector,
        assessor: StepAssessor,
    ) -> None:
        self._registry = registry
        self._selector = selector
        self._assessor = assessor

    def reason(
        self,
        context: BoardContext,
        progress: ProgressTracker,
        signature: ArchetypeSignature,
        variance: VarianceHints,
    ) -> StepIntent:
        """One reasoning cycle: assess → select → delegate.

        1. Assess: derive a StepAssessment from board state + progress
        2. Select: pick the strategy name matching the assessment
        3. Delegate: call the strategy's plan_step to produce a StepIntent
        """
        assessment = self._assessor.assess(context, progress, signature)
        strategy_name = self._selector.select(assessment)
        strategy = self._registry.get(strategy_name)
        return strategy.plan_step(context, progress, signature, variance)
