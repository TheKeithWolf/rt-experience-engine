"""Step Reasoner — reactive per-step decision engine for the Experience Engine.

Replaces the sequence planner (upfront abstract plans) with a reactive reasoner
that observes real board state, decides what should happen next, and delegates
to CSP + WFC for execution.

Data types (Step 3):
- StepType, StepIntent — what the reasoner wants a step to achieve
- DormantBooster, BoardContext — observable board state snapshot
- ClusterRecord, ProgressTracker — cumulative cascade progress
- SpawnRecord, FireRecord, StepResult — completed step outcomes

Assessment & Selection (Step 5):
- StepAssessment, StepAssessor — situation analysis from board state
- SelectionRule, StrategySelector, DEFAULT_SELECTION_RULES — data-driven dispatch
- StrategyRegistry — name-based strategy lookup
"""

from .assessor import StepAssessment, StepAssessor
from .context import BoardContext, DormantBooster
from .intent import StepIntent, StepType
from .progress import ClusterRecord, ProgressTracker
from .reasoner import StepReasoner
from .registry import StrategyRegistry
from .results import FireRecord, SpawnRecord, StepResult
from .selector import DEFAULT_SELECTION_RULES, SelectionRule, StrategySelector

__all__ = [
    "BoardContext",
    "ClusterRecord",
    "DEFAULT_SELECTION_RULES",
    "DormantBooster",
    "FireRecord",
    "ProgressTracker",
    "SelectionRule",
    "SpawnRecord",
    "StepAssessment",
    "StepAssessor",
    "StepIntent",
    "StepReasoner",
    "StepResult",
    "StepType",
    "StrategyRegistry",
    "StrategySelector",
]
