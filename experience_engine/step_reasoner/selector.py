"""Strategy selection — data-driven dispatch from assessment to strategy name.

SelectionRule pairs a condition (predicate on StepAssessment) with a
strategy name and a priority. StrategySelector evaluates rules in
descending priority order — first match wins. If no rule matches, the
default strategy ("cascade_cluster") is used.

No if/elif chains: adding a new strategy means registering a new rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .assessor import StepAssessment


@dataclass(frozen=True)
class SelectionRule:
    """Maps a condition on the assessment to a strategy name.

    Higher priority rules are evaluated first. The first rule whose
    condition returns True determines the strategy.

    No slots=True — Callable fields interact poorly with __slots__
    on some Python versions.
    """

    strategy_name: str
    condition: Callable[[StepAssessment], bool]
    priority: int


class StrategySelector:
    """Picks the right strategy name from a StepAssessment.

    Rules are sorted by descending priority at init time (O(n log n) once).
    select() is a linear scan — fast for the expected rule count (~10).
    """

    __slots__ = ("_rules", "_default")

    def __init__(
        self,
        rules: list[SelectionRule],
        default_strategy: str = "cascade_cluster",
    ) -> None:
        self._rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self._default = default_strategy

    def select(self, assessment: StepAssessment) -> str:
        """Return the strategy name for the first matching rule, or the default."""
        for rule in self._rules:
            if rule.condition(assessment):
                return rule.strategy_name
        return self._default


# ---------------------------------------------------------------------------
# Default rules — covers the 7 non-default strategy triggers
# ---------------------------------------------------------------------------

DEFAULT_SELECTION_RULES: list[SelectionRule] = [
    # Terminal states — highest priority, checked when cascade must end
    SelectionRule(
        strategy_name="terminal_near_miss",
        condition=lambda a: a.must_terminate_now and a.terminal_near_misses_required is not None,
        priority=100,
    ),
    SelectionRule(
        strategy_name="terminal_dead",
        condition=lambda a: a.must_terminate_now,
        priority=99,
    ),
    # Initial step — picks the right initial strategy based on family
    SelectionRule(
        strategy_name="initial_dead",
        condition=lambda a: a.is_first_step and a.signature_is_dead_family,
        priority=90,
    ),
    SelectionRule(
        strategy_name="initial_cluster",
        condition=lambda a: a.is_first_step,
        priority=89,
    ),
    # Mid-cascade priorities — booster lifecycle and chaining
    SelectionRule(
        strategy_name="booster_arm",
        condition=lambda a: a.booster_needs_arming_soon,
        priority=80,
    ),
    SelectionRule(
        strategy_name="wild_bridge",
        condition=lambda a: len(a.wilds_available_for_bridge) > 0 and a.needs_wild_bridge,
        priority=70,
    ),
    SelectionRule(
        strategy_name="booster_setup",
        condition=lambda a: a.needs_chain,
        priority=60,
    ),
]
