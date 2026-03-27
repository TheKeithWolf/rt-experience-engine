"""Curriculum scheduler — controls which archetypes are eligible at each training stage.

Filters by required_cascade_depth.max_val from the archetype signature,
not by archetype name. Uses dict dispatch on difficulty filter strings.
"""

from __future__ import annotations

from typing import Callable

from ...archetypes.registry import ArchetypeRegistry
from ...config.schema import CurriculumPhase


# Difficulty filter dispatch — maps filter name to depth predicate.
# No if/elif chains — adding a new filter requires only a new entry.
_DIFFICULTY_FILTERS: dict[str, Callable[[int], bool]] = {
    "standard": lambda depth: 2 <= depth <= 3,
    "hard": lambda depth: depth >= 4,
    "all": lambda depth: depth >= 2,
}


class CurriculumScheduler:
    """Selects eligible archetypes based on training progress.

    At each episode count, determines which curriculum phase is active
    and filters archetypes by cascade depth accordingly.
    """

    __slots__ = ("_phases", "_registry", "_archetype_depths")

    def __init__(
        self,
        curriculum_phases: tuple[CurriculumPhase, ...],
        registry: ArchetypeRegistry,
    ) -> None:
        # Sort phases by episode threshold (ascending)
        self._phases = sorted(curriculum_phases, key=lambda p: p.episode_threshold)
        self._registry = registry
        # Pre-compute cascade depths for all archetypes
        self._archetype_depths: dict[str, int] = {
            aid: registry.get(aid).required_cascade_depth.max_val
            for aid in registry.all_ids()
        }

    def eligible_archetypes(self, episode_count: int) -> list[str]:
        """Return archetype IDs eligible for the current training stage.

        Selects the last curriculum phase whose threshold has been reached,
        then filters archetypes by the phase's difficulty predicate.
        """
        # Find the active curriculum phase (last one whose threshold is reached)
        active_filter = self._phases[0].difficulty_filter
        for phase in self._phases:
            if episode_count >= phase.episode_threshold:
                active_filter = phase.difficulty_filter
            else:
                break

        predicate = _DIFFICULTY_FILTERS[active_filter]
        return [
            aid for aid, depth in self._archetype_depths.items()
            if predicate(depth)
        ]

    def current_phase_name(self, episode_count: int) -> str:
        """Return the difficulty filter name for the current episode count."""
        active = self._phases[0].difficulty_filter
        for phase in self._phases:
            if episode_count >= phase.episode_threshold:
                active = phase.difficulty_filter
        return active
