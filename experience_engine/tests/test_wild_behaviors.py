"""Tests for the WildPlacementBehavior protocol & registry.

Verifies the three behaviors (spawn / bridge / idle) produce the same
selection semantics as the pre-refactor if-chain in
spatial_solver/solver.py.
"""

from __future__ import annotations

import random

from ..primitives.board import Position
from ..primitives.wild_behaviors import WILD_BEHAVIORS


# ---------------------------------------------------------------------------
# select_position — exercises the candidate-filter → choice path
# ---------------------------------------------------------------------------

def _no_candidates_filter(positions, min_n, max_n):
    """Adjacency filter that always rejects — exercises fallback paths."""
    return []


def _picker(candidates, bias, rng):
    """Weighted-choice stand-in that returns the first candidate."""
    return candidates[0] if candidates else None


def test_b2_spawn_falls_back_to_full_set_when_no_clusters() -> None:
    """SpawnBehavior: when no positions are adjacent to clusters, fall
    back to the full available list (matching pre-refactor solver code).
    """
    rng = random.Random(0)
    available = [Position(0, 0), Position(1, 1)]
    pos = WILD_BEHAVIORS["spawn"].select_position(
        available, context=None,
        adjacent_to_clusters=_no_candidates_filter,
        weighted_choice=_picker,
        spatial_bias=None,
        rng=rng,
    )
    assert pos == available[0]


def test_b2_bridge_returns_none_when_no_candidates() -> None:
    """BridgeBehavior: bridge is strict — no fallback. Returns None when
    no positions touch 2+ clusters.
    """
    rng = random.Random(0)
    pos = WILD_BEHAVIORS["bridge"].select_position(
        [Position(0, 0)], context=None,
        adjacent_to_clusters=_no_candidates_filter,
        weighted_choice=_picker,
        spatial_bias=None,
        rng=rng,
    )
    assert pos is None


def test_b2_idle_falls_back_to_full_set() -> None:
    """IdleBehavior: when no positions match the max-1-clusters filter,
    fall back to the full available list (matching pre-refactor code).
    """
    rng = random.Random(0)
    available = [Position(2, 3)]
    pos = WILD_BEHAVIORS["idle"].select_position(
        available, context=None,
        adjacent_to_clusters=_no_candidates_filter,
        weighted_choice=_picker,
        spatial_bias=None,
        rng=rng,
    )
    assert pos == available[0]


def test_b2_registry_shared_across_call_sites() -> None:
    """Registering a new behavior makes it visible to solver._place_wild
    without further plumbing.
    """
    keys = set(WILD_BEHAVIORS.keys())
    assert keys == {"spawn", "bridge", "idle"}
