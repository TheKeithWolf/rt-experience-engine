"""Tests for B4 — narrative.phase_builders.

Verifies the shared spawn_phase / arm_and_fire_phase builders produce
NarrativePhase instances structurally identical to the per-family
factories they replaced (snapshot of pre-refactor shapes).
"""

from __future__ import annotations

from ..narrative.arc import NarrativePhase
from ..narrative.phase_builders import arm_and_fire_phase, spawn_phase
from ..pipeline.protocols import Range


def test_b4_spawn_phase_matches_pre_refactor_rocket_shape() -> None:
    """spawn_phase("R", (Range(9, 10),)) reproduces the rocket spawn shape
    that was previously inline in archetypes/rocket.py.
    """
    phase = spawn_phase(booster_type="R", cluster_sizes=(Range(9, 10),))
    assert phase.id == "spawn_r"
    assert phase.spawns == ("R",)
    assert phase.arms is None
    assert phase.fires is None
    assert phase.cluster_sizes == (Range(9, 10),)
    assert phase.cluster_count == Range(1, 2)
    assert phase.repetitions == Range(1, 1)
    assert phase.ends_when == "always"


def test_b4_spawn_phase_matches_pre_refactor_bomb_shape() -> None:
    """spawn_phase("B", (Range(11, 12),)) reproduces the bomb spawn shape."""
    phase = spawn_phase(booster_type="B", cluster_sizes=(Range(11, 12),))
    assert phase.spawns == ("B",)
    assert phase.cluster_sizes == (Range(11, 12),)


def test_b4_arm_and_fire_phase_matches_pre_refactor_shape() -> None:
    """arm_and_fire_phase("R") reproduces the rocket fire-phase shape:
    arms and fires the same booster type on the same step.
    """
    phase = arm_and_fire_phase(booster_type="R")
    assert phase.arms == ("R",)
    assert phase.fires == ("R",)
    assert phase.spawns is None
    assert phase.cluster_sizes == (Range(5, 6),)


def test_b4_phase_id_intent_overrides_apply() -> None:
    """Custom phase_id and intent override the inferred defaults."""
    phase = spawn_phase(
        booster_type="R", cluster_sizes=(Range(9, 9),),
        phase_id="custom_id", intent="custom intent",
    )
    assert phase.id == "custom_id"
    assert phase.intent == "custom intent"


def test_b4_rocket_archetypes_register_after_refactor() -> None:
    """End-to-end smoke test: rocket archetypes (which now use the shared
    builders via thin wrappers) still register without errors.
    """
    from pathlib import Path
    from ..archetypes.registry import ArchetypeRegistry
    from ..archetypes.rocket import register_rocket_archetypes
    from ..archetypes.bomb import register_bomb_archetypes
    from ..config.loader import load_config

    config = load_config(
        Path(__file__).parent.parent / "config" / "default.yaml"
    )
    registry = ArchetypeRegistry(config)
    register_rocket_archetypes(registry)
    register_bomb_archetypes(registry)
    # No exceptions raised → both families register cleanly
