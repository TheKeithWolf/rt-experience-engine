"""Narrative data types — NarrativePhase and NarrativeArc.

A NarrativePhase is one beat in the player's experience (e.g., "clusters grow",
"booster fires", "board goes dead"). Phases carry structural constraints
(cluster count/sizes, booster lifecycle, wild behavior) and a transition rule
key that determines when the phase ends.

A NarrativeArc is the complete player experience as an ordered phase sequence
with global outcome constraints (payout, terminal state, chain depth).

The arc defines the *experience*; the ArchetypeSignature defines the *identity*.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pipeline.protocols import Range, RangeFloat, TerminalNearMissSpec
from ..primitives.symbols import SymbolTier


@dataclass(frozen=True, slots=True)
class NarrativePhase:
    """One beat in the player's experience.

    Each field drives generation constraints for steps that fall within this
    phase. The ``ends_when`` key selects a transition predicate from
    TRANSITION_RULES — the phase advances when the predicate returns True
    or max repetitions are reached.
    """

    # Identity — id is machine-readable, intent is a human-readable debug label
    id: str
    intent: str

    # How many consecutive cascade steps this phase can span
    repetitions: Range

    # Cluster constraints for each step within this phase
    cluster_count: Range
    cluster_sizes: tuple[Range, ...]
    cluster_symbol_tier: SymbolTier | None

    # Booster lifecycle — uniform tuple of type names, or None if unconstrained.
    # Using tuple[str, ...] eliminates isinstance checks that the old
    # str | tuple[str, ...] | None encoding required.
    spawns: tuple[str, ...] | None
    arms: tuple[str, ...] | None
    fires: tuple[str, ...] | None

    # Wild behavior required during this phase — "spawn", "bridge", "idle", or None
    wild_behavior: str | None

    # Key into TRANSITION_RULES — validated at registration time
    ends_when: str


@dataclass(frozen=True, slots=True)
class NarrativeArc:
    """The complete player experience as an ordered phase sequence.

    Global outcome constraints (payout, terminal state) span the full
    trajectory — they are not per-phase. Per-phase constraints live on
    NarrativePhase. Chain depth, rocket orientation, and LB target tier
    are also global because they span multiple phases.
    """

    phases: tuple[NarrativePhase, ...]

    # Outcome constraints
    payout: RangeFloat
    wild_count_on_terminal: Range
    terminal_near_misses: TerminalNearMissSpec | None
    dormant_boosters_on_terminal: tuple[str, ...] | None

    # Cross-phase constraints
    required_chain_depth: Range
    rocket_orientation: str | None
    lb_target_tier: SymbolTier | None
