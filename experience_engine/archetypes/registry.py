"""Archetype registry — signature types, validation, and lookup.

An archetype is a named experience template defining structural, cascade, booster,
and payout constraints for a single spin outcome. The registry validates signatures
against game rules (CONTRACT-SIG-1 through SIG-7) and provides O(1) lookup by id
and O(1) lookup by family.

All validation thresholds come from MasterConfig — zero hardcoded values.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import MasterConfig
from ..narrative.arc import NarrativeArc
from ..narrative.derivation import derive_constraints
from ..narrative.transitions import ALLOWED_TRANSITION_KEYS
from ..pipeline.protocols import Range, RangeFloat, TerminalNearMissSpec
from ..primitives.symbols import SymbolTier


# ---------------------------------------------------------------------------
# Family → criteria mapping (single source of truth for family names)
# ---------------------------------------------------------------------------

# B8: family identifiers sourced from the typed ArchetypeFamily StrEnum so
# the set of legal values is defined once. StrEnum's str-equality means
# downstream callers comparing sig.family to either form continue to work.
from .families import ArchetypeFamily

REGISTERED_FAMILIES: frozenset[str] = frozenset(ArchetypeFamily)

FAMILY_CRITERIA: dict[str, str] = {
    ArchetypeFamily.DEAD: "0",
    ArchetypeFamily.T1: "basegame",
    ArchetypeFamily.WILD: "basegame",
    ArchetypeFamily.ROCKET: "basegame",
    ArchetypeFamily.BOMB: "basegame",
    ArchetypeFamily.LB: "basegame",
    ArchetypeFamily.SLB: "basegame",
    ArchetypeFamily.CHAIN: "basegame",
    ArchetypeFamily.TRIGGER: "freegame",
    ArchetypeFamily.WINCAP: "wincap",
    # Reel-strip-driven basegame — the strip determines the outcome;
    # signatures are wide envelopes validated by InstanceValidator.
    ArchetypeFamily.REEL: "basegame",
}


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

# TerminalNearMissSpec re-exported from pipeline.protocols (moved there to
# break circular import with narrative.arc). Existing imports still work.

@dataclass(frozen=True, slots=True)
class CascadeStepConstraint:
    """Per-step constraints within a cascade sequence (used by ASP planner)."""

    cluster_count: Range
    cluster_sizes: tuple[Range, ...]
    cluster_symbol_tier: SymbolTier | None
    # Single booster name or tuple of names for multi-booster steps (e.g., ("R", "B"))
    must_spawn_booster: str | tuple[str, ...] | None
    must_arm_booster: str | tuple[str, ...] | None
    # Wild behavior required at this step — "spawn", "bridge", "idle", or None
    # Encoded as ASP fact required_wild_behavior(step, action_code)
    wild_behavior: str | None = None


# ---------------------------------------------------------------------------
# Archetype signature
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ArchetypeSignature:
    """Immutable specification for one archetype's structural constraints.

    Every field drives generation and validation — the solver pipeline reads
    these to know what to build, and the validator reads them to verify output.
    """

    id: str
    family: str
    # RGS criteria: "0" (dead), "basegame", "freegame", "wincap"
    criteria: str

    # Initial board constraints
    required_cluster_count: Range
    required_cluster_sizes: tuple[Range, ...]
    required_cluster_symbols: SymbolTier | None
    required_scatter_count: Range
    required_near_miss_count: Range
    required_near_miss_symbol_tier: SymbolTier | None
    # Maximum component size for any single standard symbol (dead boards)
    max_component_size: int | None

    # Cascade constraints
    required_cascade_depth: Range
    cascade_steps: tuple[CascadeStepConstraint, ...] | None

    # Booster constraints — booster_name → required count range
    required_booster_spawns: dict[str, Range]
    required_booster_fires: dict[str, Range]
    required_chain_depth: Range
    rocket_orientation: str | None
    lb_target_tier: SymbolTier | None

    # Narrative constraints
    symbol_tier_per_step: dict[int, SymbolTier] | None
    terminal_near_misses: TerminalNearMissSpec | None
    dormant_boosters_on_terminal: tuple[str, ...] | None

    # Payout constraints
    payout_range: RangeFloat

    # Feature constraints
    triggers_freespin: bool
    reaches_wincap: bool

    # Narrative definition — None for depth-0 archetypes, replaces cascade_steps
    narrative_arc: NarrativeArc | None = None


# ---------------------------------------------------------------------------
# Arc-based signature factory
# ---------------------------------------------------------------------------

def build_arc_signature(arc: NarrativeArc, **kwargs: object) -> ArchetypeSignature:
    """Build an ArchetypeSignature with derived fields computed from the arc.

    Single construction path for arc-based signatures — eliminates drift
    between the arc's phase definitions and the derived structural fields.
    Callers pass identity/feature fields (id, family, criteria, scatter, etc.)
    via kwargs; arc-derivable fields are computed here.
    """
    derived = derive_constraints(arc)
    return ArchetypeSignature(
        narrative_arc=arc,
        required_cascade_depth=derived.cascade_depth,
        required_cluster_count=derived.cluster_count,
        required_cluster_sizes=derived.cluster_sizes,
        required_booster_spawns=derived.booster_spawns,
        required_booster_fires=derived.booster_fires,
        required_chain_depth=arc.required_chain_depth,
        payout_range=arc.payout,
        terminal_near_misses=arc.terminal_near_misses,
        dormant_boosters_on_terminal=arc.dormant_boosters_on_terminal,
        rocket_orientation=arc.rocket_orientation,
        lb_target_tier=arc.lb_target_tier,
        # cascade_steps and symbol_tier_per_step set to None — superseded by arc
        cascade_steps=None,
        symbol_tier_per_step=None,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SignatureValidationError(Exception):
    """Raised when an archetype signature violates a CONTRACT-SIG rule."""

    def __init__(self, sig_id: str, contract: str, reason: str) -> None:
        self.sig_id = sig_id
        self.contract = contract
        super().__init__(f"{sig_id} [{contract}]: {reason}")


class ArchetypeRegistry:
    """Registration, validation, and lookup for archetype signatures.

    Validates each signature against CONTRACT-SIG-1 through SIG-7 using
    thresholds from MasterConfig. Provides O(1) id lookup and per-family queries.
    """

    __slots__ = ("_config", "_by_id", "_by_family")

    def __init__(self, config: MasterConfig) -> None:
        self._config = config
        self._by_id: dict[str, ArchetypeSignature] = {}
        self._by_family: dict[str, list[ArchetypeSignature]] = {
            f: [] for f in REGISTERED_FAMILIES
        }

    def register(self, sig: ArchetypeSignature) -> None:
        """Validate and register an archetype signature.

        Raises SignatureValidationError if any CONTRACT-SIG rule is violated.
        """
        self._validate_signature(sig)
        self._by_id[sig.id] = sig
        self._by_family[sig.family].append(sig)

    def get(self, archetype_id: str) -> ArchetypeSignature:
        """Look up a signature by id. Raises KeyError if not registered."""
        return self._by_id[archetype_id]

    def get_family(self, family: str) -> tuple[ArchetypeSignature, ...]:
        """Return all signatures in a family, in registration order."""
        return tuple(self._by_family.get(family, ()))

    def all_ids(self) -> frozenset[str]:
        """All registered archetype ids."""
        return frozenset(self._by_id.keys())

    def registered_families(self) -> frozenset[str]:
        """Families that have at least one registered archetype."""
        return frozenset(f for f, sigs in self._by_family.items() if sigs)

    def _validate_signature(self, sig: ArchetypeSignature) -> None:
        """Enforce CONTRACT-SIG-1 through SIG-7."""

        # SIG-1: Unique id
        if sig.id in self._by_id:
            raise SignatureValidationError(
                sig.id, "CONTRACT-SIG-1",
                f"duplicate id — already registered",
            )

        # SIG-2: Family must be a recognized family
        if sig.family not in REGISTERED_FAMILIES:
            raise SignatureValidationError(
                sig.id, "CONTRACT-SIG-2",
                f"unknown family '{sig.family}' — must be one of {sorted(REGISTERED_FAMILIES)}",
            )

        # SIG-3: Payout range min >= 0; dead family requires both 0
        if sig.payout_range.min_val < 0.0:
            raise SignatureValidationError(
                sig.id, "CONTRACT-SIG-3",
                f"payout_range.min_val ({sig.payout_range.min_val}) must be >= 0",
            )
        if sig.family == "dead":
            if sig.payout_range.min_val != 0.0 or sig.payout_range.max_val != 0.0:
                raise SignatureValidationError(
                    sig.id, "CONTRACT-SIG-3",
                    f"dead family must have payout_range (0.0, 0.0), "
                    f"got ({sig.payout_range.min_val}, {sig.payout_range.max_val})",
                )

        # SIG-4: triggers_freespin requires enough scatters
        min_scatters_to_trigger = self._config.freespin.min_scatters_to_trigger
        if sig.triggers_freespin:
            if sig.required_scatter_count.min_val < min_scatters_to_trigger:
                raise SignatureValidationError(
                    sig.id, "CONTRACT-SIG-4",
                    f"triggers_freespin=True requires scatter_count.min >= "
                    f"{min_scatters_to_trigger}, got {sig.required_scatter_count.min_val}",
                )

        # SIG-5: cascade_depth.max = 0 → narrative_arc and cascade_steps must be None/empty
        if sig.required_cascade_depth.max_val == 0:
            if sig.cascade_steps is not None and len(sig.cascade_steps) > 0:
                raise SignatureValidationError(
                    sig.id, "CONTRACT-SIG-5",
                    "cascade_depth.max=0 but cascade_steps is non-empty",
                )
            if sig.narrative_arc is not None:
                raise SignatureValidationError(
                    sig.id, "CONTRACT-SIG-5",
                    "cascade_depth.max=0 but narrative_arc is not None",
                )

        # SIG-ARC: all NarrativePhase.ends_when values must exist in TRANSITION_RULES
        if sig.narrative_arc is not None:
            for phase in sig.narrative_arc.phases:
                if phase.ends_when not in ALLOWED_TRANSITION_KEYS:
                    raise SignatureValidationError(
                        sig.id, "CONTRACT-SIG-ARC",
                        f"unknown ends_when value '{phase.ends_when}' — "
                        f"must be one of {sorted(ALLOWED_TRANSITION_KEYS)}",
                    )

        # SIG-6: booster_fires non-empty → cascade_depth.min >= 1
        if sig.required_booster_fires:
            if sig.required_cascade_depth.min_val < 1:
                raise SignatureValidationError(
                    sig.id, "CONTRACT-SIG-6",
                    "booster_fires specified but cascade_depth.min < 1",
                )

        # SIG-7: chain_depth.min > 0 → at least two distinct booster types fire
        if sig.required_chain_depth.min_val > 0:
            fire_types = set(sig.required_booster_fires.keys())
            if len(fire_types) < 2:
                raise SignatureValidationError(
                    sig.id, "CONTRACT-SIG-7",
                    f"chain_depth.min > 0 requires >= 2 booster fire types, "
                    f"got {len(fire_types)}: {fire_types}",
                )
