"""Shared constraint construction helpers — DRY propagator assembly.

Strategies that build WFC propagator lists share common concerns that
cannot be expressed as plain data (they depend on live progress state,
signature data, and config objects). Rather than duplicating the
construction across every strategy — violating DRY at the six-strategy
mark — this module exposes tiny factory functions that return ready-to-use
propagator instances.

Each helper owns a single concern, reuses upstream primitives, and
derives thresholds from game rules (spawn_thresholds, min_cluster_size,
tier membership) — no hardcoded values.
"""

from __future__ import annotations

from ...archetypes.registry import ArchetypeSignature
from ...board_filler.spawn_cap_propagator import BoosterSpawnCapPropagator
from ...config.schema import SymbolConfig
from ...pipeline.protocols import Range
from ...primitives.board import Position
from ...primitives.symbols import (
    Symbol,
    SymbolTier,
    symbol_from_name,
    symbols_in_tier,
)
from ..evaluators import SpawnEvaluator
from ..progress import ProgressTracker


def build_spawn_cap_propagator(
    spawn_evaluator: SpawnEvaluator,
    progress: ProgressTracker,
    wild_positions: frozenset[Position] | None = None,
    committed_spawns: dict[str, int] | None = None,
) -> BoosterSpawnCapPropagator:
    """Assemble a BoosterSpawnCapPropagator from live progress + spawn rules.

    The propagator blocks placements whose projected component size would
    spawn a booster whose effective remaining max has hit 0 — derived from
    the signature's required_booster_spawns and live spawn counts on the
    ProgressTracker. Wild-aware when `wild_positions` is non-empty so
    wild-bridged merges count toward the projected component size.

    `committed_spawns` — booster types the caller has already planned to
    spawn this step (not yet recorded in ProgressTracker because CSP
    pinning precedes the transition that actually counts the spawn).
    Example: InitialClusterStrategy pins a 7-8 cluster, which will spawn
    one W during transition; passing `{"W": 1}` lets the propagator block
    WFC from forming an additional 7-8 component that would exceed the
    archetype's budget. Defaults to empty — behavior unchanged for callers
    with no pre-committed plan.
    """
    remaining = progress.remaining_booster_spawns()
    if committed_spawns:
        remaining = {
            btype: Range(
                max(0, r.min_val - committed_spawns.get(btype, 0)),
                max(0, r.max_val - committed_spawns.get(btype, 0)),
            )
            for btype, r in remaining.items()
        }
    return BoosterSpawnCapPropagator(
        size_to_booster=spawn_evaluator.booster_for_size,
        remaining_spawns=remaining,
        wild_positions=wild_positions,
    )


def forbidden_near_miss_symbols_for(
    signature: ArchetypeSignature,
    symbol_config: SymbolConfig,
) -> frozenset[Symbol]:
    """Symbols forbidden from forming near-miss-sized terminal components.

    When the archetype's `terminal_near_misses.symbol_tier` pins the tier
    (HIGH or LOW), every symbol outside that tier must not form a
    near-miss-sized component on the terminal board — otherwise validation
    fails with `terminal NM symbol LX is low, expected high`.

    Returns an empty frozenset when no tier constraint exists or when
    the constraint allows both tiers (SymbolTier.ANY). Callers forward
    this into `NearMissAwareDeadPropagator`, which treats an empty set
    as "no additional tier constraint" (backward compatible).
    """
    nm_spec = signature.terminal_near_misses
    if nm_spec is None or nm_spec.symbol_tier is None:
        return frozenset()
    if nm_spec.symbol_tier is SymbolTier.ANY:
        return frozenset()
    # All standard symbols minus those in the allowed tier = forbidden set
    allowed = frozenset(symbols_in_tier(nm_spec.symbol_tier, symbol_config))
    all_standard = frozenset(
        symbol_from_name(name) for name in symbol_config.standard
    )
    return all_standard - allowed
