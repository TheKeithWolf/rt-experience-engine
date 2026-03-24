"""Instance-level validation metrics collected during validation.

InstanceMetrics is a frozen snapshot of what the validator detected on a
generated instance. Used by the diagnostics engine for population-level
statistics and by the population controller to decide accept/reject.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..primitives.board import Position


@dataclass(frozen=True, slots=True)
class InstanceMetrics:
    """Validation metrics from a single generated instance.

    cascade_depth and grid_multiplier_values are populated for cascade instances
    (cascade_depth > 0). Static instances default to 0 and empty tuple.
    """

    archetype_id: str
    family: str
    criteria: str
    sim_id: int
    payout: float
    centipayout: int
    win_level: int
    cluster_count: int
    cluster_sizes: tuple[int, ...]
    cluster_symbols: tuple[str, ...]
    scatter_count: int
    near_miss_count: int
    near_miss_symbols: tuple[str, ...]
    max_component_size: int
    is_valid: bool
    validation_errors: tuple[str, ...]
    # Cascade-specific metrics — 0 / empty for static instances
    cascade_depth: int = 0
    # Final grid multiplier values (all positions, reel-major) — for diagnostics
    grid_multiplier_values: tuple[int, ...] = ()
    # Wild-specific metrics — count of wild symbols on the initial board
    wild_count: int = 0
    # Terminal near-miss count (for fakeout archetypes)
    terminal_near_miss_count: int = 0
    # Count of dormant boosters on the terminal board
    dormant_booster_count: int = 0
    # Booster spawn counts per type — (("R", 1), ("B", 0)) format for frozen compat
    booster_spawn_counts: tuple[tuple[str, int], ...] = ()
    # Booster fire counts per type
    booster_fire_counts: tuple[tuple[str, int], ...] = ()
    # Chain depth — number of chain-triggered fires in the instance
    chain_depth: int = 0
    # Actual rocket orientation for the first R fire ("H"/"V"/None if no rockets)
    rocket_orientation_actual: str | None = None
    # LB target — which standard symbol was targeted by lightball fire (None if no LB)
    lb_target_symbol: str | None = None
    # SLB targets — which standard symbols were targeted by superlightball fire
    slb_target_symbols: tuple[str, ...] = ()
    # Whether this instance triggers freespin (from archetype signature + scatter validation)
    triggers_freespin: bool = False
    # Whether this instance reached the win cap (signature + payout validation)
    reaches_wincap: bool = False
