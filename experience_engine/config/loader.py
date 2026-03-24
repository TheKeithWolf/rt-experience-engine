"""Load and validate master config from YAML.

Handles weight renormalization, paytable tier expansion, and cross-field
validation. Raises ConfigValidationError with specific field path on failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import (
    AnticipationConfig,
    BoardConfig,
    BoosterConfig,
    CentipayoutConfig,
    ConfigValidationError,
    DiagnosticTarget,
    DiagnosticsConfig,
    FreespinConfig,
    GravityConfig,
    GridMultiplierConfig,
    MasterConfig,
    OutputConfig,
    PaytableConfig,
    PaytableEntry,
    PopulationConfig,
    ReasonerConfig,
    SolverConfig,
    SpawnThreshold,
    SymbolConfig,
    WincapConfig,
    WinLevelConfig,
    WinLevelTier,
)


def load_config(path: Path) -> MasterConfig:
    """Load and validate master config from YAML file.

    Performs:
    - YAML parsing
    - Sub-config construction with type validation
    - Family/archetype weight auto-renormalization
    - Spawn threshold non-overlap validation
    - Win level tier contiguity validation
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigValidationError("root", "YAML root must be a mapping")

    board = _build_board(raw.get("board", {}))
    gravity = _build_gravity(raw.get("gravity", {}))
    grid_multiplier = _build_grid_multiplier(raw.get("grid_multiplier", {}))
    boosters = _build_boosters(raw.get("boosters", {}))
    paytable = _build_paytable(raw.get("paytable", {}))
    centipayout = _build_centipayout(raw.get("centipayout", {}))
    win_levels = _build_win_levels(raw.get("win_levels", {}))
    freespin = _build_freespin(raw.get("freespin", {}))
    wincap = _build_wincap(raw.get("wincap", {}))
    solvers = _build_solvers(raw.get("solvers", {}))
    diagnostics = _build_diagnostics(raw.get("diagnostics", {}))
    population = _build_population(raw.get("population", {}))
    symbols = _build_symbols(raw.get("symbols", {}))
    anticipation = _build_anticipation(raw.get("anticipation", {}))
    output = _build_output(raw.get("output"))
    reasoner = _build_reasoner(raw.get("reasoner", {}))

    # Cross-field validations
    _validate_spawn_thresholds(boosters.spawn_thresholds)
    _validate_win_level_contiguity(win_levels.tiers)

    return MasterConfig(
        board=board,
        gravity=gravity,
        grid_multiplier=grid_multiplier,
        boosters=boosters,
        paytable=paytable,
        centipayout=centipayout,
        win_levels=win_levels,
        freespin=freespin,
        wincap=wincap,
        solvers=solvers,
        diagnostics=diagnostics,
        population=population,
        symbols=symbols,
        anticipation=anticipation,
        reasoner=reasoner,
        output=output,
    )


def _require(data: dict[str, Any], key: str, field_path: str) -> Any:
    """Extract a required key or raise ConfigValidationError."""
    if key not in data:
        raise ConfigValidationError(field_path, f"missing required field '{key}'")
    return data[key]


# ---------------------------------------------------------------------------
# Builder functions — one per sub-config
# ---------------------------------------------------------------------------


def _build_board(data: dict[str, Any]) -> BoardConfig:
    return BoardConfig(
        num_reels=_require(data, "num_reels", "board.num_reels"),
        num_rows=_require(data, "num_rows", "board.num_rows"),
        min_cluster_size=_require(data, "min_cluster_size", "board.min_cluster_size"),
    )


def _build_gravity(data: dict[str, Any]) -> GravityConfig:
    raw_priorities = _require(data, "donor_priorities", "gravity.donor_priorities")
    priorities = tuple(tuple(p) for p in raw_priorities)
    return GravityConfig(
        donor_priorities=priorities,
        max_settle_passes=_require(data, "max_settle_passes", "gravity.max_settle_passes"),
    )


def _build_grid_multiplier(data: dict[str, Any]) -> GridMultiplierConfig:
    return GridMultiplierConfig(
        initial_value=_require(data, "initial_value", "grid_multiplier.initial_value"),
        first_hit_value=_require(data, "first_hit_value", "grid_multiplier.first_hit_value"),
        increment=_require(data, "increment", "grid_multiplier.increment"),
        cap=_require(data, "cap", "grid_multiplier.cap"),
        minimum_contribution=_require(
            data, "minimum_contribution", "grid_multiplier.minimum_contribution"
        ),
    )


def _build_boosters(data: dict[str, Any]) -> BoosterConfig:
    raw_thresholds = _require(data, "spawn_thresholds", "boosters.spawn_thresholds")
    thresholds = tuple(
        SpawnThreshold(
            booster=t["booster"],
            min_size=t["min_size"],
            max_size=t["max_size"],
        )
        for t in raw_thresholds
    )
    return BoosterConfig(
        spawn_thresholds=thresholds,
        spawn_order=tuple(_require(data, "spawn_order", "boosters.spawn_order")),
        rocket_tie_orientation=_require(
            data, "rocket_tie_orientation", "boosters.rocket_tie_orientation"
        ),
        bomb_blast_radius=_require(data, "bomb_blast_radius", "boosters.bomb_blast_radius"),
        immune_to_rocket=tuple(
            _require(data, "immune_to_rocket", "boosters.immune_to_rocket")
        ),
        immune_to_bomb=tuple(
            _require(data, "immune_to_bomb", "boosters.immune_to_bomb")
        ),
        chain_initiators=tuple(
            _require(data, "chain_initiators", "boosters.chain_initiators")
        ),
    )


def _build_paytable(data: dict[str, Any]) -> PaytableConfig:
    raw_entries = _require(data, "entries", "paytable.entries")
    entries = tuple(
        PaytableEntry(
            symbol=e["symbol"],
            tier_min=e["tier_min"],
            tier_max=e["tier_max"],
            payout=float(e["payout"]),
        )
        for e in raw_entries
    )
    return PaytableConfig(entries=entries)


def _build_centipayout(data: dict[str, Any]) -> CentipayoutConfig:
    return CentipayoutConfig(
        multiplier=_require(data, "multiplier", "centipayout.multiplier"),
        rounding_base=_require(data, "rounding_base", "centipayout.rounding_base"),
    )


def _build_win_levels(data: dict[str, Any]) -> WinLevelConfig:
    raw_tiers = _require(data, "tiers", "win_levels.tiers")
    tiers = tuple(
        WinLevelTier(
            level=t["level"],
            min_payout=float(t["min_payout"]),
            max_payout=float(t["max_payout"]),
        )
        for t in raw_tiers
    )
    return WinLevelConfig(tiers=tiers)


def _build_freespin(data: dict[str, Any]) -> FreespinConfig:
    raw_awards = _require(data, "awards", "freespin.awards")
    awards = tuple(tuple(a) for a in raw_awards)
    return FreespinConfig(
        min_scatters_to_trigger=_require(
            data, "min_scatters_to_trigger", "freespin.min_scatters_to_trigger"
        ),
        max_scatters_on_board=_require(
            data, "max_scatters_on_board", "freespin.max_scatters_on_board"
        ),
        awards=awards,
        grid_multipliers_persist=_require(
            data, "grid_multipliers_persist", "freespin.grid_multipliers_persist"
        ),
    )


def _build_wincap(data: dict[str, Any]) -> WincapConfig:
    return WincapConfig(
        max_payout=float(_require(data, "max_payout", "wincap.max_payout")),
        halt_cascade=_require(data, "halt_cascade", "wincap.halt_cascade"),
    )


def _build_solvers(data: dict[str, Any]) -> SolverConfig:
    return SolverConfig(
        wfc_max_backtracks=_require(data, "wfc_max_backtracks", "solvers.wfc_max_backtracks"),
        wfc_min_symbol_weight=float(
            _require(data, "wfc_min_symbol_weight", "solvers.wfc_min_symbol_weight")
        ),
        csp_max_solve_time_ms=_require(
            data, "csp_max_solve_time_ms", "solvers.csp_max_solve_time_ms"
        ),
        asp_max_models=_require(data, "asp_max_models", "solvers.asp_max_models"),
        asp_rand_freq=float(
            _require(data, "asp_rand_freq", "solvers.asp_rand_freq")
        ),
        max_retries_per_instance=_require(
            data, "max_retries_per_instance", "solvers.max_retries_per_instance"
        ),
        max_construction_retries=_require(
            data, "max_construction_retries", "solvers.max_construction_retries"
        ),
    )


def _build_diagnostics(data: dict[str, Any]) -> DiagnosticsConfig:
    raw_targets = data.get("targets", [])
    targets = tuple(
        DiagnosticTarget(
            metric=t["metric"],
            min_value=t.get("min_value"),
            max_value=t.get("max_value"),
        )
        for t in raw_targets
    )
    return DiagnosticsConfig(targets=targets)


def _build_population(data: dict[str, Any]) -> PopulationConfig:
    raw_family = _require(data, "family_weights", "population.family_weights")
    raw_archetype = _require(data, "archetype_weights", "population.archetype_weights")

    # Renormalize family weights to sum to 1.0
    family_weights = _renormalize_weights(raw_family, "population.family_weights")

    # Renormalize per-family archetype weights
    archetype_weights_list: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for family_name, arch_dict in raw_archetype.items():
        normalized = _renormalize_weights(
            arch_dict, f"population.archetype_weights.{family_name}"
        )
        archetype_weights_list.append((family_name, normalized))

    return PopulationConfig(
        total_budget=_require(data, "total_budget", "population.total_budget"),
        min_instances_per_archetype=_require(
            data, "min_instances_per_archetype", "population.min_instances_per_archetype"
        ),
        family_weights=family_weights,
        archetype_weights=tuple(archetype_weights_list),
    )


def _build_symbols(data: dict[str, Any]) -> SymbolConfig:
    raw_rank = _require(data, "payout_rank", "symbols.payout_rank")
    payout_rank = tuple((k, v) for k, v in raw_rank.items())
    return SymbolConfig(
        standard=tuple(_require(data, "standard", "symbols.standard")),
        low_tier=tuple(_require(data, "low_tier", "symbols.low_tier")),
        high_tier=tuple(_require(data, "high_tier", "symbols.high_tier")),
        payout_rank=payout_rank,
    )


def _build_anticipation(data: dict[str, Any]) -> AnticipationConfig:
    return AnticipationConfig(
        trigger_threshold=_require(
            data, "trigger_threshold", "anticipation.trigger_threshold"
        ),
    )


def _build_output(data: dict[str, Any] | None) -> OutputConfig | None:
    """Build OutputConfig from YAML data, or None if section is absent.

    Optional section — Phases 1-13 configs omit this and get None.
    """
    if data is None:
        return None
    return OutputConfig(
        compression=_require(data, "compression", "output.compression"),
        output_dir=_require(data, "output_dir", "output.output_dir"),
        num_workers=_require(data, "num_workers", "output.num_workers"),
        mode_name=_require(data, "mode_name", "output.mode_name"),
        audit_survival_threshold=float(
            _require(data, "audit_survival_threshold", "output.audit_survival_threshold")
        ),
    )


def _build_reasoner(data: dict[str, Any]) -> ReasonerConfig:
    """Build ReasonerConfig from YAML data.

    Required section — all step reasoner thresholds and computation caps.
    """
    return ReasonerConfig(
        payout_low_fraction=float(
            _require(data, "payout_low_fraction", "reasoner.payout_low_fraction")
        ),
        payout_high_fraction=float(
            _require(data, "payout_high_fraction", "reasoner.payout_high_fraction")
        ),
        arming_urgency_horizon=int(
            _require(data, "arming_urgency_horizon", "reasoner.arming_urgency_horizon")
        ),
        terminal_dead_default_max_component=int(
            _require(data, "terminal_dead_default_max_component", "reasoner.terminal_dead_default_max_component")
        ),
        max_forward_simulations_per_step=int(
            _require(data, "max_forward_simulations_per_step", "reasoner.max_forward_simulations_per_step")
        ),
        max_strategic_cells_per_step=int(
            _require(data, "max_strategic_cells_per_step", "reasoner.max_strategic_cells_per_step")
        ),
        lookahead_depth=int(
            _require(data, "lookahead_depth", "reasoner.lookahead_depth")
        ),
    )


# ---------------------------------------------------------------------------
# Cross-field validators
# ---------------------------------------------------------------------------


def _renormalize_weights(
    weights: dict[str, float], field_path: str
) -> tuple[tuple[str, float], ...]:
    """Renormalize weights to sum to 1.0. Raises if all weights are zero."""
    if not weights:
        raise ConfigValidationError(field_path, "must not be empty")
    total = sum(weights.values())
    if total <= 0:
        raise ConfigValidationError(field_path, "weight sum must be > 0")
    return tuple((k, v / total) for k, v in weights.items())


def _validate_spawn_thresholds(thresholds: tuple[SpawnThreshold, ...]) -> None:
    """Verify spawn thresholds are non-overlapping and ascending by min_size."""
    sorted_thresholds = sorted(thresholds, key=lambda t: t.min_size)
    for i in range(len(sorted_thresholds) - 1):
        current = sorted_thresholds[i]
        next_t = sorted_thresholds[i + 1]
        if current.max_size >= next_t.min_size:
            raise ConfigValidationError(
                f"boosters.spawn_thresholds[{i}]",
                f"overlap: {current.booster} max_size={current.max_size} "
                f">= {next_t.booster} min_size={next_t.min_size}",
            )


def _validate_win_level_contiguity(tiers: tuple[WinLevelTier, ...]) -> None:
    """Verify win level tiers are contiguous — each tier's max equals the next tier's min."""
    sorted_tiers = sorted(tiers, key=lambda t: t.level)
    for i in range(len(sorted_tiers) - 1):
        current = sorted_tiers[i]
        next_t = sorted_tiers[i + 1]
        if current.max_payout != next_t.min_payout:
            raise ConfigValidationError(
                f"win_levels.tiers[{i}]",
                f"non-contiguous: tier {current.level} max_payout={current.max_payout} "
                f"!= tier {next_t.level} min_payout={next_t.min_payout}",
            )
