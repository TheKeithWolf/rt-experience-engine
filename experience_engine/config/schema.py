"""Typed frozen dataclass hierarchy for the Experience Engine master configuration.

Every tweakable constant in the engine is represented here. Modules receive their
relevant config slice via __init__ (dependency injection). No module contains
hardcoded literals — all values originate from this schema, loaded from YAML.
"""

from __future__ import annotations

from dataclasses import dataclass


class ConfigValidationError(Exception):
    """Raised when config validation fails, with field path and reason."""

    def __init__(self, field_path: str, reason: str) -> None:
        self.field_path = field_path
        self.reason = reason
        super().__init__(f"{field_path}: {reason}")


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BoardConfig:
    """Board dimensions and cluster detection threshold."""

    num_reels: int
    num_rows: int
    min_cluster_size: int

    def __post_init__(self) -> None:
        if self.num_reels < 1:
            raise ConfigValidationError("board.num_reels", "must be >= 1")
        if self.num_rows < 1:
            raise ConfigValidationError("board.num_rows", "must be >= 1")
        if self.min_cluster_size < 2:
            raise ConfigValidationError("board.min_cluster_size", "must be >= 2")


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GravityConfig:
    """Pull-model donor priority order and safety limit for settle passes."""

    # (dx, dy) offsets from empty cell to potential donor
    # (0,-1)=directly above, (1,-1)=above-right, (-1,-1)=above-left
    donor_priorities: tuple[tuple[int, int], ...]
    max_settle_passes: int

    def __post_init__(self) -> None:
        if not self.donor_priorities:
            raise ConfigValidationError("gravity.donor_priorities", "must not be empty")
        if self.max_settle_passes < 1:
            raise ConfigValidationError("gravity.max_settle_passes", "must be >= 1")


# ---------------------------------------------------------------------------
# Grid Multipliers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GridMultiplierConfig:
    """Grid position multiplier progression and bounds."""

    initial_value: int
    first_hit_value: int
    increment: int
    cap: int
    # Floor for position_multiplier_sum — ensures base payout always applies
    minimum_contribution: int

    def __post_init__(self) -> None:
        if self.cap < self.first_hit_value:
            raise ConfigValidationError(
                "grid_multiplier.cap",
                f"cap ({self.cap}) must be >= first_hit_value ({self.first_hit_value})",
            )
        if self.increment < 1:
            raise ConfigValidationError("grid_multiplier.increment", "must be >= 1")
        if self.minimum_contribution < 0:
            raise ConfigValidationError("grid_multiplier.minimum_contribution", "must be >= 0")


# ---------------------------------------------------------------------------
# Boosters
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SpawnThreshold:
    """Maps a cluster size range to a booster type."""

    booster: str
    min_size: int
    max_size: int

    def __post_init__(self) -> None:
        if self.min_size < 1:
            raise ConfigValidationError(
                f"spawn_threshold({self.booster}).min_size", "must be >= 1"
            )
        if self.max_size < self.min_size:
            raise ConfigValidationError(
                f"spawn_threshold({self.booster})",
                f"max_size ({self.max_size}) < min_size ({self.min_size})",
            )


@dataclass(frozen=True, slots=True)
class BoosterConfig:
    """Booster spawn rules, blast parameters, and chain initiators."""

    spawn_thresholds: tuple[SpawnThreshold, ...]
    spawn_order: tuple[str, ...]
    # Tie-breaking orientation when a cluster's row span equals column span
    rocket_tie_orientation: str
    # Manhattan distance for bomb blast radius (1 = 3x3 area)
    bomb_blast_radius: int
    immune_to_rocket: tuple[str, ...]
    immune_to_bomb: tuple[str, ...]
    # Only these booster types can initiate chain reactions
    chain_initiators: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.spawn_thresholds:
            raise ConfigValidationError("boosters.spawn_thresholds", "must not be empty")
        if self.rocket_tie_orientation not in ("H", "V"):
            raise ConfigValidationError(
                "boosters.rocket_tie_orientation",
                f"must be 'H' or 'V', got '{self.rocket_tie_orientation}'",
            )
        if self.bomb_blast_radius < 1:
            raise ConfigValidationError("boosters.bomb_blast_radius", "must be >= 1")


# ---------------------------------------------------------------------------
# Paytable
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PaytableEntry:
    """A single paytable entry mapping a cluster size range + symbol to a payout."""

    symbol: str
    tier_min: int
    tier_max: int
    payout: float

    def __post_init__(self) -> None:
        if self.tier_max < self.tier_min:
            raise ConfigValidationError(
                f"paytable({self.symbol})",
                f"tier_max ({self.tier_max}) < tier_min ({self.tier_min})",
            )
        if self.payout < 0:
            raise ConfigValidationError(
                f"paytable({self.symbol})", f"payout ({self.payout}) must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class PaytableConfig:
    """Complete paytable — list of entries expanded from tier ranges at load time."""

    entries: tuple[PaytableEntry, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ConfigValidationError("paytable.entries", "must not be empty")


# ---------------------------------------------------------------------------
# Centipayout
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CentipayoutConfig:
    """Centipayout conversion: round(payout * multiplier / rounding_base) * rounding_base."""

    multiplier: int
    rounding_base: int

    def __post_init__(self) -> None:
        if self.multiplier < 1:
            raise ConfigValidationError("centipayout.multiplier", "must be >= 1")
        if self.rounding_base < 1:
            raise ConfigValidationError("centipayout.rounding_base", "must be >= 1")


# ---------------------------------------------------------------------------
# Win Levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WinLevelTier:
    """A single win level tier with payout range [min_payout, max_payout)."""

    level: int
    min_payout: float
    max_payout: float

    def __post_init__(self) -> None:
        if self.max_payout <= self.min_payout and self.max_payout != float("inf"):
            raise ConfigValidationError(
                f"win_levels.tier({self.level})",
                f"max_payout ({self.max_payout}) must be > min_payout ({self.min_payout})",
            )


@dataclass(frozen=True, slots=True)
class WinLevelConfig:
    """Win level tier definitions — contiguous, non-overlapping payout ranges."""

    tiers: tuple[WinLevelTier, ...]

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ConfigValidationError("win_levels.tiers", "must not be empty")


# ---------------------------------------------------------------------------
# Freespin
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FreespinConfig:
    """Freespin trigger rules and grid multiplier persistence."""

    min_scatters_to_trigger: int
    max_scatters_on_board: int
    # (scatter_count, freespins_awarded) pairs — stored as tuple for immutability
    awards: tuple[tuple[int, int], ...]
    # Whether grid multipliers carry over between freespin spins
    grid_multipliers_persist: bool

    def __post_init__(self) -> None:
        if self.min_scatters_to_trigger < 1:
            raise ConfigValidationError(
                "freespin.min_scatters_to_trigger", "must be >= 1"
            )
        if self.max_scatters_on_board < self.min_scatters_to_trigger:
            raise ConfigValidationError(
                "freespin.max_scatters_on_board",
                "must be >= min_scatters_to_trigger",
            )


# ---------------------------------------------------------------------------
# Wincap
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WincapConfig:
    """Maximum payout cap and cascade halt behavior."""

    max_payout: float
    # Whether to stop cascade processing immediately upon reaching wincap
    halt_cascade: bool

    def __post_init__(self) -> None:
        if self.max_payout <= 0:
            raise ConfigValidationError("wincap.max_payout", "must be > 0")


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Solver limits — backtracks, time budgets, model counts, retries."""

    wfc_max_backtracks: int
    # Floor for WFC symbol selection weights — prevents zero-probability symbols
    # during weighted collapse (CONSTRAINT-VE-3: variance weights must never be zero)
    wfc_min_symbol_weight: float
    csp_max_solve_time_ms: int
    asp_max_models: int
    asp_rand_freq: float
    max_retries_per_instance: int
    max_construction_retries: int
    # Retry generation when validation fails (terminal max_component, payout overflow)
    max_validation_retries: int = 1
    # BoardSymbolNonAdjacency enforced for cascade steps 0 through this index
    board_adjacency_max_step: int = 2
    # BFS seed retry limit — when frontier exhaustion occurs on fragmented
    # available-sets, retry with a different seed before failing the attempt
    max_seed_retries: int = 5

    def __post_init__(self) -> None:
        if self.wfc_max_backtracks < 0:
            raise ConfigValidationError("solvers.wfc_max_backtracks", "must be >= 0")
        if self.wfc_min_symbol_weight <= 0.0:
            raise ConfigValidationError(
                "solvers.wfc_min_symbol_weight", "must be > 0.0"
            )
        if self.csp_max_solve_time_ms < 1:
            raise ConfigValidationError("solvers.csp_max_solve_time_ms", "must be >= 1")
        if not (0.0 <= self.asp_rand_freq <= 1.0):
            raise ConfigValidationError("solvers.asp_rand_freq", "must be in [0.0, 1.0]")
        if self.max_seed_retries < 1:
            raise ConfigValidationError("solvers.max_seed_retries", "must be >= 1")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DiagnosticTarget:
    """A single diagnostic metric with optional min/max bounds."""

    metric: str
    min_value: float | None
    max_value: float | None


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """Diagnostic targets for population-level metric validation."""

    targets: tuple[DiagnosticTarget, ...]


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PopulationConfig:
    """Budget allocation and weight distribution across archetype families."""

    total_budget: int
    min_instances_per_archetype: int
    # Relative weights per family — auto-renormalized to sum to 1.0 at load time
    family_weights: tuple[tuple[str, float], ...]
    # Per-family archetype weights — auto-renormalized per family at load time
    archetype_weights: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]

    def __post_init__(self) -> None:
        if self.total_budget < 1:
            raise ConfigValidationError("population.total_budget", "must be >= 1")
        if self.min_instances_per_archetype < 1:
            raise ConfigValidationError(
                "population.min_instances_per_archetype", "must be >= 1"
            )


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SymbolConfig:
    """Symbol classification lists and payout ranking."""

    standard: tuple[str, ...]
    low_tier: tuple[str, ...]
    high_tier: tuple[str, ...]
    # Higher rank = more valuable — used as tiebreaker in booster targeting
    payout_rank: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.standard:
            raise ConfigValidationError("symbols.standard", "must not be empty")
        # Low + high should cover all standard symbols
        low_high = set(self.low_tier) | set(self.high_tier)
        standard_set = set(self.standard)
        if low_high != standard_set:
            raise ConfigValidationError(
                "symbols",
                f"low_tier + high_tier ({low_high}) must equal standard ({standard_set})",
            )


# ---------------------------------------------------------------------------
# Anticipation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AnticipationConfig:
    """Rules for computing anticipation array from scatter positions."""

    # Scatters needed before anticipation starts on the next reel
    trigger_threshold: int

    def __post_init__(self) -> None:
        if self.trigger_threshold < 1:
            raise ConfigValidationError(
                "anticipation.trigger_threshold", "must be >= 1"
            )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Phase 14 output settings — compression, paths, parallelism, and audit."""

    # Zstandard compression for book JSONL files
    compression: bool
    # Base output directory (relative to game root)
    output_dir: str
    # Multiprocessing worker count for parallel generation
    num_workers: int
    # Bet mode name — controls output file naming (books_{mode}.jsonl, lookUpTable_{mode}.csv)
    mode_name: str
    # Archetype survival rate floor for post-optimization audit flagging
    audit_survival_threshold: float

    def __post_init__(self) -> None:
        if self.num_workers < 1:
            raise ConfigValidationError("output.num_workers", "must be >= 1")
        if not self.mode_name:
            raise ConfigValidationError("output.mode_name", "must not be empty")
        if not (0.0 <= self.audit_survival_threshold <= 1.0):
            raise ConfigValidationError(
                "output.audit_survival_threshold", "must be in [0.0, 1.0]"
            )


# ---------------------------------------------------------------------------
# Reasoner
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReasonerConfig:
    """Step reasoner thresholds, payout budgeting, and computation caps.

    payout_low_fraction / payout_high_fraction bracket the "expected spend"
    range: below low → underspending (pacing too slow), above high →
    overspending (risk of overshooting budget). arming_urgency_horizon
    controls how many steps of slack remain before dormant booster arming
    becomes mandatory (0 = must arm immediately).

    max_forward_simulations_per_step caps how many hypothetical boards a
    strategy may evaluate per step (computation budget).
    max_strategic_cells_per_step prevents over-constraining future steps
    by limiting how many "seed" cells the executor pins.
    lookahead_depth controls how many gravity-settle steps ahead strategies
    may chain-simulate (1 = immediate consequences only).
    """

    payout_low_fraction: float
    payout_high_fraction: float
    arming_urgency_horizon: int
    # Fallback max connected component size for terminal dead boards when
    # the archetype signature doesn't specify one
    terminal_dead_default_max_component: int
    # Maximum hypothetical boards a strategy may evaluate per step (computation budget)
    max_forward_simulations_per_step: int
    # Cap on strategic cells per step intent — prevents over-constraining future steps
    max_strategic_cells_per_step: int
    # How many gravity-settle steps ahead strategies may chain-simulate (1 = immediate only)
    lookahead_depth: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.payout_low_fraction <= 1.0):
            raise ConfigValidationError(
                "reasoner.payout_low_fraction", "must be in [0.0, 1.0]"
            )
        if not (0.0 <= self.payout_high_fraction <= 1.0):
            raise ConfigValidationError(
                "reasoner.payout_high_fraction", "must be in [0.0, 1.0]"
            )
        if self.payout_low_fraction >= self.payout_high_fraction:
            raise ConfigValidationError(
                "reasoner",
                "payout_low_fraction must be < payout_high_fraction",
            )
        if self.arming_urgency_horizon < 0:
            raise ConfigValidationError(
                "reasoner.arming_urgency_horizon", "must be >= 0"
            )
        if self.terminal_dead_default_max_component < 1:
            raise ConfigValidationError(
                "reasoner.terminal_dead_default_max_component", "must be >= 1"
            )
        if self.max_forward_simulations_per_step < 1:
            raise ConfigValidationError(
                "reasoner.max_forward_simulations_per_step", "must be >= 1"
            )
        if self.max_strategic_cells_per_step < 1:
            raise ConfigValidationError(
                "reasoner.max_strategic_cells_per_step", "must be >= 1"
            )
        if self.lookahead_depth < 1:
            raise ConfigValidationError(
                "reasoner.lookahead_depth", "must be >= 1"
            )


# ---------------------------------------------------------------------------
# Gravity-Aware WFC
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GravityWfcConfig:
    """Tuning parameters for gravity-aware WFC mechanisms.

    Controls how the WFC board filler suppresses symbols near clusters to
    prevent unintended post-gravity merges. Each suppression multiplier
    scales symbol weights in a spatial zone — lower values = stronger
    suppression. Changing these values trades fill success rate against
    post-gravity cluster prevention strength.
    """

    # Zone 1: multiplier for same-tier symbols at cluster boundary cells —
    # lower values more aggressively prevent same-tier survivors at explosion edges
    cluster_boundary_tier_suppression: float
    # Zone 2: BFS radius (in cells) for the extended neighborhood around clusters
    extended_neighborhood_radius: int
    # Zone 2: multiplier for same-tier symbols in the extended neighborhood —
    # softer than boundary suppression, prevents gradual tier concentration
    extended_neighborhood_suppression: float
    # Zone 3: multiplier for same-tier symbols in columns below the cluster —
    # prevents gravity from stacking same-tier symbols into post-settle merges
    compression_column_suppression: float
    # Zone 4: multiplier for the seed symbol near strategic cell positions —
    # prevents WFC from duplicating the seed symbol at adjacent cells
    strategic_cell_neighbor_suppression: float
    # Absolute floor for spatial weights — prevents any symbol from reaching
    # zero probability, which would make WFC unable to fill the cell
    min_symbol_weight: float

    def __post_init__(self) -> None:
        for field_name in (
            "cluster_boundary_tier_suppression",
            "extended_neighborhood_suppression",
            "compression_column_suppression",
            "strategic_cell_neighbor_suppression",
        ):
            value = getattr(self, field_name)
            if not (0.0 < value <= 1.0):
                raise ConfigValidationError(
                    f"gravity_wfc.{field_name}",
                    f"must be in (0.0, 1.0], got {value}",
                )
        if self.extended_neighborhood_radius < 1:
            raise ConfigValidationError(
                "gravity_wfc.extended_neighborhood_radius", "must be >= 1"
            )
        if self.min_symbol_weight <= 0.0:
            raise ConfigValidationError(
                "gravity_wfc.min_symbol_weight", "must be > 0.0"
            )


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MasterConfig:
    """Top-level configuration aggregating all sub-configs.

    Loaded once at startup from YAML, validated, then injected into every component.
    The output field is optional for backward compatibility with Phases 1-13.
    """

    board: BoardConfig
    gravity: GravityConfig
    grid_multiplier: GridMultiplierConfig
    boosters: BoosterConfig
    paytable: PaytableConfig
    centipayout: CentipayoutConfig
    win_levels: WinLevelConfig
    freespin: FreespinConfig
    wincap: WincapConfig
    solvers: SolverConfig
    diagnostics: DiagnosticsConfig
    population: PopulationConfig
    symbols: SymbolConfig
    anticipation: AnticipationConfig
    # Step 5 thresholds + Step 8 computation caps for step reasoning
    reasoner: ReasonerConfig
    # Phase 14 — optional to preserve backward compatibility with existing tests
    output: OutputConfig | None = None
    # Gravity-aware WFC tuning — optional to preserve backward compatibility
    gravity_wfc: GravityWfcConfig | None = None
