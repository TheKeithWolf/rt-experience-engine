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
    # Cluster size (inclusive) at which BFS growth switches from single-seed
    # to multi-seed. Derived from the minimum bomb-spawn threshold in
    # boosters.spawn_thresholds — clusters at or above this size are large
    # enough to exhaust a single BFS frontier on fragmented post-gravity
    # boards, so multi-seed BFS is used to widen the initial frontier.
    multi_seed_threshold: int = 11
    # Number of initial seeds when multi-seed BFS is engaged. 3 balances
    # growth-frontier width against risk of overlapping initial components.
    multi_seed_count: int = 3

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
        if self.multi_seed_threshold < 2:
            raise ConfigValidationError(
                "solvers.multi_seed_threshold", "must be >= 2"
            )
        if self.multi_seed_count < 1:
            raise ConfigValidationError("solvers.multi_seed_count", "must be >= 1")


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
# Trajectory Planner (Tier 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    """Tier-2 trajectory planner thresholds.

    max_sketch_retries — how many sketch attempts the CascadeInstanceGenerator
      makes per generation before falling back to unguided (Tier 3). A single
      miss is common when variance draws an incompatible cluster size; the
      budget gives the planner another roll before giving up.
    waypoint_feasibility_threshold — per-waypoint landing score floor. Below
      this, the planner marks the sketch infeasible and exits early. Lower
      values accept riskier plans; higher values demand tight alignment at
      every step.
    sketch_feasibility_threshold — composite (product of waypoint scores)
      floor the whole sketch must clear. Independent from per-waypoint so a
      single marginal step can still pass if later steps are strong.
    """

    max_sketch_retries: int
    waypoint_feasibility_threshold: float
    sketch_feasibility_threshold: float

    def __post_init__(self) -> None:
        if self.max_sketch_retries < 1:
            raise ConfigValidationError(
                "reasoner.trajectory.max_sketch_retries", "must be >= 1"
            )
        if not (0.0 < self.waypoint_feasibility_threshold <= 1.0):
            raise ConfigValidationError(
                "reasoner.trajectory.waypoint_feasibility_threshold",
                "must be in (0.0, 1.0]",
            )
        if not (0.0 < self.sketch_feasibility_threshold <= 1.0):
            raise ConfigValidationError(
                "reasoner.trajectory.sketch_feasibility_threshold",
                "must be in (0.0, 1.0]",
            )


# ---------------------------------------------------------------------------
# Spatial Atlas (Tier 1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AtlasDepthBand:
    """One named row-range bucket used for column profile indexing.

    The atlas keys profiles by (counts, depth_band_name). Depth bands coarsen
    the row dimension so the key space stays tractable while preserving the
    gravity differences between top-of-column and bottom-of-column explosions.
    """

    name: str
    min_row: int
    max_row: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("atlas.depth_bands[].name", "must not be empty")
        if self.min_row < 0:
            raise ConfigValidationError(
                f"atlas.depth_bands[{self.name}].min_row", "must be >= 0"
            )
        if self.max_row < self.min_row:
            raise ConfigValidationError(
                f"atlas.depth_bands[{self.name}]",
                "max_row must be >= min_row",
            )


@dataclass(frozen=True, slots=True)
class AtlasConfig:
    """Spatial atlas (Tier 1) configuration.

    enabled — atlas query is skipped when False (falls straight to Tier 2).
    path — file location of the serialized atlas, relative to the config dir.
    depth_bands — tuple in row-order; AtlasBuilder assigns each simulated
      settle to the band whose [min_row, max_row] contains its centroid row.
    region_falloff_per_column — multiplicative penalty per column of distance
      applied by RegionPreferenceFactor to cells outside the atlas region.
      Keeps the constraint soft (placement still possible outside) while
      biasing cluster shape toward the pre-validated columns.
    min_composite_score — atlas configurations below this score are rejected
      at query time; forces a fall-through to the trajectory planner when
      the atlas can only produce low-confidence guidance.
    """

    enabled: bool
    path: str
    depth_bands: tuple[AtlasDepthBand, ...]
    region_falloff_per_column: float
    min_composite_score: float

    def __post_init__(self) -> None:
        if not self.depth_bands:
            raise ConfigValidationError("atlas.depth_bands", "must not be empty")
        if not (0.0 < self.region_falloff_per_column <= 1.0):
            raise ConfigValidationError(
                "atlas.region_falloff_per_column", "must be in (0.0, 1.0]"
            )
        if not (0.0 < self.min_composite_score <= 1.0):
            raise ConfigValidationError(
                "atlas.min_composite_score", "must be in (0.0, 1.0]"
            )
        names = [band.name for band in self.depth_bands]
        if len(set(names)) != len(names):
            raise ConfigValidationError(
                "atlas.depth_bands", "band names must be unique"
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
    # Per-survivor-cell weight bonus during booster arm symbol selection —
    # counteracts merge-safety penalty for symbols with adjacent survivors
    survivor_affinity_per_cell: float
    # Minimum acceptable landing score for a spawn-cluster candidate. Below
    # this, BoosterSetupStrategy retries with reshape bias to find a landing
    # whose refill zone can fit the next step's arming cluster.
    arm_feasibility_threshold: float
    # Maximum reshape attempts after the first cluster placement fails the
    # arm_feasibility_threshold. Each attempt re-rolls via compute_reshape_bias.
    arm_feasibility_retry_budget: int
    # Tier-2 trajectory planner tuning. None disables the planner fallback;
    # the atlas (Tier 1) and unguided generation (Tier 3) remain available.
    trajectory: TrajectoryConfig | None = None

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
        if self.survivor_affinity_per_cell < 0.0:
            raise ConfigValidationError(
                "reasoner.survivor_affinity_per_cell", "must be >= 0.0"
            )
        if not (0.0 <= self.arm_feasibility_threshold <= 1.0):
            raise ConfigValidationError(
                "reasoner.arm_feasibility_threshold", "must be in [0.0, 1.0]"
            )
        if self.arm_feasibility_retry_budget < 0:
            raise ConfigValidationError(
                "reasoner.arm_feasibility_retry_budget", "must be >= 0"
            )


# ---------------------------------------------------------------------------
# Refill Strategies
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RefillConfig:
    """Tuning parameters for post-gravity refill strategies.

    adjacency_boost — per-neighbor weight multiplier for cluster-seeking
    refill.  Higher values more aggressively extend existing formations.

    depth_scale — multiplier applied to a neighbor's row index when
    computing depth-weighted adjacency score.  Higher values bias cluster
    formation toward the bottom of the board.

    terminal_max_retries — per-cell retry budget for the terminal refill
    strategy when a candidate symbol would create a cluster.
    """

    adjacency_boost: float
    depth_scale: float
    terminal_max_retries: int

    def __post_init__(self) -> None:
        if self.adjacency_boost <= 0.0:
            raise ConfigValidationError(
                "refill.adjacency_boost", "must be > 0.0"
            )
        if self.depth_scale < 0.0:
            raise ConfigValidationError(
                "refill.depth_scale", "must be >= 0.0"
            )
        if self.terminal_max_retries < 1:
            raise ConfigValidationError(
                "refill.terminal_max_retries", "must be >= 1"
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
# Spatial Intelligence — foresight for cascade seed placement
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SpatialIntelligenceConfig:
    """Tuning parameters for the spatial intelligence layer.

    Controls Gaussian influence falloff, gravity field weighting, and
    multi-objective utility scoring for cascade seed placement. Gives
    the solver foresight about where future steps need space, replacing
    retry-luck with informed first-attempt placement.
    """

    # Gaussian influence — sigma = base + cluster_size × scale_per_cell
    influence_sigma_base: float
    influence_sigma_scale_per_cell: float

    # Per-booster sigma multipliers — keyed by booster name from spawn_thresholds.
    # Wider zones for boosters whose next-step mechanics need more adjacent space.
    # e.g. W bridge needs adjacent refill across multiple columns (1.35×),
    # B blast radius extends effective zone (1.2×), R is column-linear (1.0×).
    booster_sigma_multipliers: dict[str, float]

    # Influence threshold below which a cell is NOT in the reserve zone.
    # Cells above this threshold get WFC suppression for the current step.
    reserve_threshold: float

    # Utility factor weights — keyed by factor name for registry lookup.
    # Positive weights are additive, negative weights are subtractive (e.g. merge_risk).
    utility_factor_weights: dict[str, float]

    # WFC suppression multiplier for cells inside the reserve zone.
    # Applied alongside existing suppression zones in SpatialWeightMap.
    reserve_suppression_multiplier: float

    def __post_init__(self) -> None:
        if self.influence_sigma_base <= 0:
            raise ConfigValidationError(
                "spatial_intelligence.influence_sigma_base", "must be > 0",
            )
        if self.influence_sigma_scale_per_cell <= 0:
            raise ConfigValidationError(
                "spatial_intelligence.influence_sigma_scale_per_cell",
                "must be > 0",
            )
        if not (0.0 < self.reserve_threshold < 1.0):
            raise ConfigValidationError(
                "spatial_intelligence.reserve_threshold",
                "must be in (0, 1)",
            )
        for name, mult in self.booster_sigma_multipliers.items():
            if mult <= 0:
                raise ConfigValidationError(
                    f"spatial_intelligence.booster_sigma_multipliers.{name}",
                    "must be > 0",
                )
        if not self.utility_factor_weights:
            raise ConfigValidationError(
                "spatial_intelligence.utility_factor_weights",
                "must not be empty",
            )
        if not (0.0 < self.reserve_suppression_multiplier <= 1.0):
            raise ConfigValidationError(
                "spatial_intelligence.reserve_suppression_multiplier",
                "must be in (0, 1]",
            )


# ---------------------------------------------------------------------------
# RL Archive — sub-configs for MAP-Elites archive system
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DescriptorConfig:
    """Behavioral descriptor binning for MAP-Elites archive cells.

    Controls how cascade trajectories are discretized into behavioral niches.
    Spatial bins divide the board into regions; payout bins divide the arc's
    payout range into equal-width buckets.
    """

    spatial_col_bins: int
    spatial_row_bins: int
    payout_bins: int

    def __post_init__(self) -> None:
        if self.spatial_col_bins < 1:
            raise ConfigValidationError(
                "rl_archive.descriptor.spatial_col_bins", "must be >= 1"
            )
        if self.spatial_row_bins < 1:
            raise ConfigValidationError(
                "rl_archive.descriptor.spatial_row_bins", "must be >= 1"
            )
        if self.payout_bins < 1:
            raise ConfigValidationError(
                "rl_archive.descriptor.payout_bins", "must be >= 1"
            )


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Weights for multi-component quality scoring in the MAP-Elites archive.

    Each weight controls how much that quality dimension contributes to the
    overall score. Setting a weight to 0.0 disables that component.
    Negative weights are forbidden.
    """

    payout_centering_weight: float
    escalation_weight: float
    cluster_size_weight: float
    productivity_weight: float
    multiplier_engagement_weight: float

    def __post_init__(self) -> None:
        for field_name in (
            "payout_centering_weight",
            "escalation_weight",
            "cluster_size_weight",
            "productivity_weight",
            "multiplier_engagement_weight",
        ):
            if getattr(self, field_name) < 0.0:
                raise ConfigValidationError(
                    f"rl_archive.quality.{field_name}", "must be >= 0.0"
                )


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Cascade environment parameters for RL training episodes.

    Controls episode length, reward shaping for invalid/terminal steps,
    and how feasibility and progress contribute to the per-step reward.
    """

    max_episode_steps: int
    invalid_step_penalty: float
    completion_bonus: float
    failure_penalty: float
    feasibility_weight: float
    progress_weight: float

    def __post_init__(self) -> None:
        if self.max_episode_steps < 1:
            raise ConfigValidationError(
                "rl_archive.environment.max_episode_steps", "must be >= 1"
            )
        if not (0.0 <= self.feasibility_weight <= 1.0):
            raise ConfigValidationError(
                "rl_archive.environment.feasibility_weight",
                "must be in [0.0, 1.0]",
            )
        if not (0.0 <= self.progress_weight <= 1.0):
            raise ConfigValidationError(
                "rl_archive.environment.progress_weight",
                "must be in [0.0, 1.0]",
            )


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for the phase-aware reward function.

    Each weight controls how much that reward component contributes.
    Field-iteration reward: the reward computer iterates over phase constraint
    fields and adds the corresponding weight when the step result matches.
    """

    phase_match_reward: float
    cluster_match_reward: float
    spawn_match_reward: float
    fire_match_reward: float
    wild_behavior_match_reward: float
    feasibility_empty_cell_weight: float
    feasibility_adjacency_weight: float

    def __post_init__(self) -> None:
        for field_name in (
            "feasibility_empty_cell_weight",
            "feasibility_adjacency_weight",
        ):
            val = getattr(self, field_name)
            if not (0.0 <= val <= 1.0):
                raise ConfigValidationError(
                    f"rl_archive.reward.{field_name}",
                    f"must be in [0.0, 1.0], got {val}",
                )


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Neural network architecture parameters for the cascade policy.

    All spatial dimensions are derived from BoardConfig at construction
    time — this config controls layer counts, widths, and embedding sizes.
    """

    # Extra non-symbol board channels (multiplier, empty, wild, booster, etc.)
    board_channels: int
    cnn_filters: int
    cnn_layers: int
    trunk_hidden: int
    archetype_embedding_dim: int
    phase_embedding_dim: int
    entropy_coefficient: float

    def __post_init__(self) -> None:
        if self.board_channels < 1:
            raise ConfigValidationError(
                "rl_archive.policy.board_channels", "must be >= 1"
            )
        if self.cnn_filters < 1:
            raise ConfigValidationError(
                "rl_archive.policy.cnn_filters", "must be >= 1"
            )
        if self.cnn_layers < 1:
            raise ConfigValidationError(
                "rl_archive.policy.cnn_layers", "must be >= 1"
            )
        if self.trunk_hidden < 1:
            raise ConfigValidationError(
                "rl_archive.policy.trunk_hidden", "must be >= 1"
            )
        if self.entropy_coefficient < 0.0:
            raise ConfigValidationError(
                "rl_archive.policy.entropy_coefficient", "must be >= 0.0"
            )


@dataclass(frozen=True, slots=True)
class CurriculumPhase:
    """One phase in the training curriculum schedule.

    Difficulty filters control which archetypes are eligible:
    "standard" = cascade_depth 2-3, "hard" = depth >= 4, "all" = depth >= 2.
    """

    episode_threshold: int
    difficulty_filter: str

    def __post_init__(self) -> None:
        if self.episode_threshold < 0:
            raise ConfigValidationError(
                "rl_archive.training.curriculum.episode_threshold",
                "must be >= 0",
            )
        if self.difficulty_filter not in ("standard", "hard", "all"):
            raise ConfigValidationError(
                "rl_archive.training.curriculum.difficulty_filter",
                f"must be 'standard', 'hard', or 'all', got '{self.difficulty_filter}'",
            )


@dataclass(frozen=True, slots=True)
class ReporterConfig:
    """Console reporter tuning parameters for training progress display."""

    completion_rolling_window: int
    completion_trend_buckets: int
    plateau_warn_threshold: int
    condense_above_completion: float
    report_every_n_batches: int

    def __post_init__(self) -> None:
        if self.completion_rolling_window < 1:
            raise ConfigValidationError(
                "rl_archive.training.reporter.completion_rolling_window",
                "must be >= 1",
            )
        if self.report_every_n_batches < 1:
            raise ConfigValidationError(
                "rl_archive.training.reporter.report_every_n_batches",
                "must be >= 1",
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """PPO training hyperparameters and schedule.

    Controls learning rate, GAE parameters, batch sizing, curriculum
    scheduling, and reporter verbosity.
    """

    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    epochs_per_batch: int
    batch_size: int
    max_training_episodes: int
    checkpoint_interval: int
    imitation_epochs: int
    imitation_batch_size: int
    curriculum: tuple[CurriculumPhase, ...]
    reporter: ReporterConfig

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ConfigValidationError(
                "rl_archive.training.learning_rate", "must be > 0.0"
            )
        if not (0.0 <= self.gamma <= 1.0):
            raise ConfigValidationError(
                "rl_archive.training.gamma", "must be in [0.0, 1.0]"
            )
        if self.batch_size < 1:
            raise ConfigValidationError(
                "rl_archive.training.batch_size", "must be >= 1"
            )
        if self.max_training_episodes < 1:
            raise ConfigValidationError(
                "rl_archive.training.max_training_episodes", "must be >= 1"
            )


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Configuration for the RL archive production generator.

    Controls where trained archives are stored and the minimum coverage
    level before a warning is emitted during population generation.
    """

    archive_dir: str
    min_coverage_warn: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_coverage_warn <= 1.0):
            raise ConfigValidationError(
                "rl_archive.generator.min_coverage_warn",
                "must be in [0.0, 1.0]",
            )


@dataclass(frozen=True, slots=True)
class RLArchiveDiagnosticsConfig:
    """Thresholds for archive health reporting.

    Coverage below warn_threshold → "warn" status.
    Coverage below fail_threshold → "fail" status.
    """

    coverage_warn_threshold: float
    coverage_fail_threshold: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.coverage_warn_threshold <= 1.0):
            raise ConfigValidationError(
                "rl_archive.diagnostics.coverage_warn_threshold",
                "must be in [0.0, 1.0]",
            )
        if not (0.0 <= self.coverage_fail_threshold <= 1.0):
            raise ConfigValidationError(
                "rl_archive.diagnostics.coverage_fail_threshold",
                "must be in [0.0, 1.0]",
            )
        if self.coverage_fail_threshold > self.coverage_warn_threshold:
            raise ConfigValidationError(
                "rl_archive.diagnostics",
                "coverage_fail_threshold must be <= coverage_warn_threshold",
            )


@dataclass(frozen=True, slots=True)
class RLArchiveConfig:
    """Top-level RL archive configuration aggregating all sub-configs.

    Each sub-config is optional (None) until its implementation phase lands.
    The entire rl_archive section is optional in default.yaml — omitting it
    preserves full backward compatibility.
    """

    descriptor: DescriptorConfig | None = None
    quality: QualityConfig | None = None
    environment: EnvironmentConfig | None = None
    reward: RewardConfig | None = None
    policy: PolicyConfig | None = None
    training: TrainingConfig | None = None
    generator: GeneratorConfig | None = None
    diagnostics: RLArchiveDiagnosticsConfig | None = None


# ---------------------------------------------------------------------------
# Reel Strip
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReelStripConfig:
    """Points at the CSV that backs the `reel` archetype family.

    Optional top-level section — absent when no reel archetypes are
    registered. The CSV itself is parsed and validated by
    `primitives.reel_strip.load_reel_strip` against `BoardConfig`.
    """

    csv_path: str  # Path relative to repo root (or absolute)

    def __post_init__(self) -> None:
        if not self.csv_path:
            raise ConfigValidationError("reel_strip.csv_path", "must not be empty")


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
    # Spatial intelligence — foresight for cascade seed placement
    spatial_intelligence: SpatialIntelligenceConfig | None = None
    # RL archive — optional MAP-Elites archive system for deep cascade generation
    rl_archive: RLArchiveConfig | None = None
    # Refill strategy tuning — optional to preserve backward compatibility
    refill: RefillConfig | None = None
    # Reel strip — optional; present when the `reel` archetype family is in use
    reel_strip: ReelStripConfig | None = None
    # Spatial atlas (Tier 1) — offline pre-validated per-arc region guidance.
    # Optional: when absent or disabled the generator skips the atlas tier and
    # uses only the trajectory planner (Tier 2) and unguided fallback (Tier 3).
    atlas: AtlasConfig | None = None
