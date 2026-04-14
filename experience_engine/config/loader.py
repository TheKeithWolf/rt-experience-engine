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
    DescriptorConfig,
    DiagnosticTarget,
    CurriculumPhase,
    EnvironmentConfig,
    GeneratorConfig,
    PolicyConfig,
    QualityConfig,
    RLArchiveDiagnosticsConfig,
    ReporterConfig,
    RewardConfig,
    TrainingConfig,
    DiagnosticsConfig,
    FreespinConfig,
    GravityConfig,
    GravityWfcConfig,
    GridMultiplierConfig,
    MasterConfig,
    OutputConfig,
    PaytableConfig,
    PaytableEntry,
    PopulationConfig,
    RefillConfig,
    RLArchiveConfig,
    ReasonerConfig,
    SolverConfig,
    SpatialIntelligenceConfig,
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
    gravity_wfc = _build_gravity_wfc(raw.get("gravity_wfc"))
    spatial_intelligence = _build_spatial_intelligence(
        raw.get("spatial_intelligence"),
    )
    rl_archive = _build_rl_archive(raw.get("rl_archive"))
    refill = _build_refill(raw.get("refill"))

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
        gravity_wfc=gravity_wfc,
        spatial_intelligence=spatial_intelligence,
        rl_archive=rl_archive,
        refill=refill,
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
        survivor_affinity_per_cell=float(
            data.get("survivor_affinity_per_cell", 2.0)
        ),
        arm_feasibility_threshold=float(
            _require(data, "arm_feasibility_threshold",
                     "reasoner.arm_feasibility_threshold")
        ),
        arm_feasibility_retry_budget=int(
            _require(data, "arm_feasibility_retry_budget",
                     "reasoner.arm_feasibility_retry_budget")
        ),
    )


def _build_gravity_wfc(data: dict[str, Any] | None) -> GravityWfcConfig | None:
    """Build GravityWfcConfig from YAML data, or None if section is absent.

    Optional section — configs without gravity_wfc get None, which disables
    the gravity-aware WFC mechanisms (baseline fill path used instead).
    """
    if data is None:
        return None
    return GravityWfcConfig(
        cluster_boundary_tier_suppression=float(
            _require(data, "cluster_boundary_tier_suppression",
                     "gravity_wfc.cluster_boundary_tier_suppression")
        ),
        extended_neighborhood_radius=int(
            _require(data, "extended_neighborhood_radius",
                     "gravity_wfc.extended_neighborhood_radius")
        ),
        extended_neighborhood_suppression=float(
            _require(data, "extended_neighborhood_suppression",
                     "gravity_wfc.extended_neighborhood_suppression")
        ),
        compression_column_suppression=float(
            _require(data, "compression_column_suppression",
                     "gravity_wfc.compression_column_suppression")
        ),
        strategic_cell_neighbor_suppression=float(
            _require(data, "strategic_cell_neighbor_suppression",
                     "gravity_wfc.strategic_cell_neighbor_suppression")
        ),
        min_symbol_weight=float(
            _require(data, "min_symbol_weight",
                     "gravity_wfc.min_symbol_weight")
        ),
    )


def _build_refill(data: dict[str, Any] | None) -> RefillConfig | None:
    """Build RefillConfig from YAML data, or None if section is absent.

    Optional section — configs without refill get None, which preserves
    backward compatibility with tests that don't exercise refill strategies.
    """
    if data is None:
        return None
    return RefillConfig(
        adjacency_boost=float(
            _require(data, "adjacency_boost", "refill.adjacency_boost")
        ),
        depth_scale=float(
            _require(data, "depth_scale", "refill.depth_scale")
        ),
        terminal_max_retries=int(
            _require(data, "terminal_max_retries",
                     "refill.terminal_max_retries")
        ),
    )


def _build_spatial_intelligence(
    data: dict[str, Any] | None,
) -> SpatialIntelligenceConfig | None:
    """Build SpatialIntelligenceConfig from YAML data, or None if absent.

    Optional section — configs without spatial_intelligence disable the
    foresight layer and strategies fall back to column-matching heuristics.
    """
    if data is None:
        return None
    prefix = "spatial_intelligence"
    return SpatialIntelligenceConfig(
        influence_sigma_base=float(
            _require(data, "influence_sigma_base",
                     f"{prefix}.influence_sigma_base"),
        ),
        influence_sigma_scale_per_cell=float(
            _require(data, "influence_sigma_scale_per_cell",
                     f"{prefix}.influence_sigma_scale_per_cell"),
        ),
        booster_sigma_multipliers={
            str(k): float(v)
            for k, v in _require(
                data, "booster_sigma_multipliers",
                f"{prefix}.booster_sigma_multipliers",
            ).items()
        },
        reserve_threshold=float(
            _require(data, "reserve_threshold",
                     f"{prefix}.reserve_threshold"),
        ),
        utility_factor_weights={
            str(k): float(v)
            for k, v in _require(
                data, "utility_factor_weights",
                f"{prefix}.utility_factor_weights",
            ).items()
        },
        reserve_suppression_multiplier=float(
            _require(data, "reserve_suppression_multiplier",
                     f"{prefix}.reserve_suppression_multiplier"),
        ),
    )


# ---------------------------------------------------------------------------
# RL Archive — optional MAP-Elites archive sub-configs
# ---------------------------------------------------------------------------


def _build_rl_archive(data: dict[str, Any] | None) -> RLArchiveConfig | None:
    """Build RLArchiveConfig from YAML data, or None if section is absent.

    Optional section — configs without rl_archive get None, which disables
    the MAP-Elites archive system (cascade generator used as fallback).
    """
    if data is None:
        return None

    descriptor = _build_rl_descriptor(data.get("descriptor"))
    quality = _build_rl_quality(data.get("quality"))
    environment = _build_rl_environment(data.get("environment"))
    reward = _build_rl_reward(data.get("reward"))
    policy = _build_rl_policy(data.get("policy"))
    training = _build_rl_training(data.get("training"))
    generator = _build_rl_generator(data.get("generator"))
    rl_diagnostics = _build_rl_diagnostics(data.get("diagnostics"))

    return RLArchiveConfig(
        descriptor=descriptor,
        quality=quality,
        environment=environment,
        reward=reward,
        policy=policy,
        training=training,
        generator=generator,
        diagnostics=rl_diagnostics,
    )


def _build_rl_descriptor(data: dict[str, Any] | None) -> DescriptorConfig | None:
    """Build DescriptorConfig for behavioral descriptor binning."""
    if data is None:
        return None
    return DescriptorConfig(
        spatial_col_bins=int(
            _require(data, "spatial_col_bins", "rl_archive.descriptor.spatial_col_bins")
        ),
        spatial_row_bins=int(
            _require(data, "spatial_row_bins", "rl_archive.descriptor.spatial_row_bins")
        ),
        payout_bins=int(
            _require(data, "payout_bins", "rl_archive.descriptor.payout_bins")
        ),
    )


def _build_rl_quality(data: dict[str, Any] | None) -> QualityConfig | None:
    """Build QualityConfig for MAP-Elites quality scoring weights."""
    if data is None:
        return None
    return QualityConfig(
        payout_centering_weight=float(
            _require(data, "payout_centering_weight",
                     "rl_archive.quality.payout_centering_weight")
        ),
        escalation_weight=float(
            _require(data, "escalation_weight",
                     "rl_archive.quality.escalation_weight")
        ),
        cluster_size_weight=float(
            _require(data, "cluster_size_weight",
                     "rl_archive.quality.cluster_size_weight")
        ),
        productivity_weight=float(
            _require(data, "productivity_weight",
                     "rl_archive.quality.productivity_weight")
        ),
        multiplier_engagement_weight=float(
            _require(data, "multiplier_engagement_weight",
                     "rl_archive.quality.multiplier_engagement_weight")
        ),
    )


def _build_rl_environment(data: dict[str, Any] | None) -> EnvironmentConfig | None:
    """Build EnvironmentConfig for cascade RL training episodes."""
    if data is None:
        return None
    return EnvironmentConfig(
        max_episode_steps=int(
            _require(data, "max_episode_steps",
                     "rl_archive.environment.max_episode_steps")
        ),
        invalid_step_penalty=float(
            _require(data, "invalid_step_penalty",
                     "rl_archive.environment.invalid_step_penalty")
        ),
        completion_bonus=float(
            _require(data, "completion_bonus",
                     "rl_archive.environment.completion_bonus")
        ),
        failure_penalty=float(
            _require(data, "failure_penalty",
                     "rl_archive.environment.failure_penalty")
        ),
        feasibility_weight=float(
            _require(data, "feasibility_weight",
                     "rl_archive.environment.feasibility_weight")
        ),
        progress_weight=float(
            _require(data, "progress_weight",
                     "rl_archive.environment.progress_weight")
        ),
    )


def _build_rl_reward(data: dict[str, Any] | None) -> RewardConfig | None:
    """Build RewardConfig for phase-aware reward shaping."""
    if data is None:
        return None
    return RewardConfig(
        phase_match_reward=float(
            _require(data, "phase_match_reward",
                     "rl_archive.reward.phase_match_reward")
        ),
        cluster_match_reward=float(
            _require(data, "cluster_match_reward",
                     "rl_archive.reward.cluster_match_reward")
        ),
        spawn_match_reward=float(
            _require(data, "spawn_match_reward",
                     "rl_archive.reward.spawn_match_reward")
        ),
        fire_match_reward=float(
            _require(data, "fire_match_reward",
                     "rl_archive.reward.fire_match_reward")
        ),
        wild_behavior_match_reward=float(
            _require(data, "wild_behavior_match_reward",
                     "rl_archive.reward.wild_behavior_match_reward")
        ),
        feasibility_empty_cell_weight=float(
            _require(data, "feasibility_empty_cell_weight",
                     "rl_archive.reward.feasibility_empty_cell_weight")
        ),
        feasibility_adjacency_weight=float(
            _require(data, "feasibility_adjacency_weight",
                     "rl_archive.reward.feasibility_adjacency_weight")
        ),
    )


def _build_rl_policy(data: dict[str, Any] | None) -> PolicyConfig | None:
    """Build PolicyConfig for the cascade policy network architecture."""
    if data is None:
        return None
    return PolicyConfig(
        board_channels=int(
            _require(data, "board_channels", "rl_archive.policy.board_channels")
        ),
        cnn_filters=int(
            _require(data, "cnn_filters", "rl_archive.policy.cnn_filters")
        ),
        cnn_layers=int(
            _require(data, "cnn_layers", "rl_archive.policy.cnn_layers")
        ),
        trunk_hidden=int(
            _require(data, "trunk_hidden", "rl_archive.policy.trunk_hidden")
        ),
        archetype_embedding_dim=int(
            _require(data, "archetype_embedding_dim",
                     "rl_archive.policy.archetype_embedding_dim")
        ),
        phase_embedding_dim=int(
            _require(data, "phase_embedding_dim",
                     "rl_archive.policy.phase_embedding_dim")
        ),
        entropy_coefficient=float(
            _require(data, "entropy_coefficient",
                     "rl_archive.policy.entropy_coefficient")
        ),
    )


def _build_rl_training(data: dict[str, Any] | None) -> TrainingConfig | None:
    """Build TrainingConfig for PPO training hyperparameters and schedule."""
    if data is None:
        return None

    # Build curriculum phases
    curriculum_raw = _require(data, "curriculum", "rl_archive.training.curriculum")
    if not isinstance(curriculum_raw, dict):
        raise ConfigValidationError(
            "rl_archive.training.curriculum", "must be a mapping"
        )
    phases_raw = _require(
        curriculum_raw, "phases", "rl_archive.training.curriculum.phases"
    )
    curriculum_phases = tuple(
        CurriculumPhase(
            episode_threshold=int(p.get("episode_threshold", 0)),
            difficulty_filter=str(p.get("difficulty_filter", "standard")),
        )
        for p in phases_raw
    )

    # Build reporter config
    reporter_raw = _require(data, "reporter", "rl_archive.training.reporter")
    reporter = ReporterConfig(
        completion_rolling_window=int(
            _require(reporter_raw, "completion_rolling_window",
                     "rl_archive.training.reporter.completion_rolling_window")
        ),
        completion_trend_buckets=int(
            _require(reporter_raw, "completion_trend_buckets",
                     "rl_archive.training.reporter.completion_trend_buckets")
        ),
        plateau_warn_threshold=int(
            _require(reporter_raw, "plateau_warn_threshold",
                     "rl_archive.training.reporter.plateau_warn_threshold")
        ),
        condense_above_completion=float(
            _require(reporter_raw, "condense_above_completion",
                     "rl_archive.training.reporter.condense_above_completion")
        ),
        report_every_n_batches=int(
            _require(reporter_raw, "report_every_n_batches",
                     "rl_archive.training.reporter.report_every_n_batches")
        ),
    )

    return TrainingConfig(
        learning_rate=float(
            _require(data, "learning_rate", "rl_archive.training.learning_rate")
        ),
        gamma=float(_require(data, "gamma", "rl_archive.training.gamma")),
        gae_lambda=float(
            _require(data, "gae_lambda", "rl_archive.training.gae_lambda")
        ),
        clip_epsilon=float(
            _require(data, "clip_epsilon", "rl_archive.training.clip_epsilon")
        ),
        epochs_per_batch=int(
            _require(data, "epochs_per_batch",
                     "rl_archive.training.epochs_per_batch")
        ),
        batch_size=int(
            _require(data, "batch_size", "rl_archive.training.batch_size")
        ),
        max_training_episodes=int(
            _require(data, "max_training_episodes",
                     "rl_archive.training.max_training_episodes")
        ),
        checkpoint_interval=int(
            _require(data, "checkpoint_interval",
                     "rl_archive.training.checkpoint_interval")
        ),
        imitation_epochs=int(
            _require(data, "imitation_epochs",
                     "rl_archive.training.imitation_epochs")
        ),
        imitation_batch_size=int(
            _require(data, "imitation_batch_size",
                     "rl_archive.training.imitation_batch_size")
        ),
        curriculum=curriculum_phases,
        reporter=reporter,
    )


def _build_rl_generator(data: dict[str, Any] | None) -> GeneratorConfig | None:
    """Build GeneratorConfig for the RL archive production generator."""
    if data is None:
        return None
    return GeneratorConfig(
        archive_dir=str(
            _require(data, "archive_dir", "rl_archive.generator.archive_dir")
        ),
        min_coverage_warn=float(
            _require(data, "min_coverage_warn",
                     "rl_archive.generator.min_coverage_warn")
        ),
    )


def _build_rl_diagnostics(
    data: dict[str, Any] | None,
) -> RLArchiveDiagnosticsConfig | None:
    """Build RLArchiveDiagnosticsConfig for archive health reporting."""
    if data is None:
        return None
    return RLArchiveDiagnosticsConfig(
        coverage_warn_threshold=float(
            _require(data, "coverage_warn_threshold",
                     "rl_archive.diagnostics.coverage_warn_threshold")
        ),
        coverage_fail_threshold=float(
            _require(data, "coverage_fail_threshold",
                     "rl_archive.diagnostics.coverage_fail_threshold")
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
