"""CLI entry point for RL archive training.

Trains a cascade policy via PPO and populates a MAP-Elites archive.
Reuses _build_full_registry and _build_pipeline from run.py (second caller,
rule of three not reached — no extraction yet).

Usage:
    python -m games.royal_tumble.experience_engine.train --archetype wild_bridge_small --seed 42
    python -m games.royal_tumble.experience_engine.train --family wild --seed 42 --episodes 1000
    python -m games.royal_tumble.experience_engine.train --resume library/archives/checkpoints/...
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:
    print("PyTorch is required for RL archive training. Install with: pip install torch")
    sys.exit(1)

from .config.loader import load_config
from .config.schema import MasterConfig, TrainingConfig

# Default config path — same as run.py
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RL archive for cascade instance generation",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML (default: experience_engine/config/default.yaml)",
    )
    parser.add_argument(
        "--archetype", type=str, default=None,
        help="Train a specific archetype by ID",
    )
    parser.add_argument(
        "--family", type=str, default=None,
        help="Train all archetypes in a family",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override max_training_episodes from config",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from a checkpoint file",
    )
    return parser.parse_args(argv)


def _build_training_config(
    base: TrainingConfig, episodes_override: int | None,
) -> TrainingConfig:
    """Apply CLI overrides to the training config."""
    if episodes_override is None:
        return base
    return dataclasses.replace(base, max_training_episodes=episodes_override)


def _resolve_target_archetypes(
    args: argparse.Namespace,
    registry,
) -> list[str]:
    """Determine which archetypes to train based on CLI args."""
    if args.archetype:
        aid = args.archetype
        if aid not in registry.all_ids():
            print(f"Error: archetype '{aid}' not found in registry")
            print(f"Available: {sorted(registry.all_ids())}")
            sys.exit(1)
        return [aid]

    if args.family:
        family_sigs = registry.get_family(args.family)
        if not family_sigs:
            print(f"Error: family '{args.family}' not found or empty")
            sys.exit(1)
        # Only train cascade archetypes (depth >= 2)
        return [
            sig.id for sig in family_sigs
            if sig.required_cascade_depth.max_val >= 2
        ]

    # Default: all cascade archetypes with depth >= 2
    return [
        aid for aid in sorted(registry.all_ids())
        if registry.get(aid).required_cascade_depth.max_val >= 2
    ]


def main(argv: list[str] | None = None) -> int:
    """Train RL archive policy and populate MAP-Elites archive."""
    args = _parse_args(argv)
    config = load_config(args.config)

    if config.rl_archive is None:
        print("Error: rl_archive section missing from config")
        return 1

    rl_config = config.rl_archive
    if rl_config.training is None:
        print("Error: rl_archive.training section missing from config")
        return 1
    if rl_config.environment is None:
        print("Error: rl_archive.environment section missing from config")
        return 1
    if rl_config.policy is None:
        print("Error: rl_archive.policy section missing from config")
        return 1
    if rl_config.descriptor is None:
        print("Error: rl_archive.descriptor section missing from config")
        return 1
    if rl_config.quality is None:
        print("Error: rl_archive.quality section missing from config")
        return 1
    if rl_config.reward is None:
        print("Error: rl_archive.reward section missing from config")
        return 1

    # Build registry and pipeline — reuses run.py patterns
    from .run import _build_full_registry, _build_pipeline

    print(f"Loading config from: {args.config}")
    registry = _build_full_registry(config)
    static_gen, cascade_gen, validator = _build_pipeline(config, registry)

    # Resolve target archetypes
    targets = _resolve_target_archetypes(args, registry)
    if not targets:
        print("No eligible archetypes found (need cascade_depth >= 2)")
        return 1

    print(f"Training targets: {targets}")
    print(f"Seed: {args.seed}")

    # Apply episode override
    training_config = _build_training_config(rl_config.training, args.episodes)
    print(f"Max episodes: {training_config.max_training_episodes}")

    # Build shared components
    from .primitives.booster_rules import BoosterRules
    from .primitives.gravity import GravityDAG
    from .step_reasoner.evaluators import PayoutEstimator, SpawnEvaluator
    from .step_reasoner.services.cluster_builder import ClusterBuilder
    from .step_reasoner.services.boundary_analyzer import BoundaryAnalyzer
    from .pipeline.step_executor import StepExecutor
    from .pipeline.step_validator import StepValidator
    from .pipeline.simulator import StepTransitionSimulator

    gravity_dag = GravityDAG(config.board, config.gravity)
    spawn_eval = SpawnEvaluator(config.boosters)
    payout_eval = PayoutEstimator(
        config.paytable, config.centipayout, config.win_levels,
        config.symbols, config.grid_multiplier,
    )
    boundary_analyzer = BoundaryAnalyzer(config.board, config.symbols)
    cluster_builder = ClusterBuilder(
        spawn_eval, payout_eval, config.board, config.symbols,
        boundary_analyzer, centipayout_multiplier=config.centipayout.multiplier,
        max_seed_retries=config.solvers.max_seed_retries,
    )
    booster_rules = BoosterRules(config.boosters, config.board, config.symbols)
    executor = StepExecutor(config, gravity_dag=gravity_dag)
    step_validator = StepValidator(config)
    simulator = StepTransitionSimulator(gravity_dag, config)

    # Build RL components
    from .rl_archive.observation import ObservationBuilder, ObservationEncoder
    from .rl_archive.action_space import ActionInterpreter
    from .rl_archive.environment import CascadeEnvironment
    from .rl_archive.reward import PhaseRewardComputer
    from .rl_archive.archive import MAPElitesArchive
    from .rl_archive.descriptor import CascadeDescriptorExtractor
    from .rl_archive.quality import CascadeQualityScorer
    from .rl_archive.policy.network import CascadePolicy
    from .rl_archive.training.curriculum import CurriculumScheduler
    from .rl_archive.training.reporter import (
        ConsoleTrainingReporter, TrainingStartSummary,
    )
    from .rl_archive.training.ppo import PPOTrainer

    obs_builder = ObservationBuilder(config.symbols, config.board)
    obs_encoder = ObservationEncoder(config.symbols, config.board)
    action_interpreter = ActionInterpreter(config.symbols, config.board, cluster_builder)
    reward_computer = PhaseRewardComputer(
        rl_config.reward, rl_config.environment, config.board,
    )
    descriptor_extractor = CascadeDescriptorExtractor(
        booster_rules, config.board, rl_config.descriptor,
    )
    quality_scorer = CascadeQualityScorer(
        rl_config.quality, config.board, config.grid_multiplier,
    )
    curriculum = CurriculumScheduler(training_config.curriculum, registry)
    reporter = ConsoleTrainingReporter(training_config.reporter)

    # Compute policy dimensions from registry
    num_symbols = len(config.symbols.standard)
    num_archetypes = len(registry.all_ids())
    max_phases = max(
        (len(registry.get(aid).narrative_arc.phases)
         for aid in registry.all_ids()
         if registry.get(aid).narrative_arc is not None),
        default=1,
    )

    # Train each target archetype
    for archetype_id in targets:
        sig = registry.get(archetype_id)
        if sig.narrative_arc is None:
            print(f"Skipping {archetype_id}: no narrative arc defined")
            continue

        print(f"\n{'='*60}")
        print(f"  Training archetype: {archetype_id}")
        print(f"{'='*60}")

        # Build per-archetype components
        archive = MAPElitesArchive(rl_config.descriptor, num_symbols)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed)

        policy = CascadePolicy(
            board_config=config.board,
            num_symbols=num_symbols,
            policy_config=rl_config.policy,
            num_archetypes=num_archetypes,
            max_phases=max_phases,
            obs_encoder=obs_encoder,
        ).to(device)

        param_count = sum(p.numel() for p in policy.parameters())

        env = CascadeEnvironment(
            config=config,
            arc=sig.narrative_arc,
            signature=sig,
            executor=executor,
            validator=step_validator,
            simulator=simulator,
            gravity_dag=gravity_dag,
            obs_builder=obs_builder,
            action_interpreter=action_interpreter,
            env_config=rl_config.environment,
        )

        # Report training start
        reporter.on_training_start(TrainingStartSummary(
            archetype_id=archetype_id,
            seed=args.seed,
            max_episodes=training_config.max_training_episodes,
            batch_size=training_config.batch_size,
            curriculum_phase=curriculum.current_phase_name(0),
            policy_param_count=param_count,
            archive_total_niches=archive.total_cells(),
            archive_descriptor_axes={
                "spatial_col_bins": rl_config.descriptor.spatial_col_bins,
                "spatial_row_bins": rl_config.descriptor.spatial_row_bins,
                "payout_bins": rl_config.descriptor.payout_bins,
                "symbols": num_symbols,
                "orientations": 3,
            },
            device=str(device),
        ))

        # Run PPO training
        trainer = PPOTrainer(
            policy=policy,
            environment=env,
            archive=archive,
            config=config,
            training_config=training_config,
            curriculum=curriculum,
            reporter=reporter,
            reward_computer=reward_computer,
            descriptor_extractor=descriptor_extractor,
            quality_scorer=quality_scorer,
            device=device,
        )

        rng = random.Random(args.seed)
        result = trainer.train(rng)

        # Save final archive
        from .rl_archive.archive_io import save_archive
        archive_dir = Path(
            rl_config.generator.archive_dir
            if rl_config.generator
            else "games/royal_tumble/experience_engine/library/archives"
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{archetype_id}.jsonl"
        save_archive(archive, archive_path)

        print(f"\n  Result: {result.total_episodes} episodes, "
              f"{result.archive_filled}/{result.archive_total} niches filled "
              f"({result.final_completion_rate:.1%} completion)")
        print(f"  Archive saved: {archive_path}")

    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
