"""Step validator — verifies filled boards against StepIntent expectations.

Two validation levels:
- validate_step: single step — detects clusters, computes payout, identifies
  booster spawns, and builds a StepResult. Raises on hard constraint violations.
- validate_instance: full cascade — checks total payout, terminal board state,
  near-miss requirements, and cascade depth against the archetype signature.

All thresholds from MasterConfig + ArchetypeSignature — zero hardcoded values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.schema import MasterConfig
from ..primitives.board import Board
from ..primitives.cluster_detection import detect_clusters
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.paytable import Paytable
from ..primitives.symbols import SymbolTier, tier_of
from ..step_reasoner.evaluators import SpawnEvaluator, TerminalEvaluator
from ..step_reasoner.intent import StepIntent
from ..step_reasoner.progress import ClusterRecord, ProgressTracker
from ..step_reasoner.results import SpawnRecord, StepResult
from .data_types import GeneratedInstance

if TYPE_CHECKING:
    from ..archetypes.registry import ArchetypeSignature
    from ..primitives.cluster_detection import Cluster
    from ..spatial_solver.data_types import SpatialStep


class StepValidationFailed(Exception):
    """Raised when a step or instance violates hard validation constraints."""


class StepValidator:
    """Validates filled boards against StepIntent expectations.

    Constructs Paytable, SpawnEvaluator, and TerminalEvaluator from config
    at init time — no repeated construction during the cascade loop.
    """

    __slots__ = ("_config", "_paytable", "_spawn_eval", "_terminal_eval")

    def __init__(self, config: MasterConfig) -> None:
        self._config = config
        self._paytable = Paytable(
            config.paytable, config.centipayout, config.win_levels,
        )
        self._spawn_eval = SpawnEvaluator(config.boosters)
        self._terminal_eval = TerminalEvaluator(config)

    def validate_step(
        self,
        board: Board,
        intent: StepIntent,
        progress: ProgressTracker,
        grid_mults: GridMultiplierGrid,
    ) -> StepResult:
        """Validate one filled board and produce a StepResult.

        Detects clusters on the board, computes payout per cluster using
        grid multipliers, identifies booster spawns from cluster sizes, and
        determines the symbol tier for narrative tracking.

        Terminal intents are validated to have zero clusters (dead board).
        Raises StepValidationFailed on hard constraint violations.
        """
        step_index = progress.steps_completed
        detected = detect_clusters(board, self._config)

        # Terminal steps must produce a dead board (zero clusters)
        if intent.is_terminal and detected:
            raise StepValidationFailed(
                f"Step {step_index}: terminal intent but "
                f"{len(detected)} cluster(s) detected on board"
            )

        # Build cluster records with payout from paytable
        cluster_records = self._build_cluster_records(
            detected, grid_mults, step_index,
        )

        # Identify booster spawns from qualifying cluster sizes
        spawn_records = self._build_spawn_records(detected, step_index)

        # Total centipayout for this step
        step_payout = sum(cr.payout for cr in cluster_records)

        # Determine dominant symbol tier from clusters for narrative arc
        symbol_tier = self._derive_symbol_tier(detected)

        return StepResult(
            step_index=step_index,
            clusters=tuple(cluster_records),
            spawns=tuple(spawn_records),
            # Fire records are populated during transition phase, not at validation
            fires=(),
            symbol_tier=symbol_tier,
            step_payout=step_payout,
        )

    def validate_instance(
        self,
        steps: list[StepResult],
        signature: ArchetypeSignature,
        final_board: Board,
        progress: ProgressTracker,
    ) -> GeneratedInstance:
        """Validate the complete cascade sequence against archetype constraints.

        Checks:
        1. Total payout within signature.payout_range
        2. Terminal board is dead (no clusters)
        3. Terminal near-miss requirements (if signature specifies)
        4. Cascade depth within required_cascade_depth range

        Raises StepValidationFailed on failure.
        Returns a GeneratedInstance on success.
        """
        total_centipayout = progress.cumulative_payout
        total_payout = total_centipayout / self._config.centipayout.multiplier

        # Payout within archetype range
        if not signature.payout_range.contains(total_payout):
            raise StepValidationFailed(
                f"Instance payout {total_payout:.4f} outside "
                f"range [{signature.payout_range.min_val}, "
                f"{signature.payout_range.max_val}]"
            )

        # Terminal board must be dead
        if not self._terminal_eval.is_dead(final_board):
            raise StepValidationFailed(
                "Terminal board has surviving clusters — must be dead"
            )

        # Terminal near-miss requirements
        if signature.terminal_near_misses is not None:
            if not self._terminal_eval.satisfies_terminal_near_misses(
                final_board, signature.terminal_near_misses,
            ):
                raise StepValidationFailed(
                    f"Terminal board does not satisfy near-miss requirement: "
                    f"count={signature.terminal_near_misses.count}, "
                    f"tier={signature.terminal_near_misses.symbol_tier}"
                )

        # Only steps with winning clusters are cascade events — terminal dead steps
        # (clusters=()) end the sequence but are not cascades themselves.
        # Matches ProgressTracker.remaining_cascade_steps() semantics where
        # required_cascade_depth counts winning steps, not total steps.
        cascade_depth = sum(1 for s in steps if s.clusters)
        if not signature.required_cascade_depth.contains(cascade_depth):
            raise StepValidationFailed(
                f"Cascade depth {cascade_depth} outside required range "
                f"[{signature.required_cascade_depth.min_val}, "
                f"{signature.required_cascade_depth.max_val}]"
            )

        win_level = self._paytable.get_win_level(total_payout)

        return GeneratedInstance(
            # sim_id assigned by the generator, not the validator — use 0 as placeholder
            sim_id=0,
            archetype_id=signature.id,
            family=signature.family,
            criteria=signature.criteria,
            board=final_board,
            # Cascade instances don't use the single-step SpatialStep format
            spatial_step=_EMPTY_SPATIAL_STEP,
            payout=total_payout,
            centipayout=total_centipayout,
            win_level=win_level,
        )

    # -- Private helpers -------------------------------------------------------

    def _build_cluster_records(
        self,
        clusters: list[Cluster],
        grid_mults: GridMultiplierGrid,
        step_index: int,
    ) -> list[ClusterRecord]:
        """Convert detected clusters into ClusterRecords with computed payouts."""
        records: list[ClusterRecord] = []
        for cluster in clusters:
            payout = self._paytable.compute_cluster_payout(cluster, grid_mults)
            centipayout = self._paytable.to_centipayout(payout)
            records.append(ClusterRecord(
                symbol=cluster.symbol,
                size=cluster.size,
                positions=cluster.positions | cluster.wild_positions,
                step_index=step_index,
                payout=centipayout,
                wild_positions=cluster.wild_positions,
            ))
        return records

    def _build_spawn_records(
        self,
        clusters: list[Cluster],
        step_index: int,
    ) -> list[SpawnRecord]:
        """Identify which clusters qualify for booster spawns."""
        records: list[SpawnRecord] = []
        for cluster_idx, cluster in enumerate(clusters):
            booster_type = self._spawn_eval.booster_for_size(cluster.size)
            if booster_type is not None:
                # Validator identifies the spawn — position resolved during transition
                records.append(SpawnRecord(
                    booster_type=booster_type,
                    # Position determined by BoosterRules during transition, not validation
                    position=_centroid_estimate(cluster),
                    source_cluster_index=cluster_idx,
                    step_index=step_index,
                ))
        return records

    def _derive_symbol_tier(self, clusters: list[Cluster]) -> SymbolTier | None:
        """Determine the dominant symbol tier from detected clusters.

        Returns the tier of the first cluster's symbol. None if no clusters.
        Uses config.symbols to classify the symbol into LOW or HIGH tier.
        """
        if not clusters:
            return None
        first_symbol = clusters[0].symbol
        # Only standard symbols have tier classification
        if first_symbol.name in self._config.symbols.standard:
            try:
                return tier_of(first_symbol, self._config.symbols)
            except ValueError:
                return None
        return None


def _centroid_estimate(cluster: Cluster) -> Position:
    """Approximate cluster centroid as the position closest to the mean.

    Used as a placeholder — the actual centroid is computed by BoosterRules
    during the transition phase with collision resolution.
    """
    from ..primitives.board import Position

    all_positions = list(cluster.positions | cluster.wild_positions)
    avg_reel = sum(p.reel for p in all_positions) / len(all_positions)
    avg_row = sum(p.row for p in all_positions) / len(all_positions)

    # Pick the actual position closest to the average
    return min(
        all_positions,
        key=lambda p: (p.reel - avg_reel) ** 2 + (p.row - avg_row) ** 2,
    )


# Sentinel empty SpatialStep for cascade instances — they don't use single-step format
def _make_empty_spatial_step() -> SpatialStep:
    """Construct an empty SpatialStep placeholder for cascade-generated instances."""
    from ..spatial_solver.data_types import SpatialStep

    return SpatialStep(
        clusters=(),
        near_misses=(),
        scatter_positions=frozenset(),
        boosters=(),
    )


_EMPTY_SPATIAL_STEP = _make_empty_spatial_step()
