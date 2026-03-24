"""Diagnostics engine — computes population-level metrics and compares to targets.

All metric computations are pure functions of the collected InstanceMetrics.
Targets from config.diagnostics.targets. Uses dict dispatch for metric computers.

All thresholds from MasterConfig — zero hardcoded values.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..config.schema import MasterConfig
from ..primitives.board import Position
from ..validation.metrics import InstanceMetrics


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """Result for a single diagnostic metric."""

    metric: str
    value: float
    target_min: float | None
    target_max: float | None
    status: str  # "pass", "warn", "fail"


@dataclass(frozen=True, slots=True)
class DiagnosticsReport:
    """Complete diagnostics report for a population run."""

    results: tuple[DiagnosticResult, ...]
    spatial_heatmap: dict[Position, int]
    cluster_size_distribution: dict[int, int]
    symbol_win_contribution: dict[str, float]
    payout_percentiles: dict[str, float]
    archetype_distribution: dict[str, int]
    win_level_distribution: dict[int, int]
    failure_rates: dict[str, int]
    # Cascade-specific: count of instances per tumble depth (0, 1, 2, 3, ...)
    tumble_depth_distribution: dict[int, int]
    # Cascade-specific: histogram of final grid multiplier values across all positions
    grid_multiplier_distribution: dict[int, int]
    # Booster-specific (Phase 9+): per-type rates and chain metrics
    booster_spawn_rate: dict[str, float]
    booster_fire_rate: dict[str, float]
    rocket_orientation_balance: dict[str, float]
    chain_trigger_rate: float
    # LB/SLB targeting distributions — which symbols were targeted and how often
    lb_target_distribution: dict[str, int]
    slb_target_distribution: dict[str, int]
    # Fraction of population that triggers freespin
    freespin_trigger_rate: float
    # Fraction of population that reaches the win cap
    wincap_hit_rate: float


class DiagnosticsEngine:
    """Computes population-level metrics and compares against config targets.

    Metric computation functions are dispatched via dict for extensibility —
    adding a new metric requires only adding a function and a dict entry.
    """

    __slots__ = ("_config", "_targets")

    def __init__(self, config: MasterConfig) -> None:
        self._config = config
        # Build target lookup: metric_name → (min, max)
        self._targets: dict[str, tuple[float | None, float | None]] = {
            t.metric: (t.min_value, t.max_value)
            for t in config.diagnostics.targets
        }

    def analyze(
        self,
        metrics: tuple[InstanceMetrics, ...],
        failure_log: tuple[tuple[str, int, str], ...] = (),
    ) -> DiagnosticsReport:
        """Compute all diagnostics from collected instance metrics."""
        if not metrics:
            return DiagnosticsReport(
                results=(),
                spatial_heatmap={},
                cluster_size_distribution={},
                symbol_win_contribution={},
                payout_percentiles={},
                archetype_distribution={},
                win_level_distribution={},
                failure_rates={},
                tumble_depth_distribution={},
                grid_multiplier_distribution={},
                booster_spawn_rate={},
                booster_fire_rate={},
                rocket_orientation_balance={},
                chain_trigger_rate=0.0,
                lb_target_distribution={},
                slb_target_distribution={},
                freespin_trigger_rate=0.0,
                wincap_hit_rate=0.0,
            )

        n = len(metrics)

        # Compute scalar metrics
        rtp_total = sum(m.payout for m in metrics) / n
        hit_rate = sum(1 for m in metrics if m.payout > 0) / n
        dead_spin_rate = sum(1 for m in metrics if m.payout == 0.0) / n

        # Spatial heatmap — which positions participate in wins
        spatial_heatmap: dict[Position, int] = defaultdict(int)
        # Not directly available from InstanceMetrics, so we track cluster count
        # per archetype as a proxy. Full heatmap requires board data.

        # Spatial coefficient of variation from cluster counts per archetype
        spatial_cv = self._compute_spatial_cv(metrics)

        # Near-miss + scatter co-occurrence rate
        nm_scatter_count = sum(
            1 for m in metrics
            if m.near_miss_count > 0 and m.scatter_count > 0
        )
        near_miss_scatter_rate = nm_scatter_count / n

        # Average scatters per instance
        scatter_landing_frequency = sum(m.scatter_count for m in metrics) / n

        # Distribution metrics
        cluster_size_distribution: dict[int, int] = Counter()
        for m in metrics:
            for size in m.cluster_sizes:
                cluster_size_distribution[size] += 1

        symbol_win_contribution = self._compute_symbol_contribution(metrics)

        payout_percentiles = self._compute_payout_percentiles(metrics)

        archetype_distribution: dict[str, int] = Counter(
            m.archetype_id for m in metrics
        )

        win_level_distribution: dict[int, int] = Counter(
            m.win_level for m in metrics
        )

        failure_rates: dict[str, int] = Counter(
            arch_id for arch_id, _, _ in failure_log
        )

        # Cascade-specific metrics: tumble depth and grid multiplier distributions
        tumble_depth_distribution: dict[int, int] = Counter(
            m.cascade_depth for m in metrics
        )

        grid_multiplier_distribution: dict[int, int] = Counter()
        for m in metrics:
            for val in m.grid_multiplier_values:
                grid_multiplier_distribution[val] += 1

        # Average tumble depth across cascade instances (cascade_depth > 0)
        cascade_metrics = [m for m in metrics if m.cascade_depth > 0]
        avg_tumble_depth = (
            sum(m.cascade_depth for m in cascade_metrics) / len(cascade_metrics)
            if cascade_metrics else 0.0
        )

        # Wild-specific metrics — fraction of instances with wilds, and wild assist rate
        wild_frequency = sum(1 for m in metrics if m.wild_count > 0) / n
        # Wild assist rate: fraction of winning instances where wilds were present
        winning_metrics = [m for m in metrics if m.payout > 0]
        wild_assist_rate = (
            sum(1 for m in winning_metrics if m.wild_count > 0) / len(winning_metrics)
            if winning_metrics else 0.0
        )

        # Booster-specific metrics (Phase 9+)
        booster_spawn_rate = self._compute_booster_spawn_rate(metrics, n)
        booster_fire_rate = self._compute_booster_fire_rate(metrics, n)
        rocket_orientation_balance = self._compute_rocket_orientation_balance(metrics)
        chain_trigger_rate = sum(1 for m in metrics if m.chain_depth > 0) / n

        # Trigger/wincap metrics — fraction of population in each category
        freespin_trigger_rate = sum(1 for m in metrics if m.triggers_freespin) / n
        wincap_hit_rate = sum(1 for m in metrics if m.reaches_wincap) / n

        # LB/SLB targeting distributions — which symbols get cleared
        lb_target_distribution: dict[str, int] = Counter(
            m.lb_target_symbol for m in metrics if m.lb_target_symbol is not None
        )
        slb_target_distribution: dict[str, int] = Counter()
        for m in metrics:
            for sym in m.slb_target_symbols:
                slb_target_distribution[sym] += 1

        # Build results with target comparison
        scalar_metrics = {
            "rtp_total": rtp_total,
            "hit_rate": hit_rate,
            "dead_spin_rate": dead_spin_rate,
            "spatial_cv": spatial_cv,
            "near_miss_scatter_rate": near_miss_scatter_rate,
            "scatter_landing_frequency": scatter_landing_frequency,
            "avg_tumble_depth": avg_tumble_depth,
            "wild_frequency": wild_frequency,
            "wild_assist_rate": wild_assist_rate,
            "chain_trigger_rate": chain_trigger_rate,
            "freespin_trigger_rate": freespin_trigger_rate,
            "wincap_hit_rate": wincap_hit_rate,
        }

        results: list[DiagnosticResult] = []
        for metric_name, value in sorted(scalar_metrics.items()):
            target = self._targets.get(metric_name)
            if target is not None:
                status = self._evaluate_status(value, target[0], target[1])
                results.append(DiagnosticResult(
                    metric=metric_name,
                    value=value,
                    target_min=target[0],
                    target_max=target[1],
                    status=status,
                ))
            else:
                results.append(DiagnosticResult(
                    metric=metric_name,
                    value=value,
                    target_min=None,
                    target_max=None,
                    status="pass",
                ))

        return DiagnosticsReport(
            results=tuple(results),
            spatial_heatmap=dict(spatial_heatmap),
            cluster_size_distribution=dict(cluster_size_distribution),
            symbol_win_contribution=symbol_win_contribution,
            payout_percentiles=payout_percentiles,
            archetype_distribution=dict(archetype_distribution),
            win_level_distribution=dict(win_level_distribution),
            failure_rates=dict(failure_rates),
            tumble_depth_distribution=dict(tumble_depth_distribution),
            grid_multiplier_distribution=dict(grid_multiplier_distribution),
            booster_spawn_rate=booster_spawn_rate,
            booster_fire_rate=booster_fire_rate,
            rocket_orientation_balance=rocket_orientation_balance,
            chain_trigger_rate=chain_trigger_rate,
            lb_target_distribution=dict(lb_target_distribution),
            slb_target_distribution=dict(slb_target_distribution),
            freespin_trigger_rate=freespin_trigger_rate,
            wincap_hit_rate=wincap_hit_rate,
        )

    def _compute_spatial_cv(
        self, metrics: tuple[InstanceMetrics, ...],
    ) -> float:
        """Coefficient of variation for archetype distribution.

        Lower CV means more uniform distribution across archetypes.
        """
        counts = Counter(m.archetype_id for m in metrics)
        if not counts:
            return 0.0
        values = list(counts.values())
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)
        return std_dev / mean

    def _compute_symbol_contribution(
        self, metrics: tuple[InstanceMetrics, ...],
    ) -> dict[str, float]:
        """Fraction of total payout contributed by each symbol."""
        symbol_payout: dict[str, float] = defaultdict(float)
        total_payout = 0.0

        for m in metrics:
            # Distribute payout evenly among cluster symbols in this instance
            if m.cluster_symbols and m.payout > 0:
                per_cluster = m.payout / len(m.cluster_symbols)
                for sym_name in m.cluster_symbols:
                    symbol_payout[sym_name] += per_cluster
                total_payout += m.payout

        if total_payout == 0:
            return dict(symbol_payout)

        return {sym: pay / total_payout for sym, pay in symbol_payout.items()}

    def _compute_payout_percentiles(
        self, metrics: tuple[InstanceMetrics, ...],
    ) -> dict[str, float]:
        """Compute p50, p90, p95, p99 of payout distribution."""
        payouts = sorted(m.payout for m in metrics)
        n = len(payouts)
        if n == 0:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

        def percentile(pct: float) -> float:
            idx = int(pct / 100.0 * (n - 1))
            return payouts[min(idx, n - 1)]

        return {
            "p50": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
        }

    def _evaluate_status(
        self,
        value: float,
        target_min: float | None,
        target_max: float | None,
    ) -> str:
        """Compare metric value against target bounds."""
        if target_min is not None and value < target_min:
            return "fail"
        if target_max is not None and value > target_max:
            return "fail"
        return "pass"

    def _compute_booster_spawn_rate(
        self,
        metrics: tuple[InstanceMetrics, ...],
        n: int,
    ) -> dict[str, float]:
        """Fraction of instances where each booster type was spawned."""
        spawn_counts: dict[str, int] = defaultdict(int)
        for m in metrics:
            seen: set[str] = set()
            for btype, count in m.booster_spawn_counts:
                if count > 0 and btype not in seen:
                    spawn_counts[btype] += 1
                    seen.add(btype)
        return {btype: count / n for btype, count in spawn_counts.items()}

    def _compute_booster_fire_rate(
        self,
        metrics: tuple[InstanceMetrics, ...],
        n: int,
    ) -> dict[str, float]:
        """Fraction of instances where each booster type fired."""
        fire_counts: dict[str, int] = defaultdict(int)
        for m in metrics:
            seen: set[str] = set()
            for btype, count in m.booster_fire_counts:
                if count > 0 and btype not in seen:
                    fire_counts[btype] += 1
                    seen.add(btype)
        return {btype: count / n for btype, count in fire_counts.items()}

    def _compute_rocket_orientation_balance(
        self,
        metrics: tuple[InstanceMetrics, ...],
    ) -> dict[str, float]:
        """H vs V balance across all instances where a rocket fired.

        Returns fractions summing to 1.0 (e.g., {"H": 0.52, "V": 0.48}).
        Empty dict if no rockets fired.
        """
        orientation_counts: dict[str, int] = Counter()
        for m in metrics:
            if m.rocket_orientation_actual is not None:
                orientation_counts[m.rocket_orientation_actual] += 1

        total = sum(orientation_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in orientation_counts.items()}
