"""Instance validation against archetype signatures.

Reuses the same shared primitives (cluster_detection, paytable) as the
generation pipeline — identical detection logic ensures no drift (DRY).
All validation thresholds come from the archetype signature + MasterConfig.
"""

from __future__ import annotations

from ..archetypes.registry import ArchetypeRegistry, ArchetypeSignature
from ..config.schema import MasterConfig
from ..pipeline.data_types import CascadeStepRecord, GeneratedInstance
from ..primitives.board import Board, Position
from ..primitives.cluster_detection import (
    Cluster,
    detect_clusters,
    detect_components,
)
from ..primitives.paytable import Paytable
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.symbols import (
    Symbol,
    SymbolTier,
    is_booster,
    is_scatter,
    is_standard,
    is_wild,
    symbol_from_name,
    symbols_in_tier,
    tier_of,
)
from .metrics import InstanceMetrics


def _count_consumed_wilds(
    cascade_steps: tuple[CascadeStepRecord, ...] | None,
) -> int:
    """Count wild positions cleared by cluster explosions across a cascade.

    Deduplicates per step — a wild bridging multiple clusters in one step
    (R-WILD-5) is cleared once.
    """
    if not cascade_steps:
        return 0
    return sum(
        len({
            pos
            for cluster in step_rec.clusters
            for pos in cluster.wild_positions
        })
        for step_rec in cascade_steps
    )


class InstanceValidator:
    """Validates generated instances against their archetype signatures.

    Uses the same shared primitives as the generation pipeline so detection
    logic is identical. All thresholds from config + signature.

    For cascade instances (cascade_steps not None), initial board checks run
    against step 0's board, while terminal checks run on instance.board.
    """

    __slots__ = ("_config", "_registry", "_paytable")

    def __init__(
        self,
        config: MasterConfig,
        registry: ArchetypeRegistry,
    ) -> None:
        self._config = config
        self._registry = registry
        self._paytable = Paytable(
            config.paytable, config.centipayout, config.win_levels,
        )

    def validate(self, instance: GeneratedInstance) -> InstanceMetrics:
        """Validate an instance and collect metrics.

        For static instances: checks against instance.board.
        For cascade instances: initial board checks use step 0's board_after,
        terminal checks use instance.board (final dead board).
        """
        sig = self._registry.get(instance.archetype_id)
        errors: list[str] = []
        is_cascade = instance.cascade_steps is not None and len(instance.cascade_steps) > 0

        # Determine which board to use for initial-state checks
        # Cascade: step 0's board_after has the filled initial board
        # Static: instance.board is the only board
        if is_cascade:
            initial_board = instance.cascade_steps[0].board_after
        else:
            initial_board = instance.board

        terminal_board = instance.board

        # 1. Board completeness — no None cells (check both boards for cascade)
        empty = terminal_board.empty_positions()
        if empty:
            errors.append(f"terminal board has {len(empty)} empty cells")
        if is_cascade:
            init_empty = initial_board.empty_positions()
            if init_empty:
                errors.append(f"initial board has {len(init_empty)} empty cells")

        # 2. Detect clusters on the initial board (where clusters are expected)
        clusters = detect_clusters(initial_board, self._config)

        # 3. Cluster count within range
        cluster_count = len(clusters)
        if not sig.required_cluster_count.contains(cluster_count):
            errors.append(
                f"cluster_count={cluster_count} outside "
                f"[{sig.required_cluster_count.min_val}, {sig.required_cluster_count.max_val}]"
            )

        # 4. Cluster sizes — each within some signature size range
        cluster_sizes = tuple(c.size for c in clusters)
        for i, size in enumerate(cluster_sizes):
            in_any_range = any(
                r.contains(size) for r in sig.required_cluster_sizes
            ) if sig.required_cluster_sizes else (size >= 0)
            if not in_any_range and sig.required_cluster_sizes:
                errors.append(
                    f"cluster[{i}] size={size} not in any required size range"
                )

        # 5. Cluster symbols match tier constraint
        cluster_symbols = tuple(c.symbol.name for c in clusters)
        if sig.required_cluster_symbols is not None:
            allowed_tier = sig.required_cluster_symbols
            for i, c in enumerate(clusters):
                if allowed_tier is not SymbolTier.ANY:
                    try:
                        actual_tier = tier_of(c.symbol, self._config.symbols)
                        if actual_tier is not allowed_tier:
                            errors.append(
                                f"cluster[{i}] symbol {c.symbol.name} is "
                                f"{actual_tier.value}, expected {allowed_tier.value}"
                            )
                    except ValueError:
                        errors.append(
                            f"cluster[{i}] symbol {c.symbol.name} has no tier"
                        )

        # 6. Scatter count — count S symbols on the initial board
        scatter_count = sum(
            1 for pos in initial_board.all_positions()
            if initial_board.get(pos) is Symbol.S
        )
        if not sig.required_scatter_count.contains(scatter_count):
            errors.append(
                f"scatter_count={scatter_count} outside "
                f"[{sig.required_scatter_count.min_val}, {sig.required_scatter_count.max_val}]"
            )

        # 6b. Trigger validation — scatters must meet minimum for freespin
        if sig.triggers_freespin:
            min_required = self._config.freespin.min_scatters_to_trigger
            if scatter_count < min_required:
                errors.append(
                    f"triggers_freespin=True but scatter_count={scatter_count} "
                    f"< min_scatters_to_trigger={min_required}"
                )

        # 7. Near-miss detection — components of size min_cluster_size-1
        near_miss_size = self._config.board.min_cluster_size - 1
        near_misses = self._detect_near_misses(initial_board, near_miss_size)
        near_miss_count = len(near_misses)
        near_miss_symbols = tuple(sym.name for sym, _ in near_misses)

        if not sig.required_near_miss_count.contains(near_miss_count):
            errors.append(
                f"near_miss_count={near_miss_count} outside "
                f"[{sig.required_near_miss_count.min_val}, {sig.required_near_miss_count.max_val}]"
            )

        # 8. Near-miss symbol tier check
        if sig.required_near_miss_symbol_tier is not None and near_misses:
            allowed_tier = sig.required_near_miss_symbol_tier
            if allowed_tier is not SymbolTier.ANY:
                for nm_sym, _ in near_misses:
                    try:
                        actual_tier = tier_of(nm_sym, self._config.symbols)
                        if actual_tier is not allowed_tier:
                            errors.append(
                                f"near_miss symbol {nm_sym.name} is "
                                f"{actual_tier.value}, expected {allowed_tier.value}"
                            )
                    except ValueError:
                        errors.append(
                            f"near_miss symbol {nm_sym.name} has no tier"
                        )

        # 9. Max component size check (dead boards — static only)
        # When near-misses are expected, the max allowed component is at least
        # the near-miss size (min_cluster_size - 1), since CSP places intentional
        # NMs that exceed max_component_size
        max_comp = self._compute_max_component(initial_board)
        if sig.max_component_size is not None and not is_cascade:
            effective_max = sig.max_component_size
            if sig.required_near_miss_count.max_val > 0:
                nm_size = self._config.board.min_cluster_size - 1
                effective_max = max(effective_max, nm_size)
            if max_comp > effective_max:
                errors.append(
                    f"max_component_size={max_comp} exceeds limit {effective_max}"
                )

        # 10. Payout within range
        if is_cascade:
            # Cumulative payout from all cascade steps
            computed_payout = sum(
                s.step_payout for s in instance.cascade_steps
            )
        else:
            # Static: recompute from detected clusters
            computed_payout = self._compute_payout(clusters)

        if not sig.payout_range.contains(computed_payout):
            errors.append(
                f"payout={computed_payout:.4f} outside "
                f"[{sig.payout_range.min_val}, {sig.payout_range.max_val}]"
            )

        # 10b. Wincap validation — payout must equal max_payout exactly
        if sig.reaches_wincap:
            expected = self._config.wincap.max_payout
            if abs(computed_payout - expected) > 0.001:
                errors.append(
                    f"reaches_wincap=True but payout={computed_payout:.4f} "
                    f"!= max_payout={expected}"
                )

        # 11. No booster or wild symbols on board (dead/t1 families only)
        # Wild family legitimately uses wilds; later phases add booster families
        if sig.family in ("dead", "t1"):
            for check_board in ([initial_board, terminal_board] if is_cascade else [terminal_board]):
                for pos in check_board.all_positions():
                    sym = check_board.get(pos)
                    if sym is not None:
                        if is_wild(sym) or is_booster(sym):
                            errors.append(
                                f"unexpected {sym.name} at ({pos.reel}, {pos.row}) — "
                                f"{sig.family} boards must not contain wilds or boosters"
                            )

        # 11b. Wild-family — count wilds on terminal board (validated after 11d
        # once booster_spawn_counts is available for the spawned−consumed formula)
        wild_count = 0
        if sig.family == "wild":
            wild_count = sum(
                1 for pos in terminal_board.all_positions()
                if terminal_board.get(pos) is Symbol.W
            )

        # 11c. Terminal constraints — shared across wild, rocket, bomb families
        # DRY: terminal NMs and dormant boosters validated identically for all families
        terminal_near_miss_count_val = 0
        dormant_booster_count = 0

        if sig.terminal_near_misses is not None:
            terminal_nms = self._detect_near_misses(terminal_board, near_miss_size)
            terminal_near_miss_count_val = len(terminal_nms)
            if not sig.terminal_near_misses.count.contains(terminal_near_miss_count_val):
                errors.append(
                    f"terminal_near_miss_count={terminal_near_miss_count_val} outside "
                    f"[{sig.terminal_near_misses.count.min_val}, "
                    f"{sig.terminal_near_misses.count.max_val}]"
                )
            if (sig.terminal_near_misses.symbol_tier is not None
                    and sig.terminal_near_misses.symbol_tier is not SymbolTier.ANY):
                for nm_sym, _ in terminal_nms:
                    try:
                        actual_tier = tier_of(nm_sym, self._config.symbols)
                        if actual_tier is not sig.terminal_near_misses.symbol_tier:
                            errors.append(
                                f"terminal NM symbol {nm_sym.name} is "
                                f"{actual_tier.value}, expected "
                                f"{sig.terminal_near_misses.symbol_tier.value}"
                            )
                    except ValueError:
                        pass

        if sig.dormant_boosters_on_terminal:
            for booster_name in sig.dormant_boosters_on_terminal:
                booster_sym = symbol_from_name(booster_name)
                found = any(
                    terminal_board.get(pos) is booster_sym
                    for pos in terminal_board.all_positions()
                )
                if found:
                    dormant_booster_count += 1
                else:
                    errors.append(
                        f"dormant {booster_name} not found on terminal board"
                    )

        # 11d. Booster spawn/fire validation — rocket and bomb families
        booster_spawn_counts: dict[str, int] = {}
        booster_fire_counts: dict[str, int] = {}
        chain_depth_val = 0
        rocket_orientation_val: str | None = None

        if is_cascade and instance.cascade_steps:
            # Aggregate spawn and fire counts from cascade step records
            for step_rec in instance.cascade_steps:
                for btype in step_rec.booster_spawn_types:
                    booster_spawn_counts[btype] = (
                        booster_spawn_counts.get(btype, 0) + 1
                    )
                for fire_rec in step_rec.booster_fire_records:
                    booster_fire_counts[fire_rec.booster_type] = (
                        booster_fire_counts.get(fire_rec.booster_type, 0) + 1
                    )
                    # Track chain depth as total chain-triggered fires
                    chain_depth_val += fire_rec.chain_target_count
                    # Capture first rocket orientation for diagnostics
                    if (fire_rec.booster_type == "R"
                            and rocket_orientation_val is None):
                        rocket_orientation_val = fire_rec.orientation

            # Validate booster spawn counts against signature requirements
            for btype, required_range in sig.required_booster_spawns.items():
                actual = booster_spawn_counts.get(btype, 0)
                if not required_range.contains(actual):
                    errors.append(
                        f"booster_spawn({btype})={actual} outside "
                        f"[{required_range.min_val}, {required_range.max_val}]"
                    )

            # Validate booster fire counts against signature requirements
            for btype, required_range in sig.required_booster_fires.items():
                actual = booster_fire_counts.get(btype, 0)
                if not required_range.contains(actual):
                    errors.append(
                        f"booster_fire({btype})={actual} outside "
                        f"[{required_range.min_val}, {required_range.max_val}]"
                    )

            # Validate chain depth
            if not sig.required_chain_depth.contains(chain_depth_val):
                errors.append(
                    f"chain_depth={chain_depth_val} outside "
                    f"[{sig.required_chain_depth.min_val}, "
                    f"{sig.required_chain_depth.max_val}]"
                )

            # Validate rocket orientation if the archetype constrains it
            if sig.rocket_orientation is not None and rocket_orientation_val is not None:
                if rocket_orientation_val != sig.rocket_orientation:
                    errors.append(
                        f"rocket_orientation={rocket_orientation_val}, "
                        f"expected {sig.rocket_orientation}"
                    )

        # 11d-post. Wild-family terminal validation — empirical consumption
        # from cluster detection records replaces signature-derived estimates.
        if sig.family == "wild":
            wild_spawned = booster_spawn_counts.get("W", 0)
            wild_consumed = _count_consumed_wilds(instance.cascade_steps)
            expected_terminal = wild_spawned - wild_consumed
            if wild_count != expected_terminal:
                errors.append(
                    f"wild_count={wild_count} on terminal board, "
                    f"expected {expected_terminal} "
                    f"(spawned={wild_spawned}, consumed={wild_consumed})"
                )
            # Archetype-intent range check — independent of conservation
            if (
                sig.narrative_arc is not None
                and sig.narrative_arc.wild_count_on_terminal is not None
            ):
                if not sig.narrative_arc.wild_count_on_terminal.contains(wild_count):
                    errors.append(
                        f"wild_count={wild_count} outside signature range "
                        f"[{sig.narrative_arc.wild_count_on_terminal.min_val}, "
                        f"{sig.narrative_arc.wild_count_on_terminal.max_val}]"
                    )

        # 11e. LB/SLB-specific validation
        lb_target_symbol: str | None = None
        slb_target_symbols: tuple[str, ...] = ()

        if is_cascade and instance.cascade_steps:
            for step_rec in instance.cascade_steps:
                for fire_rec in step_rec.booster_fire_records:
                    # LB/SLB must never initiate chains — zero chain targets
                    if fire_rec.booster_type in ("LB", "SLB"):
                        if fire_rec.chain_target_count > 0:
                            errors.append(
                                f"{fire_rec.booster_type} at "
                                f"({fire_rec.position_reel}, {fire_rec.position_row}) "
                                f"has chain_target_count={fire_rec.chain_target_count} "
                                f"— LB/SLB cannot initiate chains"
                            )
                    # Capture LB target for metrics/diagnostics
                    if fire_rec.booster_type == "LB" and fire_rec.target_symbols:
                        lb_target_symbol = fire_rec.target_symbols[0]
                    # Capture SLB targets for metrics/diagnostics
                    if fire_rec.booster_type == "SLB" and fire_rec.target_symbols:
                        slb_target_symbols = fire_rec.target_symbols

            # Validate LB target tier if the archetype constrains it
            if (sig.lb_target_tier is not None
                    and sig.lb_target_tier is not SymbolTier.ANY
                    and lb_target_symbol is not None):
                try:
                    actual_tier = tier_of(
                        symbol_from_name(lb_target_symbol), self._config.symbols,
                    )
                    if actual_tier is not sig.lb_target_tier:
                        errors.append(
                            f"lb_target_symbol={lb_target_symbol} is "
                            f"{actual_tier.value}, expected {sig.lb_target_tier.value}"
                        )
                except ValueError:
                    errors.append(
                        f"lb_target_symbol={lb_target_symbol} has no tier"
                    )

        # 12. Cascade-specific validation
        cascade_depth = 0
        grid_multiplier_values: tuple[int, ...] = ()

        if is_cascade:
            # Step 0 is the initial board reveal, not a refill cycle —
            # required_cascade_depth counts refill cycles only
            cascade_depth = len(instance.cascade_steps) - 1

            # Cascade depth within signature range
            if not sig.required_cascade_depth.contains(cascade_depth):
                errors.append(
                    f"cascade_depth={cascade_depth} outside "
                    f"[{sig.required_cascade_depth.min_val}, "
                    f"{sig.required_cascade_depth.max_val}]"
                )

            # Per-step cluster validation against cascade_steps constraints
            if sig.cascade_steps is not None:
                for i, (step_record, step_constraint) in enumerate(
                    zip(instance.cascade_steps, sig.cascade_steps)
                ):
                    step_cluster_count = len(step_record.clusters)
                    if not step_constraint.cluster_count.contains(step_cluster_count):
                        errors.append(
                            f"step[{step_record.step_index}] cluster_count="
                            f"{step_cluster_count} outside "
                            f"[{step_constraint.cluster_count.min_val}, "
                            f"{step_constraint.cluster_count.max_val}]"
                        )

                    if step_constraint.cluster_sizes:
                        for j, cluster in enumerate(step_record.clusters):
                            in_any = any(
                                r.contains(cluster.size)
                                for r in step_constraint.cluster_sizes
                            )
                            if not in_any:
                                errors.append(
                                    f"step[{step_record.step_index}] "
                                    f"cluster[{j}] size={cluster.size} "
                                    f"not in any required size range"
                                )

                    if step_constraint.cluster_symbol_tier is not None:
                        allowed_tier = step_constraint.cluster_symbol_tier
                        if allowed_tier is not SymbolTier.ANY:
                            for j, cluster in enumerate(step_record.clusters):
                                try:
                                    actual_tier = tier_of(
                                        cluster.symbol, self._config.symbols,
                                    )
                                    if actual_tier is not allowed_tier:
                                        errors.append(
                                            f"step[{step_record.step_index}] "
                                            f"cluster[{j}] symbol "
                                            f"{cluster.symbol.name} is "
                                            f"{actual_tier.value}, expected "
                                            f"{allowed_tier.value}"
                                        )
                                except ValueError:
                                    pass

            # symbol_tier_per_step validation (narrative arc constraints)
            if sig.symbol_tier_per_step is not None:
                for step_record in instance.cascade_steps:
                    if step_record.step_index in sig.symbol_tier_per_step:
                        required_tier = sig.symbol_tier_per_step[
                            step_record.step_index
                        ]
                        if required_tier is not SymbolTier.ANY:
                            for j, cluster in enumerate(step_record.clusters):
                                try:
                                    actual_tier = tier_of(
                                        cluster.symbol, self._config.symbols,
                                    )
                                    if actual_tier is not required_tier:
                                        errors.append(
                                            f"step[{step_record.step_index}] "
                                            f"narrative arc: cluster[{j}] "
                                            f"symbol {cluster.symbol.name} "
                                            f"is {actual_tier.value}, expected "
                                            f"{required_tier.value}"
                                        )
                                except ValueError:
                                    pass

            # Terminal board must be dead — no cluster >= min_cluster_size
            terminal_max_comp = self._compute_max_component(terminal_board)
            max_allowed = self._config.board.min_cluster_size - 1
            if terminal_max_comp > max_allowed:
                errors.append(
                    f"terminal board max_component={terminal_max_comp} exceeds "
                    f"dead threshold {max_allowed}"
                )

            # Extract final grid multiplier snapshot for diagnostics
            if instance.cascade_steps:
                grid_multiplier_values = instance.cascade_steps[-1].grid_multipliers_snapshot

        return InstanceMetrics(
            archetype_id=instance.archetype_id,
            family=instance.family,
            criteria=instance.criteria,
            sim_id=instance.sim_id,
            payout=computed_payout,
            centipayout=self._paytable.to_centipayout(computed_payout),
            win_level=self._paytable.get_win_level(computed_payout),
            cluster_count=cluster_count,
            cluster_sizes=cluster_sizes,
            cluster_symbols=cluster_symbols,
            scatter_count=scatter_count,
            near_miss_count=near_miss_count,
            near_miss_symbols=near_miss_symbols,
            max_component_size=max_comp,
            is_valid=len(errors) == 0,
            validation_errors=tuple(errors),
            cascade_depth=cascade_depth,
            wild_count=wild_count,
            terminal_near_miss_count=terminal_near_miss_count_val,
            dormant_booster_count=dormant_booster_count,
            grid_multiplier_values=grid_multiplier_values,
            booster_spawn_counts=tuple(booster_spawn_counts.items()),
            booster_fire_counts=tuple(booster_fire_counts.items()),
            chain_depth=chain_depth_val,
            rocket_orientation_actual=rocket_orientation_val,
            lb_target_symbol=lb_target_symbol,
            slb_target_symbols=slb_target_symbols,
            triggers_freespin=sig.triggers_freespin,
            reaches_wincap=(
                sig.reaches_wincap
                and computed_payout >= self._config.wincap.max_payout
            ),
        )

    def _detect_near_misses(
        self,
        board: Board,
        target_size: int,
    ) -> list[tuple[Symbol, frozenset[Position]]]:
        """Find connected components of exactly target_size for each standard symbol.

        Uses detect_components() from cluster_detection (DRY).
        """
        result: list[tuple[Symbol, frozenset[Position]]] = []
        for name in self._config.symbols.standard:
            sym = symbol_from_name(name)
            components = detect_components(board, sym, self._config.board)
            for comp in components:
                if len(comp) == target_size:
                    result.append((sym, comp))
        return result

    def _compute_max_component(self, board: Board) -> int:
        """Find the largest connected component of any standard symbol."""
        max_size = 0
        for name in self._config.symbols.standard:
            sym = symbol_from_name(name)
            components = detect_components(board, sym, self._config.board)
            for comp in components:
                max_size = max(max_size, len(comp))
        return max_size

    def _compute_payout(self, clusters: list[Cluster]) -> float:
        """Compute total payout from detected clusters.

        Static boards have grid multipliers at initial value (0), so
        position_multiplier_sum returns minimum_contribution (1).
        Effective payout = sum(base_payout * 1) for each cluster.
        """
        total = 0.0
        for cluster in clusters:
            base = self._paytable.get_payout(cluster.size, cluster.symbol)
            # Grid multipliers at initial state — minimum_contribution applies
            grid_mults = GridMultiplierGrid(
                self._config.grid_multiplier, self._config.board,
            )
            all_positions = cluster.positions | cluster.wild_positions
            mult_sum = grid_mults.position_multiplier_sum(all_positions)
            total += base * mult_sum
        return total
