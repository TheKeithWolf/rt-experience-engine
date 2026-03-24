"""TEST-P1-015 through TEST-P1-019: Paytable lookup, payout, centipayout, win levels."""

from ..config.schema import MasterConfig
from ..primitives.cluster_detection import Cluster
from ..primitives.board import Position
from ..primitives.grid_multipliers import GridMultiplierGrid
from ..primitives.paytable import Paytable
from ..primitives.symbols import Symbol


def _make_paytable(config: MasterConfig) -> Paytable:
    return Paytable(config.paytable, config.centipayout, config.win_levels)


def test_p1_015_paytable_lookups(default_config: MasterConfig) -> None:
    """TEST-P1-015: Paytable: L1 size 5 → 0.1, H3 size 15 → 25.0."""
    pt = _make_paytable(default_config)
    assert pt.get_payout(5, Symbol.L1) == 0.1
    assert pt.get_payout(15, Symbol.H3) == 25.0
    # Size outside paytable range returns 0
    assert pt.get_payout(3, Symbol.L1) == 0.0


def test_p1_016_same_payout_within_tier(default_config: MasterConfig) -> None:
    """TEST-P1-016: Same payout for sizes within the same tier."""
    pt = _make_paytable(default_config)
    # Tier 1: sizes 5-6 should have the same payout for each symbol
    for sym in (Symbol.L1, Symbol.L2, Symbol.H1, Symbol.H3):
        assert pt.get_payout(5, sym) == pt.get_payout(6, sym), (
            f"{sym.name}: size 5 and 6 should have same payout"
        )


def test_p1_017_compute_cluster_payout(default_config: MasterConfig) -> None:
    """TEST-P1-017: compute_cluster_payout: base * max(mult_sum, minimum_contribution)."""
    pt = _make_paytable(default_config)
    grid = GridMultiplierGrid(default_config.grid_multiplier, default_config.board)

    # Cluster of 5 L1 symbols — all multipliers at 0
    positions = frozenset({Position(0, 0), Position(1, 0), Position(2, 0),
                           Position(3, 0), Position(4, 0)})
    cluster = Cluster(
        symbol=Symbol.L1,
        positions=positions,
        wild_positions=frozenset(),
        size=5,
    )
    # base=0.1, mult_sum=max(0, 1)=1, total=0.1
    assert pt.compute_cluster_payout(cluster, grid) == 0.1

    # Now increment some multipliers
    grid.increment(Position(0, 0))  # 0→1
    grid.increment(Position(1, 0))  # 0→1
    grid.increment(Position(1, 0))  # 1→2
    # mult_sum = 1 + 2 + 0 + 0 + 0 = 3
    assert pt.compute_cluster_payout(cluster, grid) == 0.1 * 3


def test_p1_018_centipayout_conversion(default_config: MasterConfig) -> None:
    """TEST-P1-018: to_centipayout: 2.6x → 260, 0.1x → 10, 0.0x → 0."""
    pt = _make_paytable(default_config)
    assert pt.to_centipayout(2.6) == 260
    assert pt.to_centipayout(0.1) == 10
    assert pt.to_centipayout(0.0) == 0


def test_p1_019_win_level_mapping(default_config: MasterConfig) -> None:
    """TEST-P1-019: get_win_level: 0.5x → 2, 50.0x → 7, 5000.0x → 10."""
    pt = _make_paytable(default_config)
    assert pt.get_win_level(0.05) == 1   # [0.0, 0.1)
    assert pt.get_win_level(0.5) == 2    # [0.1, 1.0)
    assert pt.get_win_level(1.5) == 3    # [1.0, 2.0)
    assert pt.get_win_level(50.0) == 7   # [25.0, 100.0)
    assert pt.get_win_level(5000.0) == 10  # [5000.0, inf)
