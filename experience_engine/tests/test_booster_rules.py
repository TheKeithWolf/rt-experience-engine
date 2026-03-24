"""TEST-P1-039 through TEST-P1-046: Booster rules — thresholds, centroid, orientation, paths."""

from ..config.schema import MasterConfig
from ..primitives.board import Position
from ..primitives.booster_rules import BoosterRules
from ..primitives.symbols import Symbol


def _make_rules(config: MasterConfig) -> BoosterRules:
    return BoosterRules(config.boosters, config.board, config.symbols)


def test_p1_039_booster_type_for_size(default_config: MasterConfig) -> None:
    """TEST-P1-039: Spawn thresholds from config."""
    rules = _make_rules(default_config)
    assert rules.booster_type_for_size(4) is None    # Below threshold
    assert rules.booster_type_for_size(6) is None    # Still below
    assert rules.booster_type_for_size(7) is Symbol.W
    assert rules.booster_type_for_size(8) is Symbol.W
    assert rules.booster_type_for_size(9) is Symbol.R
    assert rules.booster_type_for_size(10) is Symbol.R
    assert rules.booster_type_for_size(11) is Symbol.B
    assert rules.booster_type_for_size(12) is Symbol.B
    assert rules.booster_type_for_size(13) is Symbol.LB
    assert rules.booster_type_for_size(14) is Symbol.LB
    assert rules.booster_type_for_size(15) is Symbol.SLB
    assert rules.booster_type_for_size(49) is Symbol.SLB


def test_p1_040_compute_centroid(default_config: MasterConfig) -> None:
    """TEST-P1-040: Centroid of a 3x3 block is the center cell."""
    rules = _make_rules(default_config)
    positions = frozenset({
        Position(2, 2), Position(3, 2), Position(4, 2),
        Position(2, 3), Position(3, 3), Position(4, 3),
        Position(2, 4), Position(3, 4), Position(4, 4),
    })
    centroid = rules.compute_centroid(positions)
    assert centroid == Position(3, 3)


def test_p1_041_resolve_collision(default_config: MasterConfig) -> None:
    """TEST-P1-041: Centroid occupied → next-nearest cluster member."""
    rules = _make_rules(default_config)
    positions = frozenset({Position(0, 0), Position(1, 0), Position(2, 0),
                           Position(3, 0), Position(4, 0)})
    centroid = rules.compute_centroid(positions)

    # Mark centroid as occupied
    occupied = frozenset({centroid})
    result = rules.resolve_collision(centroid, positions, occupied)
    assert result not in occupied
    assert result in positions


def test_p1_042_rocket_orientation_taller(default_config: MasterConfig) -> None:
    """TEST-P1-042: Row span > col span → "H" (fires perpendicular to vertical axis)."""
    rules = _make_rules(default_config)
    # Vertical cluster: 5 rows, 1 column → row_span=5, col_span=1
    positions = frozenset({
        Position(3, 0), Position(3, 1), Position(3, 2),
        Position(3, 3), Position(3, 4),
    })
    assert rules.compute_rocket_orientation(positions) == "H"


def test_p1_043_rocket_orientation_tie(default_config: MasterConfig) -> None:
    """TEST-P1-043: Equal row_span and col_span → config default ("H")."""
    rules = _make_rules(default_config)
    # Square cluster: 2x2
    positions = frozenset({
        Position(3, 3), Position(4, 3),
        Position(3, 4), Position(4, 4),
    })
    assert rules.compute_rocket_orientation(positions) == "H"


def test_p1_044_rocket_path_horizontal(default_config: MasterConfig) -> None:
    """TEST-P1-044: Horizontal rocket clears all 7 positions in the row."""
    rules = _make_rules(default_config)
    path = rules.rocket_path(Position(3, 3), "H")
    assert len(path) == 7
    for reel in range(7):
        assert Position(reel, 3) in path


def test_p1_044b_rocket_path_vertical(default_config: MasterConfig) -> None:
    """Vertical rocket clears all 7 positions in the column."""
    rules = _make_rules(default_config)
    path = rules.rocket_path(Position(3, 3), "V")
    assert len(path) == 7
    for row in range(7):
        assert Position(3, row) in path


def test_p1_045_bomb_blast_clipping(default_config: MasterConfig) -> None:
    """TEST-P1-045: Bomb blast clipped at edges and corners."""
    rules = _make_rules(default_config)

    # Center: full 3x3 = 9 positions
    center_blast = rules.bomb_blast(Position(3, 3))
    assert len(center_blast) == 9

    # Corner (0,0): clipped to 2x2 = 4 positions
    corner_blast = rules.bomb_blast(Position(0, 0))
    assert len(corner_blast) == 4
    assert Position(0, 0) in corner_blast
    assert Position(1, 0) in corner_blast
    assert Position(0, 1) in corner_blast
    assert Position(1, 1) in corner_blast

    # Edge (0,3): clipped to 2x3 = 6 positions
    edge_blast = rules.bomb_blast(Position(0, 3))
    assert len(edge_blast) == 6


def test_p1_046_immune_sets(default_config: MasterConfig) -> None:
    """TEST-P1-046: Immune sets match config values."""
    rules = _make_rules(default_config)
    assert Symbol.W in rules.immune_to_rocket
    assert Symbol.S in rules.immune_to_rocket
    assert Symbol.W in rules.immune_to_bomb
    assert Symbol.S in rules.immune_to_bomb
    assert Symbol.R in rules.chain_initiators
    assert Symbol.B in rules.chain_initiators
