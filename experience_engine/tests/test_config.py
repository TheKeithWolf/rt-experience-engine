"""Config loading, validation, and normalization tests.

TEST-P1-001 through TEST-P1-005: Phase 1 config tests.
TEST-R8-001, TEST-R8-002: Step 8 reasoner config tests.
TEST-REFILL-030 through TEST-REFILL-033: Refill strategy config tests.
"""

from pathlib import Path

import pytest
import yaml

from ..config.loader import load_config
from ..config.schema import ConfigValidationError, MasterConfig, RefillConfig

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def test_p1_001_load_default_config_succeeds(default_config: MasterConfig) -> None:
    """TEST-P1-001: MasterConfig loads from default.yaml without errors."""
    # All 14 sub-configs should be present
    assert default_config.board is not None
    assert default_config.gravity is not None
    assert default_config.grid_multiplier is not None
    assert default_config.boosters is not None
    assert default_config.paytable is not None
    assert default_config.centipayout is not None
    assert default_config.win_levels is not None
    assert default_config.freespin is not None
    assert default_config.wincap is not None
    assert default_config.solvers is not None
    assert default_config.diagnostics is not None
    assert default_config.population is not None
    assert default_config.symbols is not None
    assert default_config.anticipation is not None

    # Verify some specific default values
    assert default_config.board.num_reels == 7
    assert default_config.board.num_rows == 7
    assert default_config.board.min_cluster_size == 5


def test_p1_002_config_rejects_missing_required_fields(tmp_path: Path) -> None:
    """TEST-P1-002: Config rejects missing required fields."""
    # Write a YAML missing the board section entirely
    config_data = {"gravity": {"donor_priorities": [[0, -1]], "max_settle_passes": 100}}
    config_file = tmp_path / "bad_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    with pytest.raises(ConfigValidationError, match="board"):
        load_config(config_file)


def test_p1_003_config_rejects_overlapping_spawn_thresholds(tmp_path: Path) -> None:
    """TEST-P1-003: Config rejects spawn thresholds with overlapping ranges."""
    # Load default, modify spawn thresholds to overlap
    with open(DEFAULT_CONFIG_PATH, "r") as fh:
        config_data = yaml.safe_load(fh)

    # Make W overlap with R: W goes up to 10, R starts at 9
    config_data["boosters"]["spawn_thresholds"][0]["max_size"] = 10

    config_file = tmp_path / "overlap_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    with pytest.raises(ConfigValidationError, match="overlap"):
        load_config(config_file)


def test_p1_004_config_auto_renormalizes_family_weights(default_config: MasterConfig) -> None:
    """TEST-P1-004: Config auto-renormalizes family weights to 1.0."""
    total = sum(w for _, w in default_config.population.family_weights)
    assert abs(total - 1.0) < 1e-9


def test_p1_005_config_expands_paytable_entries(default_config: MasterConfig) -> None:
    """TEST-P1-005: Config has paytable entries for all 7 symbols x 6 tiers."""
    # 7 symbols * 6 tiers = 42 entries in the YAML
    assert len(default_config.paytable.entries) == 42


# ---------------------------------------------------------------------------
# TEST-R8-001: MasterConfig loads with reasoner section — all 7 fields present
# ---------------------------------------------------------------------------

def test_r8_001_config_loads_with_reasoner(default_config: MasterConfig) -> None:
    """TEST-R8-001: MasterConfig loads with all 7 reasoner fields from default.yaml."""
    r = default_config.reasoner
    # Step 5 payout pacing and booster urgency thresholds
    assert r.payout_low_fraction == 0.3
    assert r.payout_high_fraction == 0.85
    assert r.arming_urgency_horizon == 1
    assert r.terminal_dead_default_max_component == 4
    # Step 8 computation caps
    assert r.max_forward_simulations_per_step == 10
    assert r.max_strategic_cells_per_step == 16
    assert r.lookahead_depth == 2


# ---------------------------------------------------------------------------
# TEST-R8-002: MasterConfig rejects YAML with missing reasoner section
# ---------------------------------------------------------------------------

def test_r8_002_config_rejects_missing_reasoner(tmp_path: Path) -> None:
    """TEST-R8-002: MasterConfig rejects YAML with no reasoner section."""
    with open(DEFAULT_CONFIG_PATH, "r") as fh:
        config_data = yaml.safe_load(fh)
    del config_data["reasoner"]
    config_file = tmp_path / "no_reasoner.yaml"
    config_file.write_text(yaml.dump(config_data))

    with pytest.raises(ConfigValidationError, match="reasoner"):
        load_config(config_file)


# ---------------------------------------------------------------------------
# TEST-REFILL-030 through TEST-REFILL-033: Refill strategy config tests
# ---------------------------------------------------------------------------

def test_refill_030_loads_from_default_yaml(default_config: MasterConfig) -> None:
    """TEST-REFILL-030: RefillConfig loads from default.yaml with expected values."""
    assert default_config.refill is not None
    assert default_config.refill.adjacency_boost == 3.0
    assert default_config.refill.depth_scale == 0.3
    assert default_config.refill.terminal_max_retries == 10


def test_refill_031_rejects_non_positive_adjacency_boost() -> None:
    """TEST-REFILL-031: RefillConfig rejects adjacency_boost <= 0."""
    with pytest.raises(ConfigValidationError, match="adjacency_boost"):
        RefillConfig(adjacency_boost=0.0, depth_scale=0.3, terminal_max_retries=10)
    with pytest.raises(ConfigValidationError, match="adjacency_boost"):
        RefillConfig(adjacency_boost=-1.0, depth_scale=0.3, terminal_max_retries=10)


def test_refill_032_rejects_invalid_terminal_max_retries() -> None:
    """TEST-REFILL-032: RefillConfig rejects terminal_max_retries < 1."""
    with pytest.raises(ConfigValidationError, match="terminal_max_retries"):
        RefillConfig(adjacency_boost=3.0, depth_scale=0.3, terminal_max_retries=0)


def test_refill_033_missing_section_yields_none(tmp_path: Path) -> None:
    """TEST-REFILL-033: Missing refill section in YAML yields config.refill is None."""
    with open(DEFAULT_CONFIG_PATH, "r") as fh:
        config_data = yaml.safe_load(fh)
    del config_data["refill"]
    config_file = tmp_path / "no_refill.yaml"
    config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)
    assert config.refill is None
