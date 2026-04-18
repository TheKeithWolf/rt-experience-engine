"""Config loading, validation, and normalization tests.

TEST-P1-001 through TEST-P1-005: Phase 1 config tests.
TEST-R8-001, TEST-R8-002: Step 8 reasoner config tests.
TEST-REFILL-030 through TEST-REFILL-033: Refill strategy config tests.
"""

from pathlib import Path

import pytest
import yaml

from ..config.loader import load_config
from ..config.schema import ConfigValidationError, MasterConfig, RefillConfig, SolverConfig

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


def test_solver_config_loads_optional_retry_fields(default_config: MasterConfig) -> None:
    """Solver tuning fields in default.yaml must override dataclass defaults."""
    solvers = default_config.solvers
    assert solvers.max_validation_retries == 250
    assert solvers.max_seed_retries == 50
    assert solvers.board_adjacency_max_step == 2
    assert solvers.multi_seed_threshold == 11
    assert solvers.multi_seed_count == 3


def test_solver_config_honors_yaml_overrides(tmp_path: Path) -> None:
    """Non-default solver retry and seed values should round-trip from YAML."""
    with open(DEFAULT_CONFIG_PATH, "r") as fh:
        config_data = yaml.safe_load(fh)

    config_data["solvers"]["max_validation_retries"] = 7
    config_data["solvers"]["board_adjacency_max_step"] = 4
    config_data["solvers"]["max_seed_retries"] = 9
    config_data["solvers"]["multi_seed_threshold"] = 13
    config_data["solvers"]["multi_seed_count"] = 5

    config_file = tmp_path / "solver_overrides.yaml"
    config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)
    assert config.solvers.max_validation_retries == 7
    assert config.solvers.board_adjacency_max_step == 4
    assert config.solvers.max_seed_retries == 9
    assert config.solvers.multi_seed_threshold == 13
    assert config.solvers.multi_seed_count == 5


def test_solver_config_uses_defaults_when_optional_fields_missing(tmp_path: Path) -> None:
    """Older configs without new optional solver fields keep schema defaults."""
    with open(DEFAULT_CONFIG_PATH, "r") as fh:
        config_data = yaml.safe_load(fh)

    for key in (
        "max_validation_retries",
        "board_adjacency_max_step",
        "max_seed_retries",
        "multi_seed_threshold",
        "multi_seed_count",
    ):
        del config_data["solvers"][key]

    config_file = tmp_path / "old_solver_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)
    assert config.solvers.max_validation_retries == 1
    assert config.solvers.board_adjacency_max_step == 2
    assert config.solvers.max_seed_retries == 5
    assert config.solvers.multi_seed_threshold == 11
    assert config.solvers.multi_seed_count == 3


def _valid_solver_config(**overrides: object) -> SolverConfig:
    data = {
        "wfc_max_backtracks": 500,
        "wfc_min_symbol_weight": 0.01,
        "csp_max_solve_time_ms": 5000,
        "asp_max_models": 10,
        "asp_rand_freq": 0.5,
        "max_retries_per_instance": 150,
        "max_construction_retries": 100,
        "max_validation_retries": 250,
        "board_adjacency_max_step": 2,
        "max_seed_retries": 50,
        "multi_seed_threshold": 11,
        "multi_seed_count": 3,
    }
    data.update(overrides)
    return SolverConfig(**data)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("max_retries_per_instance", 0),
        ("max_construction_retries", 0),
        ("max_validation_retries", 0),
        ("board_adjacency_max_step", -1),
        ("max_seed_retries", 0),
    ),
)
def test_solver_config_rejects_invalid_retry_values(
    field: str,
    bad_value: int,
) -> None:
    """Retry and adjacency budgets should fail fast when misconfigured."""
    with pytest.raises(ConfigValidationError, match=field):
        _valid_solver_config(**{field: bad_value})


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
    assert r.lookahead_depth == 3


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
    assert default_config.refill.depth_scale == 0.5
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
