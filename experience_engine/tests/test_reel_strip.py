"""Reel strip data model tests (TEST-REEL-001 through TEST-REEL-009).

Covers load_reel_strip CSV parsing + validation and read_circular circular
indexing. All tests use the reference 9x7 CSV shipped in experience_engine/data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..config.schema import BoardConfig, ConfigValidationError
from ..primitives.reel_strip import ReelStrip, load_reel_strip, read_circular
from ..primitives.symbols import Symbol

# Board matches default.yaml — 7 reels, 7 rows, min_cluster=5
BOARD_CONFIG = BoardConfig(num_reels=7, num_rows=7, min_cluster_size=5)

# Reference CSV location — single source of truth; tests read the same file
# that production will load.
REFERENCE_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "reel_strip.csv"
)


def _write_csv(tmp_path: Path, rows: list[list[str]]) -> Path:
    """Helper: write a CSV with the given rows and return its path."""
    csv_path = tmp_path / "strip.csv"
    csv_path.write_text("\n".join(",".join(row) for row in rows) + "\n")
    return csv_path


# ---------------------------------------------------------------------------
# load_reel_strip
# ---------------------------------------------------------------------------

class TestLoadReelStrip:

    def test_reel_001_parses_reference_csv(self) -> None:
        """TEST-REEL-001: reference CSV → num_reels=7, strip_length=9."""
        strip = load_reel_strip(REFERENCE_CSV, BOARD_CONFIG)
        assert strip.num_reels == 7
        assert strip.strip_length == 9
        assert len(strip.columns) == 7
        assert all(len(col) == 9 for col in strip.columns)

    def test_reel_009_all_symbols_resolve(self) -> None:
        """TEST-REEL-009: every cell resolves to a valid Symbol member."""
        strip = load_reel_strip(REFERENCE_CSV, BOARD_CONFIG)
        for column in strip.columns:
            for sym in column:
                assert isinstance(sym, Symbol)

    def test_reel_006_wrong_column_count_rejected(self, tmp_path: Path) -> None:
        """TEST-REEL-006: CSV with wrong column count → ConfigValidationError."""
        csv_path = _write_csv(tmp_path, [
            ["L1", "L2", "L3"],  # only 3 columns; board expects 7
        ])
        with pytest.raises(ConfigValidationError):
            load_reel_strip(csv_path, BOARD_CONFIG)

    def test_reel_007_unknown_symbol_rejected(self, tmp_path: Path) -> None:
        """TEST-REEL-007: CSV with unknown symbol name → ValueError."""
        csv_path = _write_csv(tmp_path, [
            ["L1", "L2", "L3", "L4", "H1", "H2", "ZZ"],  # ZZ invalid
        ] * 7)
        with pytest.raises(ValueError):
            load_reel_strip(csv_path, BOARD_CONFIG)

    def test_reel_008_strip_shorter_than_num_rows_rejected(
        self, tmp_path: Path,
    ) -> None:
        """TEST-REEL-008: strip_length < num_rows → ConfigValidationError."""
        # 3 rows × 7 columns; board requires >= 7 rows deep
        rows = [["L1", "L2", "L3", "L4", "H1", "H2", "H3"] for _ in range(3)]
        csv_path = _write_csv(tmp_path, rows)
        with pytest.raises(ConfigValidationError):
            load_reel_strip(csv_path, BOARD_CONFIG)

    def test_ragged_columns_rejected(self, tmp_path: Path) -> None:
        """Uneven row widths surface as column-count errors per row."""
        rows = [
            ["L1", "L2", "L3", "L4", "H1", "H2", "H3"],
            ["L1", "L2", "L3", "L4", "H1", "H2"],  # short row
        ]
        csv_path = _write_csv(tmp_path, rows)
        with pytest.raises(ConfigValidationError):
            load_reel_strip(csv_path, BOARD_CONFIG)


# ---------------------------------------------------------------------------
# read_circular
# ---------------------------------------------------------------------------

class TestReadCircular:

    def _strip(self) -> ReelStrip:
        return load_reel_strip(REFERENCE_CSV, BOARD_CONFIG)

    def test_reel_002_no_wrap_first_seven(self) -> None:
        """TEST-REEL-002: start=0, count=7 → first 7 symbols verbatim."""
        strip = self._strip()
        got = read_circular(strip, reel=0, start=0, count=7)
        expected = tuple(strip.columns[0][:7])
        assert got == expected

    def test_reel_003_wraps_from_stop_seven(self) -> None:
        """TEST-REEL-003: start=7, count=7 → indices 7,8,0,1,2,3,4."""
        strip = self._strip()
        got = read_circular(strip, reel=0, start=7, count=7)
        col = strip.columns[0]
        expected = (col[7], col[8], col[0], col[1], col[2], col[3], col[4])
        assert got == expected

    def test_reel_004_full_length_exact(self) -> None:
        """TEST-REEL-004: start=0, count=strip_length → full column."""
        strip = self._strip()
        got = read_circular(strip, reel=0, start=0, count=strip.strip_length)
        assert got == strip.columns[0]

    def test_reel_005_partial_midrange(self) -> None:
        """TEST-REEL-005: start=5, count=2 → indices 5,6 (no wrap)."""
        strip = self._strip()
        got = read_circular(strip, reel=0, start=5, count=2)
        col = strip.columns[0]
        assert got == (col[5], col[6])

    def test_negative_start_wraps(self) -> None:
        """Cursors advanced upward produce negative starts — must wrap."""
        strip = self._strip()
        # start = -1 → equivalent to strip_length - 1
        got = read_circular(strip, reel=0, start=-1, count=2)
        col = strip.columns[0]
        assert got == (col[strip.strip_length - 1], col[0])
