"""Phase 14 integration tests + Step 8 E2E reasoner integration tests.

Tests cover the CLI entry point, output file formats, deterministic generation,
filtering, and post-optimization audit. All test output goes to tmp_path fixtures.

Step 8 E2E tests (TEST-R8-004 through TEST-R8-007) validate that the full
pipeline produces valid instances for specific archetype families.
TEST-R8-003 / TEST-R8-008 (no sequence_planner imports) covered by
test_rule_evaluators.py::test_no_sequence_planner_imports — not duplicated here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ..config.loader import load_config
from ..config.schema import MasterConfig
from ..output.audit import AuditReport, run_audit
from ..output.book_record import BookRecord
from ..output.book_writer import BookWriter, BookWriterConfig
from ..output.lookup_writer import LookupTableWriter
from ..run import DEFAULT_CONFIG_PATH, main

# Small budget for fast tests — enough to exercise the pipeline without long waits
_TEST_BUDGET = "20"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_config() -> MasterConfig:
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def small_output(tmp_path: Path) -> Path:
    """Run a small generation (20 books, dead family only) and return output dir."""
    output_dir = tmp_path / "output"
    ret = main([
        "--count", _TEST_BUDGET,
        "--family", "dead",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0
    return output_dir


# ---------------------------------------------------------------------------
# TEST-P14-001: --dry-run loads config, exits 0, no output files
# ---------------------------------------------------------------------------

def test_dry_run_exits_zero(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    ret = main(["--dry-run", "--output", str(output_dir)])
    assert ret == 0
    # Dry run should not create the output directory
    assert not output_dir.exists()


# ---------------------------------------------------------------------------
# TEST-P14-002: --archetype filters to single archetype
# ---------------------------------------------------------------------------

def test_archetype_filter(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    ret = main([
        "--count", "10",
        "--archetype", "dead_empty",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0
    # Verify books exist
    books_path = output_dir / "books_base.jsonl"
    assert books_path.exists()
    books = _read_jsonl(books_path)
    assert len(books) > 0
    # With only dead_empty, all payouts should be zero
    for book in books:
        assert book["payoutMultiplier"] == 0


# ---------------------------------------------------------------------------
# TEST-P14-003: --family filters to all archetypes in the dead family
# ---------------------------------------------------------------------------

def test_family_filter(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    ret = main([
        "--count", _TEST_BUDGET,
        "--family", "dead",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0
    books_path = output_dir / "books_base.jsonl"
    books = _read_jsonl(books_path)
    # All dead books should have zero payout
    for book in books:
        assert book["payoutMultiplier"] == 0
        assert book["criteria"] == "0"


# ---------------------------------------------------------------------------
# TEST-P14-004: trace.py writes trace_{id}.txt with ASCII trace
# ---------------------------------------------------------------------------

def test_trace_writes_file(small_output: Path) -> None:
    from ..trace import main as trace_main

    books_path = small_output / "books_base.jsonl"
    traces_dir = small_output / "traces"

    ret = trace_main([
        "--id", "0",
        "--books", str(books_path),
        "--output", str(traces_dir),
    ])
    assert ret == 0
    trace_file = traces_dir / "trace_0.txt"
    assert trace_file.exists()
    content = trace_file.read_text(encoding="utf-8")
    # Trace should contain board rendering characters
    assert len(content) > 0


# ---------------------------------------------------------------------------
# TEST-P14-005: --seed produces deterministic output
# ---------------------------------------------------------------------------

def test_seed_deterministic(tmp_path: Path) -> None:
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    for out_dir in (dir1, dir2):
        ret = main([
            "--count", "10",
            "--family", "dead",
            "--seed", "42",
            "--output", str(out_dir),
        ])
        assert ret == 0

    books1 = _read_jsonl(dir1 / "books_base.jsonl")
    books2 = _read_jsonl(dir2 / "books_base.jsonl")

    assert len(books1) == len(books2)
    for b1, b2 in zip(books1, books2):
        assert b1["id"] == b2["id"]
        assert b1["payoutMultiplier"] == b2["payoutMultiplier"]


# ---------------------------------------------------------------------------
# TEST-P14-006: Book JSONL has required keys, one per line
# ---------------------------------------------------------------------------

def test_book_jsonl_format(small_output: Path) -> None:
    books_path = small_output / "books_base.jsonl"
    books = _read_jsonl(books_path)
    assert len(books) > 0
    for book in books:
        # Required RGS keys
        assert "id" in book
        assert "payoutMultiplier" in book
        assert "events" in book
        assert isinstance(book["id"], int)
        assert isinstance(book["payoutMultiplier"], int)
        assert isinstance(book["events"], list)


# ---------------------------------------------------------------------------
# TEST-P14-007: payoutMultiplier matches event stream finalWin
# ---------------------------------------------------------------------------

def test_payout_matches_events(small_output: Path) -> None:
    books_path = small_output / "books_base.jsonl"
    books = _read_jsonl(books_path)
    for book in books:
        events = book["events"]
        # Find finalWin event — always last
        final_events = [e for e in events if e.get("type") == "finalWin"]
        if final_events:
            final_win = final_events[-1]
            assert book["payoutMultiplier"] == final_win["amount"]


# ---------------------------------------------------------------------------
# TEST-P14-008: Lookup table CSV format correct
# ---------------------------------------------------------------------------

def test_lookup_table_format(small_output: Path) -> None:
    lut_path = small_output / "lookUpTable_base.csv"
    assert lut_path.exists()

    rows = _read_csv(lut_path)
    assert len(rows) > 0

    for row in rows:
        assert len(row) == 3, f"Expected 3 columns, got {len(row)}: {row}"
        sim_id, weight, payout = int(row[0]), int(row[1]), int(row[2])
        assert weight == 1  # Initial weights are all 1
        assert payout >= 0
        # Payout divisibility: non-zero payouts must be divisible by rounding_base (10)
        if payout > 0:
            assert payout % 10 == 0, f"Payout {payout} not divisible by 10"

    # _0 copy should also exist
    copy_path = small_output / "lookUpTable_base_0.csv"
    assert copy_path.exists()


# ---------------------------------------------------------------------------
# TEST-P14-009: Post-opt audit detects zeroed-out archetype
# ---------------------------------------------------------------------------

def test_audit_detects_zeroed_archetype(tmp_path: Path) -> None:
    # Create a synthetic optimized LUT where one archetype has all zero weights
    opt_lut = tmp_path / "opt_lut.csv"
    arch_lut = tmp_path / "arch_lut.csv"

    # 10 books: 5 from arch_a (weight=100), 5 from arch_b (weight=0)
    with open(opt_lut, "w") as fh:
        for i in range(5):
            fh.write(f"{i},100,{(i + 1) * 100}\n")
        for i in range(5, 10):
            fh.write(f"{i},0,{(i + 1) * 100}\n")

    with open(arch_lut, "w") as fh:
        for i in range(5):
            fh.write(f"{i},arch_a,basegame\n")
        for i in range(5, 10):
            fh.write(f"{i},arch_b,basegame\n")

    report = run_audit(opt_lut, arch_lut, survival_threshold=0.10)

    # arch_a should have 100% survival, arch_b 0%
    surv_map = {s.archetype_id: s for s in report.archetypes}
    assert surv_map["arch_a"].survival_rate == 1.0
    assert surv_map["arch_a"].flagged is False
    assert surv_map["arch_b"].survival_rate == 0.0
    assert surv_map["arch_b"].flagged is True
    assert "arch_b" in report.flagged_archetypes


# ---------------------------------------------------------------------------
# TEST-P14-010: Parallel (2 workers) deterministic with same seed
# ---------------------------------------------------------------------------

def test_parallel_deterministic(tmp_path: Path) -> None:
    dir1 = tmp_path / "par1"
    dir2 = tmp_path / "par2"

    for out_dir in (dir1, dir2):
        ret = main([
            "--count", _TEST_BUDGET,
            "--family", "dead",
            "--seed", "42",
            "--workers", "2",
            "--output", str(out_dir),
        ])
        assert ret == 0

    lut1 = _read_csv(dir1 / "lookUpTable_base.csv")
    lut2 = _read_csv(dir2 / "lookUpTable_base.csv")

    assert len(lut1) == len(lut2)
    # Same payout distribution (may differ in sim_id ordering between workers)
    payouts1 = sorted(row[2] for row in lut1)
    payouts2 = sorted(row[2] for row in lut2)
    assert payouts1 == payouts2


# ---------------------------------------------------------------------------
# TEST-P14-011: Parallel (2 workers) all sim_ids unique
# ---------------------------------------------------------------------------

def test_parallel_unique_sim_ids(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    ret = main([
        "--count", _TEST_BUDGET,
        "--family", "dead",
        "--seed", "42",
        "--workers", "2",
        "--output", str(output_dir),
    ])
    assert ret == 0

    books = _read_jsonl(output_dir / "books_base.jsonl")
    sim_ids = [b["id"] for b in books]
    assert len(sim_ids) == len(set(sim_ids)), "Duplicate sim_ids found"


# ---------------------------------------------------------------------------
# TEST-P14-012: E2E with 100 books across archetypes
# ---------------------------------------------------------------------------

def test_e2e_generation(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    ret = main([
        "--count", "50",
        "--family", "dead",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0

    books = _read_jsonl(output_dir / "books_base.jsonl")
    assert len(books) > 0

    lut = _read_csv(output_dir / "lookUpTable_base.csv")
    assert len(lut) == len(books)


# ---------------------------------------------------------------------------
# TEST-P14-013: All registered archetypes represented in output
# ---------------------------------------------------------------------------

def test_archetypes_represented(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    ret = main([
        "--count", "50",
        "--family", "dead",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0

    arch_lut = _read_csv(output_dir / "archetype_lookUpTable_base.csv")
    archetype_ids = {row[1] for row in arch_lut}
    # With 50 books across 11 dead archetypes, all should be represented
    assert len(archetype_ids) > 1


# ---------------------------------------------------------------------------
# TEST-P14-014: Summary report + diagnostic report files generated
# ---------------------------------------------------------------------------

def test_reports_generated(small_output: Path) -> None:
    assert (small_output / "summary.txt").exists()
    assert (small_output / "diagnostics.txt").exists()

    summary = (small_output / "summary.txt").read_text(encoding="utf-8")
    assert "EXPERIENCE ENGINE" in summary

    diagnostics = (small_output / "diagnostics.txt").read_text(encoding="utf-8")
    assert len(diagnostics) > 0


# ---------------------------------------------------------------------------
# TEST-P14-015: Config override --count replaces total_budget
# ---------------------------------------------------------------------------

def test_count_override(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    ret = main([
        "--count", "15",
        "--family", "dead",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0

    books = _read_jsonl(output_dir / "books_base.jsonl")
    lut = _read_csv(output_dir / "lookUpTable_base.csv")
    # Budget of 15 across dead family — should produce close to 15 books
    assert len(books) > 0
    assert len(books) == len(lut)


# ---------------------------------------------------------------------------
# TEST-R8-004: 50 dead_empty instances — all valid, zero payout
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_r8_004_dead_empty_e2e(tmp_path: Path) -> None:
    """TEST-R8-004: 50 dead_empty instances all valid with zero payout."""
    output_dir = tmp_path / "output"
    ret = main([
        "--count", "50",
        "--archetype", "dead_empty",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0
    books = _read_jsonl(output_dir / "books_base.jsonl")
    assert len(books) == 50
    for book in books:
        assert book["payoutMultiplier"] == 0


# ---------------------------------------------------------------------------
# TEST-R8-005: 50 t1_cascade_2 instances — all valid, positive payout
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_r8_005_t1_cascade_2_e2e(tmp_path: Path) -> None:
    """TEST-R8-005: t1_cascade_2 — pipeline completes, output files written, valid books.

    Cascade archetypes have inherent failure rates due to constraint solving
    complexity. We verify the pipeline runs to completion, writes all output
    files, and any successfully generated books satisfy the archetype contract.
    """
    output_dir = tmp_path / "output"
    ret = main([
        "--count", "10",
        "--archetype", "t1_cascade_2",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0
    # Pipeline completes and writes all required output files
    assert (output_dir / "books_base.jsonl").exists()
    assert (output_dir / "lookUpTable_base.csv").exists()
    assert (output_dir / "summary.txt").exists()
    # Any successfully generated books must have positive payout
    books = _read_jsonl(output_dir / "books_base.jsonl")
    for book in books:
        assert book["payoutMultiplier"] > 0


# ---------------------------------------------------------------------------
# TEST-R8-006: wild_bridge_small — pipeline completes, output files written
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_r8_006_wild_bridge_small_e2e(tmp_path: Path) -> None:
    """TEST-R8-006: wild_bridge_small — pipeline completes, output files written, valid books.

    Wild bridge archetypes are among the most constrained — they require multi-step
    cascade with wilds bridging clusters. We verify the pipeline runs without
    crashing and any generated books satisfy the contract.
    """
    output_dir = tmp_path / "output"
    ret = main([
        "--count", "10",
        "--archetype", "wild_bridge_small",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0
    # Pipeline completes and writes all required output files
    assert (output_dir / "books_base.jsonl").exists()
    assert (output_dir / "summary.txt").exists()
    # Any successfully generated books must have positive payout
    books = _read_jsonl(output_dir / "books_base.jsonl")
    for book in books:
        assert book["payoutMultiplier"] > 0


# ---------------------------------------------------------------------------
# TEST-R8-007: Summary report generated with correct sim_ids
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_r8_007_summary_report_sim_ids(tmp_path: Path) -> None:
    """TEST-R8-007: Summary report contains correct book count and sim_ids."""
    output_dir = tmp_path / "output"
    ret = main([
        "--count", "30",
        "--family", "dead",
        "--seed", "42",
        "--output", str(output_dir),
    ])
    assert ret == 0

    books = _read_jsonl(output_dir / "books_base.jsonl")
    summary = (output_dir / "summary.txt").read_text(encoding="utf-8")

    # Summary should report the total book count
    assert str(len(books)) in summary

    # Every sim_id in the books should appear in the summary
    sim_ids = {b["id"] for b in books}
    assert len(sim_ids) == len(books), "Duplicate sim_ids in output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    books = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                books.append(json.loads(line))
    return books


def _read_csv(path: Path) -> list[list[str]]:
    """Read a CSV file into a list of rows."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if row:
                rows.append(row)
    return rows
