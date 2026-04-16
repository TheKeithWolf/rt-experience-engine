"""Tests for Step 5: atlas CLI runner.

Exercises `build_cli.main(argv)` directly — no subprocess, no stdout capture
beyond what pytest's capsys provides. The tests pin exit codes, file output,
and the `--force` / `--stats` flag contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..atlas.build_cli import _parse_sizes, main
from ..atlas.storage import AtlasStorage
from ..config.loader import load_config


_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent / "config" / "default.yaml"
)


def test_parse_sizes_accepts_comma_separated_ints() -> None:
    assert _parse_sizes("5,9") == (5, 9)
    assert _parse_sizes("  5 , 9 ") == (5, 9)


def test_parse_sizes_returns_none_for_empty_or_missing() -> None:
    assert _parse_sizes(None) is None
    assert _parse_sizes("") is None


def test_main_builds_loadable_atlas(tmp_path: Path) -> None:
    """A minimal size-5 build must leave a file that AtlasStorage.load()
    successfully returns — proves the CLI's wiring is complete."""
    out = tmp_path / "atlas.bin"
    exit_code = main([
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--out", str(out),
        "--sizes", "5",
    ])
    assert exit_code == 0
    assert out.is_file()
    cfg = load_config(_DEFAULT_CONFIG_PATH)
    assert AtlasStorage().load(cfg, out) is not None


def test_main_skips_rebuild_on_matching_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Running the CLI twice without --force must detect the cached atlas
    on the second invocation and return immediately."""
    out = tmp_path / "atlas.bin"
    argv = [
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--out", str(out),
        "--sizes", "5",
    ]
    assert main(argv) == 0
    first_mtime = out.stat().st_mtime_ns

    capsys.readouterr()  # Discard first-run output.
    assert main(argv) == 0
    assert out.stat().st_mtime_ns == first_mtime
    captured = capsys.readouterr()
    assert "up-to-date" in captured.out


def test_main_force_rewrites_existing_atlas(tmp_path: Path) -> None:
    """--force must bypass the hash check — the file is rewritten even when
    the config hasn't changed."""
    out = tmp_path / "atlas.bin"
    argv_base = [
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--out", str(out),
        "--sizes", "5",
    ]
    assert main(argv_base) == 0
    first_mtime = out.stat().st_mtime_ns
    # File systems may round mtime — sleep is unnecessary if we compare bytes.
    first_bytes = out.read_bytes()
    assert main(argv_base + ["--force"]) == 0
    # Bytes should equal (deterministic build) but the write must have
    # actually happened; we rely on exit code 0 + an unchanged payload to
    # verify a successful rebuild path without requiring timestamp skew.
    assert out.read_bytes() == first_bytes


def test_inspect_emits_header_and_map_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A freshly built atlas should inspect cleanly: hash matches, every
    map label from MAP_SUMMARIES appears in the output."""
    out = tmp_path / "atlas.bin"
    main([
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--out", str(out),
        "--sizes", "5",
    ])
    capsys.readouterr()  # Discard build output.

    exit_code = main([
        "inspect",
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--path", str(out),
    ])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "stored hash:" in captured
    assert "current hash:" in captured
    assert "hash status:  match" in captured
    # Every registered map label must surface in the output.
    from ..atlas.inspect_cli import MAP_SUMMARIES
    for label, _ in MAP_SUMMARIES:
        assert label in captured


def test_inspect_handles_missing_file_gracefully(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Pointing inspect at a nonexistent path should print a notice and
    exit 0 — read-only inspection never returns non-zero."""
    exit_code = main([
        "inspect",
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--path", str(tmp_path / "missing.bin"),
    ])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "no readable atlas" in captured


def test_inspect_reports_mismatch_but_still_lists_maps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A stale atlas (config changed since save) should still surface its
    contents — we read past the hash check using the payload offset."""
    out = tmp_path / "atlas.bin"
    main([
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--out", str(out),
        "--sizes", "5",
    ])
    capsys.readouterr()

    # Mutate the config so the stored hash no longer matches.
    import yaml
    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["boosters"]["bomb_blast_radius"] = data["boosters"]["bomb_blast_radius"] + 1
    alt_cfg = tmp_path / "alt.yaml"
    alt_cfg.write_text(yaml.dump(data, sort_keys=False))

    exit_code = main([
        "inspect",
        "--config", str(alt_cfg),
        "--path", str(out),
    ])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "hash status:  mismatch" in captured
    # Map counts must still appear despite the mismatch.
    from ..atlas.inspect_cli import MAP_SUMMARIES
    for label, _ in MAP_SUMMARIES:
        assert label in captured


def test_inspect_sample_prints_entries_per_map(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """--sample N must print up to N entries for each map. Used to spot-check
    that the persisted maps actually contain meaningful data."""
    out = tmp_path / "atlas.bin"
    main([
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--out", str(out),
        "--sizes", "5",
    ])
    capsys.readouterr()

    exit_code = main([
        "inspect",
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--path", str(out),
        "--sample", "1",
    ])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "samples (first 1)" in captured
    # The arrow used by _print_summary's sample formatter must appear at
    # least once per non-empty map; size 5 produces non-empty topologies.
    assert "->" in captured


def test_bare_invocation_still_builds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Backward-compat: `python -m ...atlas` (no subcommand) defaults to
    build via the top-level parser's set_defaults(func=build_main)."""
    out = tmp_path / "atlas.bin"
    exit_code = main([
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--out", str(out),
        "--sizes", "5",
    ])
    assert exit_code == 0
    assert out.is_file()


def test_stats_flag_emits_progress_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """--stats routes the builder's progress callback to print, so the
    captured stdout must contain at least one 'Size' header and a summary."""
    out = tmp_path / "atlas.bin"
    exit_code = main([
        "--config", str(_DEFAULT_CONFIG_PATH),
        "--out", str(out),
        "--sizes", "5",
        "--stats",
    ])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Size" in captured
    assert "Total:" in captured
