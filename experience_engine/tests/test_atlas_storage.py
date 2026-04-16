"""Tests for the storage refactor — peek_header + default_path.

Both surface area was added by the inspect-command work; load() already had
coverage indirectly via the builder tests, so these tests focus on the new
functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..atlas.builder import build_atlas_services
from ..atlas.storage import AtlasStorage, HeaderInfo
from ..config.schema import MasterConfig


def _saved_atlas_path(
    tmp_path: Path, default_config: MasterConfig
) -> Path:
    """Build the smallest viable atlas + persist it. Used by every header
    parse test that needs a real on-disk file."""
    out = tmp_path / "atlas.bin"
    atlas = build_atlas_services(default_config).builder.build(sizes=(5,))
    AtlasStorage().save(atlas, default_config, out)
    return out


def test_peek_header_returns_header_info_for_saved_atlas(
    tmp_path: Path, default_config: MasterConfig,
) -> None:
    path = _saved_atlas_path(tmp_path, default_config)
    header = AtlasStorage().peek_header(path)
    assert isinstance(header, HeaderInfo)
    assert header.stored_hash == AtlasStorage.config_hash(default_config)
    # Payload offset must point at the byte after the hash newline.
    raw = path.read_bytes()
    assert raw[header.payload_offset - 1:header.payload_offset] == b"\n"


def test_peek_header_returns_none_for_missing_file(
    tmp_path: Path,
) -> None:
    assert AtlasStorage().peek_header(tmp_path / "nope.bin") is None


def test_peek_header_returns_none_for_corrupted_magic(
    tmp_path: Path,
) -> None:
    """A file whose first bytes don't match the magic must fail soft —
    callers (load(), inspect) treat None as "no readable atlas"."""
    path = tmp_path / "bad.bin"
    path.write_bytes(b"not an atlas at all")
    assert AtlasStorage().peek_header(path) is None


def test_peek_header_returns_none_for_truncated_header(
    tmp_path: Path,
) -> None:
    """Magic present but no newline terminator on the hash line — the
    parser must reject without raising."""
    path = tmp_path / "trunc.bin"
    path.write_bytes(b"RT_ATLAS_V1\nabcdef")  # No closing newline after hash.
    assert AtlasStorage().peek_header(path) is None


def test_default_path_joins_under_config_directory(
    default_config: MasterConfig,
) -> None:
    """default_path must resolve atlas.path relative to the config file's
    parent — that's how the build CLI persists today and how the generator
    expects to find the atlas at startup."""
    config_path = Path("/some/dir/default.yaml")
    expected = config_path.parent / default_config.atlas.path
    assert AtlasStorage.default_path(default_config, config_path) == expected


def test_default_path_raises_when_atlas_section_missing(
    default_config: MasterConfig,
) -> None:
    """An atlas-less config must raise — silent default would point at a
    non-existent file and surprise the caller."""
    from dataclasses import replace
    cfg = replace(default_config, atlas=None)
    with pytest.raises(ValueError, match="atlas section"):
        AtlasStorage.default_path(cfg, Path("/tmp/cfg.yaml"))
