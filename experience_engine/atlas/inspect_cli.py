"""Inspect a serialized atlas file.

Prints the header (magic + stored hash + current hash + match status) and a
per-map entry-count summary. Optional `--sample N` emits N entries from each
map for spot-checking.

Composes `AtlasStorage.peek_header` + `AtlasStorage.load` — no second copy
of the file format. The single `MAP_SUMMARIES` registry drives every loop
so adding a new SpatialAtlas map means appending one row, not editing this
file's logic.
"""

from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path

from ..config.loader import load_config
from .data_types import SpatialAtlas
from .storage import AtlasStorage, HeaderInfo


# Registry of (display label, SpatialAtlas attribute name).
# Single source of truth for both count and sample iteration.
# Adding a new map = append one row; no other code in this file changes.
MAP_SUMMARIES: tuple[tuple[str, str], ...] = (
    ("topologies",         "topologies"),
    ("booster_landings",   "booster_landings"),
    ("arm_adjacencies",    "arm_adjacencies"),
    ("fire_zones",         "fire_zones"),
    ("dormant_survivals",  "dormant_survivals"),
)


def inspect_main(args: argparse.Namespace) -> int:
    """Inspect entrypoint dispatched by build_cli.main's subparser.

    Always returns 0 — inspection is read-only. Missing or corrupt files
    emit a notice but don't exit non-zero so wrapping shells can still
    chain commands without special-casing the inspect step.
    """
    config = load_config(args.config)
    path = (
        Path(args.path) if args.path is not None
        else AtlasStorage.default_path(config, args.config)
    )
    print(f"atlas: {path}")

    storage = AtlasStorage()
    header = storage.peek_header(path)
    if header is None:
        print(f"  no readable atlas at {path}")
        return 0

    current_hash = AtlasStorage.config_hash(config)
    _print_header(header, current_hash)

    atlas = _load_with_fallback(storage, header, config, path)
    _print_summary(atlas, args.sample)
    return 0


def _print_header(header: HeaderInfo, current_hash: str) -> None:
    """Header lines — derived entirely from HeaderInfo + current config hash."""
    print(f"  magic:        {header.magic!r}")
    print(f"  stored hash:  {header.stored_hash}")
    print(f"  current hash: {current_hash}")
    status = "match" if header.stored_hash == current_hash else "mismatch"
    print(f"  hash status:  {status}")


def _load_with_fallback(
    storage: AtlasStorage,
    header: HeaderInfo,
    config,
    path: Path,
) -> SpatialAtlas:
    """Return the deserialized atlas regardless of hash status.

    On a hash match, AtlasStorage.load handles the deserialization. On a
    mismatch, load returns None — but inspection is still useful, so we
    bypass the hash check by reading the pickle blob directly using the
    payload offset that peek_header already computed (single source of
    truth for the file layout).
    """
    atlas = storage.load(config, path)
    if atlas is not None:
        return atlas
    return pickle.loads(path.read_bytes()[header.payload_offset:])


def _print_summary(atlas: SpatialAtlas, sample: int) -> None:
    """One pass over MAP_SUMMARIES for counts; a second pass for samples
    only when the user requested any. No per-map branching — every map
    flows through the same accessor and formatter."""
    print("maps:")
    for label, attr in MAP_SUMMARIES:
        items = getattr(atlas, attr)
        print(f"  {label:<20} {len(items):>10,}")

    if sample <= 0:
        return
    print(f"samples (first {sample}):")
    for label, attr in MAP_SUMMARIES:
        items = getattr(atlas, attr)
        print(f"  {label}:")
        for key, value in itertools.islice(items.items(), sample):
            print(f"    {key!r} -> {value!r}")
