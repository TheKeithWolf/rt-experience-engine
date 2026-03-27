"""Archive I/O — save and load MAP-Elites archives as JSONL files.

Follows the stream-write pattern from output/book_writer.py: one JSON line
per archive entry, compact separators. One JSONL file per archetype.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..archetypes.registry import ArchetypeRegistry
from ..config.schema import DescriptorConfig
from .archive import ArchiveEntry, MAPElitesArchive
from .descriptor import TrajectoryDescriptor


def save_archive(archive: MAPElitesArchive, path: Path) -> None:
    """Save a MAP-Elites archive to a JSONL file.

    One JSON line per ArchiveEntry. Uses compact separators to minimize
    file size, following the book_writer.py pattern.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for key in sorted(archive.occupied_keys()):
            entry = archive.get(key)
            if entry is None:
                continue
            line = json.dumps(
                _serialize_entry(entry), separators=(",", ":"),
            )
            fh.write(line)
            fh.write("\n")


def load_archive(
    path: Path,
    descriptor_config: DescriptorConfig,
    num_symbols: int,
) -> MAPElitesArchive:
    """Load a MAP-Elites archive from a JSONL file.

    Each line is deserialized into an ArchiveEntry and inserted via
    try_insert(). Returns a fully populated archive.
    """
    archive = MAPElitesArchive(descriptor_config, num_symbols)

    if not path.exists():
        return archive

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entry = _deserialize_entry(data)
            archive.try_insert(entry.instance, entry.descriptor, entry.quality)

    return archive


def load_archives(
    archive_dir: Path,
    registry: ArchetypeRegistry,
    descriptor_config: DescriptorConfig,
    num_symbols: int,
) -> dict[str, MAPElitesArchive]:
    """Load all archive files from a directory.

    Globs *.jsonl, uses stem as archetype_id, filters by registry.
    """
    archives: dict[str, MAPElitesArchive] = {}
    if not archive_dir.exists():
        return archives

    registered_ids = registry.all_ids()
    for path in sorted(archive_dir.glob("*.jsonl")):
        archetype_id = path.stem
        if archetype_id in registered_ids:
            archives[archetype_id] = load_archive(
                path, descriptor_config, num_symbols,
            )

    return archives


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_entry(entry: ArchiveEntry) -> dict[str, Any]:
    """Serialize an ArchiveEntry to a JSON-compatible dict."""
    return {
        "descriptor": _serialize_descriptor(entry.descriptor),
        "quality": entry.quality,
        "instance": _serialize_instance_summary(entry.instance),
    }


def _serialize_descriptor(desc: TrajectoryDescriptor) -> dict[str, Any]:
    return {
        "archetype_id": desc.archetype_id,
        "step0_symbol": desc.step0_symbol,
        "spatial_bin": list(desc.spatial_bin),
        "cluster_orientation": desc.cluster_orientation,
        "payout_bin": desc.payout_bin,
    }


def _serialize_instance_summary(instance: Any) -> dict[str, Any]:
    """Serialize the essential fields of a GeneratedInstance.

    Stores enough to reconstruct the instance for population sampling.
    Board state is stored as a flat list of symbol names.
    """
    return {
        "sim_id": instance.sim_id,
        "archetype_id": instance.archetype_id,
        "family": instance.family,
        "criteria": instance.criteria,
        "payout": instance.payout,
        "centipayout": instance.centipayout,
        "win_level": instance.win_level,
    }


def _deserialize_entry(data: dict[str, Any]) -> ArchiveEntry:
    """Deserialize a JSON dict into an ArchiveEntry.

    The instance field stores a summary dict — the full GeneratedInstance
    is reconstructed by the generator when sampling from the archive.
    For archive health reporting and quality tracking, we store a
    lightweight placeholder.
    """
    desc_data = data["descriptor"]
    descriptor = TrajectoryDescriptor(
        archetype_id=desc_data["archetype_id"],
        step0_symbol=desc_data["step0_symbol"],
        spatial_bin=tuple(desc_data["spatial_bin"]),
        cluster_orientation=desc_data["cluster_orientation"],
        payout_bin=desc_data["payout_bin"],
    )

    # Store the raw instance dict as-is — the generator will handle
    # full reconstruction when needed. For archive operations (quality
    # comparison, coverage), the descriptor and quality are sufficient.
    return ArchiveEntry(
        instance=data["instance"],  # type: ignore[arg-type]
        descriptor=descriptor,
        quality=data["quality"],
    )
