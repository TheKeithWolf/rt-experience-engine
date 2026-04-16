"""Atlas persistence — hash-invalidated load/save.

Atlas contents depend only on board geometry, gravity rules, and booster
spawn configuration. Other config fields (paytable, symbols, RL parameters)
are irrelevant. AtlasStorage hashes exactly those sections so that
unrelated config tweaks don't invalidate a built atlas.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config.schema import MasterConfig
from .data_types import SpatialAtlas


# Magic header so we can reject pickles from foreign sources cleanly.
_MAGIC = b"RT_ATLAS_V1\n"


@dataclass(frozen=True, slots=True)
class HeaderInfo:
    """Parsed atlas-file header — single source of truth for the file layout.

    magic — the bytes preceding the hash; preserved so callers can verify
      they're looking at an atlas this engine version produced.
    stored_hash — sha256 hex string written by save() at build time.
    payload_offset — byte index where the pickle blob begins; consumers that
      want to bypass hash validation (e.g. inspect on a stale atlas) read
      from here.
    """

    magic: bytes
    stored_hash: str
    payload_offset: int


class AtlasStorage:
    """Serialize SpatialAtlas to disk with a config-hash header.

    The header is written before the pickle blob. load() reads the header,
    compares it to the hash of the supplied config, and returns None on
    mismatch — forcing a rebuild without raising.
    """

    def save(
        self, atlas: SpatialAtlas, config: MasterConfig, path: Path
    ) -> None:
        """Write the atlas bytes to `path`, creating parent dirs as needed.

        The file is overwritten if it exists; callers are expected to treat
        the atlas file as build output, not versioned data.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(atlas, protocol=pickle.HIGHEST_PROTOCOL)
        header = _MAGIC + self.config_hash(config).encode("ascii") + b"\n"
        path.write_bytes(header + payload)

    def load(self, config: MasterConfig, path: Path) -> SpatialAtlas | None:
        """Return the stored atlas if the header matches, else None.

        Composes peek_header + payload deserialization — the file format
        knowledge lives entirely in peek_header, so any future format change
        only touches one place.
        """
        header = self.peek_header(path)
        if header is None or header.stored_hash != self.config_hash(config):
            return None
        return pickle.loads(path.read_bytes()[header.payload_offset:])

    def peek_header(self, path: Path) -> HeaderInfo | None:
        """Parse the atlas-file header without deserializing the pickle.

        Returns None for missing file, wrong magic, or truncated header.
        Used by load() (for the hash check) and by inspect_cli (for header
        display + raw-pickle fallback when the hash has moved). One parser,
        two consumers — no second copy of the file layout in any caller.
        """
        if not path.is_file():
            return None
        blob = path.read_bytes()
        if not blob.startswith(_MAGIC):
            return None
        remainder = blob[len(_MAGIC):]
        newline = remainder.find(b"\n")
        if newline < 0:
            return None
        stored_hash = remainder[:newline].decode("ascii")
        return HeaderInfo(
            magic=_MAGIC,
            stored_hash=stored_hash,
            payload_offset=len(_MAGIC) + newline + 1,
        )

    @staticmethod
    def default_path(config: MasterConfig, config_path: Path) -> Path:
        """Resolve the atlas file location from the loaded config.

        config.atlas.path is interpreted relative to the directory containing
        the YAML config; that's how the build CLI persists today and how the
        generator expects to find the atlas at startup. Centralizing the
        join here keeps build, inspect, and any future tooling in lockstep.
        """
        if config.atlas is None:
            raise ValueError(
                "config has no atlas section — cannot resolve a default path"
            )
        return config_path.parent / config.atlas.path

    @staticmethod
    def config_hash(config: MasterConfig) -> str:
        """Digest of the config fields that influence atlas content.

        Only board, gravity, and boosters contribute — changing paytable or
        any cosmetic field must not trigger a rebuild.
        """
        relevant = {
            "board": asdict(config.board),
            "gravity": _gravity_signature(config.gravity),
            "boosters": _booster_signature(config.boosters),
            "atlas": _atlas_signature(config.atlas),
        }
        canonical = json.dumps(relevant, sort_keys=True, default=_default)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _gravity_signature(gravity) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Gravity config contributes via donor priorities and settle cap.

    max_settle_passes affects how far the simulation propagates; different
    values can produce different topologies for edge-case inputs.
    """
    return {
        "donor_priorities": [list(pair) for pair in gravity.donor_priorities],
        "max_settle_passes": gravity.max_settle_passes,
    }


def _booster_signature(boosters) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Booster config contributes spawn thresholds (cluster-size -> booster)
    plus rocket/bomb rules — anything that changes landing or fire semantics.
    """
    return {
        "spawn_thresholds": [
            {"booster": t.booster, "min": t.min_size, "max": t.max_size}
            for t in boosters.spawn_thresholds
        ],
        "spawn_order": list(boosters.spawn_order),
        "rocket_tie_orientation": boosters.rocket_tie_orientation,
        "bomb_blast_radius": boosters.bomb_blast_radius,
        "chain_initiators": list(boosters.chain_initiators),
        "immune_to_rocket": list(boosters.immune_to_rocket),
        "immune_to_bomb": list(boosters.immune_to_bomb),
    }


def _atlas_signature(atlas) -> dict[str, object] | None:  # type: ignore[no-untyped-def]
    """Depth bands affect how ColumnProfile keys are carved, so they belong
    to the hash even though the atlas section itself is optional."""
    if atlas is None:
        return None
    return {
        "depth_bands": [
            {"name": band.name, "min": band.min_row, "max": band.max_row}
            for band in atlas.depth_bands
        ],
    }


def _default(obj: object) -> object:
    """JSON fallback for tuples/frozensets — converts them to lists.

    The digest is deterministic because asdict flattens dataclasses and
    json.dumps(sort_keys=True) fixes key ordering.
    """
    if isinstance(obj, (tuple, frozenset, set)):
        return sorted(obj) if isinstance(obj, (frozenset, set)) else list(obj)
    raise TypeError(f"Cannot serialize {type(obj).__name__} for config hash")
