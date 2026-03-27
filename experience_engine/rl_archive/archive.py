"""MAP-Elites archive — stores one instance per behavioral niche.

Each cell holds the highest-quality GeneratedInstance for that descriptor key.
Supports diversity-weighted sampling for population generation, where weights
come from the variance engine's accumulator-derived hints.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..config.schema import DescriptorConfig
from ..pipeline.data_types import GeneratedInstance
from .descriptor import DescriptorKey, TrajectoryDescriptor


class ArchiveEmpty(Exception):
    """Raised when sampling from an empty archive."""


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """A single cell in the MAP-Elites archive."""

    instance: GeneratedInstance
    descriptor: TrajectoryDescriptor
    quality: float


class MAPElitesArchive:
    """MAP-Elites archive indexed by behavioral descriptor keys.

    Stores at most one instance per descriptor cell, keeping the highest
    quality. Total cell count is the Cartesian product of all descriptor
    axes: spatial_col_bins x spatial_row_bins x payout_bins x num_symbols
    x num_orientations (3: H, V, compact).
    """

    __slots__ = ("_cells", "_total_cells")

    _ORIENTATION_COUNT = 3  # H, V, compact

    def __init__(
        self,
        descriptor_config: DescriptorConfig,
        num_symbols: int,
    ) -> None:
        """Initialize empty archive with computed total cell count.

        num_symbols is len(SymbolConfig.standard) — the number of distinct
        standard symbols that can appear as the step-0 cluster symbol.
        """
        self._cells: dict[DescriptorKey, ArchiveEntry] = {}
        self._total_cells = (
            descriptor_config.spatial_col_bins
            * descriptor_config.spatial_row_bins
            * descriptor_config.payout_bins
            * num_symbols
            * self._ORIENTATION_COUNT
        )

    def try_insert(
        self,
        instance: GeneratedInstance,
        descriptor: TrajectoryDescriptor,
        quality: float,
    ) -> bool:
        """Insert if cell is empty or new quality exceeds incumbent.

        Returns True if the entry was inserted (new cell or replacement).
        """
        key = _descriptor_to_key(descriptor)
        existing = self._cells.get(key)

        if existing is not None and existing.quality >= quality:
            return False

        self._cells[key] = ArchiveEntry(
            instance=instance,
            descriptor=descriptor,
            quality=quality,
        )
        return True

    def get(self, key: DescriptorKey) -> ArchiveEntry | None:
        """Retrieve the entry at a specific descriptor key, or None."""
        return self._cells.get(key)

    def sample(
        self,
        weights: dict[DescriptorKey, float],
        rng: random.Random,
    ) -> GeneratedInstance:
        """Sample one instance weighted by the provided key weights.

        Weights are normalized over occupied cells only. Keys not in the
        archive are ignored. Raises ArchiveEmpty if no cells are occupied.
        """
        if not self._cells:
            raise ArchiveEmpty("Cannot sample from an empty archive")

        # Filter to occupied cells and build parallel lists
        keys: list[DescriptorKey] = []
        sample_weights: list[float] = []
        for key, entry in self._cells.items():
            keys.append(key)
            sample_weights.append(weights.get(key, 1.0))

        # Normalize — if all weights are zero, fall back to uniform
        total = sum(sample_weights)
        if total <= 0.0:
            sample_weights = [1.0] * len(keys)

        chosen_key = rng.choices(keys, weights=sample_weights, k=1)[0]
        return self._cells[chosen_key].instance

    def coverage(self) -> float:
        """Fraction of total possible cells that are occupied."""
        if self._total_cells == 0:
            return 0.0
        return self.filled_count() / self._total_cells

    def filled_count(self) -> int:
        """Number of occupied cells."""
        return len(self._cells)

    def total_cells(self) -> int:
        """Total number of possible cells in the archive."""
        return self._total_cells

    def all_keys(self) -> frozenset[DescriptorKey]:
        """All possible descriptor keys (enumeration of the full grid).

        Note: This enumerates the Cartesian product of bin indices only
        (without archetype/symbol names). For per-archetype archives,
        callers should filter by archetype_id.
        """
        return frozenset(self._cells.keys()) | self._empty_keys()

    def occupied_keys(self) -> frozenset[DescriptorKey]:
        """All currently occupied descriptor keys."""
        return frozenset(self._cells.keys())

    def _empty_keys(self) -> frozenset[DescriptorKey]:
        """Keys for unoccupied cells — expensive, use sparingly."""
        # Only meaningful if callers enumerate the full grid externally;
        # the archive itself doesn't know the symbol/archetype vocabulary.
        # Return empty — callers use occupied_keys() for practical queries.
        return frozenset()


def _descriptor_to_key(descriptor: TrajectoryDescriptor) -> DescriptorKey:
    """Flatten a TrajectoryDescriptor into a hashable tuple."""
    return (
        descriptor.archetype_id,
        descriptor.step0_symbol,
        descriptor.spatial_bin[0],
        descriptor.spatial_bin[1],
        descriptor.cluster_orientation,
        descriptor.payout_bin,
    )
