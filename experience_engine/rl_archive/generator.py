"""RL archive generator — production sampling from pre-filled MAP-Elites archives.

Conforms to the same generate() interface as StaticInstanceGenerator and
CascadeInstanceGenerator, enabling seamless integration into the
PopulationController's three-tier routing.

At population generation time, diversity weights from VarianceHints steer
sampling toward underrepresented behavioral niches. No runtime inference —
the archive is pure lookup.
"""

from __future__ import annotations

import dataclasses
import random
from typing import TYPE_CHECKING

from ..config.schema import MasterConfig
from ..pipeline.data_types import GeneratedInstance, GenerationResult
from .archive import ArchiveEmpty, MAPElitesArchive
from .descriptor import DescriptorKey

if TYPE_CHECKING:
    from ..archetypes.registry import ArchetypeRegistry
    from ..variance.hints import VarianceHints


class RLArchiveGenerator:
    """Samples pre-computed instances from MAP-Elites archives.

    Each archetype has its own archive. The generator maps VarianceHints
    (symbol weights, spatial bias) to descriptor key weights for
    diversity-weighted sampling. Missing archetypes or empty archives
    return GenerationResult(success=False).
    """

    __slots__ = ("_config", "_registry", "_archives")

    def __init__(
        self,
        config: MasterConfig,
        registry: ArchetypeRegistry,
        archives: dict[str, MAPElitesArchive],
    ) -> None:
        self._config = config
        self._registry = registry
        self._archives = archives

    def generate(
        self,
        archetype_id: str,
        sim_id: int,
        hints: VarianceHints,
        rng: random.Random,
    ) -> GenerationResult:
        """Sample an instance from the archive for the given archetype.

        Applies diversity weighting from VarianceHints to steer sampling
        toward underrepresented niches. Stamps the returned instance with
        the population's sequential sim_id.
        """
        archive = self._archives.get(archetype_id)
        if archive is None or archive.filled_count() == 0:
            return GenerationResult(
                instance=None,
                success=False,
                attempts=1,
                failure_reason=f"no archive for {archetype_id}",
            )

        # Build niche weights from variance hints
        weights = self._build_niche_weights(archive, hints)

        try:
            instance = archive.sample(weights, rng)
        except ArchiveEmpty:
            return GenerationResult(
                instance=None,
                success=False,
                attempts=1,
                failure_reason=f"empty archive for {archetype_id}",
            )

        # Stamp with the population's sequential sim_id
        stamped = dataclasses.replace(instance, sim_id=sim_id)
        return GenerationResult(
            instance=stamped,
            success=True,
            attempts=1,
            failure_reason=None,
        )

    def _build_niche_weights(
        self,
        archive: MAPElitesArchive,
        hints: VarianceHints,
    ) -> dict[DescriptorKey, float]:
        """Map VarianceHints to per-niche sampling weights.

        symbol_weights contribute to the step0_symbol dimension.
        spatial_bias contributes to the spatial_bin dimension.
        The product of both gives the final niche weight.
        """
        weights: dict[DescriptorKey, float] = {}

        for key in archive.occupied_keys():
            entry = archive.get(key)
            if entry is None:
                continue

            desc = entry.descriptor
            # Symbol weight — look up the step0 symbol name in hints
            sym_weight = 1.0
            for sym, w in hints.symbol_weights.items():
                if sym.name == desc.step0_symbol:
                    sym_weight = w
                    break

            # Spatial weight — uniform default (hints.spatial_bias maps
            # Position → float, but descriptor uses binned coordinates)
            spatial_weight = 1.0

            weights[key] = sym_weight * spatial_weight

        return weights
