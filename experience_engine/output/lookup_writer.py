"""Lookup table writer — CSV files matching SDK format for optimizer consumption.

Produces two files:
1. Main lookup table: sim_id,weight,payout — one row per book, initial weight=1
   Also writes a _0 copy (initial optimized weights file consumed by Rust optimizer)
2. Archetype lookup table: sim_id,archetype_id,family,criteria — for post-opt audit

Format constraints (from utils/rgs_verification.py):
- No header row, comma-separated
- Payouts: integers, >= 0, divisible by centipayout rounding_base
- Weights: integers >= 0
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from .book_record import BookRecord


class LookupTableWriter:
    """Writes CSV lookup tables in SDK-compatible format.

    The main table feeds the Rust weight optimizer. The archetype table enables
    post-optimization survival auditing per archetype.
    """

    __slots__ = ("_output_dir", "_mode_name")

    def __init__(self, output_dir: Path, mode_name: str) -> None:
        self._output_dir = output_dir
        self._mode_name = mode_name

    def write_initial_table(self, books: Iterable[BookRecord]) -> Path:
        """Write lookUpTable_{mode}.csv with initial weight=1 for all books.

        Also writes lookUpTable_{mode}_0.csv as a copy — the Rust optimizer reads
        the _0 file as its starting point and writes adjusted weights back to it.
        Returns the path to the main (non-_0) table.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        main_path = self._output_dir / f"lookUpTable_{self._mode_name}.csv"

        with open(main_path, "w", encoding="utf-8") as fh:
            for book in books:
                # SDK format: sim_id,weight,payout — no header, integer values
                fh.write(f"{book.id},1,{book.payoutMultiplier}\n")

        # _0 copy — initial optimized weights file for the Rust optimizer
        copy_path = self._output_dir / f"lookUpTable_{self._mode_name}_0.csv"
        shutil.copy2(main_path, copy_path)

        return main_path

    def write_archetype_table(
        self,
        books: Iterable[BookRecord],
        archetype_ids: Iterable[str],
    ) -> Path:
        """Write archetype_lookUpTable_{mode}.csv mapping sim_id to archetype metadata.

        Format: sim_id,archetype_id,family,criteria — one row per book.
        The post-optimization audit joins this with the optimized LUT to compute
        per-archetype survival rates.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"archetype_lookUpTable_{self._mode_name}.csv"

        with open(path, "w", encoding="utf-8") as fh:
            for book, archetype_id in zip(books, archetype_ids, strict=True):
                fh.write(f"{book.id},{archetype_id},{book.criteria}\n")

        return path
