"""Book writer — serializes BookRecord objects to JSONL files.

Writes one JSON object per line, optionally compressed with Zstandard.
Stream-writes to handle 1M+ books without loading all into memory.
Output format matches SDK RGS requirements from utils/rgs_verification.py:
- Required keys per line: id, payoutMultiplier, events
- payoutMultiplier: integer, divisible by centipayout rounding_base
- One JSON object per line, newline-terminated
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .book_record import BookRecord


@dataclass(frozen=True, slots=True)
class BookWriterConfig:
    """Injected writer settings — derived from MasterConfig.output at init."""

    # Directory for output files
    output_dir: Path
    # Bet mode name — controls file naming: books_{mode_name}.jsonl[.zst]
    mode_name: str
    # Whether to apply zstandard compression to the output file
    compression: bool


class BookWriter:
    """Writes BookRecord objects to JSONL format for RGS consumption.

    Stream-writes one book per line to avoid holding the full book list in memory.
    Compression uses zstandard when enabled, producing .jsonl.zst files that the
    SDK's book loader and RGS pipeline can decompress.
    """

    __slots__ = ("_config",)

    def __init__(self, config: BookWriterConfig) -> None:
        self._config = config

    def write_books(self, books: Iterable[BookRecord]) -> Path:
        """Write all books to a single JSONL file, return the output path.

        Creates the output directory if it doesn't exist. Each book is serialized
        via BookRecord.to_dict() and written as one JSON line.
        """
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

        if self._config.compression:
            return self._write_compressed(books)
        return self._write_plain(books)

    def _write_plain(self, books: Iterable[BookRecord]) -> Path:
        """Write uncompressed JSONL — one JSON object per line."""
        path = self._config.output_dir / f"books_{self._config.mode_name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for book in books:
                line = json.dumps(book.to_dict(), separators=(",", ":"))
                fh.write(line)
                fh.write("\n")
        return path

    def _write_compressed(self, books: Iterable[BookRecord]) -> Path:
        """Write zstandard-compressed JSONL — matches RGS .jsonl.zst format."""
        import zstandard as zstd

        path = self._config.output_dir / f"books_{self._config.mode_name}.jsonl.zst"
        compressor = zstd.ZstdCompressor()
        with open(path, "wb") as fh:
            with compressor.stream_writer(fh) as writer:
                for book in books:
                    line = json.dumps(book.to_dict(), separators=(",", ":"))
                    writer.write(line.encode("utf-8"))
                    writer.write(b"\n")
        return path
