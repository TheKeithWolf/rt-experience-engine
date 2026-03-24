"""Trace a single book from JSONL output by sim_id — writes to file.

Standalone CLI that reads a book from the JSONL output produced by the book writer,
renders its ASCII trace via EventTracer, and writes the trace to a .txt file.

Supports three input formats (detected by file extension):
- .json: JSON array of book objects
- .jsonl: newline-delimited JSON, one book per line
- .jsonl.zst: zstandard-compressed JSONL

Usage:
    python -m games.royal_tumble.experience_engine.trace --id 42
    python -m games.royal_tumble.experience_engine.trace --id 42 --books path/to/books.jsonl
    python -m games.royal_tumble.experience_engine.trace --id 42 --output traces/
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from .config.loader import load_config
from .config.schema import MasterConfig
from .output.book_record import BookRecord
from .tracer.tracer import EventTracer

# Resolved relative to this file
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments — only --id is required."""
    parser = argparse.ArgumentParser(
        description="Trace a book from JSONL output — writes ASCII trace to file",
    )
    parser.add_argument(
        "--id", type=int, required=True,
        help="Sim ID of the book to trace",
    )
    parser.add_argument(
        "--books", type=Path, default=None,
        help="Path to books file (default: {output_dir}/books_{mode}.jsonl from config)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for trace file (default: {output_dir}/traces/)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=f"Master config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser


def _load_jsonl(path: Path, sim_id: int) -> dict:
    """Stream-read plain JSONL, return the book dict matching sim_id."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            book = json.loads(line)
            if book.get("id") == sim_id:
                return book
    raise KeyError(f"Book with id={sim_id} not found in {path}")


def _load_jsonl_zst(path: Path, sim_id: int) -> dict:
    """Decompress zstandard JSONL and stream-search for sim_id."""
    import zstandard as zstd

    decompressor = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with decompressor.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                line = line.strip()
                if not line:
                    continue
                book = json.loads(line)
                if book.get("id") == sim_id:
                    return book
    raise KeyError(f"Book with id={sim_id} not found in {path}")


def _load_json_array(path: Path, sim_id: int) -> dict:
    """Load a JSON array of books, search for the matching sim_id."""
    with open(path, "r", encoding="utf-8") as fh:
        books = json.load(fh)
    if not isinstance(books, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(books).__name__}")
    for book in books:
        if book.get("id") == sim_id:
            return book
    raise KeyError(f"Book with id={sim_id} not found in {path}")


def load_book_by_id(books_path: Path, sim_id: int) -> BookRecord:
    """Load a book by sim_id from .json, .jsonl, or .jsonl.zst file.

    Format detected by file extension. Raises KeyError if sim_id not found.
    """
    name = books_path.name

    if name.endswith(".jsonl.zst"):
        book_dict = _load_jsonl_zst(books_path, sim_id)
    elif name.endswith(".jsonl"):
        book_dict = _load_jsonl(books_path, sim_id)
    elif name.endswith(".json"):
        book_dict = _load_json_array(books_path, sim_id)
    else:
        raise ValueError(
            f"Unsupported file format: {name} "
            f"(expected .json, .jsonl, or .jsonl.zst)"
        )

    return BookRecord(
        id=book_dict["id"],
        payoutMultiplier=book_dict["payoutMultiplier"],
        events=tuple(book_dict["events"]),
        criteria=book_dict.get("criteria", ""),
        baseGameWins=book_dict.get("baseGameWins", 0.0),
        freeGameWins=book_dict.get("freeGameWins", 0.0),
    )


def main(argv: list[str] | None = None) -> int:
    """Load a book by sim_id, render its ASCII trace, write to file."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config or DEFAULT_CONFIG_PATH)

    # Derive default paths from config when CLI args not provided
    output_cfg = config.output
    if output_cfg is not None:
        default_books = Path(output_cfg.output_dir) / f"books_{output_cfg.mode_name}.jsonl"
        default_traces = Path(output_cfg.output_dir) / "traces"
    else:
        default_books = Path("library") / "books_base.jsonl"
        default_traces = Path("library") / "traces"

    books_path = args.books or default_books
    output_dir = args.output or default_traces
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the specific book by sim_id
    book = load_book_by_id(books_path, args.id)

    # Render trace to string via StringIO, then write to file
    trace_path = output_dir / f"trace_{args.id}.txt"
    buffer = io.StringIO()
    tracer = EventTracer(config)
    tracer.trace(book, output=buffer)
    trace_path.write_text(buffer.getvalue(), encoding="utf-8")

    print(f"Wrote trace: {trace_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
