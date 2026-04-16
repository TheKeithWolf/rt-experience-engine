"""Command-line runner for the spatial atlas — build and inspect.

Usage:
    python -m games.royal_tumble.experience_engine.atlas              # build
    python -m games.royal_tumble.experience_engine.atlas build [...]  # explicit
    python -m games.royal_tumble.experience_engine.atlas inspect [...]

Composes `build_atlas_services` + `AtlasStorage` for build, and the
`inspect_main` helper for inspection. Subcommand dispatch is argparse
`set_defaults(func=…)` — no if/elif on the command name. Bare invocation
keeps the historical behaviour by defaulting `func` to the build handler.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config.loader import load_config
from .builder import build_atlas_services
from .inspect_cli import inspect_main
from .storage import AtlasStorage


_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "default.yaml"
)


def _parse_sizes(value: str | None) -> tuple[int, ...] | None:
    """Convert the comma-separated --sizes argument to a tuple, or None.

    Kept as a pure string → tuple conversion so argparse stays declarative
    and the CLI can validate input with a single call.
    """
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    return tuple(int(p) for p in parts)


def build_main(args: argparse.Namespace) -> int:
    """Build the atlas and save it. Returns the CLI exit code.

    Exit codes:
      0 — atlas written (or up-to-date and no rebuild requested)
      1 — config or disk error (raised as SystemExit before reaching here)
    """
    config = load_config(args.config)
    if config.atlas is None:
        raise SystemExit("Config has no atlas section — nothing to build.")

    out_path = (
        Path(args.out) if args.out is not None
        else AtlasStorage.default_path(config, args.config)
    )
    storage = AtlasStorage()

    if not args.force and storage.load(config, out_path) is not None:
        print(f"atlas up-to-date at {out_path}")
        return 0

    services = build_atlas_services(config)
    progress = print if args.stats else None
    atlas = services.builder.build(
        sizes=_parse_sizes(args.sizes), progress=progress,
    )
    storage.save(atlas, config, out_path)
    print(f"atlas written to {out_path}")
    return 0


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Both subcommands share --config — extract once so they can't drift."""
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to the MasterConfig YAML (default: bundled default.yaml).",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Top-level parser with `build` and `inspect` subcommands.

    Bare invocation (no subcommand) routes to build via the top-level
    parser's `set_defaults(func=build_main)` — preserves backward-compat
    for `python -m ...atlas` with build flags directly attached.
    """
    parser = argparse.ArgumentParser(
        prog="python -m games.royal_tumble.experience_engine.atlas",
        description="Build or inspect the spatial atlas for royal_tumble.",
    )
    # Top-level build flags so bare invocation (no subcommand) accepts them.
    _add_config_arg(parser)
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path for the atlas binary. Defaults to config.atlas.path.",
    )
    parser.add_argument(
        "--sizes", type=str, default=None,
        help=(
            "Comma-separated cluster sizes to index "
            "(e.g. '5,9'). Omit to use atlas_cluster_sizes()."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even when the stored atlas's config hash still matches.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print per-size timing and final totals.",
    )
    parser.set_defaults(func=build_main)

    subparsers = parser.add_subparsers(dest="command")

    # `build` mirrors the top-level flags so explicit `build` works the
    # same way as a bare invocation.
    build_p = subparsers.add_parser(
        "build", help="Build the spatial atlas (default).",
    )
    _add_config_arg(build_p)
    build_p.add_argument("--out", type=str, default=None)
    build_p.add_argument("--sizes", type=str, default=None)
    build_p.add_argument("--force", action="store_true")
    build_p.add_argument("--stats", action="store_true")
    build_p.set_defaults(func=build_main)

    inspect_p = subparsers.add_parser(
        "inspect", help="Print the atlas header and per-map counts.",
    )
    _add_config_arg(inspect_p)
    inspect_p.add_argument(
        "--path", type=str, default=None,
        help="Atlas file to inspect. Defaults to config.atlas.path.",
    )
    inspect_p.add_argument(
        "--sample", type=int, default=0,
        help="Print this many entries per map (0 = no samples, the default).",
    )
    inspect_p.set_defaults(func=inspect_main)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint: parse argv, dispatch via argparse `func` attribute.

    Single-line dispatch (`return args.func(args)`) so adding a future
    subcommand is two changes: register a subparser, set its func.
    """
    args = _build_parser().parse_args(argv)
    return args.func(args)
