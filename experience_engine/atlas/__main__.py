"""Module entrypoint — lets `python -m ...atlas` drive build_cli.main()."""

from __future__ import annotations

import sys

from .build_cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
