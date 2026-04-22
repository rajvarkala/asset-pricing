"""Compatibility wrapper — enables `python -m sector-tree.cli` from the components directory."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sector_tree.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
