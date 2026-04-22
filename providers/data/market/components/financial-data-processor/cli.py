"""Compatibility CLI entrypoint for `python -m financial-data-processor.cli`."""

from __future__ import annotations

import sys
from pathlib import Path


# Make the real package importable when running from the component root.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from financial_data_processor.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
