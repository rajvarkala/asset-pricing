"""CLI for sector-tree."""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .settings import settings
from .tree import (
    build_sector_tree,
    collect_nse_codes,
    find_node,
    list_all_sectors,
    print_subtree,
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Query the company sector hierarchy built from company_info.company_sector. "
            "Given any sector name, returns all NSE codes in that subtree."
        )
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sector",
        metavar="NAME",
        help="Sector/sub-sector name to query (e.g. 'Industrials', 'Capital Goods').",
    )
    group.add_argument(
        "--list-sectors",
        action="store_true",
        help="List every sector name present in the tree.",
    )

    parser.add_argument(
        "--tree",
        action="store_true",
        help="Also print the subtree structure (use with --sector).",
    )

    args = parser.parse_args()

    try:
        engine = create_engine(settings.database_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

        with SessionLocal() as session:
            root = build_sector_tree(session)

            if args.list_sectors:
                sectors = list_all_sectors(root)
                print(f"Found {len(sectors)} unique sector name(s):\n")
                for s in sectors:
                    print(f"  {s}")
                return 0

            # --sector path
            node = find_node(root, args.sector)
            if node is None:
                print(
                    f"Sector '{args.sector}' not found in the tree.",
                    file=sys.stderr,
                )
                print(
                    "Tip: run with --list-sectors to see all available sector names.",
                    file=sys.stderr,
                )
                return 1

            nse_codes = collect_nse_codes(node)

            if args.tree:
                print(f"Subtree rooted at '{args.sector}':\n")
                print_subtree(node)
                print()

            print(f"Companies under '{args.sector}' ({len(nse_codes)}):\n")
            for code in nse_codes:
                print(f"  {code}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
