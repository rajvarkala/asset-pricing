from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from sector_tree.tree import build_sector_tree, collect_nse_codes, find_node, list_all_sectors


def list_index_names(session: Session) -> list[str]:
    """Return all distinct index names that have at least one mappable NSE code."""
    rows = session.execute(
        text(
            """
            SELECT DISTINCT im.index_name
            FROM index_memberships im
            JOIN company_info ci ON ci.company_id = im.company_id
            WHERE ci.nse_code IS NOT NULL
              AND ci.nse_code != ''
            ORDER BY im.index_name
            """
        )
    ).fetchall()

    return [row[0] for row in rows]


def get_index_constituent_nse_codes(session: Session, index_name: str) -> list[str]:
    """
    Return all NSE codes for a given index name.

    Constraints:
    - Joins index_memberships.company_id with company_info.company_id
    - Returns only rows where company_info.nse_code is non-null and non-empty
    - Returns NSE codes (not company_id)
    """
    rows = session.execute(
        text(
            """
            SELECT DISTINCT ci.nse_code
            FROM index_memberships im
            JOIN company_info ci ON ci.company_id = im.company_id
            WHERE im.index_name = :index_name
              AND ci.nse_code IS NOT NULL
              AND ci.nse_code != ''
            ORDER BY ci.nse_code
            """
        ),
        {"index_name": index_name},
    ).fetchall()

    return [row[0] for row in rows]


def get_sector_index_universe_nse_codes(
    session: Session,
    requested_sector: str,
    index_name: str,
) -> tuple[str, list[str], int, int]:
    """
    Build an NSE-code universe as intersection of:
    - sector subtree from sector-tree
    - index constituents from index-constituents

    Returns:
    - resolved sector name (may differ from requested for fuzzy fallback)
    - sorted universe nse codes
    - sector code count
    - index code count
    """
    root = build_sector_tree(session)
    node = find_node(root, requested_sector)

    resolved_sector = requested_sector
    sector_codes = set()
    index_codes = set()
    universe = []
    if node is not None:
        sector_codes = set(collect_nse_codes(node))
        index_codes = set(get_index_constituent_nse_codes(session, index_name))
        universe = sorted(sector_codes & index_codes)

    return resolved_sector, universe, len(sector_codes), len(index_codes)
