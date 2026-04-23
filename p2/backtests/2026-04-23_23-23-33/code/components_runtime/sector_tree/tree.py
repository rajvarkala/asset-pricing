"""Core sector tree: build from company_info.company_sector and query by sector name."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)


@dataclass
class SectorNode:
    """A node in the sector hierarchy tree."""

    name: str
    children: dict[str, "SectorNode"] = field(default_factory=dict)
    # nse_codes are only populated at the deepest node for each company path
    nse_codes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"SectorNode(name={self.name!r}, children={list(self.children)}, nse_codes={self.nse_codes})"


def build_sector_tree(session: Session) -> SectorNode:
    """
    Build a sector hierarchy tree from company_info.

    - Reads every row that has a non-empty nse_code and a valid company_sector JSON.
    - Parses sector_path from company_sector: each element is a level of the tree.
    - The nse_code is attached to the deepest (rightmost) node of its path.
    - Rows with empty/null nse_code are skipped.

    Returns the root SectorNode (name='__root__').
    """
    root = SectorNode(name="__root__")

    rows = session.execute(text("""
        SELECT nse_code, company_sector
        FROM company_info
        WHERE nse_code IS NOT NULL
          AND nse_code != ''
          AND company_sector IS NOT NULL
    """)).fetchall()

    logger.info("Found %d company_info rows with nse_code and company_sector", len(rows))

    skipped = 0
    inserted = 0
    for nse_code, company_sector in rows:
        if not company_sector:
            skipped += 1
            continue

        # company_sector may arrive as a dict (psycopg JSON) or a string
        if isinstance(company_sector, str):
            try:
                data = json.loads(company_sector)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON for nse_code=%s, skipping", nse_code)
                skipped += 1
                continue
        else:
            data = company_sector

        sector_path: list[str] = data.get("sector_path", [])
        if not sector_path:
            skipped += 1
            continue

        # Walk / create nodes along the path
        node = root
        for part in sector_path:
            if part not in node.children:
                node.children[part] = SectorNode(name=part)
            node = node.children[part]

        # Attach nse_code to the bottom (leaf) node of this path
        if nse_code not in node.nse_codes:
            node.nse_codes.append(nse_code)

        inserted += 1

    logger.info("Tree built: %d companies inserted, %d skipped", inserted, skipped)
    return root


def find_node(root: SectorNode, sector: str) -> SectorNode | None:
    """
    Find the first node whose name matches `sector` (depth-first, case-sensitive).

    Returns None if not found.
    """
    if root.name == sector:
        return root
    for child in root.children.values():
        result = find_node(child, sector)
        if result is not None:
            return result
    return None


def collect_nse_codes(node: SectorNode) -> list[str]:
    """
    Collect all nse_codes in a node's subtree (including the node itself).

    Returns a sorted, deduplicated list.
    """
    codes: list[str] = list(node.nse_codes)
    for child in node.children.values():
        codes.extend(collect_nse_codes(child))
    return sorted(set(codes))


def list_all_sectors(root: SectorNode) -> list[str]:
    """
    Return all unique sector names present anywhere in the tree (excluding root).
    """
    names: list[str] = []

    def _walk(node: SectorNode) -> None:
        if node.name != "__root__":
            names.append(node.name)
        for child in node.children.values():
            _walk(child)

    _walk(root)
    return sorted(set(names))


def print_subtree(node: SectorNode, indent: int = 0) -> None:
    """Print a human-readable view of the subtree rooted at `node`."""
    prefix = "  " * indent
    codes_str = f"  [{', '.join(node.nse_codes)}]" if node.nse_codes else ""
    print(f"{prefix}{node.name}{codes_str}")
    for child in node.children.values():
        print_subtree(child, indent + 1)
