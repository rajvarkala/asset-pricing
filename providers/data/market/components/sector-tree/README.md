# sector-tree

Builds and queries a company sector hierarchy from the `company_info` table.

## Data model

Each row in `company_info` with a non-empty `nse_code` and valid `company_sector` JSON contributes to the tree.  
The `company_sector` field has the shape:

```json
{
  "sector_path": ["Industrials", "Capital Goods", "Electrical Equipment", "Heavy Electrical Equipment"],
  "extraction_method": "...",
  "extraction_timestamp": "..."
}
```

The `sector_path` array is interpreted left-to-right as parent → child → ... → leaf.  
The `nse_code` is attached at the deepest (rightmost) node of the path.

## Usage

Run from within the component directory using the workspace venv Python:

```bash
# List every sector name in the tree
python -m sector_tree.cli --list-sectors

# Get all NSE codes under a sector (at any depth)
python -m sector_tree.cli --sector "Industrials"
python -m sector_tree.cli --sector "Capital Goods"

# Show the subtree structure alongside the NSE codes
python -m sector_tree.cli --sector "Capital Goods" --tree
```

All commands must be run from `sector-tree/src/` (or with `src/` on `PYTHONPATH`) using the components workspace venv:

```bash
cd providers/data/market/components/sector-tree/src
/path/to/market/components/.venv/bin/python -m sector_tree.cli --list-sectors
```

## Component structure

```
sector-tree/
├── cli.py                      # Compatibility wrapper (python -m sector-tree.cli)
├── __init__.py
├── pyproject.toml
├── README.md
└── src/
    └── sector_tree/
        ├── __init__.py
        ├── settings.py         # DATABASE_URL from .env
        ├── tree.py             # SectorNode, build_sector_tree, find_node, collect_nse_codes
        └── cli.py              # argparse CLI
```

## Dependencies

- `db-interface` (workspace) — SQLAlchemy models and DB utilities
- `sqlalchemy>=2.0.38`
- `psycopg[binary]>=3.2.6`
- `pydantic-settings>=2.8.1`
