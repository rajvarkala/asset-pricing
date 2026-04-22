# Financial Data Processor

Single-source summary for the financial data processing pipeline.

This component reads hierarchical financial statements from PostgreSQL, builds clean pandas DataFrames, and persists processed output either:
- as legacy filesystem pickle files, or
- as a pickled DataFrame payload inside PostgreSQL (`processed_financial_data`).

## What It Does

1. Reads dynamic period columns from `information_schema` (no hardcoded years).
2. Loads section rows for `(company_id, section_id)`.
3. Builds hierarchical row order from `parent_row` using DFS.
4. Sorts period columns chronologically (keeps `TTM` at the end).
5. Drops only columns that are all `NULL` for that specific section.
6. Writes processed output:
   - legacy path: `{SERIALIZED_DATA_PATH}/{company_id}/{section_id}.pkl`
   - database path: one pickled DataFrame blob per `(company_id, nse_code, section_id)`.

## Current Storage Model (Database)

Processed output is stored as a DataFrame pickle payload in `processed_financial_data`.

Key columns:
- `company_id`
- `nse_code`
- `section_id`
- `dataframe_pickle` (`BYTEA`)
- `created_at`, `updated_at`

Uniqueness:
- `(company_id, nse_code, section_id)`

This means each section writes exactly one row (one payload), not one row per financial line item.

## Required Source Table

`financial_data` must include:
- base columns: `id`, `company_id`, `section_id`, `row_name`, `row_type`, `parent_row`, `created_at`, `updated_at`
- dynamic period columns like `Mar 2024`, `Sep 2025`, `TTM`, etc.

## CLI Usage

From `components` directory:

```bash
# Process all companies/sections and write to DB payloads
python -m financial-data-processor.cli

# Process one symbol only (recommended for validation)
python -m financial-data-processor.cli --nse-code INFY
```

Also available as installed script:

```bash
financial-data-processor
```

Notes:
- `python -m financial-data-processor.cli` is supported via compatibility wrapper.
- After symbol-scoped runs, CLI reads back the stored DB pickle and prints the restored DataFrame.
- Intermediate processing logs are shown (candidate/kept/dropped all-null columns per section).

## Programmatic API

### Main processing/write APIs

```python
from financial_data_processor.service import (
	process_and_write_financial_data_database,
	process_all_financial_data_database,
)
```

### Read processed DataFrame payload back from DB

```python
from db_interface import load_processed_financial_dataframe_from_db

df = load_processed_financial_dataframe_from_db(
	session,
	company_id="1489",
	section_id="profit-loss",
	nse_code="INFY",
)
print(df)
```

### Raw processing DataFrame (before DB write)

```python
from db_interface import get_financial_data_dataframe

df = get_financial_data_dataframe(session, company_id="1489", section_id="profit-loss")
print(df)
```

## Configuration

Environment variables:
- `DATABASE_URL`
- `SERIALIZED_DATA_PATH` (legacy pickle-to-disk flow)

This component resolves `.env` from the market root automatically:
- `/Users/raj/ws/quantconnect/providers/data/market/.env`

So running from `components` still picks up the correct DB credentials.

## Minimal Setup

```bash
cd /Users/raj/ws/quantconnect/providers/data/market/components
pip install -e ./db-interface
pip install -e ./financial-data-processor
```

Ensure DB table has payload column:

```sql
ALTER TABLE processed_financial_data
ADD COLUMN IF NOT EXISTS dataframe_pickle BYTEA;
```

## Troubleshooting

### Password auth fails for postgres user

Cause:
- wrong `.env` loaded or default URL fallback used.

Fix:
- set valid `DATABASE_URL` in market root `.env`.
- run from `components` using `python -m financial-data-processor.cli ...`.

### DataFrame shows no year columns

Cause:
- year columns were all null after section-scoped filtering, or dynamic column retrieval mismatch.

Current behavior:
- dynamic columns are fetched via SQL using discovered column names.
- drop is section-specific (`dropna(axis=1, how="all")` on that section only).
- logs print exactly which columns were dropped per section.

### Duplicate key on rerun

Handled:
- existing section row is deleted and flushed before reinsert, so reruns are idempotent.

## Development

```bash
cd /Users/raj/ws/quantconnect/providers/data/market/components/financial-data-processor
pip install -e ".[dev]"
pytest
ruff check src tests
mypy src
```

## Status

Production-ready for:
- processing and writing by NSE symbol (`--nse-code`),
- persisting processed output as DB pickled payloads,
- restoring and printing DataFrames from stored payloads.
