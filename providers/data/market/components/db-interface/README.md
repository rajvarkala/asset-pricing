# db-interface

Independent SQLAlchemy DB interface component.

## Purpose

- Defines database schema for instruments and daily OHLC data.
- Exposes a simple CLI to initialize DB tables.

## Commands

```bash
uv sync
uv run python -m db_interface.cli init-db
```

## Environment

- `DATABASE_URL` (required)
