# provider-api

FastAPI service that replicates Kite historical API semantics from local DB.

## Endpoint

- `GET /instruments/historical/{instrument_token}/day`
- Query: `from`, `to`, `continuous`, `oi`

## Commands

```bash
uv sync
uv run uvicorn provider_api.main:app --host 0.0.0.0 --port 8080
```

## Environment

- `DATABASE_URL`
- `LOG_LEVEL` (optional)
