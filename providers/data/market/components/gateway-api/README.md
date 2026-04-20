# gateway-api

Overarching API server for current/future market data providers.

## Features

- Request logging middleware
- Proxies provider-compatible API at `/providers/kite/*`
- No auth and no rate limiting (as requested)

## Commands

```bash
uv sync
uv run uvicorn gateway_api.main:app --host 0.0.0.0 --port 8090
```

## Environment

- `PROVIDER_API_BASE_URL` (default: `http://provider-api:8080`)
- `LOG_LEVEL` (default: `INFO`)
