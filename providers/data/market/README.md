# Market Data System (Kite-Compatible Historical API)

This folder contains an independent, multi-component system that runs Python services locally and PostgreSQL in Docker:

- Crawls NSE instruments (equity + indices) using `pykiteconnect`
- Crawls last 25 years of daily candles
- Stores data in PostgreSQL through SQLAlchemy models
- Replicates Kite historical API semantics from local DB via FastAPI
- Exposes an overarching gateway API for future data providers

## Components

- `components/db-interface`: SQLAlchemy models, DB session, schema bootstrap
- `components/instruments-crawler`: NSE instrument crawler
- `components/historical-crawler`: Historical candle crawler (25 years, daily)
- `components/provider-api`: Kite-compatible historical endpoint backed by DB
- `components/gateway-api`: Overarching API with request logging and provider proxying

Each component is independent and has its own `pyproject.toml` managed with `uv`.

## Quick Start

1. Copy env file:

```bash
cp .env.example .env
```

2. Fill in Kite credentials in `.env`:

- `KITE_API_KEY`
- `KITE_ACCESS_TOKEN`

3. Start PostgreSQL only:

```bash
docker compose up -d postgres
```

4. Create one shared local virtual environment for all components:

```bash
cd components
uv venv ../.venv
uv pip install --python ../.venv/bin/python -e db-interface -e "instruments-crawler[dev]" -e "historical-crawler[dev]" -e "provider-api[dev]" -e "gateway-api[dev]"
```

5. Run services locally (in order):

```bash
# terminal 1
source .venv/bin/activate
python -m db_interface.cli init-db

# terminal 2
source .venv/bin/activate
python -m instruments_crawler.cli sync

# terminal 3
source .venv/bin/activate
python -m historical_crawler.cli sync

# terminal 4
source .venv/bin/activate
uvicorn provider_api.main:app --host 0.0.0.0 --port 8080 --reload

# terminal 5
source .venv/bin/activate
uvicorn gateway_api.main:app --host 0.0.0.0 --port 8090 --reload
```

6. Gateway API:

- `http://localhost:8090/health`
- Provider proxy path: `http://localhost:8090/providers/kite/...`

7. Provider-compatible API direct:

- `http://localhost:8080/health`
- `GET /instruments/historical/{instrument_token}/day?from=YYYY-MM-DD&to=YYYY-MM-DD&continuous=0&oi=0`

## Refresh Modes (Historical Crawler)

- Incremental all instruments:

```bash
source .venv/bin/activate
python -m historical_crawler.cli sync
```

- Incremental one instrument:

```bash
source .venv/bin/activate
python -m historical_crawler.cli sync --instrument-token 738561
```

- Full refresh one instrument (from 25 years window start):

```bash
source .venv/bin/activate
python -m historical_crawler.cli sync --full-refresh --instrument-token 738561
```

- Full refresh all instruments:

```bash
source .venv/bin/activate
python -m historical_crawler.cli sync --full-refresh
```

## Notes

- All timestamps are stored and served in UTC.
- No auth/rate limiting is enforced in local APIs.
- Integration tests against live provider are intentionally omitted (unit tests only).
