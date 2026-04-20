# historical-crawler

Independent historical daily candle crawler backed by `pykiteconnect`.

## Data Range

- Last 25 years
- Daily interval
- Stored in UTC date/time

## Commands

```bash
uv sync
uv run python -m historical_crawler.cli sync
uv run python -m historical_crawler.cli sync --instrument-token 738561
uv run python -m historical_crawler.cli sync --full-refresh
uv run python -m historical_crawler.cli sync --full-refresh --instrument-token 738561
uv run python -m historical_crawler.cli run-daemon
```

## Environment

- `DATABASE_URL`
- `KITE_API_KEY`
- `KITE_ACCESS_TOKEN`
- `HISTORICAL_SYNC_UTC_HOUR` (default: 1)
