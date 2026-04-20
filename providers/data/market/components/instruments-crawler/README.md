# instruments-crawler

Independent crawler for NSE instruments using `pykiteconnect`.

## Scope

- NSE Equity
- NSE Indices

## Commands

```bash
uv sync
uv run python -m instruments_crawler.cli sync
uv run python -m instruments_crawler.cli run-daemon
```

## Environment

- `DATABASE_URL`
- `KITE_API_KEY`
- `KITE_ACCESS_TOKEN`
- `INSTRUMENTS_SYNC_INTERVAL_HOURS` (default: 24)
