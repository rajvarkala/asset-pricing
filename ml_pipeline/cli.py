from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

from kiteconnect.exceptions import TokenException

from .db import SessionLocal
from .service import build_kite_client, crawl_one_instrument, list_target_instruments
from .settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("historical-crawler")


def sync(instrument_token: int | None = None, full_refresh: bool = False) -> None:
    kite = build_kite_client()

    with SessionLocal() as session:
        tokens = list_target_instruments(session, instrument_token)

    total = 0
    failures = 0
    for token in tokens:
        try:
            with SessionLocal() as session:
                changed = crawl_one_instrument(session, kite, token, full_refresh=full_refresh)
            total += changed
        except TokenException as exc:
            logger.error(
                "Kite authentication failed: %s. Refresh KITE_ACCESS_TOKEN in "
                "/Users/raj/ws/asset-pricing/providers/data/market/.env",
                exc,
            )
            raise SystemExit(1) from exc
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.exception("Failed instrument %s: %s", token, exc)

    logger.info(
        "Historical sync complete for %s instruments. Rows changed=%s",
        len(tokens),
        total,
    )
    if failures > 0:
        raise SystemExit(1)


def _seconds_until_next_utc_hour(target_hour: int) -> int:
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=target_hour % 24, minute=0, second=0, microsecond=0)
    if next_run <= now:
        next_run = next_run + timedelta(days=1)
    return int((next_run - now).total_seconds())


def run_daemon() -> None:
    while True:
        try:
            sync(instrument_token=None, full_refresh=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Daemon sync failed: %s", exc)

        sleep_for = _seconds_until_next_utc_hour(settings.historical_sync_utc_hour)
        logger.info("Next historical sync in %s seconds", sleep_for)
        time.sleep(max(sleep_for, 60))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Historical crawler CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="One-time historical sync")
    sync_parser.add_argument("--instrument-token", type=int, default=None)
    sync_parser.add_argument("--full-refresh", action="store_true")

    subparsers.add_parser("run-daemon", help="Run incremental sync daily")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sync":
        sync(instrument_token=args.instrument_token, full_refresh=args.full_refresh)
    elif args.command == "run-daemon":
        run_daemon()


if __name__ == "__main__":
    main()
