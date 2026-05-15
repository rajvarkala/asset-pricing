import argparse
import logging
import time

from sqlalchemy.exc import OperationalError

from .db import SessionLocal
from .service import build_kite_client, fetch_nse_equity_and_indices, upsert_instruments
from .settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("instruments-crawler")


def run_sync() -> None:
    kite = build_kite_client()
    rows = fetch_nse_equity_and_indices(kite)

    try:
        with SessionLocal() as session:
            count = upsert_instruments(session, rows)
    except OperationalError as exc:
        logger.error(
            "Database connection failed. Start postgres first: "
            "cd /Users/raj/ws/asset-pricing/providers/data/market && docker compose up -d postgres"
        )
        raise SystemExit(1) from exc

    logger.info("Synced %s NSE equity/index instruments", count)


def run_daemon() -> None:
    interval_seconds = max(settings.instruments_sync_interval_hours, 1) * 3600
    while True:
        try:
            run_sync()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Instrument sync failed: %s", exc)
        time.sleep(interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instruments crawler CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sync", help="Run one-time sync")
    subparsers.add_parser("run-daemon", help="Run periodic sync")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sync":
        run_sync()
    elif args.command == "run-daemon":
        run_daemon()


if __name__ == "__main__":
    main()
