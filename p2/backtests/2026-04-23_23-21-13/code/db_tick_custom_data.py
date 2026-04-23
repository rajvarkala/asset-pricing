from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

from AlgorithmImports import *
from sqlalchemy import select, text

from db_interface.database import SessionLocal
from db_interface.models import Instrument


class DbDailyByTradingsymbol(PythonData):
    """Load daily candle custom data for a tradingsymbol from Postgres via db_interface."""

    _cache_dir = Path("/tmp/lean-db-daily")
    _prepared_symbols: set[str] = set()
    ENABLE_LOGS = True

    @classmethod
    def _log(cls, message: str) -> None:
        if cls.ENABLE_LOGS:
            print(f"[DbDailyByTradingsymbol] {message}")

    @classmethod
    def _safe_name(cls, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", value)

    @classmethod
    def _export_symbol_daily(cls, tradingsymbol: str, out_path: Path) -> None:
        cls._log(f"Resolving instrument for tradingsymbol={tradingsymbol}")
        with SessionLocal() as session:
            instrument = session.execute(
                select(Instrument).where(Instrument.tradingsymbol == tradingsymbol, Instrument.active.is_(True))
            ).scalar_one_or_none()

            if instrument is None:
                raise RuntimeError(f"No active instrument found for tradingsymbol={tradingsymbol}")

            cls._log(f"Resolved instrument_token={instrument.instrument_token}")

            query = text(
                """
                SELECT candle_date, open, high, low, close, volume
                FROM daily_candles
                WHERE instrument_token = :instrument_token
                ORDER BY candle_date
                """
            )
            rows = session.execute(query, {"instrument_token": instrument.instrument_token}).all()

        row_count = len(rows)
        if row_count > 0:
            min_date = rows[0].candle_date
            max_date = rows[-1].candle_date
            cls._log(f"Fetched {row_count} rows from daily_candles, date range [{min_date} -> {max_date}]")
            cls._log(
                f"Latest row close/volume = {rows[-1].close}/{rows[-1].volume} on {rows[-1].candle_date}"
            )
        else:
            cls._log("Fetched 0 rows from daily_candles")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["candle_date", "open", "high", "low", "close", "volume"])
            for row in rows:
                writer.writerow([row.candle_date.isoformat(), row.open, row.high, row.low, row.close, row.volume])
        cls._log(f"Wrote cache file: {out_path}")

    def get_source(self, config: SubscriptionDataConfig, date: datetime, is_live_mode: bool) -> SubscriptionDataSource:
        tradingsymbol = config.symbol.value
        file_name = f"{self._safe_name(tradingsymbol)}.csv"
        path = self._cache_dir / file_name

        if tradingsymbol not in self._prepared_symbols or not path.exists():
            self._log(f"Refreshing cache for {tradingsymbol}")
            self._export_symbol_daily(tradingsymbol, path)
            self._prepared_symbols.add(tradingsymbol)

        return SubscriptionDataSource(str(path), SubscriptionTransportMedium.LOCAL_FILE, FileFormat.CSV)

    def reader(self, config: SubscriptionDataConfig, line: str, date: datetime, is_live_mode: bool) -> BaseData:
        if not line or not line.strip() or line.lower().startswith("candle_date"):
            return None

        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            return None

        bar = DbDailyByTradingsymbol()
        bar.symbol = config.symbol

        try:
            bar_time = datetime.fromisoformat(parts[0])
            bar.time = bar_time
            # daily_candles.candle_date already represents the close date.
            bar.end_time = bar_time
            bar["open"] = float(parts[1])
            bar["high"] = float(parts[2])
            bar["low"] = float(parts[3])
            bar["close"] = float(parts[4])
            bar["volume"] = float(parts[5])
            bar.value = bar["close"]
        except (TypeError, ValueError):
            return None

        return bar
