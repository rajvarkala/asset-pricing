from __future__ import annotations

import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from kiteconnect import KiteConnect
from kiteconnect.exceptions import NetworkException
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from db_interface.models import DailyCandle, Instrument
from .settings import settings

START_YEARS_BACK = 25
MAX_HISTORICAL_RETRIES = 5
BASE_RETRY_SECONDS = 2
MAX_DAYS_PER_REQUEST = 1900


def build_kite_client() -> KiteConnect:
    api_key = settings.kite_api_key.strip()
    access_token = settings.kite_access_token.strip()

    if not api_key or not access_token:
        raise SystemExit(
            "KITE_API_KEY and KITE_ACCESS_TOKEN must be set in "
            "/Users/raj/ws/quantconnect/providers/data/market/.env"
        )

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def start_date_25_years_ago(today: date | None = None) -> date:
    base = today or datetime.now(timezone.utc).date()
    try:
        return base.replace(year=base.year - START_YEARS_BACK)
    except ValueError:
        # Handle leap-day rollover by using Feb 28 in non-leap years.
        return base.replace(month=2, day=28, year=base.year - START_YEARS_BACK)


def list_target_instruments(session: Session, instrument_token: int | None = None) -> list[int]:
    stmt = select(Instrument.instrument_token).where(Instrument.exchange == "NSE", Instrument.active.is_(True))
    if instrument_token is not None:
        stmt = stmt.where(Instrument.instrument_token == instrument_token)
    return [token for token in session.scalars(stmt).all()]


def latest_candle_date(session: Session, instrument_token: int) -> date | None:
    stmt = select(func.max(DailyCandle.candle_date)).where(DailyCandle.instrument_token == instrument_token)
    return session.scalar(stmt)


def delete_candles(session: Session, instrument_token: int) -> None:
    session.execute(delete(DailyCandle).where(DailyCandle.instrument_token == instrument_token))
    session.commit()


def upsert_candles(session: Session, instrument_token: int, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    now = datetime.utcnow()
    # Deduplicate by candle_date — keep last occurrence (latest data wins)
    seen: dict[date, dict[str, Any]] = {}
    for item in rows:
        candle_dt = item["date"]
        candle_date = candle_dt.date() if hasattr(candle_dt, "date") else candle_dt
        seen[candle_date] = {
            "instrument_token": instrument_token,
            "candle_date": candle_date,
            "open": float(item["open"]),
            "high": float(item["high"]),
            "low": float(item["low"]),
            "close": float(item["close"]),
            "volume": int(item.get("volume", 0)),
            "oi": int(item["oi"]) if item.get("oi") is not None else None,
            "created_at": now,
            "updated_at": now,
        }
    values = list(seen.values())

    insert_stmt = pg_insert(DailyCandle).values(values)
    stmt = insert_stmt.on_conflict_do_update(
            constraint="uq_candles_instr_date",
            set_={
                "open": insert_stmt.excluded.open,
                "high": insert_stmt.excluded.high,
                "low": insert_stmt.excluded.low,
                "close": insert_stmt.excluded.close,
                "volume": insert_stmt.excluded.volume,
                "oi": insert_stmt.excluded.oi,
                "updated_at": now,
            },
        )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount


def crawl_one_instrument(
    session: Session,
    kite: KiteConnect,
    instrument_token: int,
    full_refresh: bool,
) -> int:
    today = datetime.now(timezone.utc).date()
    start_date = start_date_25_years_ago(today)

    if full_refresh:
        delete_candles(session, instrument_token)
        from_date = start_date
    else:
        latest = latest_candle_date(session, instrument_token)
        from_date = (latest + timedelta(days=1)) if latest else start_date

    if from_date > today:
        return 0

    candles: list[dict[str, Any]] = []
    chunk_start = from_date
    while chunk_start <= today:
        chunk_end = min(chunk_start + timedelta(days=MAX_DAYS_PER_REQUEST), today)
        request_from = datetime.combine(chunk_start, time.min, tzinfo=timezone.utc)
        request_to = datetime.combine(chunk_end, time.max, tzinfo=timezone.utc)

        last_error: Exception | None = None
        chunk_candles: list[dict[str, Any]] = []
        for attempt in range(1, MAX_HISTORICAL_RETRIES + 1):
            try:
                chunk_candles = kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=request_from,
                    to_date=request_to,
                    interval="day",
                    continuous=False,
                    oi=False,
                )
                last_error = None
                break
            except NetworkException as exc:
                last_error = exc
                if attempt == MAX_HISTORICAL_RETRIES:
                    break

                # Exponential backoff for rate-limit and transient network responses.
                sleep_for = BASE_RETRY_SECONDS * (2 ** (attempt - 1))
                time_module.sleep(sleep_for)

        if last_error is not None:
            raise last_error

        candles.extend(chunk_candles)
        chunk_start = chunk_end + timedelta(days=1)

    return upsert_candles(session, instrument_token, candles)
