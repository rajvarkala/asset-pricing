from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from sqlalchemy import and_, select

from db_interface.models import DailyCandle

from .db import SessionLocal

app = FastAPI(title="Kite-Compatible Provider API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "message": message})


@app.get("/instruments/historical/{instrument_token}/{interval}")
def historical_data(
    instrument_token: int,
    interval: str,
    from_date: date = Query(alias="from"),
    to_date: date = Query(alias="to"),
    continuous: int = Query(default=0),
    oi: int = Query(default=0),
) -> Any:
    if interval != "day":
        return _error("Only daily interval is currently supported", 400)

    if from_date > to_date:
        return _error("Invalid date range: from must be <= to", 400)

    _ = continuous

    with SessionLocal() as session:
        stmt = (
            select(DailyCandle)
            .where(
                and_(
                    DailyCandle.instrument_token == instrument_token,
                    DailyCandle.candle_date >= from_date,
                    DailyCandle.candle_date <= to_date,
                )
            )
            .order_by(DailyCandle.candle_date.asc())
        )
        rows = session.scalars(stmt).all()

    payload: list[dict[str, Any]] = []
    for row in rows:
        dt = datetime.combine(row.candle_date, time.min, tzinfo=timezone.utc)
        item: dict[str, Any] = {
            "date": dt.isoformat(),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        if oi:
            item["oi"] = row.oi
        payload.append(item)

    return {"status": "success", "data": payload}
