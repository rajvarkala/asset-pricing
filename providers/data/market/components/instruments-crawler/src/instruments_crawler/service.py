from __future__ import annotations

from datetime import datetime
from typing import Any

from kiteconnect import KiteConnect
from sqlalchemy import select
from sqlalchemy.orm import Session

from db_interface.models import Instrument
from .settings import settings


def build_kite_client() -> KiteConnect:
    kite = KiteConnect(api_key=settings.kite_api_key)
    kite.set_access_token(settings.kite_access_token)
    return kite


def _is_equity_or_index(instrument: dict[str, Any]) -> bool:
    exchange = (instrument.get("exchange") or "").upper()
    segment = (instrument.get("segment") or "").upper()
    instrument_type = (instrument.get("instrument_type") or "").upper()

    if exchange != "NSE":
        return False

    is_equity = instrument_type == "EQ"
    is_index = "INDICES" in segment or instrument_type == "INDEX"
    return is_equity or is_index


def fetch_nse_equity_and_indices(kite: KiteConnect) -> list[dict[str, Any]]:
    all_instruments = kite.instruments()
    return [item for item in all_instruments if _is_equity_or_index(item)]


def upsert_instruments(session: Session, rows: list[dict[str, Any]]) -> int:
    seen_tokens: set[int] = set()
    updated = 0

    for row in rows:
        token = int(row["instrument_token"])
        seen_tokens.add(token)
        existing = session.scalar(select(Instrument).where(Instrument.instrument_token == token))

        if existing is None:
            existing = Instrument(
                instrument_token=token,
                exchange_token=row.get("exchange_token"),
                tradingsymbol=row.get("tradingsymbol") or "",
                name=row.get("name"),
                exchange=row.get("exchange") or "",
                segment=row.get("segment") or "",
                instrument_type=row.get("instrument_type"),
                tick_size=row.get("tick_size"),
                lot_size=row.get("lot_size"),
                active=True,
            )
            session.add(existing)
            updated += 1
            continue

        existing.exchange_token = row.get("exchange_token")
        existing.tradingsymbol = row.get("tradingsymbol") or existing.tradingsymbol
        existing.name = row.get("name")
        existing.exchange = row.get("exchange") or existing.exchange
        existing.segment = row.get("segment") or existing.segment
        existing.instrument_type = row.get("instrument_type")
        existing.tick_size = row.get("tick_size")
        existing.lot_size = row.get("lot_size")
        existing.active = True
        existing.updated_at = datetime.utcnow()
        updated += 1

    all_existing = session.scalars(select(Instrument).where(Instrument.exchange == "NSE")).all()
    for item in all_existing:
        if item.instrument_token not in seen_tokens:
            item.active = False
            item.updated_at = datetime.utcnow()

    session.commit()
    return updated
