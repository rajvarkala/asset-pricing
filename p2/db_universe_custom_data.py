from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

from AlgorithmImports import *
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def _extract_sector_names(company_sector: object) -> list[str]:
    """Extract all sector names from company_sector JSON payload."""
    if not company_sector:
        return []

    if isinstance(company_sector, str):
        try:
            payload = json.loads(company_sector)
        except json.JSONDecodeError:
            return []
    elif isinstance(company_sector, dict):
        payload = company_sector
    else:
        return []

    path = payload.get("sector_path", [])
    if not isinstance(path, list):
        return []

    return [str(x).strip() for x in path if str(x).strip()]


def build_session_local():
    """Build a SQLAlchemy session factory from environment variables."""
    db_host = os.getenv("DB_HOST", "host.docker.internal")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "market")
    db_password = os.getenv("DB_PASSWORD", "market")
    db_name = os.getenv("DB_NAME", "market")

    url = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(url, future=True)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def build_universe_payload(session_local) -> dict[str, object]:
    """
    Build a full universe payload for all available NSE codes.

    For each NSE code, attach all index names and all sector names found for that symbol.
    The selector function in research can then filter by desired index/sector combinations.
    """
    with session_local() as session:
        rows = session.execute(
            text(
                """
                SELECT
                    ci.nse_code,
                    ci.company_sector,
                    im.index_name
                FROM company_info ci
                LEFT JOIN index_memberships im
                    ON im.company_id = ci.company_id
                WHERE ci.nse_code IS NOT NULL
                  AND ci.nse_code != ''
                """
            )
        ).fetchall()

    by_nse: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"indexes": set(), "sectors": set()}
    )

    for nse_code, company_sector, index_name in rows:
        code = str(nse_code).strip()
        if not code:
            continue

        if index_name:
            by_nse[code]["indexes"].add(str(index_name).strip())

        for sector in _extract_sector_names(company_sector):
            by_nse[code]["sectors"].add(sector)

        # Ensure symbol exists in map even when both sets are empty.
        _ = by_nse[code]

    universe = []
    for code in sorted(by_nse.keys()):
        indexes = sorted(x for x in by_nse[code]["indexes"] if x)
        sectors = sorted(x for x in by_nse[code]["sectors"] if x)
        universe.append(
            {
                "nse_code": code,
                "indexes": indexes,
                "sectors": sectors,
            }
        )

    return {
        "total_symbols": len(universe),
        "universe": universe,
        "universe_nse_codes": [x["nse_code"] for x in universe],
    }


def write_universe_to_object_store(
    qb: object,
    payload: dict[str, object],
    key: str = "portfolio-targets.csv",
    lookback_days: int = 30,
) -> str:
    """
    Write full universe payload to Object Store in LEAN custom-universe CSV format.

    ``qb`` accepts both ``QuantBook`` (research) and ``QCAlgorithm`` (backtest) — any
    object exposing ``.object_store.save(key, text)``.

    Columns: date,symbol,weight,indexes_json,sectors_json
    """
    print(f"Writing universe payload to Object Store with key='{key}'...")
    universe = payload.get("universe", [])
    if not isinstance(universe, list):
        raise TypeError("payload['universe'] must be a list")

    utc_today = datetime.utcnow().date()
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["date", "symbol", "weight", "indexes_json", "sectors_json"])

    if universe:
        weight = 1.0 / len(universe)
        for i in range(lookback_days):
            d = (utc_today - timedelta(days=i)).strftime("%Y-%m-%d")
            for row in universe:
                writer.writerow(
                    [
                        d,
                        row.get("nse_code", ""),
                        f"{weight:.8f}",
                        json.dumps(row.get("indexes", []), separators=(",", ":")),
                        json.dumps(row.get("sectors", []), separators=(",", ":")),
                    ]
                )

    csv_text = csv_buffer.getvalue()
    qb.object_store.save(key, csv_text)
    return key

 
class ComponentUniverseData(PythonData):
    """Custom universe data reader compatible with LEAN selector methodology."""

    OBJECT_STORE_KEY = "portfolio-targets.csv"

    def get_source(self, config: SubscriptionDataConfig, date: datetime, is_live_mode: bool) -> SubscriptionDataSource:
        return SubscriptionDataSource(
            self.OBJECT_STORE_KEY,
            SubscriptionTransportMedium.OBJECT_STORE,
            FileFormat.CSV,
        )

    def reader(self, config: SubscriptionDataConfig, line: str, date: datetime, is_live_mode: bool) -> BaseData:
        if not line or not line[0].isnumeric():
            return None

        items = next(csv.reader([line]))
        if len(items) < 5:
            return None

        data = ComponentUniverseData()
        data.end_time = datetime.strptime(items[0], "%Y-%m-%d")
        data.time = data.end_time - timedelta(1)
        data.symbol = Symbol.create(items[1], SecurityType.EQUITY, Market.INDIA)
        data["weight"] = float(items[2])
        data["indexes_json"] = items[3]
        data["sectors_json"] = items[4]
        return data
