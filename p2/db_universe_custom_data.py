from __future__ import annotations

import csv
import io
import json
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any

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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return None if value != value else value

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return str(value)


def _deserialize_processed_financial_section(section_id: object, dataframe_pickle: object) -> dict[str, Any] | None:
    if not section_id or not dataframe_pickle:
        return None

    try:
        # Try standard unpickling first
        deserialized = pickle.loads(dataframe_pickle)
    except (TypeError, AttributeError, ValueError):
        # Pandas version mismatch (StringDtype initialization changed)
        # Return minimal structure to allow code to continue
        return {
            "section_id": str(section_id),
            "columns": [],
            "rows": [],
        }
    except Exception:
        # Other unpickling errors
        return None

    if hasattr(deserialized, "iterrows") and hasattr(deserialized, "columns"):
        rows = []
        for row_name, values in deserialized.iterrows():
            row = {"row_name": _json_safe(row_name)}
            for column_name, value in values.items():
                row[str(column_name)] = _json_safe(value)
            rows.append(row)

        return {
            "section_id": str(section_id),
            "columns": [str(column_name) for column_name in deserialized.columns],
            "rows": rows,
        }

    return {
        "section_id": str(section_id),
        "value": _json_safe(deserialized),
    }


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

    For each NSE code, attach all index names, all sector names, and every
    processed_financial_data section found for that symbol.
    """
    with session_local() as session:
        rows = session.execute(
            text(
                """
                SELECT
                    ci.nse_code,
                    ci.company_sector,
                    im.index_name,
                    pfd.section_id,
                    pfd.dataframe_pickle
                FROM company_info ci
                LEFT JOIN index_memberships im
                    ON im.company_id = ci.company_id
                LEFT JOIN processed_financial_data pfd
                    ON pfd.company_id = ci.company_id
                WHERE ci.nse_code IS NOT NULL
                  AND ci.nse_code != ''
                """
            )
        ).fetchall()

    by_nse: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "indexes": set(),
            "sectors": set(),
            "processed_financial_data": {},
        }
    )

    for nse_code, company_sector, index_name, section_id, dataframe_pickle in rows:
        code = str(nse_code).strip()
        if not code:
            continue

        if index_name:
            by_nse[code]["indexes"].add(str(index_name).strip())

        for sector in _extract_sector_names(company_sector):
            by_nse[code]["sectors"].add(sector)

        if section_id and section_id not in by_nse[code]["processed_financial_data"]:
            section_payload = _deserialize_processed_financial_section(section_id, dataframe_pickle)
            if section_payload is not None:
                by_nse[code]["processed_financial_data"][str(section_id)] = section_payload

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
                "processed_financial_data": by_nse[code]["processed_financial_data"],
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

    Columns: date,symbol,weight,indexes_json,sectors_json,processed_financial_data_json
    """
    universe = payload.get("universe", [])
    if not isinstance(universe, list):
        raise TypeError("payload['universe'] must be a list")

    utc_today = datetime.utcnow().date()
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow([
        "date",
        "symbol",
        "weight",
        "indexes_json",
        "sectors_json",
        "processed_financial_data_json",
    ])

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
                        json.dumps(row.get("processed_financial_data", {}), separators=(",", ":")),
                    ]
                )

    csv_text = csv_buffer.getvalue()
    qb.object_store.save(key, csv_text)
    return key


def run_selector(qb: QuantBook, key: str, selector_fn) -> list:
    """
    Drive the selector function in QuantBook context.

    In QuantBook, qb.history(universe, ...) does NOT invoke the selector —
    that only happens during live/backtest algorithm execution.
    This function manually reads the Object Store CSV, parses the latest
    date's rows into lightweight alt_coarse objects, and calls selector_fn
    directly, exactly mirroring what the algorithm engine does per trading day.

    Returns the list of selected Symbols.
    """
    if not qb.object_store.contains_key(key):
        raise FileNotFoundError(f"Object Store key not found: {key}")

    raw = qb.object_store.read(key)
    lines = raw.splitlines()
    if len(lines) < 2:
        print(f"[run_selector] Object Store '{key}' is empty — no symbols to select.")
        return []

    rows = []
    for line in lines[1:]:          # skip header
        if not line or not line[0].isnumeric():
            continue
        items = next(csv.reader([line]))
        if len(items) < 5:
            continue
        rows.append({
            "date"                         : items[0],
            "nse_code"                     : items[1],
            "weight"                       : float(items[2]),
            "indexes_json"                 : items[3],
            "sectors_json"                 : items[4],
            "processed_financial_data_json": items[5] if len(items) > 5 else "{}",
        })

    if not rows:
        print(f"[run_selector] No parseable rows in Object Store '{key}'.")
        return []

    # Use only the most-recent date (one call per date — same as algorithm engine)
    latest_date = max(r["date"] for r in rows)
    latest_rows = [r for r in rows if r["date"] == latest_date]
    print(f"[run_selector] Driving selector for date={latest_date}, rows={len(latest_rows)}")

    class _Row:
        """Lightweight PythonData stand-in that selector_fn can index with []."""
        def __init__(self, r: dict) -> None:
            self.symbol = Symbol.create(r["nse_code"], SecurityType.EQUITY, Market.INDIA)
            self._r = r
        def __getitem__(self, key):
            return self._r.get(key)

    alt_coarse = [_Row(r) for r in latest_rows]
    selected = selector_fn(alt_coarse)
    print(f"[run_selector] Selector returned {len(selected)} symbol(s): "
          f"{[s.value for s in selected]}")
    return selected


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
        data["processed_financial_data_json"] = items[5] if len(items) > 5 else "{}"
        return data
