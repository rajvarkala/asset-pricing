from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from collections import defaultdict

from AlgorithmImports import *
from sqlalchemy import text

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
            "date"        : items[0],
            "nse_code"    : items[1],
            "weight"      : float(items[2]),
            "indexes_json": items[3],
            "sectors_json": items[4],
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
        return data
