# region imports
from __future__ import annotations

from datetime import timedelta
from typing import Any, List

from AlgorithmImports import *

import db_universe_custom_data
from db_universe_custom_data import ComponentUniverseData
# endregion

OBJECT_STORE_KEY = "portfolio-targets.csv"
USE_DB_UNIVERSE = False
FINANCIAL_SCORE_SECTION = "derived_fundamentals"
FINANCIAL_SCORE_ROW = "Market Cap"

# Selector filter sets — edit here to change universe scope.
INDEX_FILTERS: frozenset[str] = frozenset({"BSE 1000"})
SECTOR_FILTERS: frozenset[str] = frozenset({"Information Technology"})


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number != number:
        return None

    return number


def _extract_latest_metric_value(row: dict[str, Any], section_id: str, metric_name: str) -> float | None:
    processed = row.get("processed_financial_data")
    if not isinstance(processed, dict):
        return None

    section = processed.get(section_id)
    if not isinstance(section, dict):
        return None

    rows = section.get("rows")
    if not isinstance(rows, list):
        return None

    metric_row = None
    for candidate in rows:
        if isinstance(candidate, dict) and str(candidate.get("row_name", "")).strip() == metric_name:
            metric_row = candidate
            break

    if metric_row is None:
        return None

    columns = section.get("columns")
    ordered_columns = [str(column) for column in columns] if isinstance(columns, list) else []
    if not ordered_columns:
        ordered_columns = [
            str(column)
            for column in metric_row.keys()
            if str(column) not in {"row_name", "section_id", "columns", "rows"}
        ]

    for column_name in reversed(ordered_columns):
        value = _coerce_float(metric_row.get(column_name))
        if value is not None:
            return value

    return None


class UniverseBehaviorAlgorithm(QCAlgorithm):
    """
    Backtest: selector-driven universe with date-parity selector logic.

      Odd calendar day  → index OR sector filter  (wider universe)
      Even calendar day → index AND sector filter (narrower universe)

    Universe candidates are rebuilt each time the selector is called.
    If DB access fails inside Docker, a deterministic fallback list is used.
    """

    def initialize(self) -> None:
        self.set_start_date(2016, 4, 23)
        self.set_end_date(2026, 4, 23)
        self.set_cash(10_000_000)

        # Universe settings: refresh every calendar day at market open.
        self.universe_settings.schedule.on(self.date_rules.every_day())
        self.universe_settings.resolution = Resolution.DAILY
        self.universe_settings.minimum_time_in_universe = timedelta(1)
        self.universe_settings.data_normalization_mode = DataNormalizationMode.SPLIT_ADJUSTED

        # Fallback rows keep the algorithm deterministic if DB connectivity is
        # unavailable in the LEAN Docker runtime.
        self._fallback_rows: list[dict[str, Any]] = [
            {
                "nse_code": "INFY",
                "indexes": ["BSE 1000"],
                "sectors": ["Information Technology"],
                "processed_financial_data": {
                    "derived_fundamentals": {
                        "section_id": "derived_fundamentals",
                        "columns": ["Apr 2026"],
                        "rows": [
                            {
                                "row_name": "Market Cap",
                                "Apr 2026": 7_850_000.0,
                            }
                        ],
                    },
                    "balance-sheet": {
                        "section_id": "balance-sheet",
                        "columns": ["Mar 2024", "Mar 2025"],
                        "rows": [
                            {
                                "row_name": "Total Assets",
                                "Mar 2024": 192340.0,
                                "Mar 2025": 205110.0,
                            },
                            {
                                "row_name": "Total Liabilities",
                                "Mar 2024": 71420.0,
                                "Mar 2025": 75680.0,
                            },
                        ],
                    },
                    "profit-loss": {
                        "section_id": "profit-loss",
                        "columns": ["Mar 2024", "Mar 2025", "TTM"],
                        "rows": [
                            {
                                "row_name": "Revenue",
                                "Mar 2024": 153670.0,
                                "Mar 2025": 164220.0,
                                "TTM": 167005.0,
                            },
                            {
                                "row_name": "Net Profit",
                                "Mar 2024": 26120.0,
                                "Mar 2025": 27880.0,
                                "TTM": 28110.0,
                            },
                        ],
                    },
                },
            },
            {
                "nse_code": "KSOLVES",
                "indexes": [],
                "sectors": ["Information Technology"],
                "processed_financial_data": {
                    "derived_fundamentals": {
                        "section_id": "derived_fundamentals",
                        "columns": ["Apr 2026"],
                        "rows": [
                            {
                                "row_name": "Market Cap",
                                "Apr 2026": 185_000.0,
                            }
                        ],
                    },
                    "balance-sheet": {
                        "section_id": "balance-sheet",
                        "columns": ["Mar 2024", "Mar 2025"],
                        "rows": [
                            {
                                "row_name": "Total Assets",
                                "Mar 2024": 842.0,
                                "Mar 2025": 913.0,
                            }
                        ],
                    },
                },
            },
            {
                "nse_code": "SANOFICONR",
                "indexes": ["BSE 1000"],
                "sectors": ["Healthcare"],
                "processed_financial_data": {
                    "derived_fundamentals": {
                        "section_id": "derived_fundamentals",
                        "columns": ["Apr 2026"],
                        "rows": [
                            {
                                "row_name": "Market Cap",
                                "Apr 2026": 465_000.0,
                            }
                        ],
                    },
                    "balance-sheet": {
                        "section_id": "balance-sheet",
                        "columns": ["Mar 2024", "Mar 2025"],
                        "rows": [
                            {
                                "row_name": "Total Assets",
                                "Mar 2024": 6210.0,
                                "Mar 2025": 6475.0,
                            }
                        ],
                    },
                    "profit-loss": {
                        "section_id": "profit-loss",
                        "columns": ["Mar 2024", "Mar 2025", "TTM"],
                        "rows": [
                            {
                                "row_name": "Revenue",
                                "Mar 2024": 3180.0,
                                "Mar 2025": 3328.0,
                                "TTM": 3364.0,
                            }
                        ],
                    },
                },
            },
        ]

        # Keep selector callbacks alive via custom-universe plumbing,
        # but actual universe membership is rebuilt inside the selector.
        db_universe_custom_data.write_universe_to_object_store(
            qb=self,
            payload={
                "total_symbols": len(self._fallback_rows),
                "universe": self._fallback_rows,
                "universe_nse_codes": [x["nse_code"] for x in self._fallback_rows],
            },
            key=OBJECT_STORE_KEY,
            lookback_days=3_700,
        )
        ComponentUniverseData.OBJECT_STORE_KEY = OBJECT_STORE_KEY

        self._weight_by_symbol: dict[Symbol, float] = {}
        self._score_by_symbol: dict[Symbol, float] = {}
        self._db_failed = False

        self._universe = self.add_universe(
            ComponentUniverseData,
            "ComponentUniverse",
            Resolution.DAILY,
            self._selector,
        )

    def _build_rows_for_selector(self) -> tuple[list[dict[str, Any]], str]:
        if not USE_DB_UNIVERSE:
            return self._fallback_rows, "fallback"

        if not self._db_failed:
            try:
                session_local = db_universe_custom_data.build_session_local()
                payload = db_universe_custom_data.build_universe_payload(session_local=session_local)
                rows = payload.get("universe", [])
                if isinstance(rows, list) and rows:
                    return rows, "db"
            except Exception as exc:
                self._db_failed = True
                self.debug(f"DB universe rebuild failed. Falling back to local rows. Reason: {exc}")

        return self._fallback_rows, "fallback"

    # ------------------------------------------------------------------ #
    #  Selector — OR on odd days, AND on even days                        #
    # ------------------------------------------------------------------ #

    def _selector(self, alt_coarse: List[ComponentUniverseData]) -> List[Symbol]:
        is_odd = self.time.day % 2 == 1
        logic = "OR  (odd)" if is_odd else "AND (even)"
        rows, source = self._build_rows_for_selector()

        candidates: dict[Symbol, float | None] = {}
        for row in rows:
            code = str(row.get("nse_code", "")).strip()
            if not code:
                continue

            indexes = set(str(x) for x in row.get("indexes", []) if str(x).strip())
            sectors = set(str(x) for x in row.get("sectors", []) if str(x).strip())

            matches_index = not INDEX_FILTERS or bool(indexes & INDEX_FILTERS)
            matches_sector = not SECTOR_FILTERS or bool(sectors & SECTOR_FILTERS)

            passes = (matches_index or matches_sector) if is_odd else (matches_index and matches_sector)
            if passes:
                symbol = Symbol.create(code, SecurityType.EQUITY, Market.INDIA)
                candidates[symbol] = _extract_latest_metric_value(
                    row,
                    FINANCIAL_SCORE_SECTION,
                    FINANCIAL_SCORE_ROW,
                )

        scored = {symbol: score for symbol, score in candidates.items() if score is not None and score > 0}
        if scored:
            score_sum = sum(scored.values())
            self._weight_by_symbol = {symbol: score / score_sum for symbol, score in scored.items()}
            for symbol in candidates:
                if symbol not in self._weight_by_symbol:
                    self._weight_by_symbol[symbol] = 0.0
            self._score_by_symbol = dict(scored)
        elif candidates:
            equal_weight = 1.0 / len(candidates)
            self._weight_by_symbol = {symbol: equal_weight for symbol in candidates}
            self._score_by_symbol = {}
        else:
            self._weight_by_symbol = {}
            self._score_by_symbol = {}

        score_summary = ", ".join(
            f"{symbol.value}:{self._score_by_symbol[symbol]:.2f}"
            for symbol in sorted(self._score_by_symbol.keys(), key=lambda item: item.value)
        ) or "no financial scores"
        self.debug(
            f"[{self.time.date()}] [{logic}] source={source} selected {len(self._weight_by_symbol)} symbols | "
            f"metric={FINANCIAL_SCORE_ROW} | {score_summary}"
        )
        return list(self._weight_by_symbol.keys())

    # ------------------------------------------------------------------ #
    #  Universe membership changes                                        #
    # ------------------------------------------------------------------ #

    def on_securities_changed(self, changes: SecurityChanges) -> None:
        for removed in changes.removed_securities:
            nse = removed.symbol.value
            self.liquidate(removed.symbol)
            self.debug(f"  REMOVED {nse}  | portfolio: {len(self._weight_by_symbol)} symbol(s)")

        for added in changes.added_securities:
            nse = added.symbol.value
            weight = self._weight_by_symbol.get(added.symbol, 0.0)
            score = self._score_by_symbol.get(added.symbol)
            if score is None:
                self.debug(f"  ADDED   {nse}  | weight={weight:.4f} | no {FINANCIAL_SCORE_ROW} score")
            else:
                self.debug(f"  ADDED   {nse}  | weight={weight:.4f} | {FINANCIAL_SCORE_ROW}={score:.2f}")

    # ------------------------------------------------------------------ #
    #  Trading — equal-weight the active universe when price data arrives #
    # ------------------------------------------------------------------ #

    def on_data(self, data: Slice) -> None:
        if not self._weight_by_symbol:
            return

        for equity_symbol, weight in self._weight_by_symbol.items():
            if data.bars.contains_key(equity_symbol):
                self.set_holdings(equity_symbol, weight)
