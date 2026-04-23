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

# Selector filter sets — edit here to change universe scope.
INDEX_FILTERS: frozenset[str] = frozenset({"BSE 1000"})
SECTOR_FILTERS: frozenset[str] = frozenset({"Information Technology"})


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
            },
            {
                "nse_code": "KSOLVES",
                "indexes": [],
                "sectors": ["Information Technology"],
            },
            {
                "nse_code": "SANOFICONR",
                "indexes": ["BSE 1000"],
                "sectors": ["Healthcare"],
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

        chosen: dict[Symbol, float] = {}
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
                chosen[Symbol.create(code, SecurityType.EQUITY, Market.INDIA)] = 1.0

        if chosen:
            equal_weight = 1.0 / len(chosen)
            self._weight_by_symbol = {s: equal_weight for s in chosen}
        else:
            self._weight_by_symbol = {}

        self.debug(f"[{self.time.date()}] [{logic}] source={source} selected {len(self._weight_by_symbol)} symbols")
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
            self.debug(f"  ADDED   {nse}  | portfolio: {len(self._weight_by_symbol)} symbol(s)")

    # ------------------------------------------------------------------ #
    #  Trading — equal-weight the active universe when price data arrives #
    # ------------------------------------------------------------------ #

    def on_data(self, data: Slice) -> None:
        if not self._weight_by_symbol:
            return

        for equity_symbol, weight in self._weight_by_symbol.items():
            if data.bars.contains_key(equity_symbol):
                self.set_holdings(equity_symbol, weight)
