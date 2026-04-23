# region imports
from __future__ import annotations

import json
from datetime import timedelta
from typing import List

from AlgorithmImports import *

import db_universe_custom_data
from db_universe_custom_data import ComponentUniverseData
from db_tick_custom_data import DbDailyByTradingsymbol, SessionLocal
# endregion

OBJECT_STORE_KEY = "portfolio-targets.csv"

# Selector filter sets — edit here to change universe scope.
INDEX_FILTERS: frozenset[str] = frozenset({"BSE 1000"})
SECTOR_FILTERS: frozenset[str] = frozenset({"Information Technology"})


class UniverseBehaviorAlgorithm(QCAlgorithm):
    """
    Backtest: custom DB-backed universe with date-parity selector logic.

      Odd calendar day  → index OR sector filter  (wider universe)
      Even calendar day → index AND sector filter (narrower universe)

    Universe is refreshed daily. on_securities_changed logs entries/exits.
    on_data equal-weights holdings whenever price data is available.
    """

    def initialize(self) -> None:
        self.set_start_date(2016, 4, 23)
        self.set_end_date(2026, 4, 23)
        self.set_cash(10_000_000)

        # Build current universe from DB and write to Object Store with a
        # lookback large enough to cover the full 10-year backtest window.
        payload = db_universe_custom_data.build_universe_payload(session_local=SessionLocal)
        db_universe_custom_data.write_universe_to_object_store(
            qb=self,
            payload=payload,
            key=OBJECT_STORE_KEY,
            lookback_days=3_700,
        )
        ComponentUniverseData.OBJECT_STORE_KEY = OBJECT_STORE_KEY

        # Universe settings: refresh every calendar day at market open.
        self.universe_settings.schedule.on(self.date_rules.every_day())
        self.universe_settings.resolution = Resolution.DAILY
        self.universe_settings.minimum_time_in_universe = timedelta(1)
        self.universe_settings.data_normalization_mode = DataNormalizationMode.SPLIT_ADJUSTED

        self._weight_by_symbol: dict[Symbol, float] = {}
        self._custom_data_symbols: dict[str, Symbol] = {}   # nse_code → custom-data symbol

        self.add_universe(
            ComponentUniverseData,
            "ComponentUniverse",
            Resolution.DAILY,
            self._selector,
        )

    # ------------------------------------------------------------------ #
    #  Selector — OR on odd days, AND on even days                        #
    # ------------------------------------------------------------------ #

    def _selector(self, alt_coarse: List[ComponentUniverseData]) -> List[Symbol]:
        is_odd = self.time.day % 2 == 1
        logic = "OR  (odd)" if is_odd else "AND (even)"

        chosen: dict[Symbol, float] = {}
        for d in alt_coarse:
            indexes = set(json.loads(d["indexes_json"] or "[]"))
            sectors = set(json.loads(d["sectors_json"] or "[]"))

            matches_index = not INDEX_FILTERS or bool(indexes & INDEX_FILTERS)
            matches_sector = not SECTOR_FILTERS or bool(sectors & SECTOR_FILTERS)

            passes = (matches_index or matches_sector) if is_odd else (matches_index and matches_sector)
            if passes:
                chosen[d.symbol] = float(d["weight"])

        self.debug(f"[{self.time.date()}] [{logic}] selected {len(chosen)} symbols")
        self._weight_by_symbol = chosen
        return list(chosen.keys())

    # ------------------------------------------------------------------ #
    #  Universe membership changes                                        #
    # ------------------------------------------------------------------ #

    def on_securities_changed(self, changes: SecurityChanges) -> None:
        for removed in changes.removed_securities:
            nse = removed.symbol.value
            self.liquidate(removed.symbol)
            self._custom_data_symbols.pop(nse, None)
            self.debug(f"  REMOVED {nse}  | portfolio: {len(self._weight_by_symbol)} symbol(s)")

        for added in changes.added_securities:
            nse = added.symbol.value
            # Subscribe to DB daily price data for this NSE code so on_data
            # can fill orders.  add_data is idempotent for a given name.
            if nse not in self._custom_data_symbols:
                sub = self.add_data(DbDailyByTradingsymbol, nse, Resolution.DAILY)
                self._custom_data_symbols[nse] = sub.symbol
            self.debug(f"  ADDED   {nse}  | portfolio: {len(self._weight_by_symbol)} symbol(s)")

    # ------------------------------------------------------------------ #
    #  Trading — equal-weight the active universe when price data arrives #
    # ------------------------------------------------------------------ #

    def on_data(self, data: Slice) -> None:
        if not self._weight_by_symbol:
            return

        for equity_symbol, weight in self._weight_by_symbol.items():
            nse = equity_symbol.value
            custom_sym = self._custom_data_symbols.get(nse)
            if custom_sym and data.contains_key(custom_sym):
                self.set_holdings(equity_symbol, weight)
