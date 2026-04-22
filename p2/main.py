# region imports
from AlgorithmImports import *
from db_tick_custom_data import DbTickByTradingsymbol
# endregion


class P2(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2024, 1, 31)
        self.set_cash(100000)

        # Pick ticker from parameter; it maps to Instrument.tradingsymbol in Postgres.
        tradingsymbol = self.get_parameter("tradingsymbol")
        self._tick_symbol = self.add_data(DbTickByTradingsymbol, tradingsymbol, Resolution.TICK).symbol
        self._last_log_time = None

    def on_data(self, data: Slice):
        if not data.contains_key(self._tick_symbol):
            return

        tick = data[self._tick_symbol]
        minute_bucket = self.time.replace(second=0, microsecond=0)
        if self._last_log_time == minute_bucket:
            return

        self._last_log_time = minute_bucket
        bid = tick.get_property("bid_price")
        ask = tick.get_property("ask_price")
        qty = tick.get_property("quantity")
        self.debug(
            f"{self.time} {self._tick_symbol.value} last={tick.value} bid={bid} ask={ask} qty={qty}"
        )
