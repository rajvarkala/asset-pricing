"""Transformer pipeline for deriving fundamentals from processed financial data.

Each transformer receives a ``TransformContext`` (containing all processed section
DataFrames plus the stock's daily price history) together with the accumulated
``result`` DataFrame built by previous transformers in the chain.  The transformer
appends its own computed metric rows and returns the updated DataFrame.

Usage::

    from financial_data_processor.transformers import (
        DEFAULT_TRANSFORMERS,
        TransformContext,
        PERatioTransformer,
        PEGRatioTransformer,
    )

    context = TransformContext(
        nse_code="INFY",
        sections={"profit_loss": df_profit_loss, ...},
        price_data=df_price,
        company_info={"no_eq_shares": 123456789.0},
    )

    result = pd.DataFrame()
    for transformer in DEFAULT_TRANSFORMERS:
        result = transformer.transform(context, result)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row-name candidates for EPS across common data-source naming conventions
# ---------------------------------------------------------------------------
_EPS_ROW_CANDIDATES: list[str] = [
    "EPS in Rs.",
    "EPS",
    "Earnings Per Share",
    "Basic EPS (Rs.)",
    "Diluted EPS (Rs.)",
    "Adjusted EPS (Rs.)",
]

# Sections searched in preference order when looking for EPS
_EPS_SECTION_CANDIDATES: list[str] = [
    "profit_loss",
    "income_statement",
    "profit-loss",
    "standalone_profit_loss",
    "consolidated_profit_loss",
]


# ---------------------------------------------------------------------------
# Context & base class
# ---------------------------------------------------------------------------

@dataclass
class TransformContext:
    """Immutable input bundle passed to every transformer in the chain.

    Attributes:
        nse_code:   NSE trading symbol of the company being processed.
        sections:   Mapping of ``section_id`` → DataFrame where the DataFrame
                    has **row names as its index** and **period labels**
                    (e.g. ``"Mar 2023"``, ``"TTM"``) as its columns.
        price_data: Daily OHLCV DataFrame indexed by ``candle_date`` (``date``
                    objects), sorted in ascending chronological order.  Columns:
                    ``open``, ``high``, ``low``, ``close``, ``volume``.
        company_info: Company metadata dictionary from ``company_info`` table.
                      Used by transformers that need static fundamentals such
                      as ``no_eq_shares``.
    """

    nse_code: str
    sections: dict[str, pd.DataFrame]
    price_data: pd.DataFrame
    company_info: dict[str, Any]


class BaseTransformer(ABC):
    """Abstract base class for all derived-fundamentals transformers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable transformer name used in log messages."""
        ...

    @abstractmethod
    def transform(self, context: TransformContext, result: pd.DataFrame) -> pd.DataFrame:
        """Compute derived metric(s) and append them to *result*.

        Args:
            context: Immutable context with raw financial sections and price history.
            result:  Accumulated result so far — index = metric names, columns =
                     period labels.  May be empty for the first transformer in the chain.

        Returns:
            Updated ``result`` DataFrame with any new metric rows appended.  The
            original ``result`` must **not** be mutated in-place; return a new
            DataFrame via ``pd.concat``.
        """
        ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _period_to_date(period: str) -> date | None:
    """Map a period label such as ``'Mar 2023'`` to the last calendar day of that month.

    ``'TTM'`` maps to ``date.today()`` so the most recent available price is used.
    Returns ``None`` when the label cannot be parsed.
    """
    if period == "TTM":
        return date.today()

    months: dict[str, int] = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    parts = period.split()
    if len(parts) >= 2:
        month = months.get(parts[0])
        try:
            year = int(parts[1])
            if month:
                last_day = monthrange(year, month)[1]
                return date(year, month, last_day)
        except ValueError:
            pass
    return None


def _price_at_or_before(price_data: pd.DataFrame, target_date: date) -> float | None:
    """Return the closing price on or most-recently before *target_date*.

    Returns ``None`` when *price_data* is empty or all available data post-dates
    *target_date*.
    """
    if price_data.empty:
        return None
    mask = price_data.index <= target_date
    if not mask.any():
        return None
    return float(price_data.loc[mask, "close"].iloc[-1])


def _find_eps_series(sections: dict[str, pd.DataFrame]) -> pd.Series | None:
    """Search known sections for the best available EPS row.

    Returns a ``pd.Series`` keyed by period label, or ``None`` when not found.
    """
    # Preferred sections first
    for section_id in _EPS_SECTION_CANDIDATES:
        if section_id not in sections:
            continue
        df = sections[section_id]
        for candidate in _EPS_ROW_CANDIDATES:
            if candidate in df.index:
                return df.loc[candidate]

    # Fallback: scan all remaining sections
    for section_id, df in sections.items():
        for candidate in _EPS_ROW_CANDIDATES:
            if candidate in df.index:
                return df.loc[candidate]

    return None


def _append_row(result: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Append *row* to *result*, returning a new DataFrame."""
    row_df = row.to_frame().T
    if result.empty:
        return row_df
    return pd.concat([result, row_df], axis=0)


# ---------------------------------------------------------------------------
# Concrete transformers
# ---------------------------------------------------------------------------

class PERatioTransformer(BaseTransformer):
    """Compute the trailing Price-to-Earnings ratio for each fiscal period.

    For every period column the transformer:

    1. Reads the reported EPS value from the financial statements.
    2. Looks up the closing market price at (or just before) the fiscal year-end
       date corresponding to that period label.
    3. Emits ``P/E Ratio = price / EPS``.

    Periods with missing price data, zero EPS, or negative EPS are left as
    ``NaN`` rather than raising an error.
    """

    @property
    def name(self) -> str:
        return "P/E Ratio"

    def transform(self, context: TransformContext, result: pd.DataFrame) -> pd.DataFrame:
        print(f"\n{'='*60}")
        print(f"[PERatioTransformer] nse_code={context.nse_code}")
        print(f"  sections available: {list(context.sections.keys())}")
        print(f"  price_data shape:   {context.price_data.shape}")
        if not context.price_data.empty:
            print(f"  price_data range:   {context.price_data.index[0]} → {context.price_data.index[-1]}")
            print(f"  price_data tail:\n{context.price_data.tail(3)}")
        for section_id, df in context.sections.items():
            print(f"\n  [{section_id}] shape={df.shape}  columns={list(df.columns)}")
            print(f"  index (rows): {list(df.index)}")
            print(df)
        print(f"  result so far:\n{result}")
        print(f"{'='*60}\n")

        eps_series = _find_eps_series(context.sections)
        if eps_series is None:
            logger.warning(
                "[%s] EPS row not found in any section for nse_code=%s — skipping",
                self.name, context.nse_code,
            )
            return result

        pe_values: dict[str, float | None] = {}
        for period in eps_series.index:
            eps_raw = eps_series[period]
            try:
                eps = float(eps_raw)
            except (TypeError, ValueError):
                pe_values[str(period)] = None
                continue

            if eps <= 0:
                pe_values[str(period)] = None
                continue

            target_date = _period_to_date(str(period))
            if target_date is None:
                pe_values[str(period)] = None
                continue

            price = _price_at_or_before(context.price_data, target_date)
            if price is None:
                pe_values[str(period)] = None
                continue

            pe_values[str(period)] = round(price / eps, 2)

        pe_row = pd.Series(pe_values, name="P/E Ratio")
        return _append_row(result, pe_row)


class PEGRatioTransformer(BaseTransformer):
    """Compute the Price/Earnings-to-Growth (PEG) ratio for each fiscal period.

    PEG = P/E ÷ EPS growth rate (%)

    This transformer must run **after** ``PERatioTransformer`` because it reads
    the ``'P/E Ratio'`` row from the accumulated *result* DataFrame.

    EPS growth is calculated year-over-year as::

        growth_pct = (eps_curr − eps_prev) / |eps_prev| × 100

    Periods with non-positive or zero EPS growth, missing P/E, or the first
    period (no prior data) are emitted as ``NaN``.
    """

    @property
    def name(self) -> str:
        return "PEG Ratio"

    def transform(self, context: TransformContext, result: pd.DataFrame) -> pd.DataFrame:
        if result.empty or "P/E Ratio" not in result.index:
            logger.warning(
                "[%s] 'P/E Ratio' row not found in accumulated result for nse_code=%s — "
                "ensure PERatioTransformer runs before PEGRatioTransformer",
                self.name, context.nse_code,
            )
            return result

        eps_series = _find_eps_series(context.sections)
        if eps_series is None:
            logger.warning(
                "[%s] EPS row not found for nse_code=%s — skipping",
                self.name, context.nse_code,
            )
            return result

        pe_row = result.loc["P/E Ratio"]
        periods = list(eps_series.index)
        peg_values: dict[str, float | None] = {}

        for i, period in enumerate(periods):
            period_str = str(period)
            if i == 0:
                # No prior period available for growth calculation
                peg_values[period_str] = None
                continue

            prev_period = periods[i - 1]
            try:
                eps_curr = float(eps_series[period])
                eps_prev = float(eps_series[prev_period])
            except (TypeError, ValueError):
                peg_values[period_str] = None
                continue

            if eps_prev == 0:
                peg_values[period_str] = None
                continue

            eps_growth_pct = ((eps_curr - eps_prev) / abs(eps_prev)) * 100
            if eps_growth_pct <= 0:
                peg_values[period_str] = None
                continue

            pe_val = pe_row.get(period_str)
            if pe_val is None or pd.isna(pe_val):
                peg_values[period_str] = None
                continue

            peg_values[period_str] = round(float(pe_val) / eps_growth_pct, 4)

        peg_row = pd.Series(peg_values, name="PEG Ratio")
        return _append_row(result, peg_row)


class MarketCapMonthlyTransformer(BaseTransformer):
    """Compute monthly market capitalization using month-end close prices.

    Formula:

        market_cap_month = month_end_close_price * no_eq_shares

    The number of equity shares is read from ``context.company_info['no_eq_shares']``.
    If shares are missing or invalid, this transformer is skipped.
    """

    @property
    def name(self) -> str:
        return "Market Cap Monthly"

    def transform(self, context: TransformContext, result: pd.DataFrame) -> pd.DataFrame:
        shares_raw = context.company_info.get("no_eq_shares")
        try:
            shares = float(shares_raw) or 0.0
        except (TypeError, ValueError):
            logger.warning(
                "[%s] Invalid no_eq_shares=%r for nse_code=%s - skipping",
                self.name,
                shares_raw,
                context.nse_code,
            )
            return result

        if shares <= 0:
            logger.warning(
                "[%s] Non-positive no_eq_shares=%s for nse_code=%s - skipping",
                self.name,
                shares,
                context.nse_code,
            )
            return result

        if context.price_data.empty:
            logger.warning("[%s] No price data for nse_code=%s - skipping", self.name, context.nse_code)
            return result

        price_data = context.price_data.copy()
        price_data.index = pd.to_datetime(price_data.index)
        month_end_prices = price_data["close"].resample("ME").last().dropna()
        if month_end_prices.empty:
            logger.warning(
                "[%s] No month-end prices available for nse_code=%s - skipping",
                self.name,
                context.nse_code,
            )
            return result

        market_cap_values: dict[str, float] = {}
        for dt, close_price in month_end_prices.items():
            period_label = dt.strftime("%b %Y")
            market_cap_values[period_label] = round(float(close_price) * shares, 2)

        market_cap_row = pd.Series(market_cap_values, name="Market Cap")
        return _append_row(result, market_cap_row)
 

# ---------------------------------------------------------------------------
# Default pipeline
# ---------------------------------------------------------------------------

#: The default ordered list of transformers applied when no custom chain is given.
DEFAULT_TRANSFORMERS: list[BaseTransformer] = [
    PERatioTransformer(),
    PEGRatioTransformer(),
    MarketCapMonthlyTransformer(),
]
