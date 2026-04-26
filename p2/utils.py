"""Universe selection and weighting utilities for research and backtesting."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd


FINANCIAL_SCORE_SECTION = "derived_fundamentals"
FINANCIAL_SCORE_ROW = "Market Cap"


def coerce_float(value: Any) -> float | None:
    """Safely convert a value to float, returning None if not possible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # Check for NaN
    if number != number:
        return None
    return number


def extract_latest_metric(
    processed_payload: dict[str, Any],
    section_id: str,
    metric_name: str,
) -> float | None:
    """
    Extract the latest (most recent) value of a specific metric from processed financial data.

    Args:
        processed_payload: Nested dict of {section_id: {columns: [...], rows: [...]}}
        section_id: The section to search (e.g., "derived_fundamentals")
        metric_name: The row name to extract (e.g., "Market Cap")

    Returns:
        The latest numeric value, or None if not found or invalid.
    """
    metric_series = extract_metric_series(processed_payload, section_id, metric_name)
    if metric_series is None or metric_series.empty:
        return None

    for column_name in reversed(list(metric_series.index)):
        value = coerce_float(metric_series.get(column_name))
        if value is not None:
            return value

    return None


def extract_metric_series(
    processed_payload: dict[str, Any],
    section_id: str,
    metric_name: str,
) -> pd.Series | None:
    """Extract a metric row as a timestamp-indexed numeric Series."""
    if not isinstance(processed_payload, dict):
        return None

    section = processed_payload.get(section_id)
    if not isinstance(section, dict):
        return None

    section_value = section.get("value")
    if section_value is None:
        return None

    if isinstance(section_value, str):
        section_frame = pd.read_json(StringIO(section_value), orient="split")
    elif isinstance(section_value, dict):
        section_frame = pd.DataFrame(
            data=section_value.get("data", []),
            columns=section_value.get("columns", []),
            index=section_value.get("index", []),
        )
    else:
        return None

    if section_frame.empty:
        return None

    if metric_name in section_frame.index:
        metric_row = section_frame.loc[metric_name]
    elif "row_name" in section_frame.columns:
        matched = section_frame[section_frame["row_name"].astype(str).str.strip() == metric_name]
        if matched.empty:
            return None
        metric_row = matched.iloc[0]
    else:
        return None

    if isinstance(metric_row, pd.DataFrame):
        metric_row = metric_row.iloc[0]

    if "row_name" in metric_row.index:
        metric_row = metric_row.drop(labels=["row_name"])

    metric_series = pd.to_numeric(metric_row, errors="coerce").dropna()
    metric_series.index = metric_series.index.map(str)
    return metric_series
