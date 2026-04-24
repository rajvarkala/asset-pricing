"""Universe selection and weighting utilities for research and backtesting."""

from __future__ import annotations

from typing import Any


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
    if not isinstance(processed_payload, dict):
        return None

    section = processed_payload.get(section_id)
    if not isinstance(section, dict):
        return None

    rows = section.get("rows", [])
    metric_row = next(
        (
            row
            for row in rows
            if isinstance(row, dict) and str(row.get("row_name", "")).strip() == metric_name
        ),
        None,
    )

    if metric_row is None:
        return None

    # Get column order from section, or infer from row keys
    ordered_columns = section.get("columns", [])
    if not isinstance(ordered_columns, list) or not ordered_columns:
        ordered_columns = [
            key
            for key in metric_row.keys()
            if key not in {"row_name", "section_id", "columns", "rows"}
        ]

    # Iterate columns in reverse (most recent first)
    for column_name in reversed(ordered_columns):
        value = coerce_float(metric_row.get(column_name))
        if value is not None:
            return value

    return None
