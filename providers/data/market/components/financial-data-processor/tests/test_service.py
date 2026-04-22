"""Tests for financial data processor service."""

import pytest

from financial_data_processor.service import (
    build_hierarchical_row_order,
    parse_date_column_name,
    sort_date_columns,
)


def test_parse_date_column_name() -> None:
    """Test date column name parsing."""
    assert parse_date_column_name("Mar 2023") == (2023, 3, False)
    assert parse_date_column_name("Dec 2024") == (2024, 12, False)
    assert parse_date_column_name("Jan 2025") == (2025, 1, False)
    assert parse_date_column_name("TTM") == (0, 0, True)
    assert parse_date_column_name("Dec 2023 8m") == (2023, 12, False)


def test_sort_date_columns() -> None:
    """Test date column sorting."""
    columns = [
        "Sep 2025",
        "Mar 2023",
        "Jun 2024",
        "TTM",
        "Dec 2024",
        "Jan 2023",
    ]
    sorted_cols = sort_date_columns(columns)
    expected = [
        "Jan 2023",
        "Mar 2023",
        "Jun 2024",
        "Dec 2024",
        "Sep 2025",
        "TTM",
    ]
    assert sorted_cols == expected


def test_build_hierarchical_row_order() -> None:
    """Test hierarchical row ordering."""
    rows_data = [
        {"row_name": "Total Assets", "parent_row": None},
        {"row_name": "Current Assets", "parent_row": "Total Assets"},
        {"row_name": "Cash", "parent_row": "Current Assets"},
        {"row_name": "Accounts Receivable", "parent_row": "Current Assets"},
        {"row_name": "Fixed Assets", "parent_row": "Total Assets"},
        {"row_name": "PPE", "parent_row": "Fixed Assets"},
    ]

    order = build_hierarchical_row_order(rows_data)

    # Total Assets should come first (root)
    assert order[0] == "Total Assets"

    # Current Assets should come before Fixed Assets (same parent)
    current_assets_idx = order.index("Current Assets")
    fixed_assets_idx = order.index("Fixed Assets")
    assert current_assets_idx < fixed_assets_idx

    # Cash and Accounts Receivable should be after Current Assets
    cash_idx = order.index("Cash")
    ar_idx = order.index("Accounts Receivable")
    assert current_assets_idx < cash_idx
    assert current_assets_idx < ar_idx

    # PPE should be after Fixed Assets
    ppe_idx = order.index("PPE")
    assert fixed_assets_idx < ppe_idx
