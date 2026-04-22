"""Service for processing financial data with hierarchical ordering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from db_interface import (
    build_hierarchical_row_order,
    deserialize_dataframe,
    get_date_columns,
    get_financial_data_dataframe,
    get_financial_data_with_nse_code,
    process_all_financial_data_to_db,
    process_and_write_financial_data_to_db,
    serialize_dataframe,
    sort_date_columns,
)
from db_interface.models import FinancialData


def process_and_serialize_financial_data(
    session: Session,
    company_id: str,
    section_id: str,
    output_base_path: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Process financial data and serialize it to disk (legacy method - pickle).
    
    Returns the DataFrame and the path where it was saved.
    """
    from .settings import settings

    if output_base_path is None:
        output_base_path = settings.serialized_data_path

    output_path = Path(output_base_path) / company_id / f"{section_id}.pkl"

    df = get_financial_data_dataframe(session, company_id, section_id)
    serialize_dataframe(df, output_path)

    return df, output_path


def process_all_financial_data(
    session: Session,
    output_base_path: str | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Process all financial data and serialize to disk (legacy method - pickle).
    
    Returns a mapping of {company_id: {section_id: output_path}}.
    """
    from .settings import settings

    if output_base_path is None:
        output_base_path = settings.serialized_data_path

    # Get unique company_id and section_id combinations
    from sqlalchemy import select

    stmt = select(FinancialData.company_id, FinancialData.section_id).distinct()
    combinations = session.execute(stmt).all()

    results: dict[str, dict[str, Path]] = {}

    for company_id, section_id in combinations:
        _, output_path = process_and_serialize_financial_data(
            session,
            company_id,
            section_id,
            output_base_path,
        )

        if company_id not in results:
            results[company_id] = {}

        results[company_id][section_id] = output_path

    return results


def process_and_write_financial_data_database(
    session: Session,
    company_id: str,
    section_id: str,
) -> tuple[pd.DataFrame, int]:
    """
    Process financial data and write to PostgreSQL database.
    
    Uses nse_code from company_info as the trading symbol.
    Writes hierarchically ordered, date-sorted rows to processed_financial_data table.
    
    Returns:
        Tuple of (DataFrame, rows_written)
    """
    return process_and_write_financial_data_to_db(session, company_id, section_id)


def process_all_financial_data_database(session: Session) -> dict[str, dict[str, int]]:
    """
    Process all financial data and write to PostgreSQL database.
    
    Uses nse_code from company_info as the trading symbol for each company.
    Writes hierarchically ordered, date-sorted rows to processed_financial_data table.
    
    Returns:
        Mapping of {company_id: {section_id: rows_written}}
    """
    return process_all_financial_data_to_db(session)

