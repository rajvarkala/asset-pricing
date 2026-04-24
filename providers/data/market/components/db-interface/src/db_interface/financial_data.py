"""Financial data retrieval and serialization utilities."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.orm import Session

from .models import CompanyInfo, FinancialData, ProcessedFinancialData
from .settings import settings


logger = logging.getLogger(__name__)


def _quote_identifier(identifier: str) -> str:
    """Safely quote SQL identifiers (column names)."""
    return '"' + identifier.replace('"', '""') + '"'


def get_date_columns(session: Session, table_name: str = "financial_data") -> list[str]:
    """
    Get all date columns from the financial_data table using information_schema.
    
    Excludes system columns (id, company_id, section_id, row_name, row_type, 
    parent_row, created_at, updated_at).
    """
    system_columns = {
        "id",
        "company_id",
        "section_id",
        "row_name",
        "row_type",
        "parent_row",
        "created_at",
        "updated_at",
    }

    inspector = inspect(session.bind)
    all_columns = inspector.get_columns(table_name)

    date_columns = [col["name"] for col in all_columns if col["name"] not in system_columns]

    return sorted(date_columns)


def parse_date_column_name(column_name: str) -> tuple[int, int, bool]:
    """
    Parse a date column name and return (year, month, is_ttm).
    
    Examples:
        "Mar 2023" -> (2023, 3, False)
        "Dec 2024" -> (2024, 12, False)
        "TTM" -> (0, 0, True)
    """
    if column_name == "TTM":
        return (0, 0, True)

    months = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    parts = column_name.split()
    if len(parts) >= 2:
        month_str = parts[0]
        # Handle cases like "Dec 2023 8m" where we only care about the date part
        year_str = parts[1]

        try:
            month = months.get(month_str, 0)
            year = int(year_str)
            return (year, month, False)
        except ValueError:
            pass

    return (0, 0, False)


def sort_date_columns(columns: list[str]) -> list[str]:
    """
    Sort date columns chronologically, with TTM always at the end.
    
    Columns are sorted by (year, month), and TTM is appended last.
    """
    ttm_col = None
    date_cols = []

    for col in columns:
        if col == "TTM":
            ttm_col = col
        else:
            date_cols.append(col)

    # Sort by parsed date (year, month)
    sorted_cols = sorted(date_cols, key=lambda x: parse_date_column_name(x)[:2])

    if ttm_col:
        sorted_cols.append(ttm_col)

    return sorted_cols


def build_hierarchical_row_order(
    rows_data: list[dict[str, Any]],
) -> list[str]:
    """
    Build hierarchical row ordering from flat list using parent_row references.
    
    Returns list of row_names in hierarchical order where children come after parents.
    """
    # Create lookup maps
    row_by_name: dict[str, dict[str, Any]] = {row["row_name"]: row for row in rows_data}
    children_by_parent: dict[str | None, list[str]] = {}

    for row in rows_data:
        parent = row.get("parent_row")
        if parent == "":
            parent = None
        if parent not in children_by_parent:
            children_by_parent[parent] = []
        children_by_parent[parent].append(row["row_name"])

    # DFS traversal to build hierarchical order
    result: list[str] = []

    def dfs(parent_name: str | None) -> None:
        """Depth-first traversal of row hierarchy."""
        if parent_name in children_by_parent:
            for child_name in children_by_parent[parent_name]:
                result.append(child_name)
                dfs(child_name)

    # Start from root rows (parent_row is None)
    dfs(None)

    # Include orphaned/cyclic rows that were not reached from roots.
    if len(result) < len(row_by_name):
        for row_name in row_by_name:
            if row_name not in result:
                result.append(row_name)
                dfs(row_name)

    return result


def get_financial_data_dataframe(
    session: Session,
    company_id: str,
    section_id: str,
) -> pd.DataFrame:
    """
    Get financial data for a company and section as a DataFrame.
    
    Returns a DataFrame where:
    - Index is row_name
    - Columns are date columns (sorted chronologically with TTM last)
    - Rows are ordered hierarchically based on parent_row relationships
    - All-NULL columns are removed
    """
    # Get all date columns
    date_columns = get_date_columns(session)

    base_columns = ["row_name", "parent_row", "row_type"]
    selected_columns = base_columns + date_columns
    select_clause = ", ".join(_quote_identifier(column) for column in selected_columns)

    # Query raw rows so dynamically discovered date columns are actually retrieved.
    stmt = text(
        f"SELECT {select_clause} "
        "FROM financial_data "
        "WHERE company_id = :company_id AND section_id = :section_id"
    )
    rows_data = [
        dict(row)
        for row in session.execute(
            stmt,
            {"company_id": company_id, "section_id": section_id},
        ).mappings()
    ]

    if not rows_data:
        return pd.DataFrame()

    # Build hierarchical row ordering
    hierarchical_rows = build_hierarchical_row_order(rows_data)

    # Create DataFrame with hierarchical row order
    row_lookup = {row["row_name"]: row for row in rows_data}
    data_list = [row_lookup[row_name] for row_name in hierarchical_rows if row_name in row_lookup]

    df = pd.DataFrame(data_list)

    if df.empty:
        return df

    # Set index to row_name
    df = df.set_index("row_name")

    # Remove metadata columns
    df = df.drop(columns=["parent_row", "row_type"], errors="ignore")

    # Reorder columns chronologically
    sorted_cols = sort_date_columns(df.columns.tolist())
    df = df[[col for col in sorted_cols if col in df.columns]]

    pre_drop_columns = df.columns.tolist()

    # Remove columns that are all NULL for this section.
    df = df.dropna(axis=1, how="all")

    dropped_columns = [column for column in pre_drop_columns if column not in df.columns]
    logger.info(
        "Processed section company_id=%s section_id=%s rows=%d candidate_cols=%d kept_cols=%d dropped_all_null_cols=%d",
        company_id,
        section_id,
        len(df.index),
        len(pre_drop_columns),
        len(df.columns),
        len(dropped_columns),
    )
    if dropped_columns:
        logger.info("Dropped all-null columns for %s/%s: %s", company_id, section_id, dropped_columns)

    return df


def serialize_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Serialize DataFrame to disk using pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(df, f)


def deserialize_dataframe(input_path: Path) -> pd.DataFrame:
    """Deserialize DataFrame from disk."""
    if not input_path.exists():
        raise FileNotFoundError(f"Dataframe file not found: {input_path}")

    with open(input_path, "rb") as f:
        return pickle.load(f)


def load_financial_dataframe(
    company_id: str,
    section_id: str,
    data_path: str | None = None,
) -> pd.DataFrame:
    """
    Load a serialized financial dataframe for a company and section.
    
    Args:
        company_id: The company identifier
        section_id: The section identifier
        data_path: Optional base path for serialized data (uses settings default if not provided)
    
    Returns:
        The deserialized pandas DataFrame
    """
    if data_path is None:
        data_path = settings.serialized_data_path

    dataframe_path = Path(data_path) / company_id / f"{section_id}.pkl"
    return deserialize_dataframe(dataframe_path)


def get_financial_data_with_nse_code(
    session: Session,
    company_id: str,
    section_id: str,
) -> tuple[pd.DataFrame, str | None]:
    """
    Get financial data for a company and section, joining with company_info to get nse_code.
    
    Returns a tuple of (DataFrame, nse_code) where:
    - DataFrame has same structure as get_financial_data_dataframe
    - nse_code is the trading symbol from company_info table
    """
    # Get the company info to retrieve nse_code (trading symbol)
    stmt_company = select(CompanyInfo).where(CompanyInfo.company_id == company_id)
    company_info = session.scalar(stmt_company)

    nse_code = company_info.nse_code if company_info else None

    # Get the financial data dataframe
    df = get_financial_data_dataframe(session, company_id, section_id)

    return df, nse_code


def write_processed_financial_data_to_db(
    session: Session,
    company_id: str,
    section_id: str,
    df: pd.DataFrame,
    nse_code: str | None = None,
) -> int:
    """
    Write processed financial data to the processed_financial_data table.
    
    Args:
        session: Database session
        company_id: Company identifier
        section_id: Section identifier
        df: Processed DataFrame with hierarchical rows and sorted date columns
        nse_code: Trading symbol (from company_info). If None, will query from company_info
    
    Returns:
        Number of rows written (always 1 when data is present)
    """
    # Get nse_code if not provided
    if not nse_code:
        stmt_company = select(CompanyInfo).where(CompanyInfo.company_id == company_id)
        company_info = session.scalar(stmt_company)
        nse_code = company_info.nse_code if company_info else None

    if not nse_code:
        raise ValueError(f"nse_code not found for company_id: {company_id}")

    # Delete existing records for this company/section and flush immediately
    # so re-inserts in the same transaction do not violate unique constraints.
    session.execute(
        delete(ProcessedFinancialData).where(
            ProcessedFinancialData.company_id == company_id,
            ProcessedFinancialData.section_id == section_id,
        )
    )
    session.flush()

    # Persist a single serialized dataframe payload for this company/section.
    payload = pickle.dumps(df)
    processed_record = ProcessedFinancialData(
        company_id=company_id,
        nse_code=nse_code,
        section_id=section_id,
        dataframe_pickle=payload,
        row_name="__dataframe_pickle__",
        row_type="pickle",
        parent_row=None,
        row_order=0,
    )
    session.add(processed_record)

    session.commit()
    return 1


def load_processed_financial_dataframe_from_db(
    session: Session,
    company_id: str,
    section_id: str,
    nse_code: str | None = None,
) -> pd.DataFrame:
    """Load a processed financial dataframe stored as a pickled payload in the database."""
    stmt = select(ProcessedFinancialData).where(
        ProcessedFinancialData.company_id == company_id,
        ProcessedFinancialData.section_id == section_id,
    )

    if nse_code:
        stmt = stmt.where(ProcessedFinancialData.nse_code == nse_code)

    record = session.scalar(stmt)
    if record is None or not record.dataframe_pickle:
        return pd.DataFrame()

    deserialized = pickle.loads(record.dataframe_pickle)
    if isinstance(deserialized, pd.DataFrame):
        return deserialized

    raise TypeError("Stored payload is not a pandas DataFrame")


def process_and_write_financial_data_to_db(
    session: Session,
    company_id: str,
    section_id: str,
) -> tuple[pd.DataFrame, int]:
    """
    Process financial data and write to the database.
    
    Returns:
        Tuple of (DataFrame, rows_written)
    """
    # Get financial data with nse_code
    df, nse_code = get_financial_data_with_nse_code(session, company_id, section_id)

    if df.index.empty:
        return df, 0

    # Write to database
    rows_written = write_processed_financial_data_to_db(session, company_id, section_id, df, nse_code)

    return df, rows_written


def process_all_financial_data_to_db(session: Session) -> dict[str, dict[str, int]]:
    """
    Process all financial data and write to database.
    
    Returns:
        Mapping of {company_id: {section_id: rows_written}}
    """
    # Get unique company_id and section_id combinations
    stmt = select(FinancialData.company_id, FinancialData.section_id).distinct()
    combinations = session.execute(stmt).all()

    results: dict[str, dict[str, int]] = {}

    for company_id, section_id in combinations:
        try:
            _, rows_written = process_and_write_financial_data_to_db(session, company_id, section_id)
        except ValueError:
            continue

        if company_id not in results:
            results[company_id] = {}

        results[company_id][section_id] = rows_written

    return results
