"""Service for processing financial data with hierarchical ordering."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
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
    write_processed_financial_data_to_db,
)
from db_interface.models import CompanyInfo, DailyCandle, FinancialData, Instrument, ProcessedFinancialData


logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Derived fundamentals pipeline
# ---------------------------------------------------------------------------

def get_price_data_for_nse_code(session: Session, nse_code: str) -> pd.DataFrame:
    """Load daily candle price history for a NSE code as a DataFrame.

    Joins ``instruments`` on ``tradingsymbol = nse_code`` and ``exchange = 'NSE'``
    then returns all ``daily_candles`` sorted ascending by date.

    Returns:
        DataFrame indexed by ``candle_date`` (``date`` objects) with columns
        ``open``, ``high``, ``low``, ``close``, ``volume``.  Empty DataFrame
        when no data is found.
    """
    stmt = (
        select(DailyCandle)
        .join(Instrument, DailyCandle.instrument_token == Instrument.instrument_token)
        .where(Instrument.tradingsymbol == nse_code)
        .where(Instrument.exchange == "NSE")
        .order_by(DailyCandle.candle_date)
    )
    candles = session.scalars(stmt).all()

    if not candles:
        logger.warning("No price data found in daily_candles for nse_code=%s", nse_code)
        return pd.DataFrame()

    records = [
        {
            "candle_date": c.candle_date,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]
    df = pd.DataFrame(records).set_index("candle_date")
    return df


def load_all_processed_sections(session: Session, nse_code: str) -> dict[str, pd.DataFrame]:
    """Load all pickled DataFrames from ``processed_financial_data`` for a NSE code.

    Returns:
        Mapping of ``{section_id: DataFrame}``.  Sections whose stored payload
        cannot be unpickled as a DataFrame are silently skipped.
    """
    stmt = select(ProcessedFinancialData).where(ProcessedFinancialData.nse_code == nse_code)
    records = session.scalars(stmt).all()

    sections: dict[str, pd.DataFrame] = {}
    for record in records:
        if not record.dataframe_pickle:
            continue
        try:
            deserialized = pickle.loads(record.dataframe_pickle)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not deserialize pickle for nse_code=%s section_id=%s: %s",
                nse_code, record.section_id, exc,
            )
            continue
        if isinstance(deserialized, pd.DataFrame):
            sections[record.section_id] = deserialized

    return sections


def derive_fundamentals_for_nse_code(
    session: Session,
    nse_code: str,
    transformers: list | None = None,
) -> pd.DataFrame:
    """Run the transformer pipeline and persist derived fundamentals for one NSE code.

    Steps:
    1. Load all processed section DataFrames from ``processed_financial_data``.
    2. Load daily price history from ``daily_candles``.
    3. Pass a :class:`~financial_data_processor.transformers.TransformContext`
       through each transformer in *transformers* (defaults to
       :data:`~financial_data_processor.transformers.DEFAULT_TRANSFORMERS`).
    4. Save the resulting DataFrame back to ``processed_financial_data`` with
       ``section_id = 'derived_fundamentals'``.

    Args:
        session:      Active SQLAlchemy session.
        nse_code:     NSE trading symbol to process.
        transformers: Optional custom transformer chain.  When ``None``, uses
                      :data:`~financial_data_processor.transformers.DEFAULT_TRANSFORMERS`.

    Returns:
        The derived-fundamentals DataFrame (index = metric names, columns =
        period labels).  Returns an empty DataFrame when no processed data is
        available for the NSE code.

    Raises:
        ValueError: When ``nse_code`` does not match any row in ``company_info``.
    """
    from .transformers import DEFAULT_TRANSFORMERS, TransformContext

    if transformers is None:
        transformers = DEFAULT_TRANSFORMERS

    sections = load_all_processed_sections(session, nse_code)
    if not sections:
        logger.warning("No processed financial data for nse_code=%s — skipping", nse_code)
        return pd.DataFrame()

    company_info = session.scalar(select(CompanyInfo).where(CompanyInfo.nse_code == nse_code))
    if company_info is None:
        raise ValueError(f"No company found in company_info for nse_code={nse_code!r}")

    price_data = get_price_data_for_nse_code(session, nse_code)

    context = TransformContext(
        nse_code=nse_code,
        sections=sections,
        price_data=price_data,
        company_info={"no_eq_shares": company_info.no_eq_shares},
    )

    result: pd.DataFrame = pd.DataFrame()
    for transformer in transformers:
        logger.info("Running transformer '%s' for nse_code=%s", transformer.name, nse_code)
        result = transformer.transform(context, result)

    if result.empty:
        logger.warning("No derived fundamentals produced for nse_code=%s", nse_code)
        return result

    write_processed_financial_data_to_db(
        session,
        company_id=company_info.company_id,
        section_id="derived_fundamentals",
        df=result,
        nse_code=nse_code,
    )

    logger.info(
        "Saved %d derived metric(s) for nse_code=%s (company_id=%s)",
        len(result.index), nse_code, company_info.company_id,
    )
    return result


def derive_fundamentals_all(
    session: Session,
    transformers: list | None = None,
) -> dict[str, int]:
    """Run the transformer pipeline for every NSE code that has processed data.

    Returns:
        Mapping of ``{nse_code: number_of_metrics_derived}``.  NSE codes for
        which the pipeline produces no output are still included with a count
        of ``0``.
    """
    nse_codes: list[str] = [
        row[0]
        for row in session.execute(
            select(ProcessedFinancialData.nse_code).distinct()
        ).all()
    ]

    results: dict[str, int] = {}
    for nse_code in nse_codes:
        result_df = derive_fundamentals_for_nse_code(session, nse_code, transformers)
        results[nse_code] = len(result_df.index)

    return results

