"""Command-line interface for financial data processor."""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_interface.models import CompanyInfo, FinancialData
from db_interface import load_processed_financial_dataframe_from_db

from .service import (
    derive_fundamentals_all,
    derive_fundamentals_for_nse_code,
    process_all_financial_data_database,
    process_and_write_financial_data_database,
)
from .settings import settings


def main() -> int:
    """Main CLI entry point."""
    try:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        parser = argparse.ArgumentParser(description="Process financial data and write to database")
        parser.add_argument(
            "--nse-code",
            dest="nse_code",
            help="Process only the company matched by NSE trading symbol (e.g. INFY)",
        )
        parser.add_argument(
            "--derive-fundamentals",
            dest="derive_fundamentals",
            action="store_true",
            default=False,
            help=(
                "Run the derived-fundamentals transformer pipeline instead of the "
                "raw-data ingestion pipeline.  Results are written back to "
                "processed_financial_data with section_id='derived_fundamentals'."
            ),
        )
        args = parser.parse_args()

        engine = create_engine(settings.database_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

        with SessionLocal() as session:
            # ----------------------------------------------------------------
            # Derived-fundamentals mode
            # ----------------------------------------------------------------
            if args.derive_fundamentals:
                if args.nse_code:
                    print(f"Deriving fundamentals for NSE code '{args.nse_code}'...")
                    result_df = derive_fundamentals_for_nse_code(session, args.nse_code)
                    if result_df.empty:
                        print("No derived fundamentals were produced.")
                    else:
                        print(f"✓ Derived {len(result_df.index)} metric(s):")
                        for metric in result_df.index:
                            print(f"  - {metric}")
                        print("\nDerived fundamentals DataFrame:")
                        print(result_df.to_string())
                else:
                    print("Deriving fundamentals for all NSE codes with processed data...")
                    results = derive_fundamentals_all(session)
                    total_metrics = sum(results.values())
                    print(
                        f"✓ Processed {len(results)} NSE code(s), "
                        f"{total_metrics} total metric row(s) derived"
                    )
                    for nse_code, count in sorted(results.items()):
                        print(f"  {nse_code}: {count} metric(s)")
                return 0

            # ----------------------------------------------------------------
            # Raw financial data ingestion mode (original behaviour)
            # ----------------------------------------------------------------
            if args.nse_code:
                print(f"Processing financial data for NSE code '{args.nse_code}' and writing to database...")
                company = session.query(CompanyInfo).filter(CompanyInfo.nse_code == args.nse_code).first()

                if company is None:
                    print(f"No company found with nse_code='{args.nse_code}'")
                    return 0

                section_rows = (
                    session.query(FinancialData.section_id)
                    .filter(FinancialData.company_id == company.company_id)
                    .distinct()
                    .all()
                )

                results: dict[str, dict[str, int]] = {company.company_id: {}}
                for (section_id,) in section_rows:
                    _, rows_written = process_and_write_financial_data_database(
                        session,
                        company.company_id,
                        section_id,
                    )
                    results[company.company_id][section_id] = rows_written
            else:
                print("Processing all financial data and writing to database...")
                results = process_all_financial_data_database(session)

            total = sum(len(sections) for sections in results.values())
            print(f"✓ Successfully processed {total} section(s) across {len(results)} company(ies)")

            for company_id, sections in results.items():
                print(f"  {company_id}:")
                for section_id, rows_written in sections.items():
                    print(f"    - {section_id}: {rows_written} rows written")

            # For symbol-scoped runs, show the persisted dataframe loaded back from DB pickle.
            if args.nse_code:
                for company_id, sections in results.items():
                    for section_id in sorted(sections.keys()):
                        restored_df = load_processed_financial_dataframe_from_db(
                            session,
                            company_id,
                            section_id,
                            nse_code=args.nse_code,
                        )
                        print(f"\nDataFrame loaded from DB pickle for {company_id}/{section_id}:")
                        print(restored_df)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
