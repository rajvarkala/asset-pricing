"""Command-line interface for financial data processor."""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_interface.models import CompanyInfo, FinancialData
from db_interface import load_processed_financial_dataframe_from_db

from .service import process_all_financial_data_database, process_and_write_financial_data_database
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
        args = parser.parse_args()

        engine = create_engine(settings.database_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

        with SessionLocal() as session:
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
