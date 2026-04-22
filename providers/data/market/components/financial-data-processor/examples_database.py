"""Example usage of the financial data processor with database writing."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from financial_data_processor.service import (
    process_and_write_financial_data_database,
    process_all_financial_data_database,
    process_and_serialize_financial_data,
    process_all_financial_data,
)
from db_interface import (
    get_financial_data_with_nse_code,
    load_financial_dataframe,
)

# Setup database connection
DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/market_data"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ============================================================================
# DATABASE WRITING EXAMPLES (NEW)
# ============================================================================


def example_1_write_single_company_section_to_db():
    """Example 1: Process and write a single company's section to database."""
    with SessionLocal() as session:
        company_id = "0000000001"  # Replace with actual company_id
        section_id = "income_statement"
        
        df, rows_written = process_and_write_financial_data_database(
            session,
            company_id,
            section_id
        )
        
        print(f"✓ Processed {company_id}/{section_id}")
        print(f"  Rows written to database: {rows_written}")
        print(f"  DataFrame shape: {df.shape}")
        print(f"\nDataFrame preview:")
        print(df.head())


def example_2_write_all_data_to_db():
    """Example 2: Process and write all company/section combinations to database."""
    with SessionLocal() as session:
        print("Writing all financial data to database...")
        results = process_all_financial_data_database(session)
        
        total = sum(len(sections) for sections in results.values())
        print(f"\n✓ Successfully processed {total} section(s) across {len(results)} company(ies)")
        
        for company_id, sections in results.items():
            print(f"\n{company_id}:")
            for section_id, rows_written in sections.items():
                print(f"  - {section_id}: {rows_written} rows written")


def example_3_get_financial_data_with_nse_code():
    """Example 3: Get financial data with nse_code (trading symbol) from company_info."""
    with SessionLocal() as session:
        company_id = "0000000001"  # Replace with actual company_id
        section_id = "income_statement"
        
        df, nse_code = get_financial_data_with_nse_code(session, company_id, section_id)
        
        print(f"Company ID: {company_id}")
        print(f"NSE Code (Trading Symbol): {nse_code}")
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nColumn names (dates in chronological order):")
        print(df.columns.tolist())


def example_4_query_processed_financial_data_from_db():
    """Example 4: Query processed financial data from the database."""
    from sqlalchemy import select
    from db_interface import ProcessedFinancialData

    with SessionLocal() as session:
        company_id = "0000000001"  # Replace with actual company_id
        section_id = "income_statement"
        
        # Query processed financial data
        stmt = select(ProcessedFinancialData).where(
            ProcessedFinancialData.company_id == company_id,
            ProcessedFinancialData.section_id == section_id,
        ).order_by(ProcessedFinancialData.row_order)
        
        rows = session.scalars(stmt).all()
        
        print(f"Processed rows for {company_id}/{section_id}:")
        for row in rows[:10]:  # Show first 10
            print(
                f"  {row.row_order:3d}. {row.row_name:30s} "
                f"(type: {row.row_type}, parent: {row.parent_row})"
            )
        
        print(f"\nTotal rows: {len(rows)}")


# ============================================================================
# LEGACY EXAMPLES (PICKLE SERIALIZATION)
# ============================================================================


def example_5_process_single_company_section_pickle():
    """Example 5: Process and serialize a single company's section (pickle)."""
    with SessionLocal() as session:
        company_id = "0000000001"  # Replace with actual company_id
        section_id = "income_statement"
        
        df, output_path = process_and_serialize_financial_data(
            session,
            company_id,
            section_id,
            output_base_path="./data/financial_data"
        )
        
        print(f"✓ Processed {company_id}/{section_id}")
        print(f"  Saved to: {output_path}")
        print(f"  DataFrame shape: {df.shape}")
        print(f"\nDataFrame preview:")
        print(df.head())


def example_6_process_all_data_pickle():
    """Example 6: Process and serialize all company/section combinations (pickle)."""
    with SessionLocal() as session:
        print("Processing all financial data (pickle format)...")
        results = process_all_financial_data(
            session,
            output_base_path="./data/financial_data"
        )
        
        total = sum(len(sections) for sections in results.values())
        print(f"\n✓ Successfully processed {total} section(s) across {len(results)} company(ies)")
        
        for company_id, sections in results.items():
            print(f"\n{company_id}:")
            for section_id, path in sections.items():
                print(f"  - {section_id}: {path}")


def example_7_load_serialized_dataframe_pickle():
    """Example 7: Load a previously serialized dataframe (pickle)."""
    company_id = "0000000001"  # Replace with actual company_id
    section_id = "income_statement"
    
    df = load_financial_dataframe(
        company_id,
        section_id,
        data_path="./data/financial_data"
    )
    
    print(f"✓ Loaded {company_id}/{section_id}")
    print(f"  DataFrame shape: {df.shape}")
    print(f"\nColumns (sorted by date):")
    print(df.columns.tolist())
    print(f"\nRows (hierarchical order):")
    print(df.index.tolist())


# ============================================================================
# COMPARISON EXAMPLE
# ============================================================================


def example_8_compare_database_vs_pickle():
    """Example 8: Compare database writing vs pickle serialization."""
    import time
    
    with SessionLocal() as session:
        company_id = "0000000001"
        section_id = "income_statement"
        
        # Time database writing
        start_db = time.time()
        df_db, rows_db = process_and_write_financial_data_database(session, company_id, section_id)
        time_db = time.time() - start_db
        
        # Time pickle serialization
        start_pickle = time.time()
        df_pickle, path_pickle = process_and_serialize_financial_data(
            session,
            company_id,
            section_id,
            output_base_path="./data/financial_data"
        )
        time_pickle = time.time() - start_pickle
        
        print("Performance Comparison:")
        print(f"  Database write: {time_db:.3f}s ({rows_db} rows)")
        print(f"  Pickle serialization: {time_pickle:.3f}s ({path_pickle})")
        print(f"\nBoth methods produce the same data:")
        print(f"  Same shape: {df_db.shape == df_pickle.shape}")
        print(f"  Same rows: {list(df_db.index) == list(df_pickle.index)}")
        print(f"  Same columns: {list(df_db.columns) == list(df_pickle.columns)}")


if __name__ == "__main__":
    print("=" * 80)
    print("DATABASE WRITING EXAMPLES")
    print("=" * 80)
    
    print("\n1. Write single company/section to database:")
    example_1_write_single_company_section_to_db()
    
    print("\n" + "=" * 80)
    print("\n2. Write all data to database:")
    # Uncomment to run: example_2_write_all_data_to_db()
    print("(Uncomment in code to run)")
    
    print("\n" + "=" * 80)
    print("\n3. Get data with NSE code (trading symbol):")
    example_3_get_financial_data_with_nse_code()
    
    print("\n" + "=" * 80)
    print("\n4. Query processed data from database:")
    example_4_query_processed_financial_data_from_db()
    
    # Uncomment below to see legacy pickle examples
    
    # print("\n" + "=" * 80)
    # print("LEGACY PICKLE EXAMPLES")
    # print("=" * 80)
    # example_5_process_single_company_section_pickle()
    # example_6_process_all_data_pickle()
    # example_7_load_serialized_dataframe_pickle()
    # example_8_compare_database_vs_pickle()
