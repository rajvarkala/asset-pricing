"""Example usage of the financial data processor."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from financial_data_processor.service import (
    process_all_financial_data,
    process_and_serialize_financial_data,
)
from db_interface import (
    get_financial_data_dataframe,
    load_financial_dataframe,
)

# Setup database connection
DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/market_data"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def example_1_process_single_company_section():
    """Example 1: Process and serialize a single company's section."""
    with SessionLocal() as session:
        company_id = "AAPL"
        section_id = "income_statement"
        
        df, output_path = process_and_serialize_financial_data(
            session,
            company_id,
            section_id,
            output_base_path="./data/financial_data"
        )
        
        print(f"Processed {company_id}/{section_id}")
        print(f"Saved to: {output_path}")
        print(f"Shape: {df.shape}")
        print(f"\nDataFrame preview:")
        print(df.head())


def example_2_process_all_data():
    """Example 2: Process and serialize all company/section combinations."""
    with SessionLocal() as session:
        results = process_all_financial_data(
            session,
            output_base_path="./data/financial_data"
        )
        
        total = sum(len(sections) for sections in results.values())
        print(f"Successfully processed {total} section(s) across {len(results)} company(ies)")
        
        for company_id, sections in results.items():
            print(f"\n{company_id}:")
            for section_id, path in sections.items():
                print(f"  - {section_id}: {path}")


def example_3_load_serialized_dataframe():
    """Example 3: Load a previously serialized dataframe."""
    company_id = "AAPL"
    section_id = "income_statement"
    
    df = load_financial_dataframe(
        company_id,
        section_id,
        data_path="./data/financial_data"
    )
    
    print(f"Loaded {company_id}/{section_id}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns (sorted by date):")
    print(df.columns.tolist())
    print(f"\nRows (hierarchical order):")
    print(df.index.tolist())


def example_4_get_dataframe_directly():
    """Example 4: Get a dataframe directly without serialization."""
    with SessionLocal() as session:
        company_id = "AAPL"
        section_id = "income_statement"
        
        df = get_financial_data_dataframe(session, company_id, section_id)
        
        print(f"Direct query for {company_id}/{section_id}")
        print(f"Shape: {df.shape}")
        print(f"\nDataFrame:")
        print(df)


def example_5_inspect_dataframe_structure():
    """Example 5: Inspect the structure of a loaded dataframe."""
    company_id = "AAPL"
    section_id = "income_statement"
    
    df = load_financial_dataframe(
        company_id,
        section_id,
        data_path="./data/financial_data"
    )
    
    print(f"DataFrame structure for {company_id}/{section_id}:")
    print(f"\nShape: {df.shape}")
    print(f"Rows (hierarchical):")
    for idx, row_name in enumerate(df.index, 1):
        print(f"  {idx:3d}. {row_name}")
    
    print(f"\nColumns (chronologically ordered with TTM last):")
    for idx, col in enumerate(df.columns, 1):
        print(f"  {idx:3d}. {col}")
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nNull values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    print("=== Example 1: Process Single Company/Section ===")
    example_1_process_single_company_section()
    
    print("\n" + "="*50)
    print("=== Example 2: Process All Data ===")
    example_2_process_all_data()
    
    # Note: Examples 3-5 require data to already be serialized
    # Uncomment after running examples 1-2
    
    # print("\n" + "="*50)
    # print("=== Example 3: Load Serialized DataFrame ===")
    # example_3_load_serialized_dataframe()
    
    # print("\n" + "="*50)
    # print("=== Example 4: Get DataFrame Directly ===")
    # example_4_get_dataframe_directly()
    
    # print("\n" + "="*50)
    # print("=== Example 5: Inspect DataFrame Structure ===")
    # example_5_inspect_dataframe_structure()
