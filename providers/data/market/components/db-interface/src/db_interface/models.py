from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, LargeBinary, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Instrument(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_token: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    exchange_token: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tradingsymbol: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exchange: Mapped[str] = mapped_column(String(32), index=True)
    segment: Mapped[str] = mapped_column(String(32), index=True)
    instrument_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tick_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    lot_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DailyCandle(Base):
    __tablename__ = "daily_candles"
    __table_args__ = (UniqueConstraint("instrument_token", "candle_date", name="uq_candles_instr_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_token: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("instruments.instrument_token", ondelete="CASCADE"),
        index=True,
    )
    candle_date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    oi: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CrawlState(Base):
    __tablename__ = "crawl_state"
    __table_args__ = (UniqueConstraint("crawler_name", "scope", name="uq_crawler_scope"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    crawler_name: Mapped[str] = mapped_column(String(64), index=True)
    scope: Mapped[str] = mapped_column(String(128), index=True)
    cursor_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class CompanyInfo(Base):
    """Company information and metadata."""

    __tablename__ = "company_info"

    company_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
    company_name: Mapped[str] = mapped_column(String(255))
    company_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(100), nullable=True)
    isin: Mapped[str | None] = mapped_column(String(20), nullable=True)
    data_warehouse_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    export_excel_link: Mapped[str | None] = mapped_column(String(500), nullable=True)
    company_sector: Mapped[str | None] = mapped_column(String, nullable=True)  # JSON stored as string
    bse_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    nse_code: Mapped[str | None] = mapped_column(String(20), index=True, nullable=True)
    company_website: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    type: Mapped[str] = mapped_column(String(50), default="company")
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_low: Mapped[str | None] = mapped_column(String(100), nullable=True)
    stock_pe: Mapped[float | None] = mapped_column(Float, nullable=True)
    book_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    dividend_yield: Mapped[float | None] = mapped_column(Float, nullable=True)
    roce: Mapped[float | None] = mapped_column(Float, nullable=True)
    roe: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    sales: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    int_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    industry_pe: Mapped[float | None] = mapped_column(Float, nullable=True)
    sales_growth: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_to_sales: Mapped[float | None] = mapped_column(Float, nullable=True)
    evebitda: Mapped[float | None] = mapped_column(Float, nullable=True)
    debt: Mapped[float | None] = mapped_column(Float, nullable=True)
    promoter_holding: Mapped[float | None] = mapped_column(Float, nullable=True)
    change_in_prom_hold: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_on_assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_growth: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_after_tax: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_yield: Mapped[float | None] = mapped_column(Float, nullable=True)
    no_eq_shares: Mapped[float | None] = mapped_column(Float, nullable=True)


class FinancialData(Base):
    """Financial data model for hierarchical financial statements."""

    __tablename__ = "financial_data"
    __table_args__ = (UniqueConstraint("company_id", "section_id", "row_name", name="financial_data_unique_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[str] = mapped_column(String(255), index=True)
    section_id: Mapped[str] = mapped_column(String(255), index=True)
    row_name: Mapped[str] = mapped_column(String(255), index=True)
    row_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    parent_row: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProcessedFinancialData(Base):
    """Processed financial data stored as a serialized DataFrame payload."""

    __tablename__ = "processed_financial_data"
    __table_args__ = (
        UniqueConstraint("company_id", "nse_code", "section_id", name="processed_financial_data_unique_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[str] = mapped_column(String(255), index=True)
    nse_code: Mapped[str] = mapped_column(String(20), index=True)  # Trading symbol from company_info
    section_id: Mapped[str] = mapped_column(String(255), index=True)
    dataframe_pickle: Mapped[bytes] = mapped_column(LargeBinary)
    row_name: Mapped[str] = mapped_column(String(255), index=True, default="__dataframe_pickle__")
    row_type: Mapped[str | None] = mapped_column(String(50), nullable=True, default="pickle")
    parent_row: Mapped[str | None] = mapped_column(String(255), nullable=True)
    row_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
