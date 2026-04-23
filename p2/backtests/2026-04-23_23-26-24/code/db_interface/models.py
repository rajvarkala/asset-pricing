from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
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
