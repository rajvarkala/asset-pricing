from db_interface.models import DailyCandle, Instrument


def test_model_tablenames() -> None:
    assert Instrument.__tablename__ == "instruments"
    assert DailyCandle.__tablename__ == "daily_candles"
