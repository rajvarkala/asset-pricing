from datetime import date

from historical_crawler.service import start_date_25_years_ago


def test_start_date_25_years_ago() -> None:
    today = date(2026, 4, 20)
    assert start_date_25_years_ago(today) == date(2001, 4, 20)
