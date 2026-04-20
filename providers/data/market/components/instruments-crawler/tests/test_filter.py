from instruments_crawler.service import _is_equity_or_index


def test_filter_equity() -> None:
    row = {"exchange": "NSE", "segment": "NSE", "instrument_type": "EQ"}
    assert _is_equity_or_index(row)


def test_filter_index() -> None:
    row = {"exchange": "NSE", "segment": "INDICES", "instrument_type": ""}
    assert _is_equity_or_index(row)


def test_filter_reject_non_nse() -> None:
    row = {"exchange": "BSE", "segment": "BSE", "instrument_type": "EQ"}
    assert not _is_equity_or_index(row)
