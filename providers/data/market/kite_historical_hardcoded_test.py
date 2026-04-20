from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import os

from kiteconnect import KiteConnect

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
INSTRUMENT_TOKEN = 738561


def main() -> None:
    if not API_KEY or not ACCESS_TOKEN:
        raise SystemExit("KITE_API_KEY and KITE_ACCESS_TOKEN must be set in .env")

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

    print("Testing profile auth...")
    profile = kite.profile()
    print("Profile OK. user_id=", profile.get("user_id"))

    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=10)

    print("Fetching historical data...")
    candles = kite.historical_data(
        instrument_token=INSTRUMENT_TOKEN,
        from_date=from_date,
        to_date=to_date,
        interval="day",
        continuous=False,
        oi=False,
    )
    print(f"Historical OK. candles={len(candles)}")
    if candles:
        print("First:", candles[0])
        print("Last:", candles[-1])


if __name__ == "__main__":
    main()
