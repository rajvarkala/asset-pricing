from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
INITIAL_UNIVERSE_SIZE = 3000
START_DATE = "2016-06-30"
END_DATE = "2026-05-31"
ADD_PER_YEAR = 60
REMOVE_PER_YEAR = 40


def make_ticker(n: int) -> str:
    return f"EQ{n:05d}"


def main() -> None:
    rng = np.random.default_rng(SEED)
    out_dir = Path(__file__).resolve().parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    months = pd.date_range(START_DATE, END_DATE, freq="ME")
    year_ends = set(m for m in months if m.month == 12)

    next_id = INITIAL_UNIVERSE_SIZE + 1
    active = [make_ticker(i) for i in range(1, INITIAL_UNIVERSE_SIZE + 1)]

    state = pd.DataFrame(
        {
            "ticker": active,
            "price": rng.uniform(8.0, 250.0, size=len(active)),
            "shares_outstanding": rng.integers(20_000_000, 2_000_000_000, size=len(active)),
            "valuation_score": rng.normal(loc=0.0, scale=1.0, size=len(active)),
            "quality": rng.normal(loc=0.0, scale=1.0, size=len(active)),
            "beta": rng.normal(loc=1.0, scale=0.25, size=len(active)).clip(0.3, 2.2),
        }
    )

    rows = []
    membership_rows = []

    market_drift = 0.006
    market_vol = 0.035

    for date in months:
        market_shock = rng.normal(market_drift, market_vol)

        valuation_noise = rng.normal(0.0, 0.08, size=len(state))
        state["valuation_score"] = 0.97 * state["valuation_score"] + valuation_noise

        alpha = 0.0025 * state["valuation_score"] + 0.0015 * state["quality"]
        eps = rng.normal(0.0, 0.055, size=len(state))
        ret = alpha + state["beta"] * market_shock + eps
        ret = np.clip(ret, -0.75, 1.5)

        state["price"] = np.clip(state["price"] * (1.0 + ret), 0.5, None)
        state["market_cap"] = state["price"] * state["shares_outstanding"]

        pe = np.exp(2.85 - 0.35 * state["valuation_score"] + rng.normal(0.0, 0.18, size=len(state)))
        pe = np.clip(pe, 4.0, 90.0)
        pb = np.exp(0.85 - 0.22 * state["valuation_score"] + rng.normal(0.0, 0.14, size=len(state)))
        pb = np.clip(pb, 0.4, 20.0)

        turnover = np.exp(rng.normal(np.log(0.22), 0.35, size=len(state)))
        turnover = np.clip(turnover, 0.01, 2.5)
        volume = (turnover * state["shares_outstanding"]).astype(np.int64)

        month_df = pd.DataFrame(
            {
                "date": date,
                "ticker": state["ticker"],
                "price": state["price"].round(4),
                "return": ret.round(6),
                "market_cap": state["market_cap"].round(2),
                "pe_ratio": pe.round(4),
                "pb_ratio": pb.round(4),
                "valuation_score": state["valuation_score"].round(6),
                "shares_outstanding": state["shares_outstanding"],
                "volume": volume,
            }
        )
        rows.append(month_df)

        membership_rows.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker": state["ticker"],
                    "in_universe": 1,
                }
            )
        )

        if date in year_ends:
            cutoff = state["valuation_score"].quantile(0.10)
            bottom_decile = state[state["valuation_score"] <= cutoff]
            remove_n = min(REMOVE_PER_YEAR, len(bottom_decile))
            to_remove = bottom_decile.nsmallest(remove_n, "valuation_score")["ticker"]
            state = state[~state["ticker"].isin(to_remove)].reset_index(drop=True)

            new_tickers = [make_ticker(i) for i in range(next_id, next_id + ADD_PER_YEAR)]
            next_id += ADD_PER_YEAR
            entrants = pd.DataFrame(
                {
                    "ticker": new_tickers,
                    "price": rng.uniform(7.0, 120.0, size=ADD_PER_YEAR),
                    "shares_outstanding": rng.integers(15_000_000, 1_500_000_000, size=ADD_PER_YEAR),
                    "valuation_score": rng.normal(loc=0.15, scale=0.95, size=ADD_PER_YEAR),
                    "quality": rng.normal(loc=0.0, scale=1.0, size=ADD_PER_YEAR),
                    "beta": rng.normal(loc=1.05, scale=0.3, size=ADD_PER_YEAR).clip(0.3, 2.4),
                }
            )
            state = pd.concat([state, entrants], ignore_index=True)

    panel = pd.concat(rows, ignore_index=True)
    membership = pd.concat(membership_rows, ignore_index=True)

    panel.to_csv(out_dir / "synthetic_equity_monthly_10y.csv", index=False)
    membership.to_csv(out_dir / "synthetic_universe_membership_10y.csv", index=False)

    yearly = (
        panel.assign(year=lambda x: pd.to_datetime(x["date"]).dt.year)
        .groupby("year", as_index=False)
        .agg(
            tickers=("ticker", "nunique"),
            median_pe=("pe_ratio", "median"),
            median_pb=("pb_ratio", "median"),
            median_market_cap=("market_cap", "median"),
        )
    )
    yearly.to_csv(out_dir / "synthetic_market_summary_by_year.csv", index=False)

    print("rows", len(panel))
    print("unique_tickers", panel["ticker"].nunique())
    print("months", panel["date"].nunique())
    print("min_tickers_per_month", panel.groupby("date")["ticker"].nunique().min())
    print("max_tickers_per_month", panel.groupby("date")["ticker"].nunique().max())
    print("out_dir", out_dir)


if __name__ == "__main__":
    main()
