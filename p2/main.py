# region imports
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import pandas as pd
from AlgorithmImports import *

# Make staged shared modules importable inside Lean Docker.
PROJECT_ROOT = Path(__file__).resolve().parent
ML_INPUTS_ROOT = PROJECT_ROOT / "ml_inputs"
if str(ML_INPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_INPUTS_ROOT))

from ml_pipeline.data_loader import DataLoader
from ml_pipeline.models import ModelRegistry
from datetime import datetime, timedelta


def _synthetic_equity_path() -> Path:
    return ML_INPUTS_ROOT / "synthetic_data" / "synthetic_equity_monthly_10y.csv"


def _synthetic_equity_split_dir() -> Path:
    return ML_INPUTS_ROOT / "synthetic_data" / "by_ticker"


def _synthetic_equity_ticker_path(ticker: str) -> Path:
    return _synthetic_equity_split_dir() / f"{ticker}.csv"


class SyntheticEquity(PythonData):
    """Monthly custom data backed by synthetic equity CSV (one row per ticker per month-end)."""

    def get_source(self, config, date, is_live_mode):
        ticker_path = _synthetic_equity_ticker_path(config.symbol.value)
        path = str(ticker_path if ticker_path.exists() else _synthetic_equity_path())
        return SubscriptionDataSource(path, SubscriptionTransportMedium.LOCAL_FILE)

    def reader(self, config, line, date, is_live_mode):
        if not line or line.startswith("date"):
            return None
        try:
            parts = line.split(",")
            if parts[1].strip() != config.symbol.value:
                return None
            obj = SyntheticEquity()
            obj.symbol = config.symbol
            dt = datetime.strptime(parts[0].strip(), "%Y-%m-%d")
            obj.time = dt
            obj.end_time = dt + timedelta(days=1)
            obj.value = float(parts[2])          # synthetic price
            obj["monthly_return"] = float(parts[3])
            obj["market_cap"] = float(parts[4])
            return obj
        except Exception:
            return None


# endregion


def _safe_float(value, default=0.0):
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _metric_str(value: float, precision: int = 8) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{float(value):.{precision}f}"


def _ols_alpha_and_betas(y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Return intercept(alpha) and betas from OLS via least squares."""
    if y.size == 0:
        return np.nan, np.full(x.shape[1] if x.ndim == 2 else 0, np.nan)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    valid = np.isfinite(y)
    valid &= np.all(np.isfinite(x), axis=1)
    yv = y[valid]
    xv = x[valid]

    if yv.size < max(3, xv.shape[1] + 2):
        return np.nan, np.full(xv.shape[1], np.nan)

    design = np.column_stack([np.ones(len(yv)), xv])
    coeffs, *_ = np.linalg.lstsq(design, yv, rcond=None)
    return float(coeffs[0]), coeffs[1:]


class MLMonthlyEventDrivenBacktest(QCAlgorithm):
    """
    Lean event-driven monthly rebalancing backtest for one model.

    - Reuse shared ml_pipeline DataLoader and ModelRegistry
    - Features are lagged by 1 month in DataLoader.prepare()
    - Portfolio each month: long top decile, short bottom decile (value-weighted)
    - Metrics exported to Lean summary statistics and local files
    """

    def initialize(self) -> None:
        self.set_cash(10_000_000)

        self.model_id = (self.get_parameter("model-id") or "OLS-3_L2").strip()
        self.report_dir_param = (self.get_parameter("report-dir") or "backtests/latest/reports").strip()
        max_tickers_raw = (self.get_parameter("max-tickers") or "").strip()
        self.max_tickers = int(max_tickers_raw) if max_tickers_raw else None

        self._project_root = PROJECT_ROOT
        self._inputs_root = ML_INPUTS_ROOT

        best_params_path = self._inputs_root / "best_params.json"
        if not best_params_path.exists():
            raise FileNotFoundError(f"Missing cached model params: {best_params_path}")

        with open(best_params_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)
        self.model_params = all_params.get(self.model_id, {})

        self._load_and_prepare_data()

        self.set_start_date(self.test_dates[0].year, self.test_dates[0].month, 1)
        self.set_end_date(self.test_dates[-1].year, self.test_dates[-1].month, 28)

        self.strategy_nav = 1.0
        self.benchmark_nav = 1.0
        self.prev_weights: Dict[str, float] = {}
        self.rebalance_records: List[dict] = []
        self._benchmark_history: List[tuple[pd.Timestamp, float]] = []
        self._month_records: Dict[str, dict] = {}
        self._month_weights: Dict[str, Dict[str, float]] = {}
        self._test_month_lookup: Dict[str, pd.Timestamp] = {
            pd.Timestamp(month_end).strftime("%Y-%m"): pd.Timestamp(month_end)
            for month_end in self.test_dates
        }
        self._last_rebalance_month: str = ""

        self.debug(f"Model={self.model_id} | Train/Test split=30/70 by months")
        self.debug(f"Universe tickers loaded: {len(self.universe_tickers)}")
        if self.max_tickers is not None:
            self.debug(f"Ticker cap enabled: {self.max_tickers}")
        self.debug(f"Test months: {len(self.test_dates)} ({self.test_dates[0].date()} to {self.test_dates[-1].date()})")

        self.set_benchmark(self._benchmark_value)

        # Split the shared monthly custom data into per-ticker files so each Lean
        # subscription reads only its own rows instead of scanning the full panel.
        self._ensure_custom_data_files()

        # Subscribe each ticker as custom data so Lean's time loop advances month by month.
        self._ticker_to_symbol: Dict[str, Symbol] = {}
        for ticker in self.universe_tickers:
            sym = self.add_data(SyntheticEquity, ticker).symbol
            self._ticker_to_symbol[ticker] = sym

    def _benchmark_value(self, time: datetime) -> float:
        current = pd.Timestamp(time).tz_localize(None)
        value = 1.0
        for month_end, nav in self._benchmark_history:
            if month_end <= current:
                value = nav
            else:
                break
        return float(value)

    def _load_and_prepare_data(self) -> None:
        loader = DataLoader(self._inputs_root)
        prepared = loader.prepare(max_tickers=self.max_tickers)

        self.base_3_predictors = prepared["base_3_predictors"]
        self.all_predictors = prepared["all_predictors"]
        predictor_type = ModelRegistry.get_predictor_type(self.model_id)
        self.predictor_cols = self.base_3_predictors if predictor_type == "base_3" else self.all_predictors

        self.data = pd.concat([prepared["train"], prepared["test"]], ignore_index=True)

        # Build event-driven split as latest 70% months for backtest loop.
        dates = sorted(self.data["date"].unique())
        split_idx = int(np.floor(len(dates) * 0.30))
        split_idx = max(1, min(split_idx, len(dates) - 1))
        self.test_dates = list(dates[split_idx:])

        # Bring market cap from raw panel because DataLoader.prepare returns return-only merge.
        _, ff, panel = loader.load_raw()
        self._panel = panel.copy()
        if "market_cap" in panel.columns:
            mcap_df = panel[["date", "ticker", "market_cap"]].copy()
            self.data = self.data.merge(mcap_df, on=["date", "ticker"], how="left")
        self.data["market_cap"] = self.data.get("market_cap", 1.0)
        self.data["market_cap"] = self.data["market_cap"].fillna(1.0).clip(lower=1.0)

        self.ff = ff.sort_values("date").copy()
        for col in ["rf", "mkt_rf", "smb", "hml", "rmw", "cma", "umd"]:
            if col in self.ff.columns:
                self.ff[col] = self.ff[col].fillna(0.0)

        self.universe_tickers = sorted(self.data["ticker"].unique().tolist())

    def _ensure_custom_data_files(self) -> None:
        split_dir = _synthetic_equity_split_dir()
        split_dir.mkdir(parents=True, exist_ok=True)

        missing = [ticker for ticker in self.universe_tickers if not _synthetic_equity_ticker_path(ticker).exists()]
        if not missing:
            return

        panel = self._panel[self._panel["ticker"].isin(missing)].copy()
        if panel.empty:
            return

        panel = panel.sort_values(["ticker", "date"])
        for ticker, ticker_df in panel.groupby("ticker", sort=False):
            ticker_path = _synthetic_equity_ticker_path(str(ticker))
            ticker_df.to_csv(ticker_path, index=False)

        self.debug(f"Prepared {len(missing)} per-ticker custom data files for Lean subscriptions.")

    def _run_backtest_event_loop(self) -> None:
        for month_end in self.test_dates:
            self._rebalance_for_month(pd.Timestamp(month_end))

    def _rebalance_for_month(self, month_end: pd.Timestamp) -> None:
        started_at = perf_counter()
        train_df = self.data[self.data["date"] < month_end]
        month_df = self.data[self.data["date"] == month_end].copy()

        min_train_rows = max(24, len(self.predictor_cols) * 2)
        if len(train_df) < min_train_rows or month_df.empty:
            return

        x_train = train_df[self.predictor_cols].fillna(0.0).values
        y_train = train_df["return_premium"].values

        model = ModelRegistry.get_model(self.model_id, hparam_override=self.model_params)
        fit_started_at = perf_counter()
        model.fit(x_train, y_train)
        fit_elapsed = perf_counter() - fit_started_at

        predict_started_at = perf_counter()
        month_df["pred"] = model.predict(month_df[self.predictor_cols].fillna(0.0).values)
        month_df = month_df.sort_values("pred", ascending=False).reset_index(drop=True)
        predict_elapsed = perf_counter() - predict_started_at

        n = max(1, len(month_df) // 10)
        top = month_df.head(n).copy()
        bottom = month_df.tail(n).copy()

        top_w = top["market_cap"] / top["market_cap"].sum()
        bottom_w = bottom["market_cap"] / bottom["market_cap"].sum()
        all_w = month_df["market_cap"] / month_df["market_cap"].sum()

        long_ret = float(np.sum(top_w.values * top["return_premium"].values))
        short_ret = float(np.sum(bottom_w.values * bottom["return_premium"].values))
        benchmark_ret = float(np.sum(all_w.values * month_df["return_premium"].values))

        # Strategy = full market VW base + 30% dollar-neutral long-short overlay
        OVERLAY = 0.30
        strategy_ret = benchmark_ret + OVERLAY * (long_ret - short_ret)

        # Effective weights: market VW base + overlay adjustments
        current_weights: Dict[str, float] = {}
        for t, w in zip(month_df["ticker"].astype(str).values, all_w.values):
            current_weights[t] = current_weights.get(t, 0.0) + float(w)
        for t, w in zip(top["ticker"].astype(str).values, top_w.values):
            current_weights[t] = current_weights.get(t, 0.0) + OVERLAY * float(w)
        for t, w in zip(bottom["ticker"].astype(str).values, bottom_w.values):
            current_weights[t] = current_weights.get(t, 0.0) - OVERLAY * float(w)

        self._month_weights[month_end.strftime("%Y-%m")] = dict(current_weights)

        if self.prev_weights:
            universe = set(self.prev_weights) | set(current_weights)
            turnover = 0.5 * sum(abs(current_weights.get(k, 0.0) - self.prev_weights.get(k, 0.0)) for k in universe)
        else:
            turnover = np.nan

        self.prev_weights = current_weights

        self.strategy_nav *= 1.0 + strategy_ret
        self.benchmark_nav *= 1.0 + benchmark_ret
        self._benchmark_history.append((month_end, self.benchmark_nav))

        record = {
            "date": month_end,
            "strategy_return": strategy_ret,
            "benchmark_return": benchmark_ret,
            "turnover": turnover,
            "strategy_nav": self.strategy_nav,
            "benchmark_nav": self.benchmark_nav,
        }
        self.rebalance_records.append(record)
        self._month_records[month_end.strftime("%Y-%m")] = record

        self.debug(
            f"Rebalance {month_end.strftime('%Y-%m')}: train_rows={len(train_df)} month_rows={len(month_df)} "
            f"fit={fit_elapsed:.2f}s predict={predict_elapsed:.2f}s total={perf_counter() - started_at:.2f}s"
        )
        self.debug(f"Benchmark return for {month_end}: {benchmark_ret}")

    def on_data(self, data: Slice) -> None:
        """Use precomputed weights to place real Lean orders each month on custom data bars."""
        current_month = self.time.strftime("%Y-%m")
        has_custom_data = any(data.contains_key(sym) for sym in self._ticker_to_symbol.values())
        if not has_custom_data:
            return

        if current_month == self._last_rebalance_month:
            return

        month_end = self._test_month_lookup.get(current_month)
        if month_end is None:
            return

        if current_month not in self._month_weights:
            self.debug(f"Computing month state for {current_month}")
            self._rebalance_for_month(month_end)

        weights = self._month_weights.get(current_month)
        if weights is None:
            return
        self._last_rebalance_month = current_month
        for ticker, weight in weights.items():
            sym = self._ticker_to_symbol.get(ticker)
            if sym is not None and self.securities.contains_key(sym) and self.securities[sym].price > 0:
                self.set_holdings(sym, weight)
        for ticker, sym in self._ticker_to_symbol.items():
            if ticker not in weights and self.portfolio[sym].invested:
                self.liquidate(sym)

        record = self._month_records.get(current_month)
        if record is None:
            return

        self.plot("Cumulative", "Strategy", float(record["strategy_nav"]))
        self.plot("Cumulative", "Benchmark", float(record["benchmark_nav"]))
        self.plot("Rebalance", "StrategyReturn", float(record["strategy_return"]))
        self.plot("Rebalance", "BenchmarkReturn", float(record["benchmark_return"]))
        turnover = record["turnover"]
        self.plot("Rebalance", "Turnover", 0.0 if np.isnan(turnover) else float(turnover))

        self.set_runtime_statistic("Model", self.model_id)
        self.set_runtime_statistic("Date", current_month)
        self.set_runtime_statistic("Ret(%)", f"{float(record['strategy_return']) * 100:.2f}")

    def on_end_of_algorithm(self) -> None:
        if not self.rebalance_records:
            self.debug("No rebalance records produced.")
            return

        result = pd.DataFrame(self.rebalance_records).sort_values("date").reset_index(drop=True)

        ff_sub = self.ff[["date", "rf", "mkt_rf", "smb", "hml", "rmw", "cma", "umd"]].copy()
        result = result.merge(ff_sub, on="date", how="left")
        for col in ["rf", "mkt_rf", "smb", "hml", "rmw", "cma", "umd"]:
            result[col] = result[col].fillna(0.0)

        strat = result["strategy_return"].values
        bench = result["benchmark_return"].values
        rf = result["rf"].values

        strat_excess = strat - rf
        bench_excess = bench - rf

        alpha_bench, _ = _ols_alpha_and_betas(strat_excess, bench_excess.reshape(-1, 1))
        ff_factors = result[["mkt_rf", "smb", "hml", "rmw", "cma", "umd"]].values
        alpha_ff, betas_ff = _ols_alpha_and_betas(strat_excess, ff_factors)

        mean_ret = float(np.mean(strat))
        vol = float(np.std(strat, ddof=1)) if len(strat) > 1 else np.nan
        sharpe = (mean_ret / vol * np.sqrt(12.0)) if vol and np.isfinite(vol) and vol > 0 else np.nan

        active = strat - bench
        active_vol = float(np.std(active, ddof=1)) if len(active) > 1 else np.nan
        info_ratio = (
            float(np.mean(active)) / active_vol * np.sqrt(12.0)
            if active_vol and np.isfinite(active_vol) and active_vol > 0
            else np.nan
        )

        turnover = float(np.nanmean(result["turnover"].values)) if "turnover" in result else np.nan

        cumulative = (1.0 + result["strategy_return"]).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1.0
        max_drawdown = float(drawdown.min())

        metrics = {
            "model_id": self.model_id,
            "n_months": int(len(result)),
            "alpha_vs_benchmark": _safe_float(alpha_bench, np.nan),
            "alpha_ff6": _safe_float(alpha_ff, np.nan),
            "beta_mkt_rf": _safe_float(betas_ff[0] if len(betas_ff) > 0 else np.nan, np.nan),
            "beta_smb": _safe_float(betas_ff[1] if len(betas_ff) > 1 else np.nan, np.nan),
            "beta_hml": _safe_float(betas_ff[2] if len(betas_ff) > 2 else np.nan, np.nan),
            "beta_rmw": _safe_float(betas_ff[3] if len(betas_ff) > 3 else np.nan, np.nan),
            "beta_cma": _safe_float(betas_ff[4] if len(betas_ff) > 4 else np.nan, np.nan),
            "beta_umd": _safe_float(betas_ff[5] if len(betas_ff) > 5 else np.nan, np.nan),
            "drawdown": max_drawdown,
            "turnover": turnover,
            "sharpe_ratio": _safe_float(sharpe, np.nan),
            "information_ratio": _safe_float(info_ratio, np.nan),
            "avg_rebalance_return": mean_ret,
            "mean_return": mean_ret,
            "cumulative_return": float(cumulative.iloc[-1] - 1.0),
        }

        # Push panel metrics into Lean statistics so lean report JSON includes them.
        self.set_summary_statistic("model_id", self.model_id)
        self.set_summary_statistic("n_months", str(metrics["n_months"]))
        for key, value in metrics.items():
            if key in {"model_id", "n_months"}:
                continue
            self.set_summary_statistic(key, _metric_str(value))

        report_dir = Path(self.report_dir_param)
        if not report_dir.is_absolute():
            report_dir = self._project_root / report_dir
        model_dir = report_dir / self.model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        monthly_out = result[["date", "strategy_return", "benchmark_return", "turnover", "strategy_nav", "benchmark_nav"]].copy()
        monthly_out["date"] = monthly_out["date"].dt.strftime("%Y-%m-%d")
        monthly_out.to_csv(model_dir / "monthly_returns.csv", index=False)

        with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        self.debug(f"Saved reports: {model_dir}")
