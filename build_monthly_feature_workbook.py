from pathlib import Path

import numpy as np
import pandas as pd


SEED = 123


def ar1_series(n: int, start: float, mean: float, phi: float, sigma: float, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(n)
    x[0] = start
    for i in range(1, n):
        x[i] = mean + phi * (x[i - 1] - mean) + rng.normal(0.0, sigma)
    return np.clip(x, lo, hi)


def main() -> None:
    rng = np.random.default_rng(SEED)

    root = Path('/Users/raj/ws/quantconnect')
    data_dir = root / 'synthetic_data'
    panel_path = data_dir / 'synthetic_equity_monthly_10y.csv'
    predictors_path = root / 'top_predictors_page33_dedup_enriched.csv'
    sic_path = root / 'top_predictors_page33_dedup_enriched.xlsx'
    out_path = data_dir / 'synthetic_predictor_feature_sheets_10y.xlsx'

    panel = pd.read_csv(panel_path, parse_dates=['date'])
    panel = panel.sort_values(['ticker', 'date']).reset_index(drop=True)

    predictor_vars = pd.read_csv(predictors_path)['predictor_variable'].tolist()
    macro_vars = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

    sic_codes = pd.read_excel(sic_path, sheet_name='SIC2_Codes_74')['sic2_code'].astype(str).str.zfill(2).tolist()

    tickers = np.array(sorted(panel['ticker'].unique()))
    ticker_sic = pd.Series(rng.choice(sic_codes, size=len(tickers), replace=True), index=tickers)
    panel['sic2_code'] = panel['ticker'].map(ticker_sic)

    months = np.array(sorted(panel['date'].unique()))
    m = len(months)

    econ = pd.DataFrame(
        {
            'date': months,
            'dp': ar1_series(m, -3.6, -3.5, 0.92, 0.06, -4.3, -2.6, rng),
            'ep': ar1_series(m, -2.9, -2.8, 0.90, 0.05, -3.5, -2.0, rng),
            'bm': ar1_series(m, 0.50, 0.55, 0.88, 0.03, 0.30, 0.90, rng),
            'ntis': ar1_series(m, 0.01, 0.00, 0.75, 0.02, -0.08, 0.10, rng),
            'tbl': ar1_series(m, 0.03, 0.02, 0.94, 0.004, 0.0, 0.08, rng),
            'tms': ar1_series(m, 0.012, 0.015, 0.90, 0.003, -0.01, 0.03, rng),
            'dfy': ar1_series(m, 0.012, 0.013, 0.92, 0.002, 0.005, 0.03, rng),
            'svar': ar1_series(m, 0.020, 0.018, 0.80, 0.005, 0.005, 0.06, rng),
        }
    )

    panel = panel.merge(econ, on='date', how='left')

    panel['turn'] = panel['volume'] / panel['shares_outstanding']
    panel['mvel1'] = np.log(panel['market_cap'].clip(lower=1.0))
    panel['ep'] = 1.0 / panel['pe_ratio'].clip(lower=1e-6)
    panel['bm'] = 1.0 / panel['pb_ratio'].clip(lower=1e-6)
    panel['sp'] = (1.15 * panel['bm'] + rng.normal(0.0, 0.03, len(panel))).clip(lower=0.01)
    panel['dy'] = (0.25 * panel['ep'] + rng.normal(0.0, 0.005, len(panel))).clip(lower=0.0)
    panel['dolvol'] = np.log1p(panel['price'] * panel['volume'])
    panel['baspread'] = (0.004 + 0.015 / np.sqrt(1.0 + panel['turn']) + rng.normal(0.0, 0.001, len(panel))).clip(lower=0.0005)
    panel['ill'] = (np.abs(panel['return']) / (panel['price'] * panel['volume']).clip(lower=1.0) * 1e8).clip(lower=0.0)
    panel['zerotrade'] = np.clip(np.exp(-12.0 * panel['turn']) + rng.normal(0.0, 0.02, len(panel)), 0.0, 1.0)

    # Time-series features by ticker.
    g = panel.groupby('ticker', group_keys=False)
    panel['mom1m'] = g['return'].shift(1)
    panel['mom6m'] = g['return'].shift(1).rolling(6).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    panel['mom12m'] = g['return'].shift(1).rolling(12).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    panel['mom36m'] = g['return'].shift(1).rolling(36).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    panel['chmom'] = panel['mom6m'] - g['mom6m'].shift(6)
    panel['std_turn'] = g['turn'].shift(1).rolling(12).std()
    panel['retvol'] = g['return'].shift(1).rolling(6).std()
    panel['maxret'] = (np.abs(panel['return']) * 1.8 + rng.normal(0.0, 0.01, len(panel))).clip(lower=0.0)

    market_ret = panel.groupby('date')['return'].mean().rename('mkt_ret')
    panel = panel.merge(market_ret, on='date', how='left')
    mkt_var = market_ret.rolling(24, min_periods=6).var().rename('mkt_var').reset_index()
    panel = panel.merge(mkt_var, on='date', how='left')

    g = panel.groupby('ticker', group_keys=False)
    beta_num = g.apply(lambda x: x['return'].rolling(24, min_periods=6).cov(x['mkt_ret'])).reset_index(level=0, drop=True)
    panel['beta'] = (beta_num / panel['mkt_var'].clip(lower=1e-8)).clip(lower=0.1, upper=3.0)
    panel['betasq'] = panel['beta'] ** 2
    panel['idiovol'] = np.sqrt(np.clip(panel['retvol'].fillna(0.0) ** 2 - (panel['beta'].fillna(1.0) * np.sqrt(panel['mkt_var'].fillna(0.0))) ** 2, 0.0, None))

    # Per-ticker static traits + slow-moving fundamentals.
    q = pd.Series(rng.normal(0.0, 1.0, len(tickers)), index=tickers)
    d = pd.Series(rng.normal(0.0, 1.0, len(tickers)), index=tickers)
    panel['quality'] = panel['ticker'].map(q)
    panel['distress'] = panel['ticker'].map(d)

    panel['shares_prev'] = g['shares_outstanding'].shift(1)
    panel['chcsho'] = ((panel['shares_outstanding'] / panel['shares_prev'].replace(0, np.nan)) - 1.0).fillna(0.0)
    panel['age'] = g.cumcount() / 12.0

    panel['agr'] = (-0.12 * panel['bm'] + 0.05 * panel['quality'] + rng.normal(0.0, 0.04, len(panel)))
    panel['invest'] = (0.7 * panel['agr'] + rng.normal(0.0, 0.03, len(panel)))
    panel['rd_mve'] = np.clip(0.04 + 0.015 * panel['quality'] + rng.normal(0.0, 0.01, len(panel)), 0.0, None)
    panel['cashpr'] = np.clip(0.3 + 0.15 * panel['quality'] - 0.1 * panel['distress'] + rng.normal(0.0, 0.05, len(panel)), 0.01, None)
    panel['depr'] = np.clip(0.05 + rng.normal(0.0, 0.01, len(panel)), 0.005, 0.2)
    panel['nincr'] = np.clip(np.round(2.0 + 1.2 * panel['quality'] + rng.normal(0.0, 1.0, len(panel))), 0, 8)
    panel['lgr'] = np.clip(0.02 + 0.08 * panel['distress'] + rng.normal(0.0, 0.04, len(panel)), -0.5, 1.0)
    panel['chinv'] = np.clip(0.01 + 0.06 * panel['invest'] + rng.normal(0.0, 0.03, len(panel)), -0.5, 1.0)
    panel['operprof'] = np.clip(0.12 + 0.07 * panel['quality'] - 0.03 * panel['distress'] + rng.normal(0.0, 0.03, len(panel)), -0.3, 0.6)
    panel['lev'] = np.clip(0.35 + 0.12 * panel['distress'] - 0.05 * panel['quality'] + rng.normal(0.0, 0.04, len(panel)), 0.0, 1.5)
    panel['ps'] = np.clip(np.round(5 + 2 * panel['quality'] - panel['distress'] + rng.normal(0.0, 1.2, len(panel))), 0, 9)
    panel['securedind'] = (panel['distress'] > 0.7).astype(int)
    panel['convind'] = (panel['quality'] > 1.0).astype(int)

    # SIC-based features.
    panel['sic2'] = panel['sic2_code'].astype(int)
    panel['bm_ia'] = panel['bm'] - panel.groupby(['date', 'sic2_code'])['bm'].transform('mean')

    # Industry momentum from SIC monthly returns.
    ind_ret = panel.groupby(['date', 'sic2_code'])['return'].mean().rename('industry_ret').reset_index()
    ind_ret = ind_ret.sort_values(['sic2_code', 'date'])
    ind_ret['indmom'] = ind_ret.groupby('sic2_code')['industry_ret'].shift(1).rolling(12).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    panel = panel.merge(ind_ret[['date', 'sic2_code', 'indmom']], on=['date', 'sic2_code'], how='left')

    # Keep only required predictor vars and fill initial NaNs.
    for c in predictor_vars:
        if c not in panel.columns:
            panel[c] = 0.0
    panel[predictor_vars] = panel[predictor_vars].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # SIC one-hot dummies.
    sic_dummies = pd.get_dummies(panel['sic2_code'], prefix='sic2', dtype=int)
    panel = pd.concat([panel, sic_dummies], axis=1)
    sic_dummy_cols = sorted(sic_dummies.columns)

    # Interactions predictor x macro (per stock per month).
    interaction_cols = []
    for p in predictor_vars:
        for e in macro_vars:
            col = f'{p}__x__{e}'
            panel[col] = panel[p] * panel[e]
            interaction_cols.append(col)

    panel['date_str'] = panel['date'].dt.strftime('%Y-%m-%d')

    base_cols = ['date_str', 'ticker'] + predictor_vars + macro_vars + interaction_cols + sic_dummy_cols

    # Write one sheet per month.
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        for dt, sub in panel.groupby('date', sort=True):
            sheet = f"M_{dt.strftime('%Y_%m')}"
            out_df = sub[base_cols].copy()
            out_df.to_excel(writer, sheet_name=sheet, index=False)

    # Summary for quick inspection.
    summary = panel.groupby('date', as_index=False)['ticker'].nunique().rename(columns={'ticker': 'n_tickers'})
    summary.to_csv(data_dir / 'synthetic_feature_workbook_month_counts.csv', index=False)

    print('output', out_path)
    print('months', panel['date'].nunique())
    print('predictor_count', len(predictor_vars))
    print('macro_count', len(macro_vars))
    print('interaction_count', len(interaction_cols))
    print('sic_dummy_count', len(sic_dummy_cols))


if __name__ == '__main__':
    main()
