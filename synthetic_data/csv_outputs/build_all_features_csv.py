import pandas as pd
import numpy as np
from pathlib import Path

# Paths
root = Path('/Users/raj/ws/quantconnect')
data_dir = root / 'synthetic_data'
panel_path = data_dir / 'synthetic_equity_monthly_10y.csv'
predictors_path = root / 'top_predictors_page33_dedup_enriched.csv'

# Output CSVs
csv_dir = data_dir / 'csv_outputs'
all_features_csv = csv_dir / 'synthetic_predictor_features_all_months.csv'
econ_csv = csv_dir / 'synthetic_macro_variables.csv'

# Load data
panel = pd.read_csv(panel_path, parse_dates=['date'])
panel = panel.sort_values(['ticker', 'date']).reset_index(drop=True)
predictor_vars = pd.read_csv(predictors_path)['predictor_variable'].tolist()
macro_vars = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

# Recompute macro variables (as in build_monthly_feature_workbook.py)
SEED = 123
rng = np.random.default_rng(SEED)
months = np.array(sorted(panel['date'].unique()))
m = len(months)
def ar1_series(n, start, mean, phi, sigma, lo, hi, rng):
    x = np.empty(n)
    x[0] = start
    for i in range(1, n):
        x[i] = mean + phi * (x[i - 1] - mean) + rng.normal(0.0, sigma)
    return np.clip(x, lo, hi)
econ = pd.DataFrame({
    'date': months,
    'dp': ar1_series(m, -3.6, -3.5, 0.92, 0.06, -4.3, -2.6, rng),
    'ep': ar1_series(m, -2.9, -2.8, 0.90, 0.05, -3.5, -2.0, rng),
    'bm': ar1_series(m, 0.50, 0.55, 0.88, 0.03, 0.30, 0.90, rng),
    'ntis': ar1_series(m, 0.01, 0.00, 0.75, 0.02, -0.08, 0.10, rng),
    'tbl': ar1_series(m, 0.03, 0.02, 0.94, 0.004, 0.0, 0.08, rng),
    'tms': ar1_series(m, 0.012, 0.015, 0.90, 0.003, -0.01, 0.03, rng),
    'dfy': ar1_series(m, 0.012, 0.013, 0.92, 0.002, 0.005, 0.03, rng),
    'svar': ar1_series(m, 0.020, 0.018, 0.80, 0.005, 0.005, 0.06, rng),
})

# Merge macro variables into panel
econ = econ.sort_values('date')
panel = panel.merge(econ, on='date', how='left')

# Feature engineering (minimal, as in build_monthly_feature_workbook.py)


# --- Feature engineering to match build_monthly_feature_workbook.py ---
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

# Time-series features by ticker
g = panel.groupby('ticker', group_keys=False)
panel['mom1m'] = g['return'].shift(1)
panel['mom6m'] = g['return'].shift(1).rolling(6).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
panel['mom12m'] = g['return'].shift(1).rolling(12).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
panel['mom36m'] = g['return'].shift(1).rolling(36).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
panel['chmom'] = panel['mom6m'] - g['mom6m'].shift(6)
panel['std_turn'] = g['turn'].shift(1).rolling(12).std()
panel['retvol'] = g['return'].shift(1).rolling(6).std()
panel['maxret'] = (np.abs(panel['return']) * 1.8 + rng.normal(0.0, 0.01, len(panel))).clip(lower=0.0)

# Market return and variance
market_ret = panel.groupby('date')['return'].mean().rename('mkt_ret')
panel = panel.merge(market_ret, on='date', how='left')
mkt_var = market_ret.rolling(24, min_periods=6).var().rename('mkt_var').reset_index()
panel = panel.merge(mkt_var, on='date', how='left')

# Beta, betasq, idiovol
g = panel.groupby('ticker', group_keys=False)
beta_num = g.apply(lambda x: x['return'].rolling(24, min_periods=6).cov(x['mkt_ret'])).reset_index(level=0, drop=True)
panel['beta'] = (beta_num / panel['mkt_var'].clip(lower=1e-8)).clip(lower=0.1, upper=3.0)
panel['betasq'] = panel['beta'] ** 2
panel['idiovol'] = np.sqrt(np.clip(panel['retvol'].fillna(0.0) ** 2 - (panel['beta'].fillna(1.0) * np.sqrt(panel['mkt_var'].fillna(0.0))) ** 2, 0.0, None))

# Static traits and slow-moving fundamentals
tickers = np.array(sorted(panel['ticker'].unique()))
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

# SIC-based features
sic_codes = [str(i).zfill(2) for i in range(10, 100)]
panel['sic2_code'] = panel['ticker'].map(pd.Series(np.random.choice(sic_codes, size=len(tickers), replace=True), index=tickers))
panel['sic2'] = panel['sic2_code'].astype(int)
panel['bm_ia'] = panel['bm'] - panel.groupby(['date', 'sic2_code'])['bm'].transform('mean')

# Industry momentum
ind_ret = panel.groupby(['date', 'sic2_code'])['return'].mean().rename('industry_ret').reset_index()
ind_ret = ind_ret.sort_values(['sic2_code', 'date'])
ind_ret['indmom'] = ind_ret.groupby('sic2_code')['industry_ret'].shift(1).rolling(12).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
panel = panel.merge(ind_ret[['date', 'sic2_code', 'indmom']], on=['date', 'sic2_code'], how='left')

# Fill missing predictors
for c in predictor_vars:
    if c not in panel.columns:
        panel[c] = 0.0
panel[predictor_vars] = panel[predictor_vars].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# SIC one-hot dummies (consistent per ticker)
sic_dummies = pd.get_dummies(panel['sic2_code'], prefix='sic2', dtype=int)
panel = pd.concat([panel, sic_dummies], axis=1)
sic_dummy_cols = sorted(sic_dummies.columns)

# Add interaction terms: predictor × macro
interaction_cols = []
for p in predictor_vars:
    for e in macro_vars:
        col = f'{p}__x__{e}'
        panel[col] = panel[p] * panel[e]
        interaction_cols.append(col)

# Only keep required columns for all-features CSV
base_cols = ['date', 'ticker'] + predictor_vars + macro_vars + interaction_cols + sic_dummy_cols
panel_out = panel[base_cols].copy()
panel_out = panel_out.sort_values(['date', 'ticker'])
panel_out.to_csv(all_features_csv, index=False)

econ.to_csv(econ_csv, index=False)

print(f'Wrote all features to: {all_features_csv}')
print(f'Wrote macro variables to: {econ_csv}')
