"""
Fama-French 6-Factor Construction (2x3 sort, value-weighted)
Based on: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_5_factors_2x3.html

Factors:
  Mkt-RF : Value-weighted market excess return
  SMB    : Average of SMB from B/M, OP, and INV sorts
  HML    : High Minus Low (book-to-market)
  RMW    : Robust Minus Weak (operating profitability)
  CMA    : Conservative Minus Aggressive (investment = invest variable)
  UMD    : Up Minus Down (momentum = mom12m variable)
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path('/Users/raj/ws/quantconnect')
DATA_DIR = ROOT / 'synthetic_data'
OUT_DIR = DATA_DIR / 'csv_outputs'



def vw_return(returns, weights):
    """Value-weighted portfolio return. Returns NaN if weights sum to zero."""
    w = weights.copy()
    total = w.sum()
    if total == 0:
        return np.nan
    return (returns * w).sum() / total


def sort3_masks(series, lo_pct=0.30, hi_pct=0.70):
    """Return boolean masks for low, neutral, high buckets."""
    p_lo = series.quantile(lo_pct)
    p_hi = series.quantile(hi_pct)
    low = series <= p_lo
    high = series > p_hi
    neutral = ~low & ~high
    return low, neutral, high


def compute_2x3_portfolios(ret, mktcap, is_small, is_big, char, lo_pct=0.30, hi_pct=0.70):
    """
    Compute 6 value-weighted portfolio returns from a 2x3 Size x Characteristic sort.
    Returns a dict with keys: SL, SN, SH, BL, BN, BH  (Small/Big x Low/Neutral/High)
    """
    lo_mask, ne_mask, hi_mask = sort3_masks(char, lo_pct, hi_pct)
    portfolios = {}
    for size_name, size_mask in [('S', is_small), ('B', is_big)]:
        for char_name, char_mask in [('L', lo_mask), ('N', ne_mask), ('H', hi_mask)]:
            mask = size_mask & char_mask
            portfolios[size_name + char_name] = vw_return(ret[mask], mktcap[mask])
    return portfolios


def main():
    # ------------------------------------------------------------------ #
    # 1. Load pre-computed predictor features (bm, operprof, invest, mom12m,
    #    tbl already computed with consistent seeds)
    # ------------------------------------------------------------------ #
    features = pd.read_csv(
        OUT_DIR / 'synthetic_predictor_features_all_months.csv',
        parse_dates=['date'],
        usecols=['date', 'ticker', 'bm', 'operprof', 'invest', 'mom12m', 'tbl'],
    )
    features = features.sort_values(['ticker', 'date']).reset_index(drop=True)
    features['rf_monthly'] = features['tbl'] / 12.0

    # ------------------------------------------------------------------ #
    # 2. Load raw panel for market_cap and return (needed for value-weighted
    #    market return; these are not stored in predictor features CSV)
    # ------------------------------------------------------------------ #
    raw = pd.read_csv(
        DATA_DIR / 'synthetic_equity_monthly_10y.csv',
        parse_dates=['date'],
        usecols=['date', 'ticker', 'market_cap', 'return'],
    )
    raw = raw.sort_values(['ticker', 'date']).reset_index(drop=True)

    panel = features.merge(raw, on=['date', 'ticker'], how='inner')

    # ------------------------------------------------------------------ #
    # 3. Lag characteristics for sorting (use prior-month values)
    #    market_cap is lagged for VW weights; characteristics are as-of
    #    prior month end (already computed that way in features CSV since
    #    mom12m uses shift(1), bm/operprof/invest are point-in-time).
    # ------------------------------------------------------------------ #
    for col in ['market_cap', 'bm', 'operprof', 'invest', 'mom12m']:
        panel[f'{col}_lag'] = panel.groupby('ticker')[col].shift(1)
        
    # ------------------------------------------------------------------ #
    # 5. Compute factor returns month by month
    # ------------------------------------------------------------------ #
    required_lags = ['market_cap_lag', 'bm_lag', 'operprof_lag', 'invest_lag', 'mom12m_lag']
    records = []

    for date, grp in panel.groupby('date'):
        g = grp.dropna(subset=required_lags).copy()
        
        if len(g) < 20:
            continue

        ret     = g['return']
        mktcap  = g['market_cap_lag']
        rf      = g['rf_monthly'].iloc[0]

        # --- Market return (value-weighted) ---
        total_mktcap = mktcap.sum()
        mkt_ret      = vw_return(ret, mktcap)
        mkt_rf       = mkt_ret - rf
        # The variable `rf` in the provided code is representing the
        # risk-free rate of return on a monthly basis. It is calculated as
        # the risk-free rate per month, which is derived from the Treasury
        # Bill rate (tbl) divided by 12. This risk-free rate is used in the
        # calculation of the market excess return (mkt_rf) as the difference
        # between the value-weighted market return (mkt_ret) and the
        # risk-free rate.
        # The variable `rf` in the provided code represents the risk-free
        # rate for each month. It is calculated as the monthly Treasury Bill
        # rate (tbl) divided by 12. This risk-free rate is used in the
        # calculation of the market excess return (mkt_rf) as the difference
        # between the value-weighted market return (mkt_ret) and the
        # risk-free rate.
        rf

        # --- Size split: median of lagged market cap ---
        size_median = mktcap.median()
        is_small    = mktcap <  size_median
        is_big      = mktcap >= size_median

        # --- HML: 2x3 on B/M ---
        p_bm = compute_2x3_portfolios(ret, mktcap, is_small, is_big, g['bm_lag'])
        hml     = 0.5 * (p_bm['SH'] + p_bm['BH']) - 0.5 * (p_bm['SL'] + p_bm['BL'])
        smb_bm  = (p_bm['SL'] + p_bm['SN'] + p_bm['SH']) / 3 - (p_bm['BL'] + p_bm['BN'] + p_bm['BH']) / 3

        # --- RMW: 2x3 on operating profitability ---
        p_op = compute_2x3_portfolios(ret, mktcap, is_small, is_big, g['operprof_lag'])
        rmw     = 0.5 * (p_op['SH'] + p_op['BH']) - 0.5 * (p_op['SL'] + p_op['BL'])
        smb_op  = (p_op['SL'] + p_op['SN'] + p_op['SH']) / 3 - (p_op['BL'] + p_op['BN'] + p_op['BH']) / 3

        # --- CMA: 2x3 on investment (low invest = conservative) ---
        p_inv = compute_2x3_portfolios(ret, mktcap, is_small, is_big, g['invest_lag'])
        cma     = 0.5 * (p_inv['SL'] + p_inv['BL']) - 0.5 * (p_inv['SH'] + p_inv['BH'])   # Low invest is conservative
        smb_inv = (p_inv['SL'] + p_inv['SN'] + p_inv['SH']) / 3 - (p_inv['BL'] + p_inv['BN'] + p_inv['BH']) / 3

        # --- SMB: average of three SMBs ---
        smb = (smb_bm + smb_op + smb_inv) / 3

        # --- UMD: 2x3 on 12-month momentum ---
        p_mom = compute_2x3_portfolios(ret, mktcap, is_small, is_big, g['mom12m_lag'])
        umd = 0.5 * (p_mom['SH'] + p_mom['BH']) - 0.5 * (p_mom['SL'] + p_mom['BL'])

        records.append({
            'date':         date,
            'total_mktcap': total_mktcap,
            'mkt_ret':      mkt_ret,
            'rf':           rf,
            'mkt_rf':       mkt_rf,
            'smb':          smb,
            'hml':          hml,
            'rmw':          rmw,
            'cma':          cma,
            'umd':          umd,
        })

    factors = pd.DataFrame(records)
    factors['date'] = pd.to_datetime(factors['date'])

    # ------------------------------------------------------------------ #
    # 6. Write output
    # ------------------------------------------------------------------ #
    out_csv  = OUT_DIR / 'ff6_factors_10y.csv'
    out_xlsx = OUT_DIR / 'ff6_factors_10y.xlsx'

    factors.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        factors.to_excel(writer, sheet_name='FF6_Factors', index=False)

    print(f"Wrote {len(factors)} months")
    print(f"CSV  : {out_csv}")
    print(f"Excel: {out_xlsx}")
    print()
    print(factors.tail(5).to_string(index=False))


if __name__ == '__main__':
    main()
