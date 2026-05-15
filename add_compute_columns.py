from pathlib import Path

import pandas as pd

root = Path('/Users/raj/ws/quantconnect')
xlsx = root / 'top_predictors_page33_dedup_enriched.xlsx'
csv = root / 'top_predictors_page33_dedup_enriched.csv'

top = pd.read_excel(xlsx, sheet_name='TopPredictors_Page33')
macro = pd.read_excel(xlsx, sheet_name='MacroIndicators_8')
sic = pd.read_excel(xlsx, sheet_name='SIC2_Codes_74')

compute = {
    'mom1m': 'r_{t-1}: prior-month stock return.',
    'chmom': 'mom6m_{t-1} - mom6m_{t-7}: change in 6-month momentum (skip latest month convention may apply).',
    'indmom': "Value-weighted return of stock's industry peers over prior 12 months (often excluding latest month).",
    'mom12m': 'Cumulative return from t-12 to t-2: prod(1+r_m)-1 over months m=t-12..t-2.',
    'std_turn': 'Std. dev. of monthly share turnover over trailing window (commonly 12 months).',
    'maxret': 'Maximum daily return within prior month.',
    'sp': 'Sales / market equity (or sales-to-price equivalent).',
    'turn': 'Share turnover = monthly volume / shares outstanding.',
    'mvel1': 'log(Market equity): log(price * shares outstanding), lagged one period.',
    'chcsho': '(Shares outstanding_t / Shares outstanding_{t-1}) - 1.',
    'mom6m': 'Cumulative return from t-6 to t-2: prod(1+r_m)-1.',
    'rd_mve': 'R&D expense / market equity.',
    'agr': '(Total assets_t / Total assets_{t-1}) - 1.',
    'ep': 'Earnings / price (or earnings / market equity).',
    'invest': 'Investment intensity proxy using capital expenditures and inventory growth components.',
    'dolvol': 'log(sum of daily dollar volume in month) or monthly dollar volume aggregate.',
    'cashpr': 'Cash productivity ratio; convention varies, commonly cash-flow productivity scaling in Table A.6 lineage.',
    'depr': 'Depreciation expense / PP&E.',
    'nincr': 'Count of recent sequential earnings increases (typically quarterly streak count).',
    'bm': 'Book equity / market equity.',
    'lgr': '(Long-term debt_t / Long-term debt_{t-1}) - 1.',
    'retvol': 'Std. dev. of daily returns over prior month (or trailing horizon).',
    'chinv': '(Inventory_t / Inventory_{t-1}) - 1.',
    'operprof': 'Operating profitability = (Revenue - COGS - SG&A - interest) / book equity (FF-style).',
    'bm_ia': 'Book-to-market minus industry average book-to-market.',
    'lev': 'Leverage ratio, commonly total debt / market equity (or debt / assets by convention).',
    'mom36m': 'Cumulative long-horizon momentum, typically months t-36..t-13.',
    'ill': 'Amihud illiquidity = avg(|daily return| / daily dollar volume) over month.',
    'ps': 'Piotroski-style financial statement score (sum of accounting signal indicators).',
    'sic2': 'First two digits of SIC code; one-hot encode to industry dummies.',
    'securedind': 'Indicator = 1 if firm reports secured debt, else 0.',
    'dy': 'Dividend yield = dividends / price.',
    'baspread': 'Bid-ask spread proxy, e.g., (ask-bid)/midquote averaged over month.',
    'convind': 'Indicator = 1 if firm has convertible debt outstanding, else 0.',
    'idiovol': 'Std. dev. of residuals from factor model (e.g., CAPM/FF) over trailing daily window.',
    'beta': 'Market beta estimated from rolling regression of excess stock returns on market excess return.',
    'betasq': 'beta^2.',
    'age': 'Years since first appearance in Compustat (or listing-age proxy).',
    'zerotrade': 'Fraction or count of zero-volume trading days in prior month.',
}

top['how_to_compute'] = top['predictor_variable'].map(compute).fillna(
    'See original paper variable definition; compute per source-convention with proper lags.'
)

macro_compute = {
    'dp': 'log(D12 / P): log dividends over trailing 12 months divided by price.',
    'ep': 'log(E12 / P): log earnings over trailing 12 months divided by price.',
    'bm': 'Book value / market value (aggregate market-level ratio).',
    'ntis': 'Net equity issuance scaled by market cap (aggregate net equity expansion).',
    'tbl': '3-month Treasury bill yield level.',
    'tms': 'Long-term Treasury yield - 3-month T-bill yield.',
    'dfy': 'BAA corporate bond yield - AAA corporate bond yield.',
    'svar': 'Stock market return variance (monthly realized variance proxy).',
}
macro['how_to_compute'] = macro['indicator'].map(macro_compute)

sic['how_to_compute'] = 'Take first two digits of firm SIC and create one-hot dummy for each sic2_code group.'

top.to_csv(csv, index=False)
with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
    top.to_excel(writer, index=False, sheet_name='TopPredictors_Page33')
    macro.to_excel(writer, index=False, sheet_name='MacroIndicators_8')
    sic.to_excel(writer, index=False, sheet_name='SIC2_Codes_74')

print('updated_csv', csv)
print('updated_xlsx', xlsx)
print('top_rows', len(top), 'macro_rows', len(macro), 'sic_rows', len(sic))
print('top_compute_missing', int(top['how_to_compute'].isna().sum()))
