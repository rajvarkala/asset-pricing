"""
Data preparation for ML pipeline.
- Load features, compute return premium
- Train/test split (9y / 1y)
- Feature selection (3-predictor vs all)
"""
from pathlib import Path
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, root: Path):
        self.root = root
        self.csv_dir = root / 'synthetic_data' / 'csv_outputs'

    def load_raw(self):
        """Load predictor features and market data."""
        features = pd.read_csv(
            self.csv_dir / 'synthetic_predictor_features_all_months.csv',
            parse_dates=['date'],
        )
        ff = pd.read_csv(
            self.csv_dir / 'ff6_factors_10y.csv',
            parse_dates=['date'],
        )
        panel = pd.read_csv(
            self.root / 'synthetic_data' / 'synthetic_equity_monthly_10y.csv',
            parse_dates=['date'],
        )
        
        features = features.sort_values(['ticker', 'date']).reset_index(drop=True)
        ff = ff.sort_values('date').reset_index(drop=True)
        panel = panel.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        return features, ff, panel

    def prepare(self, max_tickers: int | None = None, ticker_sample_seed: int = 42):
        """
        Prepare data:
        - Load features and merge with returns from panel
        - Compute return premium = return - rf_monthly (from tbl)
        - Shift predictors by one month within each ticker
        - Select predictors and labels
        - Return train (9y) / test (1y) split
        """
        features, ff, panel = self.load_raw()
        
        # Extract returns from panel
        returns_df = panel[['date', 'ticker', 'return']].copy()
        
        # Merge returns into features
        merged = features.merge(returns_df, on=['date', 'ticker'], how='left')
        
        # Compute return premium
        merged['rf_monthly'] = merged['tbl'] / 12.0
        merged['return_premium'] = merged['return'] - merged['rf_monthly']

        # Shift predictors by one month within each ticker so predictors at t-1 predict premium at t.
        predictor_feature_cols = [c for c in features.columns if c not in {'date', 'ticker'}]
        merged[predictor_feature_cols] = merged.groupby('ticker')[predictor_feature_cols].shift(1)
        
        # Sort by date, ticker
        merged = merged.sort_values(['date', 'ticker']).reset_index(drop=True)

        if max_tickers is not None:
            unique_tickers = np.array(sorted(merged['ticker'].unique()))
            if max_tickers < len(unique_tickers):
                rng = np.random.default_rng(ticker_sample_seed)
                selected = sorted(rng.choice(unique_tickers, size=max_tickers, replace=False).tolist())
                merged = merged[merged['ticker'].isin(selected)].copy()
        
        # Extract dates
        dates = sorted(merged['date'].unique())
        n_months = len(dates)
        
        # Split: first 9 years for train, last 1 year for test
        test_months = 12
        cutoff_idx = n_months - test_months
        train_cutoff_date = dates[cutoff_idx - 1]
        
        train = merged[merged['date'] <= train_cutoff_date].copy()
        test = merged[merged['date'] > train_cutoff_date].copy()

        train_months = len(sorted(train['date'].unique()))
        validation_months = 12
        initial_train_months = train_months - validation_months
        
        # Get predictor column names (exclude identifiers, dates, returns, rf, tbl)
        exclude_cols = {
            'date', 'ticker', 'return', 'rf_monthly', 'return_premium',
            'price', 'market_cap', 'pe_ratio', 'pb_ratio', 'valuation_score',
            'shares_outstanding', 'volume', 'tbl',
        }
        
        # Base predictors for OLS-3 (size, book-to-market, momentum)
        base_3_predictors = ['mvel1', 'bm', 'mom12m']
        
        # All predictors (excluding interactions for now, or include?)
        all_predictors = [c for c in merged.columns if c not in exclude_cols]
        
        return {
            'train': train,
            'test': test,
            'train_dates': sorted(train['date'].unique()),
            'test_dates': sorted(test['date'].unique()),
            'initial_train_months': initial_train_months,
            'validation_months': validation_months,
            'base_3_predictors': base_3_predictors,
            'all_predictors': all_predictors,
            'tickers': sorted(merged['ticker'].unique()),
        }


if __name__ == '__main__':
    root = Path('/Users/raj/ws/asset-pricing')
    loader = DataLoader(root)
    data = loader.prepare()
    
    print(f"Train: {len(data['train_dates'])} months, {len(data['train'])} rows")
    print(f"Test: {len(data['test_dates'])} months, {len(data['test'])} rows")
    print(f"Tickers: {len(data['tickers'])}")
    print(f"All predictors: {len(data['all_predictors'])}")
    print(f"Base 3 predictors: {data['base_3_predictors']}")
