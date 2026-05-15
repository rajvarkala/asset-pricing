"""
Time-series cross-validation and evaluation utilities.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from typing import List, Tuple


class TimeSeriesSplitter:
    """Time-series aware data splitter for rolling window CV."""
    
    def __init__(self, initial_train_months: int, cv_months: int):
        """
        Args:
            initial_train_months: Size of initial training window
            cv_months: Months to roll forward for CV (validation + expansion)
        """
        self.initial_train_months = initial_train_months
        self.cv_months = cv_months
    
    def split(self, dates: List[pd.Timestamp]) -> List[Tuple]:
        """
        Yield (train_dates, val_dates) tuples for rolling CV.
        
        Pattern:
        - Train: months 0 to initial_train_months
        - Val: next 1 month
        - Then roll forward: add 1 month to train, slide val forward
        
        Repeat for cv_months periods.
        """
        dates = sorted(dates)
        folds = []
        
        for step in range(self.cv_months):
            train_end_idx = self.initial_train_months + step
            val_end_idx = train_end_idx + 1
            
            if val_end_idx > len(dates):
                break
            
            train_dates = dates[:train_end_idx]
            val_dates = dates[train_end_idx:val_end_idx]
            
            folds.append((train_dates, val_dates))
        
        return folds


class Evaluator:
    """R² calculation and aggregation utilities."""
    
    @staticmethod
    def r2_by_ticker(y_true: np.ndarray, y_pred: np.ndarray, 
                     tickers: np.ndarray) -> pd.DataFrame:
        """
        Compute R² per ticker.
        
        Args:
            y_true: True labels (n_samples,)
            y_pred: Predicted labels (n_samples,)
            tickers: Ticker assignments (n_samples,)
        
        Returns:
            DataFrame with columns: ticker, r2
        """
        results = []
        
        for ticker in np.unique(tickers):
            mask = tickers == ticker
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            # Skip if less than 2 samples
            if len(y_t) < 2:
                r2 = np.nan
            else:
                r2 = r2_score(y_t, y_p)
            
            results.append({'ticker': ticker, 'r2': r2})
        
        return pd.DataFrame(results)
    
    @staticmethod
    def aggregate_r2(r2_by_ticker: pd.DataFrame) -> dict:
        """
        Aggregate R² across tickers.
        
        Returns:
            {'mean': float, 'median': float, 'std': float, 'count': int}
        """
        r2_valid = r2_by_ticker['r2'].dropna()
        
        return {
            'mean': r2_valid.mean(),
            'median': r2_valid.median(),
            'std': r2_valid.std(),
            'count': len(r2_valid),
        }


class ModelEvaluator:
    """Full model training and evaluation pipeline."""
    
    def __init__(self, model_factory, predictor_cols: List[str]):
        self.model_factory = model_factory
        self.predictor_cols = predictor_cols
    
    def train_eval_cv(self, train_df: pd.DataFrame, cv_folds: List[Tuple]) -> dict:
        """
        Train and evaluate on CV folds.
        
        Returns:
            {
                'fold_results': [{fold_idx, train_r2, val_r2_by_ticker, ...}, ...],
                'in_sample_r2_mean': float (average across folds),
            }
        """
        fold_results = []
        in_sample_r2s = []
        
        for fold_idx, (train_dates, val_dates) in enumerate(cv_folds):
            train_fold = train_df[train_df['date'].isin(train_dates)].copy()
            val_fold = train_df[train_df['date'].isin(val_dates)].copy()
            
            # Prepare X, y
            X_train = train_fold[self.predictor_cols].fillna(0.0).values
            y_train = train_fold['return_premium'].values
            
            X_val = val_fold[self.predictor_cols].fillna(0.0).values
            y_val = val_fold['return_premium'].values
            
            # Train
            model = self.model_factory()
            model.fit(X_train, y_train)
            
            # Evaluate on training
            y_train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, y_train_pred)
            in_sample_r2s.append(train_r2)
            
            # Evaluate on validation
            y_val_pred = model.predict(X_val)
            val_r2_by_ticker = Evaluator.r2_by_ticker(
                y_val, y_val_pred, val_fold['ticker'].values
            )
            val_r2_agg = Evaluator.aggregate_r2(val_r2_by_ticker)
            
            fold_results.append({
                'fold_idx': fold_idx,
                'train_r2': train_r2,
                'val_r2_mean': val_r2_agg['mean'],
                'val_r2_by_ticker': val_r2_by_ticker,
                'model': model,  # Save for later inspection
            })
        
        return {
            'fold_results': fold_results,
            'in_sample_r2_mean': np.mean(in_sample_r2s),
            'cv_models': [f['model'] for f in fold_results],
        }
    
    def predict_on_test_year(self, model, test_year_df: pd.DataFrame) -> np.ndarray:
        """Predict return premium for a test year."""
        X = test_year_df[self.predictor_cols].fillna(0.0).values
        return model.predict(X)


if __name__ == '__main__':
    # Test splitter
    dates = pd.date_range('2016-01-01', periods=72, freq='M')
    splitter = TimeSeriesSplitter(initial_train_months=36, cv_months=3)
    folds = splitter.split(dates)
    print(f"CV folds: {len(folds)}")
    for i, (train_d, val_d) in enumerate(folds):
        print(f"  Fold {i}: train={len(train_d)}, val={len(val_d)}")
