"""
Time-series cross-validation and evaluation utilities.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from itertools import product
import time
from typing import List, Tuple

from ml_pipeline.vip_importance import compute_vip_r2_importance


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

    def train_eval_cv(
        self,
        train_df: pd.DataFrame,
        cv_folds: List[Tuple],
        compute_importance: bool = False,
        importance_label: str | None = None,
    ) -> dict:
        """
        Train and evaluate on CV folds.
        
        Returns:
            {
                'fold_results': [{fold_idx, train_r2, val_r2_by_ticker, ...}, ...],
                'in_sample_r2_mean': float (average across folds),
                'importance_scores': dict {predictor_name: avg_importance},
            }
        """
        fold_results = []
        in_sample_r2s = []
        all_importances = []
        standalone_predictors = [col for col in self.predictor_cols if '__x__' not in col]
        
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
            val_r2_cs = r2_score(y_val, y_val_pred) if len(y_val) > 1 else np.nan

            if compute_importance:
                if importance_label is not None:
                    print(
                        f"[VIP:{importance_label}] fold {fold_idx + 1}/{len(cv_folds)}",
                        flush=True,
                    )
                val_importance = compute_vip_r2_importance(
                    model=model,
                    X=val_fold[self.predictor_cols],
                    y=val_fold['return_premium'],
                    feature_names=standalone_predictors,
                )
                all_importances.append(val_importance)
            
            fold_results.append({
                'fold_idx': fold_idx,
                'train_r2': train_r2,
                'val_r2_mean': val_r2_cs,
                'model': model,  # Save for later inspection
            })
        
        if compute_importance:
            importance_dict = pd.DataFrame(all_importances).fillna(0.0).mean(axis=0).to_dict()
        else:
            importance_dict = {}
        
        return {
            'fold_results': fold_results,
            'in_sample_r2_mean': np.mean(in_sample_r2s),
            'cv_models': [f['model'] for f in fold_results],
            'importance_scores': importance_dict,
        }

    @staticmethod
    def rolling_test_prediction_monthly(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        predictor_cols: List[str],
        model_factory,
    ) -> dict:
        """
        Monthly walk-forward out-of-sample prediction.

        Process per month:
        1. Fit model on current training set.
        2. Predict one test month.
        3. Compute cross-sectional monthly R².
        4. Add that month to training set.
        
        Returns detailed predictions per ticker for quartile analysis.
        """
        current_train = train_df.copy()
        test_dates = sorted(test_df['date'].unique())
        month_results = []
        all_predictions = []

        for d in test_dates:
            model = model_factory()

            X_train = current_train[predictor_cols].fillna(0.0).values
            y_train = current_train['return_premium'].values
            model.fit(X_train, y_train)

            month_df = test_df[test_df['date'] == d].copy()
            X_test = month_df[predictor_cols].fillna(0.0).values
            y_test = month_df['return_premium'].values
            y_pred = model.predict(X_test)

            month_r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
            month_results.append({'date': pd.Timestamp(d), 'r2_monthly': month_r2})
            
            # Store detailed predictions for each ticker
            for idx, ticker in enumerate(month_df['ticker'].values):
                all_predictions.append({
                    'date': pd.Timestamp(d),
                    'ticker': ticker,
                    'actual': y_test[idx],
                    'pred': y_pred[idx],
                })

            current_train = pd.concat([current_train, month_df], ignore_index=True)

        valid_r2s = [m['r2_monthly'] for m in month_results if not np.isnan(m['r2_monthly'])]
        out_sample_r2 = np.mean(valid_r2s) if valid_r2s else np.nan

        return {
            'out_sample_r2': out_sample_r2,
            'month_results': month_results,
            'all_predictions': pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame(),
        }
    
    def predict_on_test_year(self, model, test_year_df: pd.DataFrame) -> np.ndarray:
        """Predict return premium for a test year."""
        X = test_year_df[self.predictor_cols].fillna(0.0).values
        return model.predict(X)


class HyperparameterTuner:
    """Grid-search style tuning on rolling time-series CV folds."""

    @staticmethod
    def _param_combinations(param_grid: dict):
        keys = list(param_grid.keys())
        for values in product(*(param_grid[k] for k in keys)):
            yield dict(zip(keys, values))

    @staticmethod
    def tune(
        train_df: pd.DataFrame,
        predictor_cols: List[str],
        cv_folds: List[Tuple],
        model_factory_from_params,
        param_grid: dict,
        default_params: dict,
        model_label: str = "model",
    ) -> tuple[dict, dict]:
        """
        Tune params by mean validation R² across monthly CV folds.
        """
        best_params = None
        best_score = -np.inf
        best_cv_result = None
        combinations = list(HyperparameterTuner._param_combinations(param_grid))
        total = len(combinations)
        started_at = time.time()
        keys = list(param_grid.keys())

        print(f"[TUNE:{model_label}] Starting grid search with {total} combinations", flush=True)

        idx = 0
        prefix_state = {}
        while idx < total:
            params = combinations[idx]
            shown_idx = idx + 1
            print(f"[TUNE:{model_label}] {shown_idx}/{total} params={params}", flush=True)
            evaluator = ModelEvaluator(lambda p=params: model_factory_from_params(p), predictor_cols)
            cv_result = evaluator.train_eval_cv(train_df, cv_folds, compute_importance=False)
            scores = [f['val_r2_mean'] for f in cv_result['fold_results']]
            score = np.mean(scores)

            elapsed = time.time() - started_at
            best_so_far = best_score if np.isfinite(best_score) else np.nan
            print(
                f"[TUNE:{model_label}] {shown_idx}/{total} mean_val_r2={score:.6f} "
                f"best_so_far={best_so_far:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )

            prefix = tuple(params[k] for k in keys[:-1]) if len(keys) > 1 else tuple()
            state = prefix_state.setdefault(
                prefix,
                {
                    'best': -np.inf,
                    'prev': None,
                    'improved_once': False,
                },
            )

            if np.isnan(score):
                state['prev'] = score
                idx += 1
                continue

            if score > best_score:
                best_score = score
                best_params = params
                best_cv_result = cv_result
                print(
                    f"[TUNE:{model_label}] New best at {shown_idx}/{total}: "
                    f"score={best_score:.6f} params={best_params}",
                    flush=True,
                )

            if score > state['best']:
                state['best'] = score
                state['improved_once'] = True

            # Local-best prune by parameter change sequence:
            # when the score declines after at least one improvement under the same prefix,
            # skip remaining values for the last parameter for that prefix.
            if (
                len(keys) > 1
                and state['improved_once']
                and state['prev'] is not None
                and np.isfinite(state['prev'])
                and score < state['prev']
            ):
                next_idx = idx + 1
                skipped = 0
                while next_idx < total:
                    next_params = combinations[next_idx]
                    next_prefix = tuple(next_params[k] for k in keys[:-1])
                    if next_prefix != prefix:
                        break
                    skipped += 1
                    next_idx += 1

                if skipped > 0:
                    print(
                        f"[TUNE:{model_label}] Local-best prune at {shown_idx}/{total}; "
                        f"skipping {skipped} combos for parameter {keys[-1]} with prefix "
                        f"{dict(zip(keys[:-1], prefix))}",
                        flush=True,
                    )
                    state['prev'] = score
                    idx = next_idx
                    continue

            state['prev'] = score
            idx += 1

        if best_params is None:
            best_params = dict(default_params)
        
        evaluator = ModelEvaluator(lambda: model_factory_from_params(best_params), predictor_cols)
        best_cv_result = evaluator.train_eval_cv(
            train_df,
            cv_folds,
            compute_importance=True,
            importance_label=model_label,
        )

        total_elapsed = time.time() - started_at
        print(
            f"[TUNE:{model_label}] Completed in {total_elapsed:.1f}s. "
            f"Best score={best_score:.6f} Best params={best_params}",
            flush=True,
        )

        return best_params, best_cv_result


if __name__ == '__main__':
    # Test splitter
    dates = pd.date_range('2016-01-01', periods=72, freq='M')
    splitter = TimeSeriesSplitter(initial_train_months=36, cv_months=3)
    folds = splitter.split(dates)
    print(f"CV folds: {len(folds)}")
    for i, (train_d, val_d) in enumerate(folds):
        print(f"  Fold {i}: train={len(train_d)}, val={len(val_d)}")
