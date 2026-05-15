"""
Main ML pipeline: training, in-sample CV eval, and out-of-sample test eval.
"""
import sys
from pathlib import Path

# Add parent to path so ml_pipeline modules can be imported
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd
import numpy as np
from ml_pipeline.data_loader import DataLoader
from ml_pipeline.models import ModelRegistry
from ml_pipeline.evaluation import TimeSeriesSplitter, ModelEvaluator, Evaluator


class MLPipeline:
    def __init__(self, root: Path):
        self.root = root
        self.loader = DataLoader(root)
        self.data = self.loader.prepare()
        
        # Extract predictor lists
        self.base_3_cols = self.data['base_3_predictors']
        self.all_cols = self.data['all_predictors']
    
    def train_and_evaluate(self, model_id: str) -> dict:
        """
        Train model with rolling time-series CV on 6-year train set.
        
        Returns:
            {
                'model_id': str,
                'in_sample_r2': float,
                'cv_results': list of fold results,
                'best_model': trained model,
            }
        """
        # Get predictor columns for this model
        pred_type = ModelRegistry.get_predictor_type(model_id)
        if pred_type == 'base_3':
            predictor_cols = self.base_3_cols
        else:  # 'all'
            predictor_cols = self.all_cols
        
        # Time-series CV: 36 months initial train, 36 months rolling
        splitter = TimeSeriesSplitter(initial_train_months=36, cv_months=36)
        train_folds = splitter.split(self.data['train_dates'])
        
        # Model evaluator
        def factory():
            return ModelRegistry.get_model(model_id)
        evaluator = ModelEvaluator(factory, predictor_cols)
        
        # Train and eval
        cv_result = evaluator.train_eval_cv(self.data['train'], train_folds)
        
        # Retrain on full train set for final model
        X_full_train = self.data['train'][predictor_cols].fillna(0.0).values
        y_full_train = self.data['train']['return_premium'].values
        final_model = ModelRegistry.get_model(model_id)
        final_model.fit(X_full_train, y_full_train)
        
        return {
            'model_id': model_id,
            'predictor_type': pred_type,
            'n_predictors': len(predictor_cols),
            'in_sample_r2': cv_result['in_sample_r2_mean'],
            'cv_results': cv_result['fold_results'],
            'final_model': final_model,
            'predictor_cols': predictor_cols,
        }
    
    def rolling_test_prediction(self, model_result: dict) -> dict:
        """
        Monthly walk-forward out-of-sample prediction on 4-year test set.
        
        Process (per month):
        1. Predict cross-section for that month with current model
        2. Compute cross-sectional R² (across tickers)
        3. Retrain by adding that month to train set
        4. Advance to next month
        
        Returns:
            {
                'model_id': str,
                'out_sample_r2': float,  # mean cross-sectional R² across all test months
                'month_results': list of {month, r2_cs},
            }
        """
        model = model_result['final_model']
        predictor_cols = model_result['predictor_cols']
        model_id = model_result['model_id']
        
        test_dates = sorted(self.data['test_dates'])
        test_df = self.data['test']
        current_train = self.data['train'].copy()
        
        month_results = []
        n_test = len(test_dates)
        
        for i, month_date in enumerate(test_dates):
            month_df = test_df[test_df['date'] == month_date].copy()
            
            X_test = month_df[predictor_cols].fillna(0.0).values
            y_test = month_df['return_premium'].values
            y_pred = model.predict(X_test)
            
            # Cross-sectional R² for this month
            from sklearn.metrics import r2_score
            cs_r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
            
            month_results.append({'month': month_date, 'r2_cs': cs_r2})
            print(f"  [{model_id}] OOS month {i+1}/{n_test}: {month_date.strftime('%Y-%m')} R²={cs_r2:.4f}", flush=True)
            
            # Retrain: add this month to train set
            current_train = pd.concat([current_train, month_df], ignore_index=True)
            X_full = current_train[predictor_cols].fillna(0.0).values
            y_full = current_train['return_premium'].values
            model = ModelRegistry.get_model(model_id)
            model.fit(X_full, y_full)
        
        valid_r2s = [m['r2_cs'] for m in month_results if not np.isnan(m['r2_cs'])]
        out_sample_r2 = np.mean(valid_r2s) if valid_r2s else np.nan
        
        return {
            'model_id': model_id,
            'out_sample_r2': out_sample_r2,
            'month_results': month_results,
        }


def main():
    root = Path('/Users/raj/ws/quantconnect')
    pipeline = MLPipeline(root)
    
    print("="*70)
    print("TRAINING PHASE: Rolling Time-Series CV on 6-year train set")
    print("="*70)
    
    train_results = {}
    for model_id in ModelRegistry.list_models():
        print(f"\n{model_id}...")
        result = pipeline.train_and_evaluate(model_id)
        train_results[model_id] = result
        print(f"  In-sample R²: {result['in_sample_r2']:.4f}")
        print(f"  Predictors: {result['n_predictors']}")
    
    print("\n" + "="*70)
    print("TESTING PHASE: Rolling prediction on 4-year test set")
    print("="*70)
    
    test_results = {}
    for model_id, train_res in train_results.items():
        print(f"\n{model_id}...")
        result = pipeline.rolling_test_prediction(train_res)
        test_results[model_id] = result
        print(f"  Out-of-sample R²: {result['out_sample_r2']:.4f}")
        # Show per-year summary (average monthly R² per calendar year)
        months_df = pd.DataFrame(result['month_results'])
        months_df['year'] = months_df['month'].dt.year
        for year, grp in months_df.groupby('year'):
            print(f"    {year}: R² = {grp['r2_cs'].mean():.4f} (n={len(grp)} months)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    summary_df = pd.DataFrame([
        {
            'model_id': model_id,
            'in_sample_r2': train_results[model_id]['in_sample_r2'],
            'out_sample_r2': test_results[model_id]['out_sample_r2'],
        }
        for model_id in ModelRegistry.list_models()
    ])
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv(root / 'ml_results_summary.csv', index=False)
    print(f"\nSaved summary to: ml_results_summary.csv")


if __name__ == '__main__':
    main()
