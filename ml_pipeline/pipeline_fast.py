"""
Fast ML pipeline test: train OLS-3_L2 and OLS-all_L2, then run OOS eval.
"""
import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd
import numpy as np
from ml_pipeline.data_loader import DataLoader
from ml_pipeline.models import ModelRegistry
from ml_pipeline.evaluation import TimeSeriesSplitter, ModelEvaluator, Evaluator

class FastPipeline:
    def __init__(self, root: Path):
        self.root = root
        self.loader = DataLoader(root)
        self.data = self.loader.prepare()
        self.base_3_cols = self.data['base_3_predictors']
        self.all_cols = self.data['all_predictors']
    
    def train_and_evaluate(self, model_id: str):
        pred_type = ModelRegistry.get_predictor_type(model_id)
        predictor_cols = self.base_3_cols if pred_type == 'base_3' else self.all_cols
        
        splitter = TimeSeriesSplitter(initial_train_months=36, cv_months=36)
        train_folds = splitter.split(self.data['train_dates'])
        
        def factory():
            return ModelRegistry.get_model(model_id)
        
        evaluator = ModelEvaluator(factory, predictor_cols)
        cv_result = evaluator.train_eval_cv(self.data['train'], train_folds)
        
        # Retrain on full train set
        X_full = self.data['train'][predictor_cols].fillna(0.0).values
        y_full = self.data['train']['return_premium'].values
        final_model = ModelRegistry.get_model(model_id)
        final_model.fit(X_full, y_full)
        
        return {
            'model_id': model_id,
            'in_sample_r2': cv_result['in_sample_r2_mean'],
            'final_model': final_model,
            'predictor_cols': predictor_cols,
        }
    
    def rolling_test_prediction(self, model_result: dict):
        from sklearn.metrics import r2_score as _r2
        model = model_result['final_model']
        predictor_cols = model_result['predictor_cols']
        model_id = model_result['model_id']
        
        test_dates = sorted(self.data['test_dates'])
        test_df = self.data['test']
        current_train = self.data['train'].copy()
        
        month_results = []
        
        for month_date in test_dates:
            month_df = test_df[test_df['date'] == month_date].copy()
            
            X_test = month_df[predictor_cols].fillna(0.0).values
            y_test = month_df['return_premium'].values
            y_pred = model.predict(X_test)
            
            cs_r2 = _r2(y_test, y_pred) if len(y_test) > 1 else np.nan
            month_results.append({'month': month_date, 'r2_cs': cs_r2})
            
            # Retrain with this month added
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

# Run with just 2 models
root = Path('/Users/raj/ws/quantconnect')
pipeline = FastPipeline(root)

models_to_test = ['OLS-3_L2', 'OLS-all_L2']

print("="*70)
print("TRAINING PHASE (Fast: 2 models)")
print("="*70)

train_results = {}
for model_id in models_to_test:
    print(f"\n{model_id}...")
    result = pipeline.train_and_evaluate(model_id)
    train_results[model_id] = result
    print(f"  In-sample R²: {result['in_sample_r2']:.4f}")

print("\n" + "="*70)
print("TESTING PHASE (Out-of-sample rolling prediction)")
print("="*70)

test_results = {}
for model_id in models_to_test:
    print(f"\n{model_id}...")
    result = pipeline.rolling_test_prediction(train_results[model_id])
    test_results[model_id] = result
    print(f"  Out-of-sample R²: {result['out_sample_r2']:.4f}")
    months_df = pd.DataFrame(result['month_results'])
    months_df['year'] = months_df['month'].dt.year
    for year, grp in months_df.groupby('year'):
        print(f"    {year}: R² = {grp['r2_cs'].mean():.4f} (n={len(grp)} months)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
summary_df = pd.DataFrame([
    {
        'model_id': model_id,
        'in_sample_r2': train_results[model_id]['in_sample_r2'],
        'out_sample_r2': test_results[model_id]['out_sample_r2'],
    }
    for model_id in models_to_test
])
print(summary_df.to_string(index=False))
summary_df.to_csv(root / 'ml_results_summary_fast.csv', index=False)
print(f"\nResults saved to: ml_results_summary_fast.csv")
