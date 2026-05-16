"""
Quick test of ML pipeline with single model (OLS-3_L2).
"""
import sys
from pathlib import Path

# Add parent to path
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd
import numpy as np
from ml_pipeline.data_loader import DataLoader
from ml_pipeline.models import ModelRegistry
from ml_pipeline.evaluation import TimeSeriesSplitter, ModelEvaluator

# Load data
print("Loading data...")
pipeline_root = Path(__file__).resolve().parent.parent
loader = DataLoader(pipeline_root)
data = loader.prepare()
print(f"  Train: {len(data['train'])} rows, {len(data['train_dates'])} months")
print(f"  Test: {len(data['test'])} rows, {len(data['test_dates'])} months")
print(f"  Tickers: {len(data['tickers'])}")

# Test single model
model_id = 'OLS-3_L2'
print(f"\nTesting {model_id}...")

predictor_cols = data['base_3_predictors']
print(f"  Predictors: {predictor_cols}")

# Time-series CV: 1-year monthly validation within training window
print("\n  Running rolling CV...")
splitter = TimeSeriesSplitter(
    initial_train_months=data['initial_train_months'],
    cv_months=data['validation_months'],
)
train_folds = splitter.split(data['train_dates'])
print(f"  Generated {len(train_folds)} CV folds")

# Check first fold
first_fold = train_folds[0]
print(f"  Fold 1: {len(first_fold[0])} train months, {len(first_fold[1])} val months")

# Model evaluator
def factory():
    return ModelRegistry.get_model(model_id)

evaluator = ModelEvaluator(factory, predictor_cols)
print("\n  Training and evaluating...")
cv_result = evaluator.train_eval_cv(data['train'], train_folds)

print(f"\n  In-sample R² (mean across folds): {cv_result['in_sample_r2_mean']:.4f}")
print(f"  Fold results (showing first 5):")
for i, fold_res in enumerate(cv_result['fold_results'][:5]):
    train_r2 = fold_res['train_r2']
    val_r2 = fold_res['val_r2_mean']
    print(f"    Fold {i+1}: train R² = {train_r2:.4f}, val R² = {val_r2:.4f}")

print("\nTest successful!")
