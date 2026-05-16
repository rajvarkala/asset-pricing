"""
Main ML pipeline: training, in-sample CV eval, and out-of-sample test eval.
"""
import sys
import argparse
import warnings
import json
from pathlib import Path
from pandas.errors import PerformanceWarning

# Add parent to path so ml_pipeline modules can be imported
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd
from ml_pipeline.data_loader import DataLoader
from ml_pipeline.models import ModelRegistry
from ml_pipeline.evaluation import TimeSeriesSplitter, ModelEvaluator, HyperparameterTuner
from ml_pipeline.plot_importance import ImportancePlotter, save_importance_cache


class MLPipeline:
    def __init__(self, root: Path, max_tickers: int | None = None):
        self.root = root
        self.loader = DataLoader(root)
        self.data = self.loader.prepare(max_tickers=max_tickers)
        
        # Extract predictor lists
        self.base_3_cols = self.data['base_3_predictors']
        self.all_cols = self.data['all_predictors']
    
    def train_and_evaluate(self, model_id: str) -> dict:
        """
        Train model with rolling time-series CV on 9-year train set.
        
        Returns:
            {
                'model_id': str,
                'in_sample_r2': float,
                'cv_results': list of fold results,
                'best_model': trained model,
                'importance_scores': dict of variable importance
            }
        """
        # Get predictor columns for this model
        pred_type = ModelRegistry.get_predictor_type(model_id)
        if pred_type == 'base_3':
            predictor_cols = self.base_3_cols
        else:  # 'all'
            predictor_cols = self.all_cols
        
        # Time-series CV within train period: monthly validation for 1 year.
        splitter = TimeSeriesSplitter(
            initial_train_months=self.data['initial_train_months'],
            cv_months=self.data['validation_months'],
        )
        train_folds = splitter.split(self.data['train_dates'])
        
        best_params, cv_result = HyperparameterTuner.tune(
            train_df=self.data['train'],
            predictor_cols=predictor_cols,
            cv_folds=train_folds,
            model_factory_from_params=lambda p: ModelRegistry.get_model(model_id, hparam_override=p),
            param_grid=ModelRegistry.get_param_grid(model_id),
            default_params=ModelRegistry.MODELS[model_id]['hparams'],
            model_label=model_id,
        )
        print(f"  Best params: {best_params}", flush=True)
        
        # Extract variable importance (averaged across CV folds)
        importance_scores = cv_result['importance_scores']
        standalone_importance = {
            k: v for k, v in importance_scores.items()
            if '__x__' not in k
        }
        
        # Retrain on full train set for final model
        X_full_train = self.data['train'][predictor_cols].fillna(0.0).values
        y_full_train = self.data['train']['return_premium'].values
        final_model = ModelRegistry.get_model(model_id, hparam_override=best_params)
        final_model.fit(X_full_train, y_full_train)
        
        return {
            'model_id': model_id,
            'predictor_type': pred_type,
            'n_predictors': len(predictor_cols),
            'in_sample_r2': cv_result['in_sample_r2_mean'],
            'cv_results': cv_result['fold_results'],
            'best_params': best_params,
            'importance_scores': standalone_importance,
            'final_model': final_model,
            'predictor_cols': predictor_cols,
        }
    
    def rolling_test_prediction(self, model_result: dict) -> dict:
        """
        Monthly walk-forward out-of-sample prediction on 1-year test set.
        
        Process (per month):
        1. Fit on current training data.
        2. Predict one test month.
        3. Compute cross-sectional monthly R².
        4. Add that month and continue.
        
        Returns:
            {
                'model_id': str,
                'out_sample_r2': float,  # mean monthly cross-sectional R²
                'month_results': list of {date, r2_monthly},
            }
        """
        predictor_cols = model_result['predictor_cols']
        model_id = model_result['model_id']
        result = ModelEvaluator.rolling_test_prediction_monthly(
            train_df=self.data['train'],
            test_df=self.data['test'],
            predictor_cols=predictor_cols,
            model_factory=lambda: ModelRegistry.get_model(model_id, hparam_override=model_result['best_params']),
        )

        for i, m in enumerate(result['month_results'], 1):
            print(
                f"  [{model_id}] OOS month {i}/{len(result['month_results'])}: "
                f"{m['date'].strftime('%Y-%m')} R²={m['r2_monthly']:.4f}",
                flush=True,
            )

        return {
            'model_id': model_id,
            'out_sample_r2': result['out_sample_r2'],
            'month_results': result['month_results'],
            'all_predictions': result.get('all_predictions', pd.DataFrame()),
        }
    
    def generate_quartile_panel(self, test_results: dict) -> pd.DataFrame:
        """
        Generate test results panel by return quartiles.
        
        Rows: All, Top Quartile (top 25% by return), Bottom Quartile (bottom 25% by return)
        Columns: Model IDs
        Values: Out-of-sample R²
        
        Returns:
            DataFrame with shape (3, n_models)
        """
        import numpy as np
        from sklearn.metrics import r2_score
        
        # Compute aggregate return per ticker over entire test period
        test_df = self.data['test'].copy()
        ticker_returns = test_df.groupby('ticker')['return_premium'].sum().sort_values()
        
        # Identify quartiles
        n_tickers = len(ticker_returns)
        q_size = max(1, n_tickers // 4)  # First and last quartile
        
        bottom_q_tickers = set(ticker_returns.head(q_size).index)
        top_q_tickers = set(ticker_returns.tail(q_size).index)
        all_tickers = set(test_df['ticker'].unique())
        
        subsets = {
            'All': all_tickers,
            'Top Quartile': top_q_tickers,
            'Bottom Quartile': bottom_q_tickers,
        }
        
        print(f"Ticker quartile sizes: All={len(all_tickers)}, Top={len(top_q_tickers)}, Bottom={len(bottom_q_tickers)}", flush=True)
        
        results = {}
        for model_id, test_res in test_results.items():
            # Get predictions per ticker
            pred_df = test_res.get('all_predictions', pd.DataFrame())
            if pred_df.empty:
                # Fallback: use aggregate R² for all subsets
                for subset_name in subsets.keys():
                    results[(subset_name, model_id)] = test_res['out_sample_r2']
            else:
                # Compute R² for each subset
                for subset_name, subset_tickers in subsets.items():
                    subset_pred_df = pred_df[pred_df['ticker'].isin(subset_tickers)]
                    if len(subset_pred_df) > 1:
                        r2 = r2_score(subset_pred_df['actual'], subset_pred_df['pred'])
                    else:
                        r2 = np.nan
                    results[(subset_name, model_id)] = r2
                    print(f"  [{subset_name}] {model_id}: R²(OOS) = {r2:.4f}", flush=True)
        
        # Create grid DataFrame
        model_ids = list(test_results.keys())
        subset_names = list(subsets.keys())
        grid = pd.DataFrame(
            [[results.get((subset, model_id), np.nan) for model_id in model_ids] for subset in subset_names],
            index=subset_names,
            columns=model_ids,
        )
        return grid



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run ML pipeline')
    parser.add_argument('--max-tickers', type=int, default=None)
    parser.add_argument('--dev-100-tickers', action='store_true')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and load cached best params')
    parser.add_argument('--test-quartile-panel', action='store_true', help='Generate test results panel by return quartiles (requires --skip-training)')
    return parser


def main():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r"'ml_pipeline\.pipeline' found in sys\.modules.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=PerformanceWarning,
        message=r"DataFrame is highly fragmented.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"The use of py2rpy in module rpy2\.robjects\.conversion is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"The use of rpy2py in module rpy2\.robjects\.conversion is deprecated.*",
    )

    args = build_parser().parse_args()

    max_tickers = args.max_tickers
    if args.dev_100_tickers:
        max_tickers = 100

    root = Path(__file__).resolve().parent.parent
    pipeline = MLPipeline(root, max_tickers=max_tickers)
    if max_tickers is not None:
        print(f"Using ticker subset: {max_tickers}")
    model_ids = ModelRegistry.list_models()
    
    # Check for skip-training mode
    cache_dir = root / 'ml_results_cache'
    best_params_cache_path = cache_dir / 'best_params.json'
    
    skip_training = args.skip_training and best_params_cache_path.exists()
    if skip_training:
        print(f"--skip-training: Loading cached best parameters from {best_params_cache_path}")
    
    print("="*70)
    if not skip_training:
        print("TRAINING PHASE: Rolling Time-Series CV on 9-year train set")
    else:
        print("SKIPPING TRAINING PHASE: Using cached parameters")
    print("="*70)
    
    train_results = {}
    
    if not skip_training:
        # Training phase
        for model_id in model_ids:
            print(f"\n{model_id}...")
            result = pipeline.train_and_evaluate(model_id)
            train_results[model_id] = result
            print(f"  In-sample R²: {result['in_sample_r2']:.4f}")
            print(f"  Chosen params: {result['best_params']}")
            print(f"  Predictors: {result['n_predictors']}")
        
        # Cache best parameters and importance scores
        print("\n" + "="*70)
        print("CACHING: Saving best parameters and variable importance")
        print("="*70)
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best params
        best_params_dict = {
            model_id: result['best_params']
            for model_id, result in train_results.items()
        }
        with open(best_params_cache_path, 'w') as f:
            json.dump(best_params_dict, f, indent=2)
        print(f"Saved best parameters: {best_params_cache_path}")
        
        # Save and plot importance scores
        importance_by_model = {
            model_id: result['importance_scores']
            for model_id, result in train_results.items()
        }
        
        save_importance_cache(importance_by_model, cache_dir)
        
        plotter = ImportancePlotter(cache_dir / 'plots')
        plotter.plot_all_models(importance_by_model)
    else:
        # Load cached parameters
        with open(best_params_cache_path, 'r') as f:
            best_params_dict = json.load(f)
        
        # Reconstruct minimal train_results for testing phase
        for model_id in model_ids:
            pred_type = ModelRegistry.get_predictor_type(model_id)
            if pred_type == 'base_3':
                predictor_cols = pipeline.base_3_cols
            else:
                predictor_cols = pipeline.all_cols
            
            train_results[model_id] = {
                'model_id': model_id,
                'predictor_type': pred_type,
                'n_predictors': len(predictor_cols),
                'best_params': best_params_dict.get(model_id, {}),
                'predictor_cols': predictor_cols,
            }
        
        print(f"Loaded {len(train_results)} models from cache")

    print("\n" + "="*70)
    print("TESTING PHASE: Rolling monthly prediction on 1-year test set")
    print("="*70)

    test_results = {}
    for model_id, train_res in train_results.items():
        print(f"\n{model_id}...")
        result = pipeline.rolling_test_prediction(train_res)
        test_results[model_id] = result
        print(f"  Out-of-sample R²: {result['out_sample_r2']:.4f}")
        
        # Only print month details if not in skip-training mode (for brevity)
        if not skip_training:
            for m in result['month_results']:
                print(f"    {m['date'].strftime('%Y-%m')}: R² = {m['r2_monthly']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    summary_df = pd.DataFrame([
        {
            'model_id': model_id,
            'in_sample_r2': train_results[model_id].get('in_sample_r2', np.nan),
            'out_sample_r2': test_results[model_id]['out_sample_r2'],
        }
        for model_id in model_ids
    ])
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv(root / 'ml_results_summary.csv', index=False)
    print(f"\nSaved summary to: ml_results_summary.csv")
    
    # Generate quartile panel if requested
    if args.test_quartile_panel:
        print("\n" + "="*70)
        print("QUARTILE PANEL: Test results by return quartiles")
        print("="*70)
        quartile_panel = pipeline.generate_quartile_panel(test_results)
        print("\n" + "="*70)
        print("QUARTILE PANEL RESULTS")
        print("="*70)
        print(quartile_panel.to_string())
        
        # Export to CSV and Excel
        csv_path = root / 'ml_results_quartile_panel.csv'
        quartile_panel.to_csv(csv_path)
        print(f"\nSaved quartile panel to: {csv_path}")
        
        try:
            import openpyxl
            excel_path = root / 'ml_results_quartile_panel.xlsx'
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                quartile_panel.to_excel(writer, sheet_name='Quartile Panel')
            print(f"Saved quartile panel to: {excel_path}")
        except ImportError:
            print("openpyxl not found; skipping Excel export")
        except Exception as e:
            print(f"Error writing Excel file: {e}")


if __name__ == '__main__':
    import numpy as np
    main()
