"""
Plotting utilities for variable importance visualization.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ImportancePlotter:
    """Create and save variable importance plots."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
    
    @staticmethod
    def filter_standalone_predictors(importance_dict: dict) -> dict:
        """
        Filter to standalone predictors only (no interaction terms).
        Interaction terms are named with '__x__' pattern.
        """
        return {
            k: v for k, v in importance_dict.items()
            if '__x__' not in k
        }
    
    @staticmethod
    def get_top_10(importance_dict: dict) -> dict:
        """Get top 10 predictors by importance."""
        sorted_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        return dict(list(sorted_dict.items())[:10])
    
    def plot_model_importance(self, model_id: str, importance_dict: dict) -> None:
        """
        Plot and save variable importance for a single model.
        
        Args:
            model_id: Name of the model
            importance_dict: {predictor_name: importance_score}
        """
        # Filter and select
        standalone = self.filter_standalone_predictors(importance_dict)
        top_10 = self.get_top_10(standalone)
        
        if not top_10:
            print(f"  {model_id}: No standalone predictors found")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        predictors = list(top_10.keys())
        scores = list(top_10.values())
        
        # Create color gradient (higher importance = darker color)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(predictors)))
        
        # Horizontal bar chart
        y_pos = np.arange(len(predictors))
        ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=1.2)
        
        # Labels and formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(predictors, fontsize=11)
        ax.set_xlabel('R² Importance Drop (vip permutation, avg validation folds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_id} - Top 10 Predictors', fontsize=14, fontweight='bold', pad=20)
        
        # Invert y-axis so highest importance is at top
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, (predictor, score) in enumerate(zip(predictors, scores)):
            ax.text(score, i, f' {score:.4f}', va='center', fontsize=10)
        
        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f'{model_id}_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        
        plt.close()
    
    def plot_all_models(self, importance_by_model: dict) -> None:
        """
        Plot importance for all models.
        
        Args:
            importance_by_model: {model_id: {predictor_name: importance_score}}
        """
        print(f"\nPlotting variable importance for {len(importance_by_model)} models...")
        for model_id, importance_dict in importance_by_model.items():
            print(f"  {model_id}...")
            self.plot_model_importance(model_id, importance_dict)
        
        print(f"All plots saved to: {self.output_dir}")


def save_importance_cache(importance_by_model: dict, output_dir: Path) -> None:
    """Save variable importance to JSON cache."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save filtered (standalone only) importance
    filtered = {}
    for model_id, importance_dict in importance_by_model.items():
        standalone = {
            k: v for k, v in importance_dict.items()
            if '__x__' not in k
        }
        filtered[model_id] = standalone
    
    cache_path = output_dir / 'variable_importance_cache.json'
    with open(cache_path, 'w') as f:
        json.dump(filtered, f, indent=2)
    
    print(f"Saved importance cache: {cache_path}")


def load_importance_cache(output_dir: Path) -> dict | None:
    """Load variable importance from JSON cache."""
    cache_path = Path(output_dir) / 'variable_importance_cache.json'
    if not cache_path.exists():
        return None
    
    with open(cache_path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # Test
    test_importance = {
        'model1': {
            'pred1': 0.5,
            'pred2': 0.3,
            'pred3__x__pred1': 0.1,
            'pred4': 0.8,
            'pred5': 0.2,
            'pred6': 0.1,
            'pred7': 0.05,
            'pred8': 0.04,
            'pred9': 0.03,
            'pred10': 0.02,
            'pred11': 0.01,
            'pred12__x__pred2': 0.15,
        }
    }
    
    plotter = ImportancePlotter(Path('/tmp/test_plots'))
    plotter.plot_all_models(test_importance)
    save_importance_cache(test_importance, Path('/tmp/test_plots'))
