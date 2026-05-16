"""
Model implementations for return prediction.
- OLS-3 (L2, Huber)
- OLS-all (L2, Huber, Huber+ElasticNet)
- RF (variance reduction impurity)
- GBRF (Huber loss, variance reduction)
"""
import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


class ModelFactory:
    """Factory for instantiating models with sensible defaults."""
    
    @staticmethod
    def ols_3_l2(alpha=1.0, tol=1e-4):
        """OLS-3 with L2 loss (Ridge)."""
        return Ridge(alpha=alpha, tol=tol)
    
    @staticmethod
    def ols_3_huber(epsilon=1.35, max_iter=500, tol=1e-4):
        """OLS-3 with Huber loss (scaled for convergence)."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('huber', HuberRegressor(epsilon=epsilon, max_iter=max_iter, tol=tol)),
        ])
    
    @staticmethod
    def ols_all_l2(alpha=1.0, tol=1e-4):
        """OLS-all with L2 loss (Ridge)."""
        return Ridge(alpha=alpha, tol=tol)
    
    @staticmethod
    def ols_all_huber(epsilon=0.1, max_iter=200, tol=1e-3):
        """OLS-all with scalable Huber loss (SGD, scaled features)."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('huber_sgd', SGDRegressor(
                loss='huber',
                epsilon=epsilon,
                penalty='l2',
                alpha=0.0001,
                max_iter=max_iter,
                tol=tol,
                random_state=42,
            )),
        ])
    
    @staticmethod
    def ols_all_huber_elasticnet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-3):
        """OLS-all with Huber loss + ElasticNet regularization."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('huber_sgd_en', SGDRegressor(
                loss='huber',
                penalty='elasticnet',
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                tol=tol,
                random_state=42,
            )),
        ])
    
    @staticmethod
    def rf(n_estimators=50, max_depth=8, min_samples_split=10, 
            criterion='squared_error', random_state=42, n_jobs=-1):
        """Random Forest with variance reduction impurity (lowered min_samples_split for small universes)."""
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    
    @staticmethod
    def gbrf(n_estimators=50, max_depth=4, learning_rate=0.1,
             loss='huber', random_state=42):
        """Gradient Boosting RF with Huber loss for variance reduction."""
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state,
        )


class ModelRegistry:
    """Registry of model variants with IDs and metadata."""
    
    MODELS = {
        'OLS-3_L2': {
            'factory': 'ols_3_l2',
            'predictors': 'base_3',
            'hparams': {'alpha': 1.0, 'tol': 1e-4},
        },
        'OLS-3_Huber': {
            'factory': 'ols_3_huber',
            'predictors': 'base_3',
            'hparams': {'epsilon': 1.35, 'max_iter': 500, 'tol': 1e-4},
        },
        'OLS-all_L2': {
            'factory': 'ols_all_l2',
            'predictors': 'all',
            'hparams': {'alpha': 1.0, 'tol': 1e-4},
        },
        'OLS-all_Huber': {
            'factory': 'ols_all_huber',
            'predictors': 'all',
            'hparams': {'epsilon': 0.1, 'max_iter': 200, 'tol': 1e-3},
        },
        'OLS-all_Huber-EN': {
            'factory': 'ols_all_huber_elasticnet',
            'predictors': 'all',
            'hparams': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1000, 'tol': 1e-3},
        },
        'RF': {
            'factory': 'rf',
            'predictors': 'all',
            'hparams': {
                'n_estimators': 5,
                'max_depth': 4,
                'min_samples_split': 2000,
                'n_jobs': -1,
            },
        },
        'GBRF_Huber': {
            'factory': 'gbrf',
            'predictors': 'all',
            'hparams': {
                'n_estimators': 10,
                'max_depth': 2,
                'learning_rate': 0.1,
            },
        },
    }

    PARAM_GRIDS = {
        'OLS-3_L2': {
            'alpha': [0.1, 1.0, 10.0],
            'tol': [1e-4],
        },
        'OLS-3_Huber': {
            'epsilon': [1.2, 1.35, 1.8],
            'max_iter': [300, 500],
            'tol': [1e-4],
        },
        'OLS-all_L2': {
            'alpha': [0.1, 1.0, 10.0],
            'tol': [1e-4],
        },
        'OLS-all_Huber': {
            'epsilon': [0.05, 0.1, 0.2],
            'max_iter': [200, 400],
            'tol': [1e-3],
        },
        'OLS-all_Huber-EN': {
            'alpha': [0.0001, 0.001, 0.01],
            'l1_ratio': [0.2, 0.5, 0.8],
            'max_iter': [400, 800],
            'tol': [1e-3],
        },
        'RF': {
            'n_estimators': [50, 100],
            'max_depth': [4, 8],
            'min_samples_split': [500, 2000],
            'n_jobs': [-1],
        },
        'GBRF_Huber': {
            'n_estimators': [30, 60],
            'max_depth': [2, 3],
            'learning_rate': [0.05, 0.1],
        },
    }
    
    @classmethod
    def get_model(cls, model_id: str, hparam_override: dict | None = None):
        """Instantiate a model by ID."""
        if model_id not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        spec = cls.MODELS[model_id]
        factory_func = getattr(ModelFactory, spec['factory'])
        params = dict(spec['hparams'])
        if hparam_override:
            params.update(hparam_override)
        return factory_func(**params)
    
    @classmethod
    def list_models(cls):
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_predictor_type(cls, model_id: str):
        """Get predictor type: 'base_3' or 'all'."""
        return cls.MODELS[model_id]['predictors']

    @classmethod
    def get_param_grid(cls, model_id: str):
        return cls.PARAM_GRIDS[model_id]


if __name__ == '__main__':
    for model_id in ModelRegistry.list_models():
        print(f"{model_id}: {ModelRegistry.get_predictor_type(model_id)}")
        model = ModelRegistry.get_model(model_id)
        print(f"  Params: {model.get_params()}\n")
