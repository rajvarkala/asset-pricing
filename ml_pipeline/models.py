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
    def ols_3_l2(alpha=1.0):
        """OLS-3 with L2 loss (Ridge)."""
        return Ridge(alpha=alpha)
    
    @staticmethod
    def ols_3_huber(epsilon=1.35, max_iter=500):
        """OLS-3 with Huber loss (scaled for convergence)."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('huber', HuberRegressor(epsilon=epsilon, max_iter=max_iter)),
        ])
    
    @staticmethod
    def ols_all_l2(alpha=1.0):
        """OLS-all with L2 loss (Ridge)."""
        return Ridge(alpha=alpha)
    
    @staticmethod
    def ols_all_huber(epsilon=0.1, max_iter=200):
        """OLS-all with scalable Huber loss (SGD, scaled features)."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('huber_sgd', SGDRegressor(
                loss='huber',
                epsilon=epsilon,
                penalty='l2',
                alpha=0.0001,
                max_iter=max_iter,
                tol=1e-3,
                random_state=42,
            )),
        ])
    
    @staticmethod
    def ols_all_huber_elasticnet(alpha=0.1, l1_ratio=0.5, max_iter=1000):
        """OLS-all with Huber loss + ElasticNet regularization.
        Using sklearn's ElasticNet as base; for true Huber+ElasticNet,
        we'd need custom implementation. Here we use ElasticNet as proxy.
        """
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
    
    @staticmethod
    def rf(n_estimators=50, max_depth=8, min_samples_split=50, 
           criterion='squared_error', random_state=42):
        """Random Forest with variance reduction impurity."""
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=random_state,
            n_jobs=-1,
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
            'hparams': {'alpha': 1.0},
        },
        'OLS-3_Huber': {
            'factory': 'ols_3_huber',
            'predictors': 'base_3',
            'hparams': {'epsilon': 1.35, 'max_iter': 500},
        },
        'OLS-all_L2': {
            'factory': 'ols_all_l2',
            'predictors': 'all',
            'hparams': {'alpha': 1.0},
        },
        'OLS-all_Huber': {
            'factory': 'ols_all_huber',
            'predictors': 'all',
            'hparams': {'epsilon': 0.1, 'max_iter': 200},
        },
        'OLS-all_Huber-EN': {
            'factory': 'ols_all_huber_elasticnet',
            'predictors': 'all',
            'hparams': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1000},
        },
        'RF': {
            'factory': 'rf',
            'predictors': 'all',
            'hparams': {
                'n_estimators': 5,
                'max_depth': 4,
                'min_samples_split': 2000,
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
    
    @classmethod
    def get_model(cls, model_id: str):
        """Instantiate a model by ID."""
        if model_id not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        spec = cls.MODELS[model_id]
        factory_func = getattr(ModelFactory, spec['factory'])
        return factory_func(**spec['hparams'])
    
    @classmethod
    def list_models(cls):
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_predictor_type(cls, model_id: str):
        """Get predictor type: 'base_3' or 'all'."""
        return cls.MODELS[model_id]['predictors']


if __name__ == '__main__':
    for model_id in ModelRegistry.list_models():
        print(f"{model_id}: {ModelRegistry.get_predictor_type(model_id)}")
        model = ModelRegistry.get_model(model_id)
        print(f"  Params: {model.get_params()}\n")
