"""
Model factory for different ML algorithms
"""
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


class ModelFactory:
    """
    Factory for creating different ML models with sensible defaults
    """

    @staticmethod
    def create_model(model_name: str, random_state: int = 42):
        """
        Create a model by name

        Available models:
        - xgboost_baseline
        - xgboost_deep
        - xgboost_fast
        - lightgbm
        - random_forest
        - gradient_boosting
        - ridge
        - lasso
        - elastic_net
        """

        if model_name == "xgboost_baseline":
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                random_state=random_state,
                n_jobs=-1
            )

        elif model_name == "xgboost_deep":
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=12,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=random_state,
                n_jobs=-1
            )

        elif model_name == "xgboost_fast":
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1
            )

        elif model_name == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
                num_leaves=31,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )

        elif model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            )

        elif model_name == "gradient_boosting":
            # Use HistGradientBoostingRegressor (much faster, handles NaN natively)
            return HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=random_state
            )

        elif model_name == "ridge":
            # Ridge with StandardScaler pipeline
            return Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0, random_state=random_state))
            ])

        elif model_name == "lasso":
            # Lasso with StandardScaler pipeline
            return Pipeline([
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=0.1, random_state=random_state, max_iter=10000))
            ])

        elif model_name == "elastic_net":
            # ElasticNet with StandardScaler pipeline
            return Pipeline([
                ('scaler', StandardScaler()),
                ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=10000))
            ])

        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def get_model_type(model_name: str) -> str:
        """Get model type category"""
        if 'xgboost' in model_name:
            return 'tree_ensemble'
        elif model_name in ['lightgbm', 'random_forest', 'gradient_boosting']:
            return 'tree_ensemble'
        elif model_name in ['ridge', 'lasso', 'elastic_net']:
            return 'linear'
        else:
            return 'other'

    @staticmethod
    def get_available_models() -> list:
        """Get list of available model names"""
        return [
            'xgboost_baseline',
            'xgboost_deep',
            'xgboost_fast',
            'lightgbm',
            'random_forest',
            'gradient_boosting',
            'ridge',
            'lasso',
            'elastic_net'
        ]


# Recommended model combinations for experimentation
EXPERIMENT_CONFIGS = {
    # Freight prediction experiments
    'freight': [
        {
            'name': 'baseline',
            'model': 'xgboost_baseline',
            'features': 'baseline',
            'description': 'Simple XGBoost with basic features'
        },
        {
            'name': 'xgboost_improved',
            'model': 'xgboost_baseline',
            'features': 'improved',
            'description': 'XGBoost with engineered features'
        },
        {
            'name': 'xgboost_deep',
            'model': 'xgboost_deep',
            'features': 'advanced',
            'description': 'Deep XGBoost with all features'
        },
        {
            'name': 'lightgbm_advanced',
            'model': 'lightgbm',
            'features': 'advanced',
            'description': 'LightGBM with all features'
        },
        {
            'name': 'random_forest_improved',
            'model': 'random_forest',
            'features': 'improved',
            'description': 'Random Forest with improved features'
        },
        {
            'name': 'ridge_minimal',
            'model': 'ridge',
            'features': 'minimal',
            'description': 'Ridge regression with core features'
        }
    ],

    # Delivery prediction experiments
    'delivery': [
        {
            'name': 'baseline',
            'model': 'xgboost_baseline',
            'features': 'baseline',
            'description': 'Simple XGBoost with basic features'
        },
        {
            'name': 'xgboost_improved',
            'model': 'xgboost_baseline',
            'features': 'improved',
            'description': 'XGBoost with engineered features'
        },
        {
            'name': 'xgboost_deep',
            'model': 'xgboost_deep',
            'features': 'advanced',
            'description': 'Deep XGBoost with all features'
        },
        {
            'name': 'lightgbm_advanced',
            'model': 'lightgbm',
            'features': 'advanced',
            'description': 'LightGBM with all features'
        },
        {
            'name': 'random_forest_improved',
            'model': 'random_forest',
            'features': 'improved',
            'description': 'Random Forest with improved features'
        },
        {
            'name': 'gradient_boosting_advanced',
            'model': 'gradient_boosting',
            'features': 'advanced',
            'description': 'Gradient Boosting with all features'
        }
    ]
}


def get_experiment_configs(task: str) -> list:
    """Get experiment configurations for a task (freight or delivery)"""
    if task not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    return EXPERIMENT_CONFIGS[task]
