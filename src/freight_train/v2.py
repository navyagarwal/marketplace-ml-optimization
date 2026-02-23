"""
Improved Training Pipeline for Freight Cost Prediction
With better features and hyperparameter tuning
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from config import (
    CLEAN_DATA_DIR, TRAIN_TEST_SPLIT_DATE, VAL_TEST_SPLIT_DATE,
    RANDOM_STATE, FREIGHT_TARGET_MAE, FREIGHT_TARGET_R2
)
from data_preprocessing import (
    load_and_merge_data, temporal_train_test_split, haversine_distance
)
from experiment_tracker import get_tracker


def engineer_improved_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering with domain knowledge
    """
    print("\n=== Engineering Improved Features ===")

    df = df.copy()

    # Geographic features
    df['distance_km'] = haversine_distance(
        df['latitude'], df['longitude'],
        df['latitude_seller'], df['longitude_seller']
    )

    # Product dimensions
    df['product_volume_cm3'] = (
        df['product_length_cm'] *
        df['product_height_cm'] *
        df['product_width_cm']
    )

    # Fill missing values
    df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].median())
    df['product_volume_cm3'] = df['product_volume_cm3'].fillna(df['product_volume_cm3'].median())
    df['product_category_name'] = df['product_category_name'].fillna('unknown')

    # NEW FEATURES (the key to better performance!)

    # 1. Distance buckets (non-linear freight pricing)
    df['distance_bucket'] = pd.cut(
        df['distance_km'],
        bins=[0, 50, 100, 300, 500, 1000, 10000],
        labels=['very_short', 'short', 'medium', 'long', 'very_long', 'extreme']
    ).astype(str)

    # 2. Weight buckets (shipping weight classes)
    df['weight_bucket'] = pd.cut(
        df['product_weight_g'],
        bins=[0, 500, 1000, 5000, 10000, 1000000],
        labels=['very_light', 'light', 'medium', 'heavy', 'very_heavy']
    ).astype(str)

    # 3. Volume buckets (dimensional weight)
    df['volume_bucket'] = pd.cut(
        df['product_volume_cm3'],
        bins=[0, 1000, 5000, 10000, 50000, 1000000],
        labels=['very_small', 'small', 'medium', 'large', 'very_large']
    ).astype(str)

    # 4. Volumetric weight (freight companies often use this)
    df['volumetric_weight'] = df['product_volume_cm3'] / 5000  # Standard divisor
    df['chargeable_weight'] = np.maximum(df['product_weight_g'] / 1000, df['volumetric_weight'])

    # 5. Weight-to-volume ratio (density proxy)
    df['density_proxy'] = df['product_weight_g'] / (df['product_volume_cm3'] + 1)

    # 6. Distance interactions
    df['distance_x_weight'] = df['distance_km'] * df['product_weight_g']
    df['distance_x_volume'] = df['distance_km'] * df['product_volume_cm3']
    df['distance_x_chargeable_weight'] = df['distance_km'] * df['chargeable_weight']

    # 7. Price interactions (higher value = more insurance/handling)
    df['price_x_distance'] = df['price'] * df['distance_km']
    df['price_per_kg'] = df['price'] / (df['product_weight_g'] / 1000 + 0.001)
    df['is_high_value'] = (df['price'] > df['price'].quantile(0.75)).astype(int)

    # 8. Efficiency metrics
    df['weight_per_km'] = df['product_weight_g'] / (df['distance_km'] + 1)
    df['volume_per_km'] = df['product_volume_cm3'] / (df['distance_km'] + 1)

    # 9. Category-distance interaction
    df['category_distance_zone'] = df['product_category_name'] + '_' + df['distance_bucket']

    # 10. Temporal features
    df['order_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
    df['order_month'] = df['order_purchase_timestamp'].dt.month
    df['is_weekend'] = df['order_day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday_season'] = df['order_month'].isin([11, 12]).astype(int)  # Black Friday, Christmas

    # 11. Standardized metrics (helps model learn better)
    df['distance_log'] = np.log1p(df['distance_km'])
    df['weight_log'] = np.log1p(df['product_weight_g'])
    df['volume_log'] = np.log1p(df['product_volume_cm3'])

    print(f"Total features engineered: {len(df.columns)}")

    return df


def prepare_freight_data_improved(df: pd.DataFrame) -> tuple:
    """
    Prepare data with improved features
    """
    print("\n=== Preparing Freight Data (Improved) ===")

    # Filter valid rows
    df = df[df['freight_value'].notna()].copy()
    df = df[df['freight_value'] > 0].copy()
    df = df[df['distance_km'] > 0].copy()

    # Select improved feature set
    feature_columns = [
        # Core features
        'distance_km', 'product_weight_g', 'product_volume_cm3', 'price',

        # Categorical features
        'product_category_name', 'distance_bucket', 'weight_bucket', 'volume_bucket',

        # Engineered features
        'volumetric_weight', 'chargeable_weight', 'density_proxy',
        'distance_x_weight', 'distance_x_volume', 'distance_x_chargeable_weight',
        'price_x_distance', 'price_per_kg', 'is_high_value',
        'weight_per_km', 'volume_per_km',

        # Temporal features
        'order_month', 'is_weekend', 'is_holiday_season',

        # Log features
        'distance_log', 'weight_log', 'volume_log'
    ]

    X = df[feature_columns].copy()
    y = df['freight_value'].copy()

    print(f"Training samples: {len(X)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Target range: ${y.min():.2f} - ${y.max():.2f}")
    print(f"Target mean: ${y.mean():.2f}")

    return X, y


def preprocess_features(X_train, X_val, X_test, label_encoders=None):
    """
    Preprocess features with encoding
    """
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_test_processed = X_test.copy()

    if label_encoders is None:
        label_encoders = {}

        for col in categorical_features:
            le = LabelEncoder()
            X_train_processed[col] = le.fit_transform(X_train[col].astype(str))
            label_encoders[col] = le
    else:
        for col in categorical_features:
            X_train_processed[col] = label_encoders[col].transform(X_train[col].astype(str))

    # Transform val/test with unknown category handling
    for col in categorical_features:
        le = label_encoders[col]
        X_val_processed[col] = X_val[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        X_test_processed[col] = X_test[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    return X_train_processed, X_val_processed, X_test_processed, label_encoders


def train_with_hyperparameter_tuning(X_train, y_train, X_val, y_val, quick=False):
    """
    Train model with hyperparameter tuning

    Args:
        quick: If True, use fast search. If False, use thorough search.
    """
    print("\n=== Hyperparameter Tuning ===")

    if quick:
        print("Using quick search (3 iterations)...")
        param_distributions = {
            'n_estimators': [200, 300],
            'max_depth': [8, 10],
            'learning_rate': [0.05, 0.03],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [3, 5]
        }
        n_iter = 3
    else:
        print("Using thorough search (20 iterations)...")
        param_distributions = {
            'n_estimators': [150, 200, 250, 300, 400],
            'max_depth': [6, 8, 10, 12],
            'learning_rate': [0.01, 0.03, 0.05, 0.07],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 1.5, 2]
        }
        n_iter = 20

    base_model = xgb.XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=3,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print(f"\n✅ Best parameters found:")
    for param, value in search.best_params_.items():
        print(f"   {param}: {value}")

    print(f"\nBest CV score: MAE = ${-search.best_score_:.2f}")

    return search.best_estimator_, search.best_params_


def evaluate_model(model, X, y, split_name='Test'):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    within_3 = np.mean(np.abs(y - y_pred) <= 3) * 100
    within_5 = np.mean(np.abs(y - y_pred) <= 5) * 100

    print(f"\n📊 {split_name} Set Results:")
    print(f"  MAE:  ${mae:.2f} {'✅' if mae < FREIGHT_TARGET_MAE else '❌'}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²:   {r2:.4f} {'✅' if r2 > FREIGHT_TARGET_R2 else '❌'}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Within $3: {within_3:.1f}%")
    print(f"  Within $5: {within_5:.1f}%")

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'within_3_dollars': float(within_3),
        'within_5_dollars': float(within_5)
    }


def main(experiment_name="improved_features_tuned", tune_hyperparameters=True, quick_tune=True):
    """
    Main improved training pipeline
    """
    print("=" * 80)
    print("FREIGHT COST PREDICTION - IMPROVED TRAINING")
    print("=" * 80)

    start_time = datetime.now()

    # Load data
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING")
    print("=" * 80)

    df = load_and_merge_data(CLEAN_DATA_DIR)
    df = engineer_improved_features(df)

    # Split data
    train_df, val_df, test_df = temporal_train_test_split(
        df, TRAIN_TEST_SPLIT_DATE, VAL_TEST_SPLIT_DATE
    )

    # Prepare features
    X_train, y_train = prepare_freight_data_improved(train_df)
    X_val, y_val = prepare_freight_data_improved(val_df)
    X_test, y_test = prepare_freight_data_improved(test_df)

    # Preprocess
    X_train_proc, X_val_proc, X_test_proc, label_encoders = preprocess_features(
        X_train, X_val, X_test
    )

    # Train model
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING")
    print("=" * 80)

    if tune_hyperparameters:
        model, best_params = train_with_hyperparameter_tuning(
            X_train_proc, y_train, X_val_proc, y_val, quick=quick_tune
        )
    else:
        print("\nUsing default hyperparameters...")
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.03,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train_proc, y_train)
        best_params = {}

    # Evaluate
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATION")
    print("=" * 80)

    train_metrics = evaluate_model(model, X_train_proc, y_train, 'Train')
    val_metrics = evaluate_model(model, X_val_proc, y_val, 'Validation')
    test_metrics = evaluate_model(model, X_test_proc, y_test, 'Test')

    # Log experiment
    print("\n" + "=" * 80)
    print("STEP 4: LOGGING EXPERIMENT")
    print("=" * 80)

    tracker = get_tracker()
    experiment_id, exp_dir = tracker.log_experiment(
        model_type="freight",
        experiment_name=experiment_name,
        metrics=test_metrics,
        hyperparameters=best_params if tune_hyperparameters else {},
        features=X_train.columns.tolist(),
        notes=f"Improved features with {'tuned' if tune_hyperparameters else 'default'} hyperparameters"
    )

    # Save model
    joblib.dump(model, exp_dir / "freight_model.pkl")
    joblib.dump(label_encoders, exp_dir / "freight_preprocessor.pkl")

    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  Total time: {duration:.1f}s")
    print(f"\n✅ Experiment complete: {experiment_name}")
    print(f"   Test MAE: ${test_metrics['mae']:.2f}")
    print(f"   Test R²: {test_metrics['r2']:.4f}")

    if test_metrics['mae'] < FREIGHT_TARGET_MAE and test_metrics['r2'] > FREIGHT_TARGET_R2:
        print("\n🎉 MODEL MEETS SUCCESS CRITERIA!")
    else:
        print(f"\n📊 Improvement over baseline:")
        print(f"   MAE: $5.10 → ${test_metrics['mae']:.2f} ({((5.10 - test_metrics['mae']) / 5.10 * 100):.1f}% better)")
        print(f"   R²: 0.611 → {test_metrics['r2']:.3f} ({((test_metrics['r2'] - 0.611) / 0.611 * 100):.1f}% better)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    # Allow command-line arguments
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "improved_features_tuned"

    main(
        experiment_name=experiment_name,
        tune_hyperparameters=True,
        quick_tune=True  # Set to False for thorough tuning
    )
