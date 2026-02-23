"""
Training pipeline for Freight Cost Prediction Model
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from config import (
    CLEAN_DATA_DIR, TRAIN_TEST_SPLIT_DATE, VAL_TEST_SPLIT_DATE,
    FREIGHT_MODEL_FILE, FREIGHT_PREPROCESSOR_FILE,
    FREIGHT_MODEL_PARAMS, FREIGHT_TARGET_MAE, FREIGHT_TARGET_R2,
    MODELS_DIR, LOGS_DIR
)
from data_preprocessing import (
    load_and_merge_data, engineer_features, prepare_freight_data,
    temporal_train_test_split
)


class FreightModelTrainer:
    """
    Complete training pipeline for freight cost prediction
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        self.feature_names = None
        self.metrics = {}

    def preprocess_features(self, X_train, X_val, X_test):
        """
        Preprocess features: encode categories, scale if needed
        """
        print("\n=== Preprocessing Features ===")

        # Separate numeric and categorical features
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")

        # Encode categorical features
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        X_test_processed = X_test.copy()

        for col in categorical_features:
            le = LabelEncoder()
            # Fit on train
            X_train_processed[col] = le.fit_transform(X_train[col].astype(str))

            # Transform val/test, handle unseen categories
            X_val_processed[col] = X_val[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
            X_test_processed[col] = X_test[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

            self.label_encoders[col] = le

        self.feature_names = X_train_processed.columns.tolist()

        return X_train_processed, X_val_processed, X_test_processed

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model
        """
        print("\n=== Training XGBoost Model ===")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Preprocess
        X_train_processed, X_val_processed, _ = self.preprocess_features(
            X_train, X_val, X_val  # Pass val twice for now
        )

        # Train model
        self.model = xgb.XGBRegressor(**FREIGHT_MODEL_PARAMS)

        print("\nTraining with early stopping...")
        self.model.fit(
            X_train_processed, y_train,
            eval_set=[(X_val_processed, y_val)],
            verbose=50
        )

        print(f"\n✅ Training completed!")

        # Check if early stopping was used
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            print(f"Best iteration: {self.model.best_iteration}")
        else:
            print(f"Total iterations: {self.model.n_estimators}")

        return self.model

    def evaluate(self, X, y, split_name='Test'):
        """
        Evaluate model performance
        """
        print(f"\n=== Evaluating on {split_name} Set ===")

        # Preprocess if needed
        if split_name == 'Test':
            _, _, X_processed = self.preprocess_features(X, X, X)
        else:
            X_processed, _, _ = self.preprocess_features(X, X, X)

        # Predict
        y_pred = self.model.predict(X_processed)

        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100

        # Calculate within-$X accuracy
        within_3 = np.mean(np.abs(y - y_pred) <= 3) * 100
        within_5 = np.mean(np.abs(y - y_pred) <= 5) * 100

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'within_3_dollars': within_3,
            'within_5_dollars': within_5,
            'mean_actual': y.mean(),
            'mean_predicted': y_pred.mean()
        }

        self.metrics[split_name.lower()] = metrics

        # Print results
        print(f"\n📊 {split_name} Set Results:")
        print(f"  MAE:  ${mae:.2f} {'✅' if mae < FREIGHT_TARGET_MAE else '❌'} (target: ${FREIGHT_TARGET_MAE})")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  R²:   {r2:.4f} {'✅' if r2 > FREIGHT_TARGET_R2 else '❌'} (target: {FREIGHT_TARGET_R2})")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Within $3: {within_3:.1f}%")
        print(f"  Within $5: {within_5:.1f}%")
        print(f"  Mean Actual: ${y.mean():.2f}")
        print(f"  Mean Predicted: ${y_pred.mean():.2f}")

        return y_pred, metrics

    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance
        """
        print("\n=== Feature Importance ===")

        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print(feature_importance_df.head(top_n))

        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feature_importance_df.head(top_n),
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.title('Top Feature Importances - Freight Cost Model', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()

        # Save plot
        plot_path = MODELS_DIR / 'freight_feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {plot_path}")
        plt.close()

        return feature_importance_df

    def plot_predictions(self, y_true, y_pred, split_name='Test'):
        """
        Plot predicted vs actual
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Freight Cost ($)', fontsize=12)
        axes[0].set_ylabel('Predicted Freight Cost ($)', fontsize=12)
        axes[0].set_title(f'{split_name} Set: Predicted vs Actual', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Residual plot
        residuals = y_pred - y_true
        axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color='coral')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Freight Cost ($)', fontsize=12)
        axes[1].set_ylabel('Residuals ($)', fontsize=12)
        axes[1].set_title(f'{split_name} Set: Residual Plot', fontsize=13, fontweight='bold')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = MODELS_DIR / f'freight_predictions_{split_name.lower()}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions plot to {plot_path}")
        plt.close()

    def _convert_to_python_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_model(self):
        """
        Save trained model and preprocessor
        """
        print(f"\n=== Saving Model ===")

        # Save model
        joblib.dump(self.model, FREIGHT_MODEL_FILE)
        print(f"✅ Model saved to {FREIGHT_MODEL_FILE}")

        # Save label encoders
        joblib.dump(self.label_encoders, FREIGHT_PREPROCESSOR_FILE)
        print(f"✅ Preprocessor saved to {FREIGHT_PREPROCESSOR_FILE}")

        # Save metrics (convert numpy types to Python types)
        metrics_file = MODELS_DIR / 'freight_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self._convert_to_python_types(self.metrics), f, indent=2)
        print(f"✅ Metrics saved to {metrics_file}")

        # Save feature names
        feature_file = MODELS_DIR / 'freight_features.json'
        with open(feature_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"✅ Feature names saved to {feature_file}")

    def load_model(self):
        """
        Load trained model and preprocessor
        """
        print(f"\n=== Loading Model ===")

        self.model = joblib.load(FREIGHT_MODEL_FILE)
        self.label_encoders = joblib.load(FREIGHT_PREPROCESSOR_FILE)

        # Load feature names
        feature_file = MODELS_DIR / 'freight_features.json'
        with open(feature_file, 'r') as f:
            self.feature_names = json.load(f)

        print(f"✅ Model loaded successfully")


def main():
    """
    Main training pipeline
    """
    print("=" * 80)
    print("FREIGHT COST PREDICTION MODEL - TRAINING PIPELINE")
    print("=" * 80)

    start_time = datetime.now()

    # 1. Load and prepare data
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("=" * 80)

    df = load_and_merge_data(CLEAN_DATA_DIR)
    df = engineer_features(df, mode='train')

    # Split data temporally
    train_df, val_df, test_df = temporal_train_test_split(
        df, TRAIN_TEST_SPLIT_DATE, VAL_TEST_SPLIT_DATE
    )

    # Prepare freight-specific data
    X_train, y_train = prepare_freight_data(train_df, mode='train')
    X_val, y_val = prepare_freight_data(val_df, mode='train')
    X_test, y_test = prepare_freight_data(test_df, mode='train')

    # 2. Train model
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING")
    print("=" * 80)

    trainer = FreightModelTrainer()
    trainer.train(X_train, y_train, X_val, y_val)

    # 3. Evaluate
    print("\n" + "=" * 80)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 80)

    # Evaluate on all splits
    y_train_pred, train_metrics = trainer.evaluate(X_train, y_train, 'Train')
    y_val_pred, val_metrics = trainer.evaluate(X_val, y_val, 'Validation')
    y_test_pred, test_metrics = trainer.evaluate(X_test, y_test, 'Test')

    # 4. Visualizations
    print("\n" + "=" * 80)
    print("STEP 4: VISUALIZATIONS")
    print("=" * 80)

    trainer.plot_feature_importance()
    trainer.plot_predictions(y_test, y_test_pred, 'Test')

    # 5. Save model
    print("\n" + "=" * 80)
    print("STEP 5: SAVING MODEL")
    print("=" * 80)

    trainer.save_model()

    # 6. Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  Total training time: {duration:.1f} seconds")

    print("\n📊 Final Test Set Performance:")
    print(f"  MAE:  ${test_metrics['mae']:.2f}")
    print(f"  RMSE: ${test_metrics['rmse']:.2f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")

    if test_metrics['mae'] < FREIGHT_TARGET_MAE and test_metrics['r2'] > FREIGHT_TARGET_R2:
        print("\n✅ MODEL MEETS SUCCESS CRITERIA!")
    else:
        print("\n⚠️  Model does not meet all success criteria - consider tuning")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
