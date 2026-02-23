"""
Comprehensive Experiment Runner
Trains multiple models with different configurations
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import (
    CLEAN_DATA_DIR, TRAIN_TEST_SPLIT_DATE, VAL_TEST_SPLIT_DATE, RANDOM_STATE
)
from data_preprocessing import load_and_merge_data, temporal_train_test_split
from feature_strategies import get_feature_strategy
from model_factory import ModelFactory, get_experiment_configs
from experiment_tracker import get_tracker


class ExperimentRunner:
    """
    Run systematic ML experiments
    """

    def __init__(self, task: str, experiments_base_dir: str = "experiments"):
        """
        Args:
            task: 'freight' or 'delivery'
            experiments_base_dir: Base directory for experiments
        """
        self.task = task
        self.base_dir = Path(__file__).resolve().parent.parent / experiments_base_dir
        self.task_dir = self.base_dir / task
        self.task_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = get_tracker()
        self.label_encoders = {}

    def load_and_prepare_data(self):
        """Load and split data"""
        print("\n" + "=" * 80)
        print(f"LOADING DATA FOR {self.task.upper()}")
        print("=" * 80)

        df = load_and_merge_data(CLEAN_DATA_DIR)

        # CRITICAL: Use different split strategies for freight vs delivery
        if self.task == 'freight':
            # Freight pricing is stable over time → temporal split is appropriate
            print("Using temporal split (freight pricing is stable over time)")
            train_df, val_df, test_df = temporal_train_test_split(
                df, TRAIN_TEST_SPLIT_DATE, VAL_TEST_SPLIT_DATE
            )

        else:  # delivery
            # Delivery times improved 43% over time (12.8 → 7.3 days)
            # Temporal split causes severe distribution shift → use random split
            print("⚠️  Using random split (marketplace delivery improved 43% over time)")
            print("   Temporal split would cause distribution shift: train=12.8d, test=7.3d")

            from sklearn.model_selection import train_test_split

            # First split: separate test set (15%)
            train_val_df, test_df = train_test_split(
                df, test_size=0.15, random_state=RANDOM_STATE, shuffle=True
            )

            # Second split: separate validation set (15% of remaining)
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.15, random_state=RANDOM_STATE, shuffle=True
            )

        print(f"Train: {len(train_df)} samples")
        print(f"Val:   {len(val_df)} samples")
        print(f"Test:  {len(test_df)} samples")

        return train_df, val_df, test_df

    def prepare_target(self, df, split_name='train'):
        """Prepare target variable based on task"""
        if self.task == 'freight':
            # Filter valid freight orders
            df = df[df['freight_value'].notna()].copy()
            df = df[df['freight_value'] > 0].copy()
            target = df['freight_value']
            print(f"\n{split_name.capitalize()} freight target: ${target.mean():.2f} mean, ${target.std():.2f} std")

        elif self.task == 'delivery':
            # Filter delivered orders only
            df = df[df['order_status'] == 'delivered'].copy()
            df['delivery_days'] = (
                df['order_delivered_customer_date'] - df['order_purchase_timestamp']
            ).dt.days
            df = df[df['delivery_days'].notna()].copy()
            df = df[(df['delivery_days'] > 0) & (df['delivery_days'] < 100)].copy()
            target = df['delivery_days']
            print(f"\n{split_name.capitalize()} delivery target: {target.mean():.1f} mean, {target.std():.1f} std")

        else:
            raise ValueError(f"Unknown task: {self.task}")

        return df, target

    def preprocess_features(self, X_train, X_val, X_test, fit=True):
        """Encode categorical features and handle NaN values"""
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        X_train_proc = X_train.copy()
        X_val_proc = X_val.copy()
        X_test_proc = X_test.copy()

        # Fill NaN in numeric features (for sklearn models that don't handle NaN)
        for col in numeric_features:
            median_val = X_train[col].median() if fit else self.numeric_medians.get(col, 0)
            if fit:
                if not hasattr(self, 'numeric_medians'):
                    self.numeric_medians = {}
                self.numeric_medians[col] = median_val

            X_train_proc[col] = X_train[col].fillna(median_val)
            X_val_proc[col] = X_val[col].fillna(median_val)
            X_test_proc[col] = X_test[col].fillna(median_val)

        # Encode categorical features
        if fit:
            self.label_encoders = {}
            for col in categorical_features:
                le = LabelEncoder()
                X_train_proc[col] = le.fit_transform(X_train[col].astype(str))
                self.label_encoders[col] = le
        else:
            for col in categorical_features:
                X_train_proc[col] = self.label_encoders[col].transform(X_train[col].astype(str))

        # Transform val/test with unknown handling
        for col in categorical_features:
            le = self.label_encoders[col]
            X_val_proc[col] = X_val[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
            X_test_proc[col] = X_test[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

        return X_train_proc, X_val_proc, X_test_proc

    def evaluate(self, model, X, y):
        """Evaluate model performance"""
        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100

        if self.task == 'freight':
            within_3 = np.mean(np.abs(y - y_pred) <= 3) * 100
            within_5 = np.mean(np.abs(y - y_pred) <= 5) * 100
            extra_metrics = {
                'within_3_dollars': float(within_3),
                'within_5_dollars': float(within_5)
            }
        else:  # delivery
            within_1 = np.mean(np.abs(y - y_pred) <= 1) * 100
            within_3 = np.mean(np.abs(y - y_pred) <= 3) * 100
            within_7 = np.mean(np.abs(y - y_pred) <= 7) * 100
            extra_metrics = {
                'within_1_day': float(within_1),
                'within_3_days': float(within_3),
                'within_7_days': float(within_7)
            }

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            **extra_metrics
        }

        return metrics, y_pred

    def run_single_experiment(self, config: dict, train_df, val_df, test_df):
        """Run a single experiment with given configuration"""
        exp_name = config['name']
        model_name = config['model']
        feature_strategy_name = config['features']

        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {exp_name}")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Features: {feature_strategy_name}")
        print(f"Description: {config['description']}")

        start_time = datetime.now()

        # Engineer features
        print("\n--- Feature Engineering ---")
        feature_strategy = get_feature_strategy(feature_strategy_name)

        train_df_feat = feature_strategy.engineer(train_df)
        val_df_feat = feature_strategy.engineer(val_df)
        test_df_feat = feature_strategy.engineer(test_df)

        # Prepare targets
        train_df_feat, y_train = self.prepare_target(train_df_feat, 'train')
        val_df_feat, y_val = self.prepare_target(val_df_feat, 'val')
        test_df_feat, y_test = self.prepare_target(test_df_feat, 'test')

        # Select features
        feature_cols = feature_strategy.get_feature_names()
        X_train = train_df_feat[feature_cols].copy()
        X_val = val_df_feat[feature_cols].copy()
        X_test = test_df_feat[feature_cols].copy()

        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Training samples: {len(X_train)}")

        # Preprocess
        print("\n--- Preprocessing ---")
        X_train_proc, X_val_proc, X_test_proc = self.preprocess_features(
            X_train, X_val, X_test, fit=True
        )

        # Train model
        print("\n--- Training Model ---")
        model = ModelFactory.create_model(model_name, RANDOM_STATE)

        # Check if model is a pipeline (for linear models with scaling)
        model_type = ModelFactory.get_model_type(model_name)
        if model_type == 'linear':
            print(f"Training {model_name} with StandardScaler...")
        else:
            print(f"Training {model_name}...")

        model.fit(X_train_proc, y_train)
        print("✅ Training complete")

        # Evaluate
        print("\n--- Evaluation ---")
        train_metrics, _ = self.evaluate(model, X_train_proc, y_train)
        val_metrics, _ = self.evaluate(model, X_val_proc, y_val)
        test_metrics, _ = self.evaluate(model, X_test_proc, y_test)

        print(f"\n📊 Test Results:")
        print(f"  MAE:  {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")

        if self.task == 'freight':
            print(f"  Within $5: {test_metrics['within_5_dollars']:.1f}%")
        else:
            print(f"  Within 3 days: {test_metrics['within_3_days']:.1f}%")

        # Save experiment
        print("\n--- Saving Experiment ---")
        exp_dir = self.task_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = exp_dir / f"{self.task}_model.pkl"
        joblib.dump(model, model_path)

        # Save preprocessor
        preprocessor_path = exp_dir / f"{self.task}_preprocessor.pkl"
        joblib.dump(self.label_encoders, preprocessor_path)

        # Log to tracker
        duration = (datetime.now() - start_time).total_seconds()

        self.tracker.log_experiment(
            model_type=self.task,
            experiment_name=exp_name,
            metrics=test_metrics,
            hyperparameters={
                'model': model_name,
                'features': feature_strategy_name
            },
            features=feature_cols,
            notes=f"{config['description']} | Training time: {duration:.1f}s"
        )

        print(f"✅ Experiment saved to: {exp_dir}")
        print(f"⏱️  Time: {duration:.1f}s")

        return test_metrics

    def run_all_experiments(self):
        """Run all configured experiments for this task"""
        print("\n" + "=" * 80)
        print(f"RUNNING ALL {self.task.upper()} EXPERIMENTS")
        print("=" * 80)

        # Load data once
        train_df, val_df, test_df = self.load_and_prepare_data()

        # Get experiment configurations
        configs = get_experiment_configs(self.task)

        print(f"\nTotal experiments to run: {len(configs)}")
        for i, config in enumerate(configs, 1):
            print(f"  {i}. {config['name']}: {config['description']}")

        # Run each experiment
        all_results = []
        for i, config in enumerate(configs, 1):
            print(f"\n\n{'=' * 80}")
            print(f"EXPERIMENT {i}/{len(configs)}")
            print(f"{'=' * 80}")

            try:
                test_metrics = self.run_single_experiment(config, train_df, val_df, test_df)
                all_results.append({
                    'name': config['name'],
                    'metrics': test_metrics
                })
            except Exception as e:
                print(f"\n❌ Experiment {config['name']} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Summary
        print("\n\n" + "=" * 80)
        print(f"{self.task.upper()} EXPERIMENTS COMPLETE")
        print("=" * 80)

        if all_results:
            print("\n📊 SUMMARY:")
            print(f"{'Experiment':<30} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
            print("-" * 80)

            for result in all_results:
                metrics = result['metrics']
                print(f"{result['name']:<30} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f} {metrics['r2']:<12.4f}")

            # Find best
            best_mae = min(all_results, key=lambda x: x['metrics']['mae'])
            best_r2 = max(all_results, key=lambda x: x['metrics']['r2'])

            print("\n🏆 BEST MODELS:")
            print(f"  Best MAE: {best_mae['name']} ({best_mae['metrics']['mae']:.4f})")
            print(f"  Best R²:  {best_r2['name']} ({best_r2['metrics']['r2']:.4f})")

        print("\n" + "=" * 80)
        print("To compare all experiments:")
        print(f"  python src/experiment_tracker.py")
        print("=" * 80)


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_experiments.py [freight|delivery|both]")
        sys.exit(1)

    task = sys.argv[1].lower()

    if task == 'both':
        print("\n🚀 Running experiments for BOTH freight and delivery")
        print("=" * 80)

        # Run freight experiments
        freight_runner = ExperimentRunner('freight')
        freight_runner.run_all_experiments()

        print("\n\n")

        # Run delivery experiments
        delivery_runner = ExperimentRunner('delivery')
        delivery_runner.run_all_experiments()

    elif task in ['freight', 'delivery']:
        runner = ExperimentRunner(task)
        runner.run_all_experiments()

    else:
        print(f"❌ Unknown task: {task}")
        print("Available: freight, delivery, both")
        sys.exit(1)


if __name__ == "__main__":
    main()
