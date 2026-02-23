"""
Experiment tracking and model versioning system
"""
import json
import os
from datetime import datetime
from pathlib import Path
import shutil


class ExperimentTracker:
    """
    Track ML experiments with versioning and comparison
    """

    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(__file__).resolve().parent.parent / base_dir
        self.base_dir.mkdir(exist_ok=True)

        self.experiments_file = self.base_dir / "experiments.json"
        self.experiments = self._load_experiments()

    def _load_experiments(self):
        """Load existing experiments"""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        return {"freight": [], "delivery": []}

    def _save_experiments(self):
        """Save experiments to disk"""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def log_experiment(
        self,
        model_type: str,  # "freight" or "delivery"
        experiment_name: str,
        metrics: dict,
        hyperparameters: dict,
        features: list,
        notes: str = ""
    ):
        """
        Log or update an experiment

        Args:
            model_type: "freight" or "delivery"
            experiment_name: Descriptive name (e.g., "baseline", "tuned_v1")
            metrics: Dict of test metrics
            hyperparameters: Model hyperparameters used
            features: List of feature names
            notes: Any additional notes
        """
        experiment = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "features": features,
            "num_features": len(features),
            "notes": notes
        }

        # Check if experiment already exists (by name)
        existing_idx = None
        for idx, exp in enumerate(self.experiments[model_type]):
            if exp['experiment_name'] == experiment_name:
                existing_idx = idx
                break

        if existing_idx is not None:
            # Update existing experiment
            self.experiments[model_type][existing_idx] = experiment
            action = "Updated"
        else:
            # Add new experiment
            self.experiments[model_type].append(experiment)
            action = "Logged"

        self._save_experiments()

        # Create experiment directory
        exp_dir = self.base_dir / model_type / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment details
        with open(exp_dir / "experiment.json", 'w') as f:
            json.dump(experiment, f, indent=2)

        print(f"\n✅ {action} experiment: {experiment_name}")
        print(f"   Location: {exp_dir}")

        return experiment_name, exp_dir

    def get_best_experiment(self, model_type: str, metric: str = "mae", minimize=True):
        """
        Get best experiment based on a metric

        Args:
            model_type: "freight" or "delivery"
            metric: Metric name (e.g., "mae", "r2")
            minimize: True if lower is better, False if higher is better
        """
        if not self.experiments[model_type]:
            return None

        experiments = self.experiments[model_type]

        # Sort by metric
        sorted_exps = sorted(
            experiments,
            key=lambda x: x['metrics'].get(metric, float('inf') if minimize else float('-inf')),
            reverse=not minimize
        )

        return sorted_exps[0] if sorted_exps else None

    def compare_experiments(self, model_type: str, metric_keys=None):
        """
        Compare all experiments for a model type

        Args:
            model_type: "freight" or "delivery"
            metric_keys: List of metric keys to compare (default: all)
        """
        if not self.experiments[model_type]:
            print(f"No experiments found for {model_type}")
            return

        print(f"\n{'=' * 100}")
        print(f"EXPERIMENT COMPARISON: {model_type.upper()}")
        print(f"{'=' * 100}\n")

        experiments = self.experiments[model_type]

        # Determine metrics to show
        if metric_keys is None:
            # Get all unique metric keys
            metric_keys = set()
            for exp in experiments:
                metric_keys.update(exp['metrics'].keys())
            metric_keys = sorted(list(metric_keys))

        # Print header
        print(f"{'Experiment':<30} {'Date':<12} {'Features':<10}", end="")
        for key in metric_keys:
            print(f" {key:<12}", end="")
        print()
        print("-" * 100)

        # Print experiments (sorted by experiment name for consistency)
        for exp in sorted(experiments, key=lambda x: x['experiment_name']):
            name = exp['experiment_name'][:28]
            date = exp['timestamp'][:10]
            num_features = exp['num_features']

            print(f"{name:<30} {date:<12} {num_features:<10}", end="")

            for key in metric_keys:
                value = exp['metrics'].get(key, 'N/A')
                if isinstance(value, float):
                    print(f" {value:<12.4f}", end="")
                else:
                    print(f" {str(value):<12}", end="")
            print()

        # Show best experiments
        print("\n" + "=" * 100)
        print("BEST EXPERIMENTS")
        print("=" * 100)

        for metric in ['mae', 'rmse', 'r2']:
            if metric in metric_keys:
                minimize = metric != 'r2'
                best = self.get_best_experiment(model_type, metric, minimize)
                if best:
                    print(f"\nBest {metric.upper()}: {best['experiment_name']}")
                    print(f"  Value: {best['metrics'].get(metric, 'N/A')}")
                    print(f"  Date: {best['timestamp'][:10]}")

    def export_best_model(self, model_type: str, metric: str = "mae"):
        """
        Export best model to main models directory

        Args:
            model_type: "freight" or "delivery"
            metric: Metric to use for selection
        """
        best = self.get_best_experiment(model_type, metric, minimize=(metric != 'r2'))

        if not best:
            print(f"No experiments found for {model_type}")
            return

        print(f"\n✅ Best experiment: {best['experiment_name']}")
        print(f"   Metric ({metric}): {best['metrics'].get(metric)}")

        # Copy to main models directory
        exp_dir = self.base_dir / model_type / best['experiment_name']
        main_models_dir = Path(__file__).resolve().parent.parent / "models"

        model_file = exp_dir / f"{model_type}_model.pkl"
        preprocessor_file = exp_dir / f"{model_type}_preprocessor.pkl"

        if model_file.exists() and preprocessor_file.exists():
            shutil.copy(model_file, main_models_dir / f"{model_type}_model.pkl")
            shutil.copy(preprocessor_file, main_models_dir / f"{model_type}_preprocessor.pkl")
            print(f"   Exported to: {main_models_dir}")
        else:
            print(f"   Warning: Model files not found in {exp_dir}")


# Convenience function
def get_tracker():
    """Get global experiment tracker instance"""
    return ExperimentTracker()


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker()

    # Compare all freight experiments
    tracker.compare_experiments("freight")

    print("\n" + "=" * 100)

    # Compare all delivery experiments
    tracker.compare_experiments("delivery")
