"""
Quick script to compare all experiment results
"""
from experiment_tracker import get_tracker
import sys


def main():
    tracker = get_tracker()

    print("\n" + "=" * 100)
    print("ML EXPERIMENTATION RESULTS")
    print("=" * 100)

    # Freight comparison
    print("\n")
    tracker.compare_experiments("freight", metric_keys=['mae', 'rmse', 'r2', 'within_5_dollars'])

    print("\n\n")

    # Delivery comparison
    tracker.compare_experiments("delivery", metric_keys=['mae', 'rmse', 'r2', 'within_3_days'])

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()