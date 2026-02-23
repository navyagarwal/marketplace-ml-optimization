"""
Data preprocessing and feature engineering for both models
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate haversine distance between two points in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Earth radius in kilometers

    return km


def load_and_merge_data(data_dir) -> pd.DataFrame:
    """
    Load all datasets and create master dataset with all features
    """
    print("Loading datasets...")

    # Load all files
    orders = pd.read_csv(data_dir / "orders.csv")
    order_items = pd.read_csv(data_dir / "order_items.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    sellers = pd.read_csv(data_dir / "sellers.csv")
    products = pd.read_csv(data_dir / "products.csv")

    # Convert date columns
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])
    orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

    print(f"Orders: {len(orders)}")
    print(f"Order items: {len(order_items)}")
    print(f"Customers: {len(customers)}")
    print(f"Sellers: {len(sellers)}")
    print(f"Products: {len(products)}")

    # Merge datasets
    print("\nMerging datasets...")
    df = order_items.copy()

    # Merge with orders
    df = df.merge(orders, on='order_id', how='left')

    # Merge with customers
    df = df.merge(customers, on='customer_id', how='left', suffixes=('', '_customer'))

    # Merge with sellers
    df = df.merge(sellers, on='seller_id', how='left', suffixes=('', '_seller'))

    # Merge with products
    df = df.merge(products, on='product_id', how='left')

    print(f"Merged dataset: {len(df)} rows")

    return df


def engineer_features(df: pd.DataFrame, mode='train') -> pd.DataFrame:
    """
    Engineer features for both freight and delivery models

    Args:
        df: Merged dataframe
        mode: 'train' or 'inference' (affects which features to calculate)

    Returns:
        DataFrame with engineered features
    """
    print("\nEngineering features...")

    df = df.copy()

    # ========== GEOGRAPHIC FEATURES ==========
    print("  - Geographic features...")
    df['distance_km'] = haversine_distance(
        df['latitude'], df['longitude'],
        df['latitude_seller'], df['longitude_seller']
    )

    # Distance buckets
    df['distance_bucket'] = pd.cut(
        df['distance_km'],
        bins=[0, 100, 300, 500, 1000, 10000],
        labels=['0-100km', '100-300km', '300-500km', '500-1000km', '1000+km']
    ).astype(str)

    # ========== PRODUCT FEATURES ==========
    print("  - Product features...")

    # Volume calculation
    df['product_volume_cm3'] = (
        df['product_length_cm'] *
        df['product_height_cm'] *
        df['product_width_cm']
    )

    # Fill missing values with median
    df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].median())
    df['product_volume_cm3'] = df['product_volume_cm3'].fillna(df['product_volume_cm3'].median())

    # Weight and volume interactions
    df['weight_per_km'] = df['product_weight_g'] / (df['distance_km'] + 1)
    df['volume_per_km'] = df['product_volume_cm3'] / (df['distance_km'] + 1)

    # Price interactions
    df['price_x_distance'] = df['price'] * df['distance_km']
    df['price_per_kg'] = df['price'] / (df['product_weight_g'] / 1000 + 0.001)

    # ========== CATEGORY FEATURES ==========
    print("  - Category features...")
    df['product_category_name'] = df['product_category_name'].fillna('unknown')

    # ========== TEMPORAL FEATURES ==========
    print("  - Temporal features...")
    df['order_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
    df['order_hour'] = df['order_purchase_timestamp'].dt.hour
    df['order_month'] = df['order_purchase_timestamp'].dt.month
    df['is_weekend'] = df['order_day_of_week'].isin([5, 6]).astype(int)

    # ========== DELIVERY-SPECIFIC FEATURES (for delivery model) ==========
    if mode == 'train':
        print("  - Delivery target features...")

        # Calculate delivery days (target for delivery model)
        df['delivery_days'] = (
            df['order_delivered_customer_date'] -
            df['order_purchase_timestamp']
        ).dt.days

        # Seller delay
        df['seller_delay_days'] = (
            df['order_delivered_carrier_date'] -
            df['order_purchase_timestamp']
        ).dt.days

        # Logistics delay
        df['logistics_delay_days'] = (
            df['order_delivered_customer_date'] -
            df['order_delivered_carrier_date']
        ).dt.days

        # Promised delivery window
        df['promised_delivery_days'] = (
            df['order_estimated_delivery_date'] -
            df['order_purchase_timestamp']
        ).dt.days

    print(f"Features engineered: {len(df.columns)} columns")

    return df


def prepare_freight_data(df: pd.DataFrame, mode='train') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data specifically for freight cost prediction

    Returns:
        X: Feature dataframe
        y: Target (freight_value) or None for inference
    """
    print("\n=== Preparing Freight Model Data ===")

    # Select features for freight model
    feature_columns = [
        'distance_km',
        'product_weight_g',
        'product_volume_cm3',
        'product_category_name',
        'price',
        'weight_per_km',
        'volume_per_km',
        'price_x_distance',
        'price_per_kg',
        'order_month',
        'is_weekend'
    ]

    # Filter valid rows
    if mode == 'train':
        df = df[df['freight_value'].notna()].copy()
        df = df[df['freight_value'] > 0].copy()  # Remove zero freight
        df = df[df['distance_km'] > 0].copy()  # Remove zero distance

    X = df[feature_columns].copy()

    # Handle missing values
    X = X.fillna({
        'product_weight_g': X['product_weight_g'].median(),
        'product_volume_cm3': X['product_volume_cm3'].median(),
        'product_category_name': 'unknown',
        'price': X['price'].median()
    })

    if mode == 'train':
        y = df['freight_value'].copy()
        print(f"Training samples: {len(X)}")
        print(f"Target range: ${y.min():.2f} - ${y.max():.2f}")
        print(f"Target mean: ${y.mean():.2f}")
        return X, y
    else:
        print(f"Inference samples: {len(X)}")
        return X, None


def prepare_delivery_data(df: pd.DataFrame, mode='train') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data specifically for delivery date prediction

    Returns:
        X: Feature dataframe
        y: Target (delivery_days) or None for inference
    """
    print("\n=== Preparing Delivery Model Data ===")

    # Select features for delivery model
    feature_columns = [
        'distance_km',
        'product_weight_g',
        'product_volume_cm3',
        'product_category_name',
        'price',
        'order_day_of_week',
        'order_hour',
        'order_month',
        'is_weekend',
        'weight_per_km',
        'volume_per_km'
    ]

    # Filter valid rows (only delivered orders for training)
    if mode == 'train':
        df = df[df['order_status'] == 'delivered'].copy()
        df = df[df['delivery_days'].notna()].copy()
        df = df[df['delivery_days'] > 0].copy()  # Remove invalid delivery times
        df = df[df['delivery_days'] < 100].copy()  # Remove outliers (> 100 days)

    X = df[feature_columns].copy()

    # Handle missing values
    X = X.fillna({
        'product_weight_g': X['product_weight_g'].median(),
        'product_volume_cm3': X['product_volume_cm3'].median(),
        'product_category_name': 'unknown',
        'price': X['price'].median()
    })

    if mode == 'train':
        y = df['delivery_days'].copy()
        print(f"Training samples: {len(X)}")
        print(f"Target range: {y.min():.0f} - {y.max():.0f} days")
        print(f"Target mean: {y.mean():.1f} days")
        return X, y
    else:
        print(f"Inference samples: {len(X)}")
        return X, None


def temporal_train_test_split(
    df: pd.DataFrame,
    train_date: str,
    val_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (proper way for time-series data)

    Args:
        df: Full dataset with order_purchase_timestamp
        train_date: Date to split train/val (format: 'YYYY-MM-DD')
        val_date: Date to split val/test (format: 'YYYY-MM-DD')

    Returns:
        train, val, test dataframes
    """
    train = df[df['order_purchase_timestamp'] < train_date].copy()
    val = df[
        (df['order_purchase_timestamp'] >= train_date) &
        (df['order_purchase_timestamp'] < val_date)
    ].copy()
    test = df[df['order_purchase_timestamp'] >= val_date].copy()

    print(f"\nTemporal Split:")
    print(f"  Train: {len(train)} samples (< {train_date})")
    print(f"  Val:   {len(val)} samples ({train_date} to {val_date})")
    print(f"  Test:  {len(test)} samples (>= {val_date})")

    return train, val, test


if __name__ == "__main__":
    # Test the preprocessing pipeline
    from config import CLEAN_DATA_DIR, TRAIN_TEST_SPLIT_DATE, VAL_TEST_SPLIT_DATE

    print("Testing data preprocessing pipeline...\n")

    # Load and merge
    df = load_and_merge_data(CLEAN_DATA_DIR)

    # Engineer features
    df = engineer_features(df, mode='train')

    # Split data
    train, val, test = temporal_train_test_split(
        df, TRAIN_TEST_SPLIT_DATE, VAL_TEST_SPLIT_DATE
    )

    # Test freight data preparation
    X_freight, y_freight = prepare_freight_data(train, mode='train')
    print(f"\nFreight model features: {X_freight.shape}")

    # Test delivery data preparation
    X_delivery, y_delivery = prepare_delivery_data(train, mode='train')
    print(f"Delivery model features: {X_delivery.shape}")

    print("\n✅ Preprocessing pipeline test completed!")
