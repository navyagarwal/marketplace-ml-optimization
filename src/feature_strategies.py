"""
Different feature engineering strategies for experimentation
"""
import pandas as pd
import numpy as np
from data_preprocessing import haversine_distance


class BaselineFeatures:
    """
    Baseline: Simple, essential features only
    """
    name = "baseline"

    @staticmethod
    def engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Distance
        df['distance_km'] = haversine_distance(
            df['latitude'], df['longitude'],
            df['latitude_seller'], df['longitude_seller']
        )

        # Product dimensions
        df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].median())
        df['product_volume_cm3'] = (
            df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
        ).fillna(0)
        df['product_category_name'] = df['product_category_name'].fillna('unknown')

        # Temporal
        df['order_month'] = df['order_purchase_timestamp'].dt.month
        df['is_weekend'] = df['order_purchase_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)

        return df

    @staticmethod
    def get_feature_names() -> list:
        return [
            'distance_km', 'product_weight_g', 'product_volume_cm3',
            'product_category_name', 'price', 'order_month', 'is_weekend'
        ]


class ImprovedFeatures:
    """
    Improved: Add interaction terms and domain knowledge
    """
    name = "improved"

    @staticmethod
    def engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = BaselineFeatures.engineer(df)

        # Volumetric weight (what freight companies use)
        df['volumetric_weight'] = df['product_volume_cm3'] / 5000
        df['chargeable_weight'] = np.maximum(df['product_weight_g'] / 1000, df['volumetric_weight'])

        # Distance buckets
        df['distance_bucket'] = pd.cut(
            df['distance_km'],
            bins=[0, 100, 300, 500, 1000, 10000],
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        ).astype(str)

        # Interactions
        df['distance_x_weight'] = df['distance_km'] * df['product_weight_g']
        df['distance_x_volume'] = df['distance_km'] * df['product_volume_cm3']
        df['price_x_distance'] = df['price'] * df['distance_km']

        # Efficiency metrics
        df['weight_per_km'] = df['product_weight_g'] / (df['distance_km'] + 1)
        df['volume_per_km'] = df['product_volume_cm3'] / (df['distance_km'] + 1)

        return df

    @staticmethod
    def get_feature_names() -> list:
        base = BaselineFeatures.get_feature_names()
        return base + [
            'volumetric_weight', 'chargeable_weight', 'distance_bucket',
            'distance_x_weight', 'distance_x_volume', 'price_x_distance',
            'weight_per_km', 'volume_per_km'
        ]


class AdvancedFeatures:
    """
    Advanced: All features including bucketing, density, temporal patterns
    """
    name = "advanced"

    @staticmethod
    def engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = ImprovedFeatures.engineer(df)

        # Weight/volume categories
        df['weight_bucket'] = pd.cut(
            df['product_weight_g'],
            bins=[0, 500, 1000, 5000, 1000000],
            labels=['very_light', 'light', 'medium', 'heavy']
        ).astype(str)

        df['volume_bucket'] = pd.cut(
            df['product_volume_cm3'],
            bins=[0, 1000, 5000, 10000, 1000000],
            labels=['very_small', 'small', 'medium', 'large']
        ).astype(str)

        # Density proxy
        df['density_proxy'] = df['product_weight_g'] / (df['product_volume_cm3'] + 1)

        # Price indicators
        df['is_high_value'] = (df['price'] > df['price'].quantile(0.75)).astype(int)
        df['price_per_kg'] = df['price'] / (df['product_weight_g'] / 1000 + 0.001)

        # Temporal patterns
        df['order_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
        df['order_hour'] = df['order_purchase_timestamp'].dt.hour
        df['is_holiday_season'] = df['order_month'].isin([11, 12]).astype(int)

        # Log transformations
        df['distance_log'] = np.log1p(df['distance_km'])
        df['weight_log'] = np.log1p(df['product_weight_g'])
        df['volume_log'] = np.log1p(df['product_volume_cm3'])

        # Category-distance interaction
        df['category_distance'] = df['product_category_name'] + '_' + df['distance_bucket']

        return df

    @staticmethod
    def get_feature_names() -> list:
        base = ImprovedFeatures.get_feature_names()
        return base + [
            'weight_bucket', 'volume_bucket', 'density_proxy',
            'is_high_value', 'price_per_kg',
            'order_day_of_week', 'order_hour', 'is_holiday_season',
            'distance_log', 'weight_log', 'volume_log', 'category_distance'
        ]


class MinimalFeatures:
    """
    Minimal: Only the most important features (for fast linear models)
    """
    name = "minimal"

    @staticmethod
    def engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['distance_km'] = haversine_distance(
            df['latitude'], df['longitude'],
            df['latitude_seller'], df['longitude_seller']
        )

        df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].median())
        df['product_volume_cm3'] = (
            df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
        ).fillna(0)

        return df

    @staticmethod
    def get_feature_names() -> list:
        return ['distance_km', 'product_weight_g', 'product_volume_cm3', 'price']


# Feature strategy registry
FEATURE_STRATEGIES = {
    'baseline': BaselineFeatures,
    'minimal': MinimalFeatures,
    'improved': ImprovedFeatures,
    'advanced': AdvancedFeatures
}


def get_feature_strategy(name: str):
    """Get feature engineering strategy by name"""
    if name not in FEATURE_STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(FEATURE_STRATEGIES.keys())}")
    return FEATURE_STRATEGIES[name]
