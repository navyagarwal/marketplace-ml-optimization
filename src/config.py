"""
Configuration file for ML pipeline and API
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DATA_DIR = DATA_DIR / "clean_data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data files
ORDERS_MASTER_FILE = CLEAN_DATA_DIR / "orders_master.csv"
ORDER_ITEMS_FILE = CLEAN_DATA_DIR / "order_items.csv"
CUSTOMERS_FILE = CLEAN_DATA_DIR / "customers.csv"
SELLERS_FILE = CLEAN_DATA_DIR / "sellers.csv"
PRODUCTS_FILE = CLEAN_DATA_DIR / "products.csv"

# Model files
FREIGHT_MODEL_FILE = MODELS_DIR / "freight_model.pkl"
DELIVERY_MODEL_FILE = MODELS_DIR / "delivery_model.pkl"
FREIGHT_PREPROCESSOR_FILE = MODELS_DIR / "freight_preprocessor.pkl"
DELIVERY_PREPROCESSOR_FILE = MODELS_DIR / "delivery_preprocessor.pkl"

# Model hyperparameters
FREIGHT_MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 20
}

DELIVERY_MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': 10,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 20
}

# Training parameters
TRAIN_TEST_SPLIT_DATE = "2018-06-01"
VAL_TEST_SPLIT_DATE = "2018-08-01"
RANDOM_STATE = 42

# Success metrics targets
FREIGHT_TARGET_MAE = 3.0  # dollars
FREIGHT_TARGET_R2 = 0.70
DELIVERY_TARGET_MAE = 2.0  # days
DELIVERY_TARGET_R2 = 0.65
DELIVERY_TARGET_WITHIN_3_DAYS = 0.80

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 4))
