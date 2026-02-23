"""
FastAPI application for serving ML models
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from config import (
    FREIGHT_MODEL_FILE, DELIVERY_MODEL_FILE,
    FREIGHT_PREPROCESSOR_FILE, DELIVERY_PREPROCESSOR_FILE
)
from feature_strategies import get_feature_strategy
from data_preprocessing import haversine_distance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Marketplace ML Optimization API",
    description="API for predicting freight costs and delivery dates",
    version="1.0.0",
    docs_url=None,  # Disable /docs
    redoc_url=None  # Disable /redoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class FreightPredictionRequest(BaseModel):
    """Request schema for freight cost prediction"""
    # distance_km: Optional[float] = Field(None, description="Distance between customer and seller in km (calculated from lat/lon if not provided)", gt=0)
    latitude: float = Field(None, description="Customer latitude", ge=-90, le=90)
    longitude: float = Field(None, description="Customer longitude", ge=-180, le=180)
    latitude_seller: float = Field(None, description="Seller latitude", ge=-90, le=90)
    longitude_seller: float = Field(None, description="Seller longitude", ge=-180, le=180)
    product_weight_g: float = Field(..., description="Product weight in grams", gt=0)
    product_volume_cm3: float = Field(..., description="Product volume in cm³", gt=0)
    product_category_name: str = Field(..., description="Product category name")
    price: float = Field(..., description="Product price in dollars", gt=0)
    order_date: Optional[str] = Field(None, description="Order date (YYYY-MM-DD), defaults to today")
    # order_month: Optional[int] = Field(None, description="Order month (1-12), derived from order_date if not provided", ge=1, le=12)
    # is_weekend: Optional[int] = Field(None, description="Is order on weekend (0 or 1), derived from order_date if not provided", ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "latitude": -23.5505,
                "longitude": -46.6333,
                "latitude_seller": -22.9068,
                "longitude_seller": -43.1729,
                "product_weight_g": 500,
                "product_volume_cm3": 1000,
                "product_category_name": "health_beauty",
                "price": 50.0
            }
        }


class DeliveryPredictionRequest(BaseModel):
    """Request schema for delivery date prediction"""
    # distance_km: Optional[float] = Field(None, description="Distance between customer and seller in km (calculated from lat/lon if not provided)", gt=0)
    latitude: float = Field(None, description="Customer latitude", ge=-90, le=90)
    longitude: float = Field(None, description="Customer longitude", ge=-180, le=180)
    latitude_seller: float = Field(None, description="Seller latitude", ge=-90, le=90)
    longitude_seller: float = Field(None, description="Seller longitude", ge=-180, le=180)
    product_weight_g: float = Field(..., description="Product weight in grams", gt=0)
    product_volume_cm3: float = Field(..., description="Product volume in cm³", gt=0)
    product_category_name: str = Field(..., description="Product category name")
    price: float = Field(..., description="Product price in dollars", gt=0)
    order_date: Optional[str] = Field(None, description="Order date (YYYY-MM-DD), defaults to today")
    # order_day_of_week: Optional[int] = Field(None, description="Day of week (0=Mon, 6=Sun), derived from order_date if not provided", ge=0, le=6)
    # order_hour: Optional[int] = Field(None, description="Hour of day (0-23), defaults to 12", ge=0, le=23)
    # order_month: Optional[int] = Field(None, description="Order month (1-12), derived from order_date if not provided", ge=1, le=12)
    # is_weekend: Optional[int] = Field(None, description="Is order on weekend (0 or 1), derived from order_date if not provided", ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "latitude": -23.5505,
                "longitude": -46.6333,
                "latitude_seller": -22.9068,
                "longitude_seller": -43.1729,
                "product_weight_g": 500,
                "product_volume_cm3": 1000,
                "product_category_name": "health_beauty",
                "price": 50.0
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: float = Field(..., description="Predicted value")
    prediction_formatted: str = Field(..., description="Formatted prediction for display")
    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str


# ============================================================================
# Global Model Storage
# ============================================================================

class ModelManager:
    """Manages loading and caching of ML models"""

    def __init__(self):
        self.freight_model = None
        self.freight_preprocessor = None
        self.delivery_model = None
        self.delivery_preprocessor = None
        self.models_loaded = False

        # Feature strategies for each model
        self.freight_strategy = get_feature_strategy('improved')  # random_forest_improved
        self.delivery_strategy = get_feature_strategy('advanced')  # xgboost_deep

    def load_models(self):
        """Load all models into memory"""
        try:
            logger.info("Loading models...")

            # Load freight model
            self.freight_model = joblib.load(FREIGHT_MODEL_FILE)
            self.freight_preprocessor = joblib.load(FREIGHT_PREPROCESSOR_FILE)
            logger.info("✅ Freight model loaded")

            # Load delivery model
            self.delivery_model = joblib.load(DELIVERY_MODEL_FILE)
            self.delivery_preprocessor = joblib.load(DELIVERY_PREPROCESSOR_FILE)
            logger.info("✅ Delivery model loaded")

            self.models_loaded = True
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def preprocess_freight_features(self, data: dict) -> pd.DataFrame:
        """Preprocess features for freight model (random_forest_improved with 'improved' features)"""
        # Create DataFrame from single sample
        df = pd.DataFrame([data])

        # Handle date parameters: derive from order_date or use today
        if data.get('order_date'):
            order_dt = pd.to_datetime(data['order_date'])
        else:
            order_dt = pd.Timestamp.now()

        # Derive month and is_weekend if not provided
        if data.get('order_month') is None:
            df['order_month'] = order_dt.month
        if data.get('is_weekend') is None:
            df['is_weekend'] = 1 if order_dt.dayofweek in [5, 6] else 0

        # Calculate distance from lat/lon coordinates
        df['distance_km'] = haversine_distance(
            df['latitude'], df['longitude'],
            df['latitude_seller'], df['longitude_seller']
        )

        # Add product dimensions (use volume to back-calculate cube dimensions)
        cube_side = np.cbrt(df['product_volume_cm3'].values[0])
        df['product_length_cm'] = cube_side
        df['product_height_cm'] = cube_side
        df['product_width_cm'] = cube_side

        # Add order_purchase_timestamp for feature engineering
        df['order_purchase_timestamp'] = order_dt

        # Apply feature engineering strategy
        df = self.freight_strategy.engineer(df)

        # Get expected feature columns
        feature_columns = self.freight_strategy.get_feature_names()

        # Select only required features
        X = df[feature_columns].copy()

        # Handle numeric features - fill NaN
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_features:
            X[col] = X[col].fillna(0)

        # Encode categorical features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_features:
            if col in self.freight_preprocessor:
                encoder = self.freight_preprocessor[col]
                try:
                    X[col] = encoder.transform(X[col].astype(str))
                except ValueError:
                    X[col] = -1

        return X

    def preprocess_delivery_features(self, data: dict) -> pd.DataFrame:
        """Preprocess features for delivery model (xgboost_deep with 'advanced' features)"""
        # Create DataFrame from single sample
        df = pd.DataFrame([data])

        # Handle date parameters: derive from order_date or use today
        if data.get('order_date'):
            order_dt = pd.to_datetime(data['order_date'])
        else:
            order_dt = pd.Timestamp.now()

        # Derive temporal features if not provided
        if data.get('order_month') is None:
            df['order_month'] = order_dt.month
        if data.get('is_weekend') is None:
            df['is_weekend'] = 1 if order_dt.dayofweek in [5, 6] else 0
        if data.get('order_day_of_week') is None:
            df['order_day_of_week'] = order_dt.dayofweek
        if data.get('order_hour') is None:
            df['order_hour'] = 12  # Default to noon

        # Calculate distance from lat/lon coordinates
        df['distance_km'] = haversine_distance(
            df['latitude'], df['longitude'],
            df['latitude_seller'], df['longitude_seller']
        )

        # Add product dimensions (use volume to back-calculate cube dimensions)
        cube_side = np.cbrt(df['product_volume_cm3'].values[0])
        df['product_length_cm'] = cube_side
        df['product_height_cm'] = cube_side
        df['product_width_cm'] = cube_side

        # Add order_purchase_timestamp for feature engineering
        df['order_purchase_timestamp'] = pd.Timestamp(
            year=order_dt.year, month=order_dt.month, day=order_dt.day,
            hour=int(df['order_hour'].values[0])
        )

        # Apply feature engineering strategy
        df = self.delivery_strategy.engineer(df)

        # Get expected feature columns
        feature_columns = self.delivery_strategy.get_feature_names()

        # Select only required features
        X = df[feature_columns].copy()

        # Handle numeric features - fill NaN
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_features:
            X[col] = X[col].fillna(0)

        # Encode categorical features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_features:
            if col in self.delivery_preprocessor:
                encoder = self.delivery_preprocessor[col]
                try:
                    X[col] = encoder.transform(X[col].astype(str))
                except ValueError:
                    X[col] = -1

        return X


# Initialize model manager
model_manager = ModelManager()


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting API server...")
    try:
        model_manager.load_models()
        logger.info("API server ready")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Don't fail startup - let health check report the issue


@app.get("/", response_class=HTMLResponse)
async def root():
    """Welcome page with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Marketplace ML Optimization API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                background: #f5f5f5;
                color: #333;
                padding: 20px;
                line-height: 1.6;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .header {
                background: #2c3e50;
                color: white;
                padding: 30px;
                border-bottom: 3px solid #34495e;
            }
            .header h1 {
                font-size: 2em;
                margin-bottom: 5px;
            }
            .header p {
                font-size: 1em;
                opacity: 0.9;
            }
            .content {
                padding: 30px;
            }
            .section {
                margin-bottom: 35px;
            }
            .section h2 {
                font-size: 1.4em;
                margin-bottom: 15px;
                color: #2c3e50;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 8px;
            }
            .models-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            .model-card {
                background: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 20px;
            }
            .model-card h3 {
                font-size: 1.2em;
                margin-bottom: 12px;
                color: #2c3e50;
            }
            .model-card .metric {
                margin: 6px 0;
                font-size: 0.9em;
            }
            .endpoints {
                display: grid;
                gap: 12px;
            }
            .endpoint {
                background: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 15px;
            }
            .endpoint .method {
                display: inline-block;
                background: #27ae60;
                color: white;
                padding: 3px 10px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 0.8em;
                margin-bottom: 8px;
            }
            .endpoint .method.get {
                background: #3498db;
            }
            .endpoint h3 {
                font-size: 1.1em;
                margin-bottom: 6px;
                color: #2c3e50;
            }
            .endpoint p {
                color: #555;
                margin: 4px 0;
                font-size: 0.9em;
            }
            .code-block {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 4px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 0.85em;
                margin-top: 10px;
            }
            .button {
                display: inline-block;
                background: #2c3e50;
                color: white;
                padding: 10px 20px;
                border-radius: 4px;
                text-decoration: none;
                font-weight: 500;
                margin-top: 15px;
                margin-right: 10px;
            }
            .button:hover {
                background: #34495e;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Marketplace ML Optimization API</h1>
                <p>Machine learning predictions for freight costs and delivery times</p>
            </div>

            <div class="content">

                <div class="section">
                    <h2>API Endpoints</h2>
                    <div class="endpoints">
                        <div class="endpoint">
                            <span class="method">POST</span>
                            <h3>/predict/freight</h3>
                            <p>Predict freight cost for an order</p>
                            <p><strong>Required:</strong> latitude, longitude, latitude_seller, longitude_seller, product_weight_g, product_volume_cm3, product_category_name, price</p>
                            <p><strong>Optional:</strong> order_date (defaults to today)</p>
                        </div>

                        <div class="endpoint">
                            <span class="method">POST</span>
                            <h3>/predict/delivery</h3>
                            <p>Predict delivery time in days</p>
                            <p><strong>Required:</strong> latitude, longitude, latitude_seller, longitude_seller, product_weight_g, product_volume_cm3, product_category_name, price</p>
                            <p><strong>Optional:</strong> order_date (defaults to today)</p>
                        </div>

                        <div class="endpoint">
                            <span class="method get">GET</span>
                            <h3>/health</h3>
                            <p>Check API health status</p>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>Example Usage</h2>

                    <h3 style="margin-top: 15px; margin-bottom: 8px; font-size: 1.1em;">Freight Cost Prediction</h3>
                    <div class="code-block">curl -X POST "http://localhost:8000/predict/freight" -H "Content-Type: application/json" -d '{"latitude": -23.5505, "longitude": -46.6333, "latitude_seller": -22.9068, "longitude_seller": -43.1729, "product_weight_g": 500, "product_volume_cm3": 1000, "product_category_name": "health_beauty", "price": 50.0}'</div>

                    <h3 style="margin-top: 25px; margin-bottom: 8px; font-size: 1.1em;">Delivery Time Prediction</h3>
                    <div class="code-block">curl -X POST "http://localhost:8000/predict/delivery" -H "Content-Type: application/json" -d '{"latitude": -23.5505, "longitude": -46.6333, "latitude_seller": -22.9068, "longitude_seller": -43.1729, "product_weight_g": 500, "product_volume_cm3": 1000, "product_category_name": "health_beauty", "price": 50.0}'</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_manager.models_loaded else "unhealthy",
        models_loaded={
            "freight_model": model_manager.freight_model is not None,
            "delivery_model": model_manager.delivery_model is not None
        },
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict/freight", response_model=PredictionResponse)
async def predict_freight(request: FreightPredictionRequest):
    """
    Predict optimal freight cost

    Returns predicted freight cost in dollars
    """
    if not model_manager.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Prepare input data
        input_data = request.dict()

        # Preprocess features
        features = model_manager.preprocess_freight_features(input_data)

        # Make prediction
        prediction = model_manager.freight_model.predict(features)[0]

        # Ensure non-negative prediction
        prediction = max(0, prediction)

        logger.info(f"Freight prediction: ${prediction:.2f}")

        return PredictionResponse(
            prediction=float(prediction),
            prediction_formatted=f"${prediction:.2f}",
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in freight prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/delivery", response_model=PredictionResponse)
async def predict_delivery(request: DeliveryPredictionRequest):
    """
    Predict expected delivery days

    Returns predicted number of days until delivery
    """
    if not model_manager.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Prepare input data
        input_data = request.dict()

        # Preprocess features
        features = model_manager.preprocess_delivery_features(input_data)

        # Make prediction
        prediction = model_manager.delivery_model.predict(features)[0]

        # Ensure positive prediction and round to nearest day
        prediction = max(1, round(prediction))

        logger.info(f"Delivery prediction: {prediction} days")

        return PredictionResponse(
            prediction=float(prediction),
            prediction_formatted=f"{int(prediction)} days",
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in delivery prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/both")
async def predict_both(
    freight_request: FreightPredictionRequest,
    delivery_request: DeliveryPredictionRequest
):
    """
    Get both freight and delivery predictions in one call
    """
    if not model_manager.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Get freight prediction
        freight_pred = await predict_freight(freight_request)

        # Get delivery prediction
        delivery_pred = await predict_delivery(delivery_request)

        return {
            "freight": freight_pred,
            "delivery": delivery_pred,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in combined prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Run Server (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT

    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
