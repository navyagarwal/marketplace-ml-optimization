# ML Model Training & Deployment Guide

Complete guide for training models and deploying the API.

---

## 📁 Project Structure

```
marketplace-ml-optimization/
├── data/
│   └── clean_data/          # Preprocessed data from notebooks
├── models/                   # Trained models (generated)
├── src/
│   ├── config.py            # Configuration
│   ├── data_preprocessing.py # Feature engineering
│   ├── train_freight.py     # Freight model training
│   ├── train_delivery.py    # Delivery model training
│   └── api.py               # FastAPI application
├── notebooks/               # EDA notebooks
├── Dockerfile               # Docker container definition
├── docker-compose.yml       # Docker orchestration
├── requirements.txt         # Python dependencies
└── DEPLOYMENT.md           # This file
```

---

## 🚀 Quick Start

### Option 1: Local Development (Recommended for Training)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models
cd src
python train_freight.py    # Train freight model
python train_delivery.py   # Train delivery model

# 4. Start API locally
python api.py
# API will be available at http://localhost:8000
```

### Option 2: Docker Deployment (Recommended for Production)

```bash
# 1. Build and start container
docker-compose up -d

# 2. Check logs
docker-compose logs -f

# 3. Test API
curl http://localhost:8000/health
```

---

## 📊 Model Training

### Step 1: Ensure Data is Ready

The training scripts expect cleaned data in `data/clean_data/`:
- `orders_master.csv`
- `order_items.csv`
- `customers.csv`
- `sellers.csv`
- `products.csv`

These are generated from the EDA notebooks.

### Step 2: Train Freight Cost Model

```bash
cd src
python train_freight.py
```

**What This Does:**
- Loads and preprocesses data
- Engineers features (distance, weight, volume interactions)
- Trains XGBoost regressor
- Evaluates on train/val/test splits
- Saves model to `models/freight_model.pkl`
- Generates visualizations and metrics

**Expected Output:**
```
✅ Test Set Results:
  MAE:  $X.XX (target: $3.00)
  RMSE: $X.XX
  R²:   0.XXX (target: 0.70)
  MAPE: X.XX%
```

**Generated Files:**
- `models/freight_model.pkl` - Trained model
- `models/freight_preprocessor.pkl` - Label encoders
- `models/freight_metrics.json` - Performance metrics
- `models/freight_feature_importance.png` - Feature importance plot
- `models/freight_predictions_test.png` - Predictions visualization

### Step 3: Train Delivery Date Model

```bash
cd src
python train_delivery.py
```

**What This Does:**
- Loads and preprocesses data
- Engineers temporal and route features
- Trains XGBoost regressor
- Evaluates accuracy windows (within 1/2/3/7 days)
- Saves model to `models/delivery_model.pkl`
- Generates visualizations

**Expected Output:**
```
✅ Test Set Results:
  MAE:  X.XX days (target: 2.00 days)
  R²:   0.XXX (target: 0.65)
  Within 3 days: XX.X% (target: 80%)
```

**Generated Files:**
- `models/delivery_model.pkl` - Trained model
- `models/delivery_preprocessor.pkl` - Label encoders
- `models/delivery_metrics.json` - Performance metrics
- `models/delivery_feature_importance.png` - Feature importance plot
- `models/delivery_predictions_test.png` - Predictions visualization
- `models/delivery_error_distribution_test.png` - Error analysis

---

## 🔧 API Usage

### Starting the API

**Local:**
```bash
cd src
python api.py
# Or with uvicorn directly:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Docker:**
```bash
docker-compose up -d
```

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "freight_model": true,
    "delivery_model": true
  },
  "timestamp": "2024-02-24T12:00:00"
}
```

#### 2. Predict Freight Cost

```bash
curl -X POST http://localhost:8000/predict/freight \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 250.5,
    "product_weight_g": 500,
    "product_volume_cm3": 1000,
    "product_category_name": "health_beauty",
    "price": 50.0,
    "order_month": 6,
    "is_weekend": 0
  }'
```

**Response:**
```json
{
  "prediction": 18.45,
  "prediction_formatted": "$18.45",
  "model_version": "1.0.0",
  "timestamp": "2024-02-24T12:00:00"
}
```

#### 3. Predict Delivery Date

```bash
curl -X POST http://localhost:8000/predict/delivery \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 250.5,
    "product_weight_g": 500,
    "product_volume_cm3": 1000,
    "product_category_name": "health_beauty",
    "price": 50.0,
    "order_day_of_week": 2,
    "order_hour": 14,
    "order_month": 6,
    "is_weekend": 0
  }'
```

**Response:**
```json
{
  "prediction": 12.0,
  "prediction_formatted": "12 days",
  "model_version": "1.0.0",
  "timestamp": "2024-02-24T12:00:00"
}
```

#### 4. Interactive API Documentation

Open your browser to:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🐳 Docker Commands

### Build and Run

```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down

# Restart container
docker-compose restart
```

### Updating Models

```bash
# After retraining models, restart container to reload
docker-compose restart
```

### Debugging

```bash
# Access container shell
docker-compose exec ml-api bash

# View API logs
docker-compose logs ml-api

# Check container health
docker-compose ps
```

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model hyperparameters
FREIGHT_MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    # ... more parameters
}

# Success criteria
FREIGHT_TARGET_MAE = 3.0  # dollars
DELIVERY_TARGET_MAE = 2.0  # days

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4
```

---

## 📈 Model Performance Monitoring

### Checking Current Performance

```python
# Load saved metrics
import json

with open('models/freight_metrics.json', 'r') as f:
    freight_metrics = json.load(f)

print(f"Freight MAE: ${freight_metrics['test']['mae']:.2f}")
print(f"Freight R²: {freight_metrics['test']['r2']:.4f}")

with open('models/delivery_metrics.json', 'r') as f:
    delivery_metrics = json.load(f)

print(f"Delivery MAE: {delivery_metrics['test']['mae']:.2f} days")
print(f"Within 3 days: {delivery_metrics['test']['within_3_days']:.1f}%")
```

### Retraining Models

**When to retrain:**
- New data is available
- Model performance degrades
- Business requirements change

**How to retrain:**
1. Update data in `data/clean_data/`
2. Run training scripts again
3. Review new metrics
4. If improved, deploy new models
5. Restart API: `docker-compose restart`

---

## 🧪 Testing the API

### Python Example

```python
import requests

# Freight prediction
response = requests.post(
    'http://localhost:8000/predict/freight',
    json={
        'distance_km': 250.5,
        'product_weight_g': 500,
        'product_volume_cm3': 1000,
        'product_category_name': 'health_beauty',
        'price': 50.0,
        'order_month': 6,
        'is_weekend': 0
    }
)
print(f"Predicted freight: {response.json()['prediction_formatted']}")

# Delivery prediction
response = requests.post(
    'http://localhost:8000/predict/delivery',
    json={
        'distance_km': 250.5,
        'product_weight_g': 500,
        'product_volume_cm3': 1000,
        'product_category_name': 'health_beauty',
        'price': 50.0,
        'order_day_of_week': 2,
        'order_hour': 14,
        'order_month': 6,
        'is_weekend': 0
    }
)
print(f"Predicted delivery: {response.json()['prediction_formatted']}")
```

### JavaScript Example

```javascript
// Freight prediction
const freightResponse = await fetch('http://localhost:8000/predict/freight', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    distance_km: 250.5,
    product_weight_g: 500,
    product_volume_cm3: 1000,
    product_category_name: 'health_beauty',
    price: 50.0,
    order_month: 6,
    is_weekend: 0
  })
});
const freightData = await freightResponse.json();
console.log(`Predicted freight: ${freightData.prediction_formatted}`);
```

---

## 🚀 Production Deployment

### Environment Variables

Set these in production:

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4  # Adjust based on CPU cores
```

### Scaling

**Option 1: Multiple Workers (Same Host)**
```yaml
# docker-compose.yml
environment:
  - API_WORKERS=8  # Adjust based on CPU
```

**Option 2: Multiple Containers (Load Balancing)**
```yaml
# docker-compose.yml
services:
  ml-api:
    deploy:
      replicas: 3
    # ... rest of config
```

### Monitoring

Add logging and monitoring:
- **Logs:** Container logs via `docker-compose logs`
- **Metrics:** Track prediction latency, error rates
- **Health:** Use `/health` endpoint for uptime monitoring

---

## 🔒 Security Considerations

1. **Non-root user:** Container runs as `mluser` (UID 1000)
2. **Read-only models:** Models mounted as read-only in Docker
3. **CORS:** Configure allowed origins in production
4. **Rate limiting:** Add rate limiting middleware if needed
5. **Authentication:** Add API keys or OAuth if required

---

## 📝 Troubleshooting

### Models Not Loading

```bash
# Check if model files exist
ls -la models/

# Expected files:
# - freight_model.pkl
# - freight_preprocessor.pkl
# - delivery_model.pkl
# - delivery_preprocessor.pkl
```

### API Returns 503 Error

Check health endpoint:
```bash
curl http://localhost:8000/health
```

If models not loaded, retrain:
```bash
cd src
python train_freight.py
python train_delivery.py
```

### Docker Build Fails

Clear cache and rebuild:
```bash
docker-compose down
docker system prune -a
docker-compose build --no-cache
docker-compose up
```

### Prediction Errors

Check input data matches schema:
- All required fields present
- Numeric fields > 0
- Category names match training data
- Month in range 1-12
- Day of week in range 0-6

---

## 📚 Additional Resources

- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **XGBoost Docs:** https://xgboost.readthedocs.io/
- **Docker Docs:** https://docs.docker.com/

---

## ✅ Success Checklist

Before deployment, verify:

- [ ] Models trained successfully
- [ ] Test metrics meet targets (MAE, R², accuracy)
- [ ] API starts without errors
- [ ] Health check returns "healthy"
- [ ] Sample predictions return valid results
- [ ] Docker container builds and runs
- [ ] API documentation accessible at /docs
- [ ] Models reload after container restart

---

## 📧 Support

For issues or questions:
1. Check logs: `docker-compose logs ml-api`
2. Review metrics: `cat models/freight_metrics.json`
3. Test locally before Docker deployment
4. Verify data preprocessing matches training

---

**Last Updated:** 2024-02-24
**Version:** 1.0.0
