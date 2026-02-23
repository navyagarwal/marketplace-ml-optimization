#!/bin/bash

# Complete training pipeline script
# This script trains both ML models sequentially

set -e  # Exit on error

echo "============================================================"
echo "MARKETPLACE ML OPTIMIZATION - TRAINING PIPELINE"
echo "============================================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if data exists
if [ ! -d "data/clean_data" ]; then
    echo "❌ Error: data/clean_data directory not found"
    echo "   Run the EDA notebooks first to generate clean data"
    exit 1
fi

echo "Step 1: Training Freight Cost Prediction Model"
echo "============================================================"
cd src
python train_freight.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Freight model training completed successfully"
else
    echo "❌ Freight model training failed"
    exit 1
fi

echo ""
echo ""
echo "Step 2: Training Delivery Date Prediction Model"
echo "============================================================"
python train_delivery.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Delivery model training completed successfully"
else
    echo "❌ Delivery model training failed"
    exit 1
fi

cd ..

echo ""
echo "============================================================"
echo "✅ ALL MODELS TRAINED SUCCESSFULLY"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review model metrics in models/ directory"
echo "  2. Check visualizations (feature importance, predictions)"
echo "  3. Start the API: cd src && python api.py"
echo "  4. Test the API: python test_api.py"
echo ""
echo "For Docker deployment:"
echo "  docker-compose up -d"
echo ""
