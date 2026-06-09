#!/bin/bash
set -e

echo "========================================"
echo "  Stocks Prediction - Codespaces Setup"
echo "========================================"

echo ""
echo "1. Install dependencies..."
pip install --upgrade pip
pip install mlflow xgboost scikit-learn pandas yfinance pyyaml

echo ""
echo "2. Training model XGBoost..."
cd src
python train.py
cd ..

echo ""
echo "3. Register model dengan alias champion..."
python register_champion.py

echo ""
echo "4. Cari path model artifacts..."
MODEL_PATH=$(find mlruns -path "*/artifacts/model.ubj" | head -1 | xargs dirname)
echo "   Model artifacts: $MODEL_PATH"

echo ""
echo "========================================"
echo "  Setup Selesai!"
echo "========================================"
echo ""
echo "  ✅ Model terlatih & ter-register dengan alias 'champion'"
echo ""
echo "📌 Cara menjalankan 3 replica MLflow Serve di Docker:"
echo ""
echo "  # Update path model di docker-compose (sesuaikan run_id):"
echo "  # Lalu jalankan:"
echo "  docker compose -f docker-compose.mlflow-serve.yml up --build -d"
echo ""
echo "📌 Test endpoint:"
echo "  curl -X POST http://localhost:5001/invocations \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"dataframe_split\": {\"columns\": [\"Close\",\"RSI\",\"EMA_9\",\"SMA_5\",\"SMA_10\",\"SMA_15\",\"SMA_30\",\"MACD\",\"MACD_signal\"], \"data\": [[9250.0, 55.0, 9240.0, 9230.0, 9220.0, 9210.0, 9200.0, 10.5, 9.8]]}}'"
echo ""
echo "📌 Cek 3 replica berjalan:"
echo "  docker compose -f docker-compose.mlflow-serve.yml ps"
