#!/bin/bash
set -e

echo "=================================================="
echo "Moonshot Detector - Model Training Pipeline"
echo "=================================================="

echo ""
echo "Step 1: Checking for processed features..."
if [ ! -d "data/processed/features" ] || [ -z "$(ls -A data/processed/features 2>/dev/null)" ]; then
    echo "Error: No features found in data/processed/features/"
    echo "Please run feature engineering first:"
    echo "  python src/features/feature_pipeline.py --tickers data/universe/tradeable_stocks.txt"
    exit 1
fi

echo "Step 2: Checking for labels..."
if [ ! -d "data/processed/labels" ] || [ -z "$(ls -A data/processed/labels 2>/dev/null)" ]; then
    echo "Error: No labels found in data/processed/labels/"
    echo "Please run label generation first:"
    echo "  python src/labeling/label_generator.py --tickers data/universe/tradeable_stocks.txt"
    exit 1
fi

echo ""
echo "Step 3: Training all models..."
if [ -f "data/universe/tradeable_stocks.txt" ]; then
    python src/models/train.py --tickers data/universe/tradeable_stocks.txt --config config.yaml
else
    echo "Error: Ticker list not found at data/universe/tradeable_stocks.txt"
    exit 1
fi

echo ""
echo "Step 4: Verifying trained models..."
if [ -d "models/trained" ]; then
    MODEL_COUNT=$(ls models/trained/*.pkl 2>/dev/null | wc -l)
    echo "Found $MODEL_COUNT trained models"
else
    echo "Warning: models/trained directory not found"
fi

echo ""
echo "=================================================="
echo "Model training complete!"
echo ""
echo "Trained models saved to: models/trained/"
echo ""
echo "Next steps:"
echo "  1. Run backtest: bash scripts/run_backtest.sh"
echo "  2. Generate signals: python src/inference/signal_generator.py --tickers data/universe/tradeable_stocks.txt"
echo "=================================================="
