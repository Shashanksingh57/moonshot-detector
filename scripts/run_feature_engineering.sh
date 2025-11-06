#!/bin/bash
set -e

echo "=================================================="
echo "Moonshot Detector - Feature Engineering Pipeline"
echo "=================================================="

echo ""
echo "Step 1: Checking for raw data..."
if [ ! -d "data/raw/stocks" ] || [ -z "$(ls -A data/raw/stocks 2>/dev/null)" ]; then
    echo "Error: No stock data found in data/raw/stocks/"
    echo "Please run data download first:"
    echo "  bash scripts/download_all_data.sh"
    exit 1
fi

echo ""
echo "Step 2: Checking for tradeable universe..."
if [ ! -f "data/universe/tradeable_stocks.txt" ]; then
    echo "Error: Tradeable universe not found"
    echo "Building tradeable universe..."
    python src/data_collection/universe_builder.py --asset-type stocks --config config.yaml
fi

echo ""
echo "Step 3: Running feature engineering for stocks..."
python src/features/feature_pipeline.py \
    --tickers data/universe/tradeable_stocks.txt \
    --asset-type stocks \
    --output data/processed/features

echo ""
echo "Step 4: Running feature engineering for crypto (if available)..."
if [ -f "data/universe/tradeable_crypto.txt" ]; then
    python src/features/feature_pipeline.py \
        --tickers data/universe/tradeable_crypto.txt \
        --asset-type crypto \
        --output data/processed/features
else
    echo "Skipping crypto (no tradeable universe found)"
fi

echo ""
echo "Step 5: Generating training labels..."
python src/labeling/label_generator.py \
    --tickers data/universe/tradeable_stocks.txt \
    --config config.yaml \
    --asset-type stocks

echo ""
echo "Step 6: Verifying output..."
FEATURE_COUNT=$(ls data/processed/features/*.parquet 2>/dev/null | wc -l)
LABEL_COUNT=$(ls data/processed/labels/*.parquet 2>/dev/null | wc -l)

echo "Features generated: $FEATURE_COUNT files"
echo "Labels generated: $LABEL_COUNT files"

echo ""
echo "=================================================="
echo "Feature engineering complete!"
echo ""
echo "Next steps:"
echo "  1. Train models: bash scripts/train_all_models.sh"
echo "  2. Run backtest: bash scripts/run_backtest.sh"
echo "=================================================="
