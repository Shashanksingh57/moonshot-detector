#!/bin/bash
set -e

echo "=================================================="
echo "Moonshot Detector - Data Download Script"
echo "=================================================="

echo ""
echo "Step 1: Downloading stock data (yfinance)..."
python src/data_collection/stock_collector.py --sp500 --start-date 2015-01-01

echo ""
echo "Step 2: Downloading crypto data (CCXT + CoinGecko)..."
python src/data_collection/crypto_collector.py --top-n 50

echo ""
echo "Step 3: Building tradeable universe (applying filters)..."
python src/data_collection/universe_builder.py --asset-type stocks
python src/data_collection/universe_builder.py --asset-type crypto

echo ""
echo "Step 4: Downloading fundamentals (SEC EDGAR)..."
if [ -f "data/universe/tradeable_stocks.txt" ]; then
    python src/data_collection/fundamentals_collector.py --tickers data/universe/tradeable_stocks.txt --forms 10-K 10-Q
else
    echo "Warning: tradeable_stocks.txt not found, skipping fundamentals"
fi

echo ""
echo "Step 5: Downloading short interest (FINRA)..."
echo "Note: FINRA data requires manual download from https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data"
echo "Download files and place in data/raw/short_interest/raw/ then run:"
echo "  python src/data_collection/short_interest_collector.py"

echo ""
echo "Step 6: Downloading sentiment data (Reddit + Trends)..."
if [ -f "data/universe/tradeable_stocks.txt" ]; then
    python src/data_collection/sentiment_collector.py --tickers data/universe/tradeable_stocks.txt --subreddits wallstreetbets stocks investing
else
    echo "Warning: tradeable_stocks.txt not found, skipping sentiment"
fi

echo ""
echo "Step 7: Downloading economic data (FRED)..."
python src/data_collection/economic_collector.py --series VIXCLS DGS10 UNRATE GDP

echo ""
echo "=================================================="
echo "Data download complete!"
echo ""
echo "Next steps:"
echo "  1. Review data in data/raw/"
echo "  2. Run feature engineering: bash scripts/run_feature_engineering.sh"
echo "  3. Train models: bash scripts/train_all_models.sh"
echo "=================================================="
