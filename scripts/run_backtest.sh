#!/bin/bash
set -e

echo "=================================================="
echo "Moonshot Detector - Walk-Forward Backtest"
echo "=================================================="

echo ""
echo "Step 1: Checking for trained models..."
if [ ! -d "models/trained" ] || [ -z "$(ls -A models/trained/*.pkl 2>/dev/null)" ]; then
    echo "Error: No trained models found in models/trained/"
    echo "Please train models first:"
    echo "  bash scripts/train_all_models.sh"
    exit 1
fi

echo ""
echo "Step 2: Creating output directory..."
mkdir -p data/backtests/latest

echo ""
echo "Step 3: Running walk-forward backtest..."
python src/backtest/walk_forward.py --config config.yaml --output data/backtests/latest

echo ""
echo "Step 4: Evaluating results..."
echo ""
echo "=================================================="
echo "Backtest complete!"
echo ""
echo "Results saved to: data/backtests/latest/"
echo ""
echo "VALIDATION CHECKLIST:"
echo "  [ ] t-statistic > 3.0 (not 1.96!)"
echo "  [ ] PBO < 0.30"
echo "  [ ] Deflated Sharpe > 1.0"
echo "  [ ] Profitable at 2X transaction costs"
echo "  [ ] Min 50 trades"
echo ""
echo "If ALL checks pass:"
echo "  → Proceed to paper trading for 3-6 months"
echo ""
echo "If ANY check fails:"
echo "  → DO NOT proceed to paper trading"
echo "  → Iterate on models and features"
echo "  → Re-run backtest"
echo ""
echo "=================================================="
