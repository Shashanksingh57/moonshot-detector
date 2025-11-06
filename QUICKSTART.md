# 🚀 Moonshot Detector - Quick Start Guide

Get the system running in under 1 hour (excluding data download time).

## Prerequisites

- Python 3.8+
- Git
- 10GB+ disk space
- Internet connection

## Step-by-Step Setup

### 1. Installation (5 minutes)

```bash
# Clone and navigate
cd moonshot-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Important**: TA-Lib requires system installation:
- **Ubuntu/Debian**: `sudo apt-get install ta-lib`
- **Mac**: `brew install ta-lib`
- **Windows**: Download from [ta-lib.org](https://www.ta-lib.org/)

### 2. Configure API Keys (10 minutes)

```bash
# Copy environment template
cp .env.example .env
```

Edit `.env` and add:

#### Required (Free):
- **Reddit API**: https://www.reddit.com/prefs/apps
  - Create app → get client_id and client_secret
- **FRED API**: https://fred.stlouisfed.org/docs/api/api_key.html
  - Sign up → get API key

#### Optional (for paper trading later):
- **Alpaca**: https://alpaca.markets
  - Free paper trading account

### 3. Download Data (2-4 hours automated)

```bash
# Download all data sources
bash scripts/download_all_data.sh
```

This downloads:
- ✓ S&P 500 stocks (yfinance)
- ✓ Top 50 cryptos (Binance)
- ✓ Fundamentals (SEC EDGAR)
- ✓ Sentiment (Reddit + Google Trends)
- ✓ Economic data (FRED)

**Note**: Short interest requires manual download from [FINRA](https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data)

### 4. Feature Engineering (1-2 hours)

```bash
# Generate features and labels
bash scripts/run_feature_engineering.sh
```

This creates:
- Technical indicators (RSI, MACD, etc.)
- Training labels (20-50%+ gains in 30-60 days)

### 5. Train Models (30-60 minutes)

```bash
# Train all models
bash scripts/train_all_models.sh
```

Trains:
- Model 1: Technical Breakout (XGBoost)
- Model 2: Volatility Squeeze (SVM)
- Model 3: PEAD + Short Squeeze (XGBoost)
- Model 4: GARP Filter (Random Forest)
- Model 6: Anomaly Detector (Isolation Forest)

### 6. Run Backtest (15-30 minutes)

```bash
# Run walk-forward backtest
bash scripts/run_backtest.sh
```

**Critical Validation Checks**:
- ✅ t-statistic > 3.0
- ✅ PBO < 0.30
- ✅ Deflated Sharpe > 1.0
- ✅ Profitable at 2X costs

**If ANY check fails**: DO NOT proceed. Iterate on models.

### 7. Generate Signals (1 minute)

```bash
# Generate current trading signals
python src/inference/signal_generator.py \
    --tickers data/universe/tradeable_stocks.txt \
    --output data/signals/latest_signals.csv
```

View signals:
```bash
cat data/signals/latest_signals.csv
```

## Quick Test

Run a simple test:

```bash
python tests/test_features.py
```

Should output: "All feature tests passed! ✓"

## Troubleshooting

### "No module named 'talib'"
- **Solution**: Install TA-Lib system package first (see step 1)

### "Reddit API authentication failed"
- **Solution**: Check .env has correct REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET

### "No data found"
- **Solution**: Ensure step 3 (data download) completed successfully
- Check `data/raw/stocks/` has .parquet files

### "No features found"
- **Solution**: Run `bash scripts/run_feature_engineering.sh`

### "Models not trained"
- **Solution**: Run `bash scripts/train_all_models.sh`

## Next Steps

### If Backtest Validates ✅
1. **Paper trade for 3-6 months** (see NEXT_STEPS.md)
2. Monitor performance vs backtest
3. Track slippage
4. Only go live after paper trading validates

### If Backtest Fails ❌
1. Review model configurations
2. Try different features
3. Adjust hyperparameters
4. Re-train and re-test
5. **Do NOT skip validation**

## Daily Usage

### Update Data
```bash
bash scripts/download_all_data.sh  # Re-run to get latest
```

### Generate New Signals
```bash
python src/inference/signal_generator.py --tickers data/universe/tradeable_stocks.txt
```

### Re-train Models (monthly)
```bash
bash scripts/train_all_models.sh
```

## File Structure

```
moonshot-detector/
├── data/
│   ├── raw/          # Downloaded data
│   ├── processed/    # Features and labels
│   ├── universe/     # Tradeable tickers
│   └── signals/      # Generated signals
├── models/
│   └── trained/      # Saved models
├── scripts/          # Automation scripts
└── src/              # Source code
```

## Important Warnings

1. **This is experimental** - Start small, paper trade first
2. **Free data has limitations** - 15min delay on stocks
3. **Validation is critical** - Don't skip the checks
4. **Costs matter** - We test at 2X to be safe
5. **Paper trade 3-6 months** - No shortcuts

## Getting Help

- **Documentation**: See README.md and NEXT_STEPS.md
- **Issues**: Check logs/ directory
- **Config**: All settings in config.yaml

## Performance Expectations

**If everything works** (big if):
- Win rate: 55-65% (not 80%!)
- Sharpe ratio: 1.0-2.0
- Max drawdown: 15-25%
- Monthly return: 3-8% (highly variable)

**Reality**: 50%+ of strategies fail. Be skeptical!

---

**Total setup time**: ~1 hour active work + 3-5 hours automated processing

**Ready to go live**: 3-6 months minimum (after paper trading)

**Cost**: $0/month data + $5-15/month AWS (deployment)

Let's hunt some moonshots! 🚀
