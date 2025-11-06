# Moonshot Detector 🚀

**ML-based trading system for detecting explosive 20-50%+ gains in stocks and crypto over 4-8 weeks.**

## 🎯 Project Overview

This is the **SCRAPPY VERSION** - built with **100% FREE data sources** to prove profitability before investing in paid data.

- **Budget**: $5-15/month (AWS only)
- **Goal**: Prove the system works, then upgrade
- **Timeline**: 3-6 months paper trading before going live

## ✨ Features

- **6 ML Models** (5 active + 1 disabled):
  1. ✅ Technical Breakout Classifier (XGBoost)
  2. ✅ Volatility Squeeze Detector (SVM)
  3. ✅ PEAD + Short Squeeze Hybrid (XGBoost)
  4. ✅ GARP Fundamental Filter (Random Forest)
  5. ❌ On-Chain Analyst (DISABLED - needs Glassnode $29-799/mo)
  6. ✅ Anomaly / Regime Detector (Isolation Forest)

- **Ensemble Voting** with dynamic weight adjustment
- **Walk-Forward Validation** with strict overfitting controls
- **Free Data Sources**: yfinance, SEC EDGAR, FINRA, Reddit, Google Trends, FRED, CCXT, CoinGecko

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd moonshot-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: TA-Lib requires manual installation on some systems:
- **Linux**: `sudo apt-get install ta-lib`
- **Mac**: `brew install ta-lib`
- **Windows**: Download from [TA-Lib website](https://www.ta-lib.org/)

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# - Reddit API (free): https://www.reddit.com/prefs/apps
# - FRED API (free): https://fred.stlouisfed.org/docs/api/api_key.html
# - Alpaca (free paper trading): https://alpaca.markets
```

### 3. Download Data

```bash
# Make script executable
chmod +x scripts/download_all_data.sh

# Run data download (2-4 hours)
bash scripts/download_all_data.sh
```

This downloads:
- ~500 S&P 500 stocks (yfinance)
- Top 50 cryptocurrencies (Binance)
- Fundamentals from SEC EDGAR
- Short interest from FINRA (requires manual step)
- Sentiment from Reddit + Google Trends
- Economic indicators from FRED

### 4. Feature Engineering

```bash
# Generate technical features for all tickers
python src/features/feature_pipeline.py --tickers data/universe/tradeable_stocks.txt
```

### 5. Label Generation

```bash
# Generate training labels (30-60 day explosive move targets)
python src/labeling/label_generator.py --tickers data/universe/tradeable_stocks.txt
```

### 6. Train Models

```bash
# Train all active models
# TODO: Create training script (see NEXT_STEPS.md)
```

### 7. Backtest

```bash
# Run walk-forward backtesting
# TODO: Create backtest script (see NEXT_STEPS.md)
```

### 8. Paper Trade

```bash
# After successful backtest validation
# TODO: Set up paper trading (see NEXT_STEPS.md)
```

## 📁 Project Structure

```
moonshot-detector/
├── data/
│   ├── raw/              # Downloaded raw data (stocks, crypto, fundamentals, etc.)
│   ├── processed/        # Engineered features and labels
│   ├── universe/         # Tradeable ticker lists
│   └── backtests/        # Backtest results
├── models/
│   ├── trained/          # Saved model artifacts
│   ├── configs/          # Model hyperparameters
│   └── performance/      # Model performance tracking
├── src/
│   ├── data_collection/  # Data downloaders (7 modules)
│   ├── features/         # Feature engineering
│   ├── labeling/         # Label generation
│   ├── models/           # Model implementations
│   ├── backtest/         # Backtesting framework
│   ├── inference/        # Live prediction and paper trading
│   └── utils/            # Utilities (logging, parsers, validation)
├── scripts/              # Automation scripts
├── notebooks/            # Jupyter notebooks for analysis
├── tests/                # Unit tests
├── config.yaml           # Master configuration
├── requirements.txt      # Python dependencies
└── README.md
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Data sources and filters
- Model hyperparameters
- Risk management rules
- Backtesting parameters
- AWS deployment settings

## 📊 Data Sources (All FREE)

| Data Type | Source | Cost | Rate Limit |
|-----------|--------|------|------------|
| Stock prices | yfinance | $0 | 2 req/sec |
| Stock fundamentals | SEC EDGAR | $0 | 10 req/sec |
| Short interest | FINRA | $0 | Manual download |
| Crypto prices | CCXT (Binance) | $0 | 1 req/sec |
| Crypto market data | CoinGecko | $0 | 50 req/min |
| Sentiment | Reddit API | $0 | 60 req/min |
| Search trends | Google Trends | $0 | No limit |
| Economic data | FRED | $0 | No limit |

**Total data cost: $0/month** ✅

## ⚠️ Important Warnings

1. **This is experimental** - paper trade for 3-6 months before risking real money
2. **Past performance ≠ future results** - models can and will fail
3. **Transaction costs matter** - we test at 2X expected costs for safety
4. **Overfitting is real** - that's why we use walk-forward + strict validation
5. **Start small** - begin with $5-10k even if you have more

## 📈 Performance Expectations

Based on academic research (if everything works):
- **Win rate**: 55-65% (not 80%+!)
- **Sharpe ratio**: 1.0-2.0 (after costs)
- **Max drawdown**: 15-25%
- **Monthly return**: 3-8% (highly variable)

**Reality check**: 50%+ of published trading strategies fail to replicate. Most ML trading funds fail. Be skeptical!

## 🛣️ Upgrade Path

### After $5k Profit:
- Add **Alpha Vantage Premium** ($50/mo) for real-time stock data
- Add **Financial Modeling Prep** ($29/mo) for better fundamentals
- **Total**: $79/mo

### After $10k Profit:
- Add **Glassnode Advanced** ($29/mo) to enable Model 5 (crypto on-chain)
- **Total**: $108/mo

### After $25k Profit:
- Consider **institutional data** (Polygon.io, EODHD) for $200-400/mo

**NEVER upgrade before proving profitability!**

## 🧪 Validation Requirements

Before going live with real money:

### Backtest Validation
- ✅ t-statistic > 3.0
- ✅ PBO < 0.30
- ✅ Deflated Sharpe Ratio > 1.0
- ✅ Profitable at 2X transaction costs

### Paper Trading Validation (3-6 months)
- ✅ Performance within 20% of backtest
- ✅ Slippage within 25% of model
- ✅ No model degradation detected
- ✅ Sharpe ratio > 1.0
- ✅ Max drawdown < 25%

## 🔒 Risk Management

- **Position sizing**: Max 2% per trade
- **Portfolio limits**: Max 15% in explosive strategies
- **Stop losses**: 10% (stocks), 15% (crypto)
- **Circuit breakers**: Daily (-2%), Weekly (-5%), Monthly (-10%)
- **Volatility scaling**: Reduce exposure when volatility increases

## 🐛 Known Issues & TODOs

See `NEXT_STEPS.md` for:
- Models 2-6 implementations (templates provided)
- Backtesting framework
- Paper trading integration
- Performance monitoring dashboards
- Deployment to AWS

## 📚 Resources

- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) by Marcos Lopez de Prado
- [Quantitative Trading](http://epchan.blogspot.com/) blog by Ernie Chan
- r/algotrading, r/quantfinance on Reddit

## 📝 License

MIT License - See LICENSE file

## ⚖️ Disclaimer

**This is experimental software for educational purposes.**

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Never trade with money you can't afford to lose
- Always paper trade extensively before going live
- The authors are not responsible for financial losses

**Use at your own risk.**

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Support

For issues or questions:
- Open a GitHub issue
- Check documentation in `docs/` folder

---

**Built with 100% FREE data sources to prove profitability before upgrading. Let's hunt some moonshots! 🚀**
