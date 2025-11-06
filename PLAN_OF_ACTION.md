# Moonshot Detector - Plan of Action to Get Running

**Current Status**: System is 75% complete but cannot run backtests yet due to critical blockers

**Goal**: Get to first successful backtest with validation

**Timeline**: 5-7 days of focused work

---

## 🚦 WHAT WORKS NOW vs WHAT DOESN'T

### ✅ What Works NOW:
1. **Data Collection** - All collectors work:
   ```bash
   # You can download data right now:
   python src/data_collection/stock_collector.py --sp500 --start-date 2015-01-01
   python src/data_collection/crypto_collector.py --top-n 50
   # etc.
   ```

2. **Technical Features** - Feature engineering works for technical indicators:
   ```bash
   # This works:
   python src/features/feature_pipeline.py --tickers data/universe/tradeable_stocks.txt --asset-type stocks
   ```

3. **All 6 Models** - Models are implemented and can be trained:
   ```bash
   # This works (but only with technical features):
   python src/models/train.py --tickers data/universe/tradeable_stocks.txt --asset-type stocks
   ```

4. **Signal Generation** - Can generate signals from trained models:
   ```bash
   # This works:
   python src/inference/signal_generator.py --tickers data/universe/tradeable_stocks.txt --asset-type stocks
   ```

5. **Data Quality Validation** - All validators work:
   ```bash
   # This works:
   python scripts/validate_data.py --summary
   ```

### ❌ What DOESN'T Work:
1. **Backtesting** - walk_forward.py is just a shell, cannot run backtests
2. **Full Feature Set** - Missing fundamental and sentiment features (Models 3 & 4 need these)
3. **Paper Trading** - Not implemented yet

---

## 🎯 THE PLAN - 4 PHASES

### **PHASE 1: Complete Missing Components (Days 1-4)**
Fix the blockers so backtesting can run

### **PHASE 2: End-to-End Testing (Day 5)**
Test the full pipeline with sample data

### **PHASE 3: First Backtest (Day 6)**
Run first complete backtest on limited period

### **PHASE 4: Full Backtest & Validation (Day 7+)**
Run full historical backtest and validate

---

## 📋 PHASE 1: COMPLETE MISSING COMPONENTS (Days 1-4)

### Task 1.1: Implement walk_forward.py Core Logic (Day 1-2)
**Priority**: CRITICAL - This is the #1 blocker

**Current State**:
```python
# Lines 125-140 in walk_forward.py are just:
# NOTE: This is simplified - in production would load actual data
# Placeholder for test predictions
```

**What to Build**:

**File**: `src/backtest/walk_forward.py`

**Add these functions**:

```python
def load_split_data(
    tickers: List[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    features_dir: str = 'data/processed/features',
    labels_dir: str = 'data/processed/labels'
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Load features and labels for a walk-forward split.

    Args:
        tickers: List of tickers to load
        train_start, train_end: Training period
        test_start, test_end: Test period
        features_dir: Directory with feature parquet files
        labels_dir: Directory with label parquet files

    Returns:
        train_features, train_labels, test_features, test_actual_returns
    """
    train_features_list = []
    train_labels_list = []
    test_features_list = []
    test_returns_list = []

    for ticker in tickers:
        try:
            # Load features
            features_file = Path(features_dir) / f"{ticker}_features.parquet"
            if not features_file.exists():
                continue

            features_df = pd.read_parquet(features_file)

            # Load labels
            labels_file = Path(labels_dir) / f"{ticker}_labels.parquet"
            if not labels_file.exists():
                continue

            labels_df = pd.read_parquet(labels_file)

            # Merge on date
            df = pd.merge(features_df, labels_df, on='date', how='inner')

            # Split into train and test
            df['date'] = pd.to_datetime(df['date'])

            train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
            test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

            if len(train_df) > 0:
                train_features_list.append(train_df.drop(['label', 'actual_return'], axis=1, errors='ignore'))
                train_labels_list.append(train_df[['date', 'ticker', 'label']])

            if len(test_df) > 0:
                test_features_list.append(test_df.drop(['label', 'actual_return'], axis=1, errors='ignore'))
                test_returns_list.append(test_df[['date', 'ticker', 'actual_return', 'label']])

        except Exception as e:
            logger.warning(f"Error loading data for {ticker}: {str(e)}")
            continue

    # Combine all tickers
    train_features = pd.concat(train_features_list, ignore_index=True) if train_features_list else pd.DataFrame()
    train_labels = pd.concat(train_labels_list, ignore_index=True) if train_labels_list else pd.DataFrame()
    test_features = pd.concat(test_features_list, ignore_index=True) if test_features_list else pd.DataFrame()
    test_returns = pd.concat(test_returns_list, ignore_index=True) if test_returns_list else pd.DataFrame()

    logger.info(f"Loaded train: {len(train_features)} rows, test: {len(test_features)} rows")

    return train_features, train_labels['label'], test_features, test_returns


def run_split_backtest(
    ensemble: EnsembleVoter,
    test_features: pd.DataFrame,
    test_returns: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run backtest for a single walk-forward split.

    Args:
        ensemble: EnsembleVoter instance (already trained)
        test_features: Test period features
        test_returns: Test period actual returns
        config: Configuration dictionary

    Returns:
        predictions_df, metrics_dict
    """
    try:
        # Generate predictions
        # Extract ticker list from features
        tickers = test_features['ticker'].unique().tolist()

        # Create features dict for ensemble
        features_dict = {}
        for ticker in tickers:
            ticker_features = test_features[test_features['ticker'] == ticker]
            if len(ticker_features) > 0:
                features_dict[ticker] = ticker_features

        # Generate signals
        asset_type = 'stocks'  # TODO: detect from config or pass as parameter
        signals_df = ensemble.generate_signals(asset_type, tickers, features_dict)

        # Merge with actual returns
        predictions_df = pd.merge(
            signals_df,
            test_returns,
            on=['ticker', 'date'],
            how='inner'
        )

        # Rename for evaluator
        predictions_df = predictions_df.rename(columns={
            'weighted_score': 'proba',
            'signal': 'prediction'
        })

        # Convert signal to binary
        predictions_df['prediction'] = (predictions_df['prediction'] == 'BUY').astype(int)

        # Evaluate
        from src.backtest.evaluator import evaluate_backtest
        metrics = evaluate_backtest(predictions_df, test_returns)

        return predictions_df, metrics

    except Exception as e:
        logger.error(f"Error in split backtest: {str(e)}")
        return pd.DataFrame(), {}
```

**Update `run_walk_forward_backtest()` function**:

Replace lines 110-150 with:

```python
        # Load ticker list
        ticker_file = config.get('backtesting', {}).get('ticker_file', 'data/universe/tradeable_stocks.txt')
        with open(ticker_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(tickers)} tickers for backtest")

        # Collect all predictions
        all_predictions = []
        all_metrics = []

        # Run each split
        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            logger.info(f"")
            logger.info(f"Split {i+1}/{len(splits)}:")
            logger.info(f"  Train: {train_start} to {train_end}")
            logger.info(f"  Test: {test_start} to {test_end}")

            # Load data for this split
            train_features, train_labels, test_features, test_returns = load_split_data(
                tickers, train_start, train_end, test_start, test_end
            )

            if len(test_features) == 0:
                logger.warning(f"  No test data for split {i+1}, skipping")
                continue

            # Option 1: Use pre-trained models (faster)
            # Option 2: Retrain on this split's training data (more realistic)
            # For now, using pre-trained models

            # Run backtest on test period
            predictions_df, metrics = run_split_backtest(
                ensemble, test_features, test_returns, config
            )

            if len(predictions_df) > 0:
                predictions_df['split'] = i + 1
                all_predictions.append(predictions_df)
                all_metrics.append(metrics)

                logger.info(f"  Split {i+1} metrics:")
                logger.info(f"    Trades: {metrics.get('total_trades', 0)}")
                logger.info(f"    Win Rate: {metrics.get('win_rate', 0):.2%}")
                logger.info(f"    Avg Return: {metrics.get('avg_return', 0):.2%}")

        # Combine all out-of-sample predictions
        logger.info("="*70)
        logger.info("Combining all out-of-sample results...")

        if len(all_predictions) == 0:
            logger.error("No predictions generated - check data availability")
            return {}

        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_returns = combined_predictions[['ticker', 'date', 'actual_return', 'label']]

        # Final evaluation
        from src.backtest.evaluator import evaluate_backtest
        from src.backtest.statistics import validate_backtest
        from src.backtest.transaction_costs import calculate_total_costs, analyze_cost_impact

        final_metrics = evaluate_backtest(combined_predictions, combined_returns)

        # Apply transaction costs
        asset_type = 'stocks'  # TODO: detect from config
        combined_predictions = calculate_total_costs(
            combined_predictions, asset_type, config, include_slippage=True
        )

        # Analyze cost impact
        cost_metrics = analyze_cost_impact(combined_predictions)

        # Statistical validation
        returns_series = pd.Series(combined_predictions['net_return'].values)
        validation_results = validate_backtest(
            returns_series,
            config,
            n_trials=100,  # Assume tested 100 variations
            in_sample_returns=None,  # TODO: pass IS returns for PBO
            out_sample_returns=None
        )

        logger.info("Walk-forward backtest complete")
        logger.info("="*70)

        # Combine results
        results = {
            'n_splits': len(splits),
            'splits': splits,
            'total_trades': final_metrics.get('total_trades', 0),
            'metrics': final_metrics,
            'cost_metrics': cost_metrics,
            'validation': validation_results,
            'predictions': combined_predictions,
            'all_checks_pass': validation_results.get('all_checks_pass', False)
        }

        # Save results
        output_dir = Path('data/backtests/latest')
        output_dir.mkdir(parents=True, exist_ok=True)

        combined_predictions.to_csv(output_dir / 'predictions.csv', index=False)

        import json
        with open(output_dir / 'results.json', 'w') as f:
            # Can't serialize DataFrames, so exclude predictions
            results_to_save = {k: v for k, v in results.items() if k != 'predictions'}
            json.dump(results_to_save, f, indent=2, default=str)

        return results
```

**Testing**:
```bash
# After implementing, test with:
python src/backtest/walk_forward.py --config config.yaml
```

**Estimated Time**: 1-2 days (200-300 lines of code)

---

### Task 1.2: Build fundamental_features.py (Day 3)
**Priority**: HIGH - Models 3 & 4 need these features

**File**: `src/features/fundamental_features.py`

**What to Build**:

```python
"""
Fundamental features - financial ratios and metrics from SEC filings.
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_fundamental_features(
    fundamentals_df: pd.DataFrame,
    price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute fundamental features from SEC filing data.

    Args:
        fundamentals_df: DataFrame with fundamental data (from SEC EDGAR)
        price_df: DataFrame with price data (for P/E, P/B calculations)

    Returns:
        DataFrame with fundamental features merged with price data
    """
    try:
        if len(fundamentals_df) == 0 or len(price_df) == 0:
            return price_df

        # Ensure date columns are datetime
        fundamentals_df['filing_date'] = pd.to_datetime(fundamentals_df['filing_date'])
        fundamentals_df['period_end_date'] = pd.to_datetime(fundamentals_df['period_end_date'])
        price_df['date'] = pd.to_datetime(price_df['date'])

        # Forward-fill fundamentals (use most recent filing as of each date)
        # This ensures point-in-time correctness

        result_rows = []

        for _, price_row in price_df.iterrows():
            price_date = price_row['date']

            # Get most recent filing as of this date
            available_filings = fundamentals_df[fundamentals_df['filing_date'] <= price_date]

            if len(available_filings) == 0:
                # No filings available yet
                continue

            # Get most recent filing
            latest_filing = available_filings.iloc[-1]

            # Calculate features
            features = {}

            # Valuation Ratios (require price data)
            market_cap = price_row.get('close', 0) * latest_filing.get('shares_outstanding', 0)

            if latest_filing.get('net_income', 0) > 0:
                features['pe_ratio'] = market_cap / latest_filing['net_income']
            else:
                features['pe_ratio'] = np.nan

            if latest_filing.get('total_equity', 0) > 0:
                features['pb_ratio'] = market_cap / latest_filing['total_equity']
            else:
                features['pb_ratio'] = np.nan

            if latest_filing.get('revenue', 0) > 0:
                features['ps_ratio'] = market_cap / latest_filing['revenue']
            else:
                features['ps_ratio'] = np.nan

            # PEG ratio (P/E to growth)
            # Would need earnings growth rate - calculate from historical filings

            # Profitability Metrics
            if latest_filing.get('total_equity', 0) > 0:
                features['roe'] = latest_filing.get('net_income', 0) / latest_filing['total_equity']
            else:
                features['roe'] = np.nan

            if latest_filing.get('total_assets', 0) > 0:
                features['roa'] = latest_filing.get('net_income', 0) / latest_filing['total_assets']
            else:
                features['roa'] = np.nan

            if latest_filing.get('revenue', 0) > 0:
                features['profit_margin'] = latest_filing.get('net_income', 0) / latest_filing['revenue']
                features['operating_margin'] = latest_filing.get('operating_income', 0) / latest_filing['revenue']
            else:
                features['profit_margin'] = np.nan
                features['operating_margin'] = np.nan

            # Quality Metrics
            if latest_filing.get('total_equity', 0) > 0:
                total_debt = latest_filing.get('total_liabilities', 0) - latest_filing.get('current_liabilities', 0)
                features['debt_to_equity'] = total_debt / latest_filing['total_equity']
            else:
                features['debt_to_equity'] = np.nan

            if latest_filing.get('current_liabilities', 0) > 0:
                features['current_ratio'] = latest_filing.get('current_assets', 0) / latest_filing['current_liabilities']
            else:
                features['current_ratio'] = np.nan

            # Growth Metrics (need historical filings)
            # EPS growth YoY, Revenue growth YoY
            # Would calculate by comparing to filing from 4 quarters ago

            # Combine with price data
            row = price_row.to_dict()
            row.update(features)
            result_rows.append(row)

        result_df = pd.DataFrame(result_rows)

        logger.debug(f"Computed fundamental features: {len(result_df)} rows")

        return result_df

    except Exception as e:
        logger.error(f"Error computing fundamental features: {str(e)}")
        return price_df


def calculate_growth_metrics(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate growth metrics (EPS growth, revenue growth) from time series of filings.

    Args:
        fundamentals_df: DataFrame with fundamental data sorted by date

    Returns:
        DataFrame with growth metrics added
    """
    try:
        df = fundamentals_df.copy()
        df = df.sort_values('filing_date')

        # EPS growth YoY (compare to 4 quarters ago)
        df['eps_growth_yoy'] = df['eps'].pct_change(periods=4)

        # Revenue growth YoY
        df['revenue_growth_yoy'] = df['revenue'].pct_change(periods=4)

        # Net income growth YoY
        df['net_income_growth_yoy'] = df['net_income'].pct_change(periods=4)

        # QoQ growth
        df['eps_growth_qoq'] = df['eps'].pct_change(periods=1)
        df['revenue_growth_qoq'] = df['revenue'].pct_change(periods=1)

        return df

    except Exception as e:
        logger.error(f"Error calculating growth metrics: {str(e)}")
        return fundamentals_df
```

**Integration**:
Update `src/features/feature_pipeline.py`:

```python
from src.features.fundamental_features import compute_fundamental_features

def run_pipeline_for_ticker(...):
    # After technical features:
    df = compute_technical_features(df)

    # Add fundamental features
    fundamental_file = Path(f'data/raw/fundamentals/{ticker}_fundamentals.parquet')
    if fundamental_file.exists():
        fundamentals_df = pd.read_parquet(fundamental_file)
        df = compute_fundamental_features(fundamentals_df, df)
```

**Testing**:
```bash
# Test with single ticker:
python -c "
from src.features.fundamental_features import compute_fundamental_features
import pandas as pd

price_df = pd.read_parquet('data/raw/stocks/AAPL.parquet')
fund_df = pd.read_parquet('data/raw/fundamentals/AAPL_fundamentals.parquet')

result = compute_fundamental_features(fund_df, price_df)
print(result.columns)
print(result[['date', 'close', 'pe_ratio', 'roe']].tail())
"
```

**Estimated Time**: 1 day (200-300 lines)

---

### Task 1.3: Build sentiment_features.py (Day 4)
**Priority**: HIGH - Model 3 needs these

**File**: `src/features/sentiment_features.py`

```python
"""
Sentiment features - Reddit mentions and Google Trends.
"""

import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_sentiment_features(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute sentiment features from Reddit/Trends data.

    Args:
        sentiment_df: DataFrame with sentiment data
        price_df: DataFrame with price data

    Returns:
        DataFrame with sentiment features merged
    """
    try:
        if len(sentiment_df) == 0 or len(price_df) == 0:
            return price_df

        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        price_df['date'] = pd.to_datetime(price_df['date'])

        # Aggregate sentiment by date (if multiple readings per day)
        daily_sentiment = sentiment_df.groupby('date').agg({
            'mentions_7d': 'last',
            'mentions_30d': 'last',
            'avg_sentiment': 'mean',
            'total_score': 'sum',
            'total_comments': 'sum',
            'current_interest': 'last',  # Google Trends
            'trend_momentum': 'last'
        }).reset_index()

        # Calculate mention velocity (change over time)
        daily_sentiment = daily_sentiment.sort_values('date')
        daily_sentiment['mentions_velocity'] = daily_sentiment['mentions_7d'].pct_change(periods=1)

        # Sentiment momentum
        daily_sentiment['sentiment_ma_7'] = daily_sentiment['avg_sentiment'].rolling(7).mean()
        daily_sentiment['sentiment_change'] = daily_sentiment['avg_sentiment'] - daily_sentiment['sentiment_ma_7']

        # Merge with price data
        result_df = pd.merge(price_df, daily_sentiment, on='date', how='left')

        # Forward-fill sentiment (use most recent value)
        sentiment_cols = ['mentions_7d', 'mentions_30d', 'avg_sentiment', 'mentions_velocity',
                         'current_interest', 'trend_momentum', 'sentiment_change']
        result_df[sentiment_cols] = result_df[sentiment_cols].fillna(method='ffill')

        # Fill remaining NaNs with 0 (no mentions = 0)
        result_df[sentiment_cols] = result_df[sentiment_cols].fillna(0)

        logger.debug(f"Computed sentiment features: {len(result_df)} rows")

        return result_df

    except Exception as e:
        logger.error(f"Error computing sentiment features: {str(e)}")
        return price_df
```

**Integration**: Update `feature_pipeline.py` similarly

**Estimated Time**: 1 day (150-200 lines)

---

### Task 1.4: Integrate into feature_pipeline.py (2 hours)

Update `src/features/feature_pipeline.py`:

```python
from src.features.technical_features import compute_technical_features
from src.features.fundamental_features import compute_fundamental_features
from src.features.sentiment_features import compute_sentiment_features

def run_pipeline_for_ticker(ticker, asset_type='stocks', output_dir='data/processed/features'):
    # Load raw OHLCV
    if asset_type == 'stocks':
        raw_file = Path(f'data/raw/stocks/{ticker}.parquet')
    else:
        raw_file = Path(f'data/raw/crypto/{ticker}_USDT.parquet')

    df = pd.read_parquet(raw_file)

    # 1. Technical features
    df = compute_technical_features(df)

    # 2. Fundamental features (stocks only)
    if asset_type == 'stocks':
        fund_file = Path(f'data/raw/fundamentals/{ticker}_fundamentals.parquet')
        if fund_file.exists():
            fund_df = pd.read_parquet(fund_file)
            df = compute_fundamental_features(fund_df, df)
        else:
            logger.warning(f"No fundamentals for {ticker}")

    # 3. Sentiment features
    sent_file = Path(f'data/raw/sentiment/{ticker}_sentiment.parquet')
    if sent_file.exists():
        sent_df = pd.read_parquet(sent_file)
        df = compute_sentiment_features(sent_df, df)
    else:
        logger.debug(f"No sentiment data for {ticker}")

    # Drop NaNs and save
    df = df.dropna()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path / f"{ticker}_features.parquet", index=False)

    return True
```

---

## 📋 PHASE 2: END-TO-END TESTING (Day 5)

### Task 2.1: Test Data Collection
```bash
# Download small sample for testing
python src/data_collection/stock_collector.py \
    --tickers data/test_tickers.txt \
    --start-date 2023-01-01 \
    --output data/raw/stocks

# Create test_tickers.txt with 5-10 tickers:
echo "AAPL
MSFT
GOOGL
TSLA
NVDA" > data/test_tickers.txt
```

### Task 2.2: Test Feature Engineering
```bash
python src/features/feature_pipeline.py \
    --tickers data/test_tickers.txt \
    --asset-type stocks \
    --output data/processed/features
```

**Checkpoint**: Verify feature files exist and have all column types:
- Technical: RSI, MACD, ADX, etc.
- Fundamental: P/E, ROE, margins (if available)
- Sentiment: mentions, sentiment scores (if available)

### Task 2.3: Generate Labels
Create `scripts/generate_labels.py`:

```python
from src.labeling.label_generator import triple_barrier_labeling
import pandas as pd
from pathlib import Path

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

for ticker in tickers:
    features_file = Path(f'data/processed/features/{ticker}_features.parquet')
    if not features_file.exists():
        continue

    df = pd.read_parquet(features_file)

    # Generate labels
    labels = triple_barrier_labeling(
        df['close'],
        profit_target_pct=0.20,  # 20% profit target
        stop_loss_pct=0.10,      # 10% stop loss
        timeframe_days=30        # 30 day timeframe
    )

    # Add to dataframe
    df['label'] = labels

    # Calculate actual returns
    df['actual_return'] = df['close'].pct_change(periods=30).shift(-30)

    # Save
    labels_dir = Path('data/processed/labels')
    labels_dir.mkdir(parents=True, exist_ok=True)

    df[['date', 'ticker', 'label', 'actual_return']].to_parquet(
        labels_dir / f'{ticker}_labels.parquet',
        index=False
    )
```

**Run**:
```bash
python scripts/generate_labels.py
```

---

## 📋 PHASE 3: FIRST BACKTEST (Day 6)

### Task 3.1: Train Models on Sample
```bash
python src/models/train.py \
    --tickers data/test_tickers.txt \
    --asset-type stocks \
    --config config.yaml
```

**Checkpoint**: Verify model files created in `models/trained/`

### Task 3.2: Run Walk-Forward Backtest (Sample Period)

Update `config.yaml`:
```yaml
backtesting:
  train_window_years: 1
  test_window_months: 3
  step_size_months: 3
  ticker_file: "data/test_tickers.txt"

  validation_requirements:
    min_t_statistic: 3.0
    max_pbo: 0.30
    min_trades: 10  # Lower for test
```

**Run**:
```bash
python src/backtest/walk_forward.py --config config.yaml
```

**Expected Output**:
```
==================================================
WALK-FORWARD BACKTEST
==================================================
Created 4 walk-forward splits

Split 1/4:
  Train: 2023-01-01 to 2024-01-01
  Test: 2024-01-01 to 2024-04-01
Loaded train: 500 rows, test: 150 rows
  Split 1 metrics:
    Trades: 8
    Win Rate: 50.00%
    Avg Return: 2.5%

...

BACKTEST EVALUATION RESULTS
==================================================
Total Trades: 32
Win Rate: 46.88%
Total Return: 15.2%
Sharpe Ratio: 0.85
t-statistic: 1.8 (require > 3.0)

STATISTICAL VALIDATION
==================================================
t-statistic: 1.8 (require > 3.0)
  ⚠️  FAILED - High risk of overfitting
Deflated Sharpe Ratio: 0.4 (require > 1.0)
  ⚠️  FAILED - Performance likely due to chance

❌ VALIDATION FAILED - Strategy does not meet requirements
```

**Don't Panic if Validation Fails!**
This is expected on a small sample. The point is to verify the PIPELINE works, not to have a profitable strategy yet.

---

## 📋 PHASE 4: FULL BACKTEST (Day 7+)

### Only proceed if Phase 3 completes without errors

### Task 4.1: Download Full Data
```bash
# Use full S&P 500
bash scripts/download_all_data.sh
```

### Task 4.2: Generate Features for All Tickers
```bash
python src/features/feature_pipeline.py \
    --tickers data/universe/tradeable_stocks.txt \
    --asset-type stocks
```

### Task 4.3: Generate Labels for All
Create script to generate labels for all tickers

### Task 4.4: Train on Full Data
```bash
python src/models/train.py \
    --tickers data/universe/tradeable_stocks.txt \
    --asset-type stocks
```

### Task 4.5: Run Full Backtest
Update `config.yaml`:
```yaml
data:
  date_ranges:
    training_start: "2015-01-01"
    training_end: "2024-01-01"
    test_start: "2024-01-01"
    test_end: "2025-11-06"

backtesting:
  train_window_years: 3
  test_window_months: 6
  step_size_months: 3
  ticker_file: "data/universe/tradeable_stocks.txt"
```

**Run**:
```bash
python src/backtest/walk_forward.py --config config.yaml
```

### Task 4.6: Analyze Results

**If Validation Passes** (t-stat > 3.0, PBO < 0.30, Deflated Sharpe > 1.0):
- 🎉 Congratulations! Strategy is validated
- Next step: Build paper_trading.py
- Paper trade for 3-6 months before considering live

**If Validation Fails** (likely):
- This is NORMAL and GOOD (found issues before losing money)
- Analyze which splits performed well vs poorly
- Look at feature importance
- Consider:
  - Different features
  - Different models
  - Different parameters
  - Different target (profit_target_pct, stop_loss_pct, timeframe)
- Iterate and retest

---

## 🎯 SUCCESS CRITERIA BY PHASE

### Phase 1 Success:
- ✅ walk_forward.py runs without errors (even if results are bad)
- ✅ fundamental_features.py creates features
- ✅ sentiment_features.py creates features
- ✅ feature_pipeline.py includes all feature types

### Phase 2 Success:
- ✅ Can download data for test tickers
- ✅ Can generate features with all types
- ✅ Can generate labels
- ✅ All files created successfully

### Phase 3 Success:
- ✅ Models train without errors
- ✅ Backtest runs without errors
- ✅ Gets results (even if validation fails)
- ✅ Pipeline is complete

### Phase 4 Success (The Real Test):
- ✅ t-statistic > 3.0
- ✅ PBO < 0.30
- ✅ Deflated Sharpe > 1.0
- ✅ Profitable at 2X transaction costs
- ✅ Min 50+ trades
- ✅ Win rate > 45%

---

## 🚨 WHAT TO DO IF THINGS FAIL

### If walk_forward.py throws errors:
1. Check data files exist
2. Check date ranges are valid
3. Check ticker files exist
4. Add more logging
5. Test with single ticker first

### If features have too many NaN:
- Adjust dropna() threshold
- Fill missing values strategically
- Check raw data quality

### If validation fails:
- **Don't be discouraged!** Most strategies fail validation
- This means the validation is working (catching overfitting)
- Iterate on:
  - Features (add more predictive features)
  - Models (try different algorithms)
  - Parameters (profit target, stop loss, timeframe)
  - Universe (maybe focus on high-volume stocks)

### If can't get 50+ trades:
- Reduce profit target
- Increase timeframe
- Add more tickers
- Adjust model thresholds

---

## 📊 ESTIMATED TIMELINE

**Days 1-2**: Implement walk_forward.py (200-300 lines)
**Day 3**: Build fundamental_features.py (200-300 lines)
**Day 4**: Build sentiment_features.py (150-200 lines)
**Day 5**: End-to-end testing with sample data
**Day 6**: First backtest on sample period
**Day 7**: Full data download and feature generation
**Day 8**: Full backtest on historical data
**Day 9-10**: Analysis and iteration if validation fails

**Total**: 7-10 days to first complete backtest

---

## 💡 TIPS FOR SUCCESS

1. **Start Small**: Test with 5 tickers before 500
2. **Save Often**: Commit after each working component
3. **Log Everything**: Add logger statements to debug
4. **Test Incrementally**: Don't wait until end to test
5. **Expect Failure**: Validation is supposed to be hard
6. **Be Patient**: Good strategies take time to develop

---

## 🎯 YOUR IMMEDIATE NEXT STEP

**RIGHT NOW, start with Task 1.1**:

```bash
# Open the file:
code src/backtest/walk_forward.py

# Add the two functions:
# 1. load_split_data()
# 2. run_split_backtest()

# Update run_walk_forward_backtest() to use them

# Test with:
python src/backtest/walk_forward.py --config config.yaml
```

Start there, and work through the tasks in order. Each task builds on the previous one.

Would you like me to start implementing Task 1.1 (walk_forward.py) right now?
