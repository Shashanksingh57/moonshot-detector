"""
Label generator using triple barrier method.
Generates training labels for explosive move prediction (20-50%+ gains).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
from src.utils.logging_config import setup_logging

logger = setup_logging('labeling')


def triple_barrier_labeling(
    prices: pd.Series,
    profit_target_pct: float,
    stop_loss_pct: float,
    timeframe_days: int
) -> pd.Series:
    """
    Apply triple barrier method for labeling.

    Args:
        prices: Series of closing prices
        profit_target_pct: Profit target in percentage (e.g., 20 for +20%)
        stop_loss_pct: Stop loss in percentage (e.g., 10 for -10%)
        timeframe_days: Maximum holding period in days

    Returns:
        Series of labels (1 = hit profit target first, 0 = hit stop loss or timeout)
    """
    labels = pd.Series(index=prices.index, dtype=int)

    for i in range(len(prices) - timeframe_days):
        entry_price = prices.iloc[i]

        if pd.isna(entry_price) or entry_price <= 0:
            labels.iloc[i] = 0
            continue

        # Define barriers
        profit_barrier = entry_price * (1 + profit_target_pct / 100)
        stop_loss_barrier = entry_price * (1 - stop_loss_pct / 100)

        # Look forward up to timeframe_days
        future_prices = prices.iloc[i+1:i+1+timeframe_days]

        # Check which barrier is hit first
        hit_profit = False
        hit_stop = False

        for price in future_prices:
            if pd.isna(price):
                continue

            if price >= profit_barrier:
                hit_profit = True
                break
            elif price <= stop_loss_barrier:
                hit_stop = True
                break

        # Label: 1 if profit hit first, 0 otherwise
        labels.iloc[i] = 1 if hit_profit and not hit_stop else 0

    return labels


def generate_labels_for_ticker(
    ticker: str,
    model_config: dict,
    asset_type: str = 'stocks'
) -> bool:
    """
    Generate labels for one ticker based on model configuration.

    Args:
        ticker: Ticker symbol
        model_config: Model configuration with target parameters
        asset_type: 'stocks' or 'crypto'

    Returns:
        True if successful
    """
    try:
        # Load raw price data
        if asset_type == 'stocks':
            raw_file = Path(f'data/raw/stocks/{ticker}.parquet')
        else:
            raw_file = Path(f'data/raw/crypto/{ticker}_USDT.parquet')

        if not raw_file.exists():
            logger.warning(f"Raw data not found for {ticker}")
            return False

        df = pd.read_parquet(raw_file)

        if 'close' not in df.columns:
            logger.error(f"No 'close' column in {ticker} data")
            return False

        # Extract target configuration
        target_config = model_config.get('target', {})
        profit_target_pct = target_config.get('profit_target_pct', 20)
        stop_loss_pct = target_config.get('stop_loss_pct', 10)
        timeframe_days = target_config.get('timeframe_days', 30)

        # Generate labels
        df['label'] = triple_barrier_labeling(
            df['close'],
            profit_target_pct,
            stop_loss_pct,
            timeframe_days
        )

        # Keep only date, ticker, and label
        df_labels = df[['date', 'label']].copy()
        df_labels['ticker'] = ticker

        # Save labels
        model_name = model_config.get('name', 'model').replace(' ', '_').lower()
        output_dir = Path('data/processed/labels')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{ticker}_{model_name}_labels.parquet"
        df_labels.to_parquet(output_file, index=False)

        logger.debug(f"Generated labels for {ticker}: {df_labels['label'].sum()} positive samples")

        return True

    except Exception as e:
        logger.error(f"Error generating labels for {ticker}: {str(e)}")
        return False


def generate_all_labels(
    ticker_file: str,
    config_file: str = 'config.yaml',
    asset_type: str = 'stocks'
) -> None:
    """
    Generate labels for all tickers and all models.

    Args:
        ticker_file: Path to ticker list file
        config_file: Path to config file
        asset_type: 'stocks' or 'crypto'
    """
    logger.info(f"Starting label generation for {asset_type}...")

    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Load ticker list
    with open(ticker_file, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    logger.info(f"Generating labels for {len(tickers)} tickers...")

    # Generate labels for each model
    models_to_label = ['model_1_breakout', 'model_3_pead_squeeze']

    for model_key in models_to_label:
        model_config = config['models'].get(model_key, {})

        if not model_config.get('enabled', False):
            logger.info(f"Skipping {model_key} (disabled)")
            continue

        logger.info(f"Generating labels for {model_key}...")

        success_count = 0
        fail_count = 0

        pbar = tqdm(tickers, desc=f"Labeling {model_key}", unit="ticker")

        for ticker in pbar:
            pbar.set_postfix({'ticker': ticker, 'success': success_count, 'failed': fail_count})

            if generate_labels_for_ticker(ticker, model_config, asset_type):
                success_count += 1
            else:
                fail_count += 1

        logger.info(f"{model_key}: Success: {success_count}, Failed: {fail_count}")


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Generate training labels')

    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Path to ticker list file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--asset-type',
        type=str,
        default='stocks',
        choices=['stocks', 'crypto'],
        help='Asset type'
    )

    args = parser.parse_args()

    generate_all_labels(args.tickers, args.config, args.asset_type)


if __name__ == '__main__':
    main()
