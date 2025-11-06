"""
Feature pipeline - orchestrates all feature engineering.
Loads raw data, computes features, and saves processed features.
"""

import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
from src.features.technical_features import compute_technical_features
from src.utils.logging_config import setup_logging

logger = setup_logging('feature_engineering')


def run_pipeline_for_ticker(
    ticker: str,
    asset_type: str = 'stocks',
    output_dir: str = 'data/processed/features'
) -> bool:
    """
    Run complete feature engineering pipeline for one ticker.

    Args:
        ticker: Ticker symbol
        asset_type: 'stocks' or 'crypto'
        output_dir: Output directory

    Returns:
        True if successful
    """
    try:
        # Load raw OHLCV data
        if asset_type == 'stocks':
            raw_file = Path(f'data/raw/stocks/{ticker}.parquet')
        else:
            raw_file = Path(f'data/raw/crypto/{ticker}_USDT.parquet')

        if not raw_file.exists():
            logger.warning(f"Raw data not found for {ticker}")
            return False

        df = pd.read_parquet(raw_file)

        if len(df) < 200:  # Need enough data for indicators
            logger.warning(f"{ticker} has insufficient data ({len(df)} rows)")
            return False

        # Compute technical features
        df = compute_technical_features(df)

        # TODO: Add fundamental and sentiment features here
        # For now, just technical features

        # Remove rows with NaN (from indicator calculation)
        initial_len = len(df)
        df = df.dropna()
        logger.debug(f"{ticker}: Dropped {initial_len - len(df)} rows with NaN")

        if len(df) == 0:
            logger.warning(f"{ticker}: No data after dropna")
            return False

        # Save processed features
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"{ticker}_features.parquet"
        df.to_parquet(output_file, index=False)

        logger.debug(f"Saved features for {ticker}: {len(df)} rows, {len(df.columns)} columns")

        return True

    except Exception as e:
        logger.error(f"Error in feature pipeline for {ticker}: {str(e)}")
        return False


def run_pipeline_for_all(
    ticker_file: str,
    asset_type: str = 'stocks',
    output_dir: str = 'data/processed/features'
) -> None:
    """
    Run feature pipeline for all tickers in a file.

    Args:
        ticker_file: Path to file with ticker list (one per line)
        asset_type: 'stocks' or 'crypto'
        output_dir: Output directory
    """
    logger.info(f"Starting feature pipeline for {asset_type}...")

    # Load ticker list
    with open(ticker_file, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    logger.info(f"Processing {len(tickers)} tickers...")

    success_count = 0
    fail_count = 0

    pbar = tqdm(tickers, desc="Feature engineering", unit="ticker")

    for ticker in pbar:
        pbar.set_postfix({'ticker': ticker, 'success': success_count, 'failed': fail_count})

        if run_pipeline_for_ticker(ticker, asset_type, output_dir):
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"Feature pipeline complete!")
    logger.info(f"Success: {success_count}, Failed: {fail_count}")


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')

    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Path to ticker list file'
    )

    parser.add_argument(
        '--asset-type',
        type=str,
        default='stocks',
        choices=['stocks', 'crypto'],
        help='Asset type'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/features',
        help='Output directory'
    )

    args = parser.parse_args()

    run_pipeline_for_all(args.tickers, args.asset_type, args.output)


if __name__ == '__main__':
    main()
