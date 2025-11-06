"""
Signal generator - generates live trading signals using ensemble.
Loads latest data, generates predictions, and outputs actionable signals.
"""

import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import argparse

from src.models.ensemble import EnsembleVoter
from src.utils.logging_config import setup_logging

logger = setup_logging('inference')


def load_latest_features(ticker: str, asset_type: str = 'stocks') -> pd.DataFrame:
    """
    Load latest features for a ticker.

    Args:
        ticker: Ticker symbol
        asset_type: 'stocks' or 'crypto'

    Returns:
        DataFrame with features
    """
    try:
        features_file = Path(f'data/processed/features/{ticker}_features.parquet')

        if not features_file.exists():
            logger.warning(f"Features not found for {ticker}")
            return pd.DataFrame()

        df = pd.read_parquet(features_file)

        # Return last N rows for recency
        return df.tail(100)  # Keep last 100 days for context

    except Exception as e:
        logger.error(f"Error loading features for {ticker}: {str(e)}")
        return pd.DataFrame()


def generate_signals(
    ticker_file: str,
    asset_type: str = 'stocks',
    config_path: str = 'config.yaml',
    output_file: str = None
) -> pd.DataFrame:
    """
    Generate trading signals for all tickers.

    Args:
        ticker_file: Path to ticker list file
        asset_type: 'stocks' or 'crypto'
        config_path: Path to config file
        output_file: Optional path to save signals

    Returns:
        DataFrame with signals
    """
    try:
        logger.info(f"Generating signals for {asset_type}...")

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load tickers
        with open(ticker_file, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]

        logger.info(f"Loaded {len(tickers)} tickers")

        # Load features for all tickers
        features = {}
        for ticker in tickers:
            df = load_latest_features(ticker, asset_type)
            if len(df) > 0:
                features[ticker] = df

        logger.info(f"Loaded features for {len(features)} tickers")

        # Initialize ensemble
        ensemble = EnsembleVoter(config)
        ensemble.load_all_models()

        # Generate signals
        signals_df = ensemble.generate_signals(asset_type, list(features.keys()), features)

        if len(signals_df) == 0:
            logger.info("No signals generated")
            return pd.DataFrame()

        # Add timestamp
        signals_df['generated_at'] = datetime.now()

        # Save if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            signals_df.to_csv(output_path, index=False)
            logger.info(f"Signals saved to {output_path}")

        # Log summary
        logger.info("="*70)
        logger.info("SIGNAL SUMMARY")
        logger.info("="*70)
        logger.info(f"Total signals: {len(signals_df)}")
        logger.info(f"High confidence: {(signals_df['confidence'] == 'high').sum()}")
        logger.info(f"Medium confidence: {(signals_df['confidence'] == 'medium').sum()}")
        logger.info(f"Low confidence: {(signals_df['confidence'] == 'low').sum()}")
        logger.info("="*70)

        # Log top signals
        if len(signals_df) > 0:
            logger.info("Top 10 signals:")
            for idx, row in signals_df.head(10).iterrows():
                logger.info(f"  {row['ticker']:6s} | Confidence: {row['confidence']:6s} | Score: {row['weighted_score']:.3f}")

        return signals_df

    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        return pd.DataFrame()


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Generate trading signals')

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
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/signals/latest_signals.csv',
        help='Output file for signals'
    )

    args = parser.parse_args()

    # Generate signals
    signals = generate_signals(
        args.tickers,
        args.asset_type,
        args.config,
        args.output
    )

    if len(signals) > 0:
        logger.info(f"Generated {len(signals)} signals")
    else:
        logger.info("No signals generated")


if __name__ == '__main__':
    main()
