"""
Universe builder - filters and selects tradeable stocks and crypto.
Applies liquidity, market cap, and other filters from config.
"""

import pandas as pd
import yaml
from typing import List, Dict, Optional
from pathlib import Path
import argparse
from src.utils.logging_config import setup_logging

logger = setup_logging('data_collection')


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_universe_from_raw(asset_type: str) -> pd.DataFrame:
    """
    Load all tickers from raw data directory.

    Args:
        asset_type: 'stocks' or 'crypto'

    Returns:
        DataFrame with ticker metrics
    """
    try:
        if asset_type == 'stocks':
            data_dir = Path('data/raw/stocks')
        elif asset_type == 'crypto':
            data_dir = Path('data/raw/crypto')
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")

        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return pd.DataFrame()

        # Load all parquet files and calculate metrics
        ticker_metrics = []

        for file_path in data_dir.glob('*.parquet'):
            try:
                df = pd.read_parquet(file_path)

                if len(df) == 0:
                    continue

                # Extract ticker
                if asset_type == 'stocks':
                    ticker = file_path.stem
                else:  # crypto
                    ticker = file_path.stem.replace('_USDT', '')

                # Calculate metrics
                latest_price = df['close'].iloc[-1] if 'close' in df.columns else 0
                avg_volume = df['volume'].mean() if 'volume' in df.columns else 0

                # Estimate market cap (price * volume as proxy - not accurate but useful for filtering)
                market_cap_proxy = latest_price * avg_volume if latest_price > 0 else 0

                ticker_metrics.append({
                    'ticker': ticker,
                    'latest_price': latest_price,
                    'avg_volume': avg_volume,
                    'avg_volume_usd': latest_price * avg_volume,
                    'market_cap_proxy': market_cap_proxy,
                    'data_points': len(df)
                })

            except Exception as e:
                logger.warning(f"Error loading {file_path.name}: {str(e)}")
                continue

        if not ticker_metrics:
            logger.error(f"No ticker metrics loaded for {asset_type}")
            return pd.DataFrame()

        df = pd.DataFrame(ticker_metrics)
        logger.info(f"Loaded {len(df)} {asset_type} from raw data")

        return df

    except Exception as e:
        logger.error(f"Error loading universe for {asset_type}: {str(e)}")
        return pd.DataFrame()


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply filters to universe DataFrame.

    Args:
        df: DataFrame with ticker metrics
        filters: Filter criteria from config

    Returns:
        Filtered DataFrame
    """
    try:
        original_count = len(df)

        # Min price filter
        if 'min_price' in filters:
            df = df[df['latest_price'] >= filters['min_price']]
            logger.info(f"After min_price filter: {len(df)} tickers")

        # Min daily volume (USD) filter
        if 'min_daily_volume_usd' in filters:
            df = df[df['avg_volume_usd'] >= filters['min_daily_volume_usd']]
            logger.info(f"After min_volume filter: {len(df)} tickers")

        # Min market cap filter
        if 'min_market_cap_usd' in filters:
            df = df[df['market_cap_proxy'] >= filters['min_market_cap_usd']]
            logger.info(f"After min_market_cap filter: {len(df)} tickers")

        # Exclude sectors (would need sector data - placeholder for now)
        if 'exclude_sectors' in filters and filters['exclude_sectors']:
            logger.info(f"Sector exclusion not implemented - needs sector data")

        filtered_count = len(df)
        logger.info(f"Filtered {original_count} → {filtered_count} tickers ({filtered_count/original_count*100:.1f}% retained)")

        return df

    except Exception as e:
        logger.error(f"Error applying filters: {str(e)}")
        return df


def select_top_n_by_volume(df: pd.DataFrame, n: int) -> List[str]:
    """
    Select top N tickers by average volume.

    Args:
        df: DataFrame with ticker metrics
        n: Number of tickers to select

    Returns:
        List of top N tickers
    """
    df_sorted = df.sort_values('avg_volume_usd', ascending=False)
    top_tickers = df_sorted.head(n)['ticker'].tolist()

    logger.info(f"Selected top {len(top_tickers)} tickers by volume")

    return top_tickers


def save_tradeable_list(
    tickers: List[str],
    output_path: str,
    metadata_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Save tradeable ticker list to file.

    Args:
        tickers: List of ticker symbols
        output_path: Output file path (.txt)
        metadata_df: Optional metadata DataFrame to save as parquet
    """
    try:
        # Save ticker list
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write('\n'.join(tickers))

        logger.info(f"Saved {len(tickers)} tickers to {output_file}")

        # Save metadata if provided
        if metadata_df is not None:
            metadata_file = output_file.parent / f"{output_file.stem}_metadata.parquet"
            metadata_df[metadata_df['ticker'].isin(tickers)].to_parquet(metadata_file, index=False)
            logger.info(f"Saved metadata to {metadata_file}")

    except Exception as e:
        logger.error(f"Error saving tradeable list: {str(e)}")


def build_universe(asset_type: str, config: Dict) -> None:
    """
    Build tradeable universe for stocks or crypto.

    Args:
        asset_type: 'stocks' or 'crypto'
        config: Configuration dictionary
    """
    try:
        logger.info(f"Building {asset_type} universe...")

        # Load universe from raw data
        df = load_universe_from_raw(asset_type)

        if len(df) == 0:
            logger.error(f"No data loaded for {asset_type}")
            return

        # Apply filters
        filters = config['data']['universe'][asset_type]['filters']
        df_filtered = apply_filters(df, filters)

        if len(df_filtered) == 0:
            logger.error("No tickers passed filters")
            return

        # Select top N by volume
        max_tickers = config['data']['universe'][asset_type]['max_tickers']
        top_tickers = select_top_n_by_volume(df_filtered, max_tickers)

        # Save tradeable list
        output_file = f"data/universe/tradeable_{asset_type}.txt"
        save_tradeable_list(top_tickers, output_file, df_filtered)

        logger.info(f"Universe building complete for {asset_type}!")

    except Exception as e:
        logger.error(f"Error building universe for {asset_type}: {str(e)}")


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Build tradeable universe')

    parser.add_argument(
        '--asset-type',
        type=str,
        required=True,
        choices=['stocks', 'crypto'],
        help='Asset type to build universe for'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Build universe
    build_universe(args.asset_type, config)


if __name__ == '__main__':
    main()
