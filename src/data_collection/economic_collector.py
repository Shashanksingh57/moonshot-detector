"""
Economic data collector using FRED API (FREE).
Downloads economic indicators like VIX, Treasury rates, unemployment, etc.
"""

import pandas as pd
from fredapi import Fred
from typing import List, Dict, Optional
from pathlib import Path
import argparse
import os
from dotenv import load_dotenv
from src.utils.logging_config import setup_logging

logger = setup_logging('data_collection')
load_dotenv()


def download_fred_series(
    series_id: str,
    start_date: str,
    api_key: Optional[str] = None
) -> Optional[pd.Series]:
    """
    Download a FRED economic data series.

    Args:
        series_id: FRED series ID (e.g., 'VIXCLS', 'DGS10')
        start_date: Start date (YYYY-MM-DD)
        api_key: FRED API key (or from environment)

    Returns:
        Pandas Series with the data, or None if failed
    """
    try:
        if api_key is None:
            api_key = os.getenv('FRED_API_KEY')

        if not api_key:
            logger.error("FRED API key not found. Set FRED_API_KEY environment variable.")
            logger.info("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            return None

        fred = Fred(api_key=api_key)

        # Download series
        series = fred.get_series(series_id, observation_start=start_date)

        if series is None or len(series) == 0:
            logger.warning(f"No data returned for series {series_id}")
            return None

        logger.debug(f"Downloaded {len(series)} observations for {series_id}")

        return series

    except Exception as e:
        logger.error(f"Error downloading FRED series {series_id}: {str(e)}")
        return None


def download_all_fred_data(
    series_ids: List[str],
    start_date: str,
    output_file: str = "data/raw/economic/fred_data.parquet"
) -> bool:
    """
    Download multiple FRED series and save to a single parquet file.

    Args:
        series_ids: List of FRED series IDs
        start_date: Start date (YYYY-MM-DD)
        output_file: Output file path

    Returns:
        True if successful
    """
    try:
        logger.info(f"Downloading {len(series_ids)} FRED series...")

        series_dict = {}

        for series_id in series_ids:
            logger.info(f"Downloading {series_id}...")
            series = download_fred_series(series_id, start_date)

            if series is not None:
                series_dict[series_id] = series

        if not series_dict:
            logger.error("No series downloaded successfully")
            return False

        # Merge all series into a single DataFrame
        df = pd.DataFrame(series_dict)

        # Fill forward missing values (economic data often has gaps)
        df = df.fillna(method='ffill', limit=5)

        # Save to parquet
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path)

        logger.info(f"Saved FRED data to {output_path}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Series: {list(df.columns)}")

        return True

    except Exception as e:
        logger.error(f"Error downloading FRED data: {str(e)}")
        return False


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Download economic data from FRED')

    parser.add_argument(
        '--series',
        type=str,
        nargs='+',
        default=['VIXCLS', 'DGS10', 'UNRATE', 'GDP'],
        help='FRED series IDs to download'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2015-01-01',
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/economic/fred_data.parquet',
        help='Output file path'
    )

    args = parser.parse_args()

    # Download data
    download_all_fred_data(args.series, args.start_date, args.output)


if __name__ == '__main__':
    main()
