"""
Short interest data collector using FINRA data (FREE).
Scrapes and parses FINRA short interest reports.
"""

import pandas as pd
from typing import List
from pathlib import Path
import argparse
import requests
from bs4 import BeautifulSoup
from src.utils.logging_config import setup_logging
from src.utils.finra_parser import FINRAParser

logger = setup_logging('data_collection')


def download_finra_short_interest(
    output_dir: str = "data/raw/short_interest"
) -> bool:
    """
    Download FINRA short interest data.

    Note: This is a simplified version. In production, you would:
    1. Identify the correct FINRA data file URLs
    2. Download them programmatically
    3. Parse the data

    For now, this function provides a template.

    Args:
        output_dir: Output directory

    Returns:
        True if successful
    """
    try:
        logger.info("Downloading FINRA short interest data...")
        logger.info("Note: FINRA data requires manual download or API access")
        logger.info("Visit: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # TODO: Implement actual FINRA data download
        # For now, create placeholder
        logger.warning("FINRA download not fully implemented - requires manual data download")

        return True

    except Exception as e:
        logger.error(f"Error downloading FINRA data: {str(e)}")
        return False


def parse_finra_files(
    input_dir: str,
    output_dir: str = "data/raw/short_interest"
) -> bool:
    """
    Parse FINRA short interest files.

    Args:
        input_dir: Directory containing FINRA data files
        output_dir: Output directory

    Returns:
        True if successful
    """
    try:
        parser = FINRAParser()

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all FINRA data files
        data_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.csv"))

        if not data_files:
            logger.warning(f"No FINRA data files found in {input_dir}")
            return False

        all_data = []

        for file_path in data_files:
            logger.info(f"Parsing {file_path.name}...")
            df = parser.parse_short_interest_file(str(file_path))
            if df is not None and len(df) > 0:
                all_data.append(df)

        if not all_data:
            logger.error("No data parsed from FINRA files")
            return False

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['date', 'symbol'])

        # Sort by date
        combined_df = combined_df.sort_values(['symbol', 'date'])

        # Save to parquet
        output_file = output_path / "short_interest.parquet"
        combined_df.to_parquet(output_file, index=False)

        logger.info(f"Saved combined short interest data to {output_file}")
        logger.info(f"Total records: {len(combined_df)}")

        return True

    except Exception as e:
        logger.error(f"Error parsing FINRA files: {str(e)}")
        return False


def main():
    """Main function for CLI."""
    parser_cli = argparse.ArgumentParser(description='Download and parse FINRA short interest data')

    parser_cli.add_argument(
        '--input-dir',
        type=str,
        default='data/raw/short_interest/raw',
        help='Directory containing downloaded FINRA files'
    )

    parser_cli.add_argument(
        '--output',
        type=str,
        default='data/raw/short_interest',
        help='Output directory'
    )

    args = parser_cli.parse_args()

    # Parse FINRA files
    parse_finra_files(args.input_dir, args.output)


if __name__ == '__main__':
    main()
