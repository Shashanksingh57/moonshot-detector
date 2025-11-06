"""
Stock data collector using yfinance (FREE).
Downloads OHLCV data for stocks with rate limiting and resume capability.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional
from pathlib import Path
import time
import argparse
from datetime import datetime
from tqdm import tqdm
from src.utils.logging_config import setup_logging
from src.utils.data_validation import check_data_freshness
from src.utils.data_quality import StockPriceValidator
from src.utils.validation_reports import ValidationReportManager

logger = setup_logging('data_collection')


def get_sp500_tickers() -> List[str]:
    """
    Download S&P 500 ticker list from Wikipedia.

    Returns:
        List of S&P 500 ticker symbols
    """
    try:
        logger.info("Downloading S&P 500 ticker list from Wikipedia...")

        # Wikipedia maintains an updated S&P 500 list
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]

        tickers = sp500_table['Symbol'].tolist()

        # Clean tickers (some have special characters)
        tickers = [ticker.replace('.', '-') for ticker in tickers]

        logger.info(f"Found {len(tickers)} S&P 500 tickers")

        return tickers

    except Exception as e:
        logger.error(f"Error downloading S&P 500 list: {str(e)}")
        return []


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    rate_limit_delay: float = 0.5,
    report_manager: Optional[ValidationReportManager] = None
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single stock using yfinance.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        rate_limit_delay: Delay between requests in seconds
        report_manager: Optional ValidationReportManager for quality checks

    Returns:
        DataFrame with OHLCV data, or None if failed validation
    """
    try:
        # Add delay for rate limiting
        time.sleep(rate_limit_delay)

        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)

        if df is None or len(df) == 0:
            logger.warning(f"No data returned for {ticker}")
            return None

        # Add ticker column
        df['Ticker'] = ticker

        # Reset index to make Date a column
        df = df.reset_index()

        # Rename columns to standard format
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'stock_splits',
            'Ticker': 'ticker'
        })

        # Select relevant columns
        columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        if 'dividends' in df.columns:
            columns_to_keep.append('dividends')
        if 'stock_splits' in df.columns:
            columns_to_keep.append('stock_splits')

        df = df[columns_to_keep]

        # Comprehensive validation
        validator = StockPriceValidator(df, ticker)
        is_valid, issues = validator.run_all_checks()

        # Save validation report if manager provided
        if report_manager:
            report_manager.save_validation_report(
                ticker=ticker,
                data_type='stock_price',
                issues=issues,
                is_valid=is_valid,
                metadata={'num_rows': len(df), 'date_range': f"{start_date} to {end_date or 'today'}"}
            )

        # Check for critical issues
        critical_issues = [i for i in issues if 'CRITICAL' in i or 'DATA ERROR' in i or 'IMPOSSIBLE' in i]
        if critical_issues:
            logger.error(f"{ticker}: {len(critical_issues)} CRITICAL data quality issues - REJECTING")
            return None

        # Accept with warnings
        if not is_valid and len(issues) > 0:
            logger.warning(f"{ticker}: Data quality warnings (no critical issues) - ACCEPTING")

        logger.debug(f"Downloaded {len(df)} rows for {ticker}")

        return df

    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {str(e)}")
        return None


def save_to_parquet(df: pd.DataFrame, ticker: str, output_dir: str = "data/raw/stocks") -> bool:
    """
    Save DataFrame to parquet file.

    Args:
        df: DataFrame to save
        ticker: Stock ticker symbol
        output_dir: Output directory path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save to parquet
        file_path = output_path / f"{ticker}.parquet"
        df.to_parquet(file_path, index=False, compression='snappy')

        logger.debug(f"Saved {ticker} to {file_path}")

        return True

    except Exception as e:
        logger.error(f"Error saving {ticker} to parquet: {str(e)}")
        return False


def load_ticker_list(file_path: str) -> List[str]:
    """
    Load ticker list from a text file (one ticker per line).

    Args:
        file_path: Path to ticker list file

    Returns:
        List of ticker symbols
    """
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]

        logger.info(f"Loaded {len(tickers)} tickers from {file_path}")

        return tickers

    except Exception as e:
        logger.error(f"Error loading ticker list from {file_path}: {str(e)}")
        return []


def download_all_stocks(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    output_dir: str = "data/raw/stocks",
    rate_limit_delay: float = 0.5,
    skip_existing: bool = True,
    max_age_days: int = 7
) -> None:
    """
    Download data for multiple stocks with progress bar.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
        rate_limit_delay: Delay between requests (seconds)
        skip_existing: Skip tickers with recent data files
        max_age_days: Max age of existing files to skip (days)
    """
    logger.info(f"Starting download for {len(tickers)} stocks...")
    logger.info(f"Date range: {start_date} to {end_date or 'today'}")

    success_count = 0
    skip_count = 0
    fail_count = 0
    validation_reject_count = 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize validation report manager
    report_manager = ValidationReportManager()

    # Progress bar
    pbar = tqdm(tickers, desc="Downloading stocks", unit="ticker")

    for ticker in pbar:
        pbar.set_postfix({
            'ticker': ticker,
            'success': success_count,
            'skipped': skip_count,
            'failed': fail_count,
            'rejected': validation_reject_count
        })

        # Check if file exists and is recent
        file_path = output_path / f"{ticker}.parquet"
        if skip_existing and file_path.exists():
            if check_data_freshness(str(file_path), max_age_days):
                logger.debug(f"Skipping {ticker} - recent data exists")
                skip_count += 1
                continue

        # Download data with validation
        df = download_stock_data(ticker, start_date, end_date, rate_limit_delay, report_manager)

        if df is not None and len(df) > 0:
            # Save to parquet
            if save_to_parquet(df, ticker, output_dir):
                success_count += 1
            else:
                fail_count += 1
        else:
            # Check if rejection was due to validation
            issues = report_manager.get_issues_for_ticker(ticker, 'stock_price')
            critical_issues = [i for i in issues if 'CRITICAL' in i or 'DATA ERROR' in i or 'IMPOSSIBLE' in i]
            if critical_issues:
                validation_reject_count += 1
            else:
                fail_count += 1

    logger.info(f"Download complete!")
    logger.info(f"Success: {success_count}, Skipped: {skip_count}, Failed: {fail_count}, Rejected (validation): {validation_reject_count}")

    # Print validation summary
    logger.info("")
    logger.info("Data Quality Summary:")
    report_manager.print_summary()


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Download stock data using yfinance')

    parser.add_argument(
        '--start-date',
        type=str,
        default='2015-01-01',
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), defaults to today'
    )

    parser.add_argument(
        '--tickers',
        type=str,
        default=None,
        help='Path to ticker list file (one per line)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/stocks',
        help='Output directory'
    )

    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.5,
        help='Delay between requests (seconds)'
    )

    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-download even if recent data exists'
    )

    parser.add_argument(
        '--sp500',
        action='store_true',
        help='Download S&P 500 tickers'
    )

    args = parser.parse_args()

    # Get ticker list
    if args.sp500:
        tickers = get_sp500_tickers()
        # Save S&P 500 list
        sp500_file = Path('data/universe/sp500_tickers.txt')
        sp500_file.parent.mkdir(parents=True, exist_ok=True)
        with open(sp500_file, 'w') as f:
            f.write('\n'.join(tickers))
        logger.info(f"Saved S&P 500 ticker list to {sp500_file}")
    elif args.tickers:
        tickers = load_ticker_list(args.tickers)
    else:
        logger.error("Must specify either --tickers or --sp500")
        return

    if not tickers:
        logger.error("No tickers to download")
        return

    # Download data
    download_all_stocks(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output,
        rate_limit_delay=args.rate_limit,
        skip_existing=not args.no_skip_existing
    )


if __name__ == '__main__':
    main()
