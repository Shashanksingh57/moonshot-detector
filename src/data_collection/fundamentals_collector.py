"""
Fundamentals data collector using SEC EDGAR (FREE).
Downloads and parses 10-K and 10-Q filings for financial metrics.
"""

import pandas as pd
from typing import List, Optional, Dict
from pathlib import Path
import time
import argparse
from datetime import datetime
from tqdm import tqdm
from sec_edgar_downloader import Downloader
from src.utils.logging_config import setup_logging
from src.utils.sec_edgar_parser import SECEDGARParser

logger = setup_logging('data_collection')


def download_sec_filings(
    ticker: str,
    forms: List[str] = ['10-K', '10-Q'],
    download_dir: str = "data/raw/sec_filings",
    after_date: str = "2015-01-01",
    rate_limit_delay: float = 0.1
) -> bool:
    """
    Download SEC filings for a ticker using sec-edgar-downloader.

    Args:
        ticker: Stock ticker symbol
        forms: List of form types to download (e.g., ['10-K', '10-Q'])
        download_dir: Directory to save filings
        after_date: Only download filings after this date
        rate_limit_delay: Delay between requests (seconds)

    Returns:
        True if successful, False otherwise
    """
    try:
        time.sleep(rate_limit_delay)

        # Initialize downloader
        dl = Downloader("MoonshotDetector", "user@example.com", download_dir)

        # Download each form type
        for form in forms:
            try:
                dl.get(form, ticker, after=after_date, download_details=True)
                logger.debug(f"Downloaded {form} filings for {ticker}")
            except Exception as e:
                logger.warning(f"Error downloading {form} for {ticker}: {str(e)}")

        return True

    except Exception as e:
        logger.error(f"Error downloading SEC filings for {ticker}: {str(e)}")
        return False


def extract_fundamentals_from_filing(
    filing_dir: Path,
    ticker: str
) -> Optional[Dict]:
    """
    Extract fundamental metrics from downloaded SEC filings.

    Args:
        filing_dir: Directory containing filing files
        ticker: Stock ticker symbol

    Returns:
        Dictionary of fundamental metrics, or None if extraction failed
    """
    try:
        parser = SECEDGARParser()

        # Find all filing files (JSON or XML)
        filing_files = list(filing_dir.glob('**/*.json')) + list(filing_dir.glob('**/*.xml'))

        if not filing_files:
            logger.warning(f"No filing files found for {ticker} in {filing_dir}")
            return None

        # Parse the most recent filing
        filing_file = sorted(filing_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]

        metrics = parser.parse_xbrl_file(str(filing_file))
        filing_date = parser.extract_filing_date(str(filing_file))

        if metrics:
            metrics['filing_date'] = filing_date
            # Calculate derived metrics
            derived = parser.calculate_derived_metrics(metrics)
            metrics.update(derived)

        return metrics

    except Exception as e:
        logger.error(f"Error extracting fundamentals for {ticker}: {str(e)}")
        return None


def collect_fundamentals_for_ticker(
    ticker: str,
    forms: List[str] = ['10-K', '10-Q'],
    after_date: str = "2015-01-01",
    sec_dir: str = "data/raw/sec_filings",
    output_dir: str = "data/raw/fundamentals"
) -> bool:
    """
    Complete pipeline: download filings and extract fundamentals for one ticker.

    Args:
        ticker: Stock ticker symbol
        forms: Form types to download
        after_date: Download filings after this date
        sec_dir: Directory for SEC filings
        output_dir: Directory for processed fundamentals

    Returns:
        True if successful, False otherwise
    """
    try:
        # Step 1: Download SEC filings
        logger.debug(f"Downloading SEC filings for {ticker}...")
        if not download_sec_filings(ticker, forms, sec_dir, after_date):
            return False

        # Step 2: Extract fundamentals from filings
        filing_dir = Path(sec_dir) / "sec-edgar-filings" / ticker
        if not filing_dir.exists():
            logger.warning(f"Filing directory not found: {filing_dir}")
            return False

        # Collect all quarterly/annual fundamentals
        fundamentals_list = []

        for form_type in forms:
            form_dir = filing_dir / form_type
            if not form_dir.exists():
                continue

            # Process each filing
            for filing_subdir in form_dir.iterdir():
                if filing_subdir.is_dir():
                    metrics = extract_fundamentals_from_filing(filing_subdir, ticker)
                    if metrics:
                        metrics['ticker'] = ticker
                        metrics['form_type'] = form_type
                        fundamentals_list.append(metrics)

        if not fundamentals_list:
            logger.warning(f"No fundamentals extracted for {ticker}")
            return False

        # Convert to DataFrame
        df = pd.DataFrame(fundamentals_list)

        # Sort by filing date
        if 'filing_date' in df.columns:
            df = df.sort_values('filing_date')

        # Save to parquet
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"{ticker}_fundamentals.parquet"
        df.to_parquet(output_file, index=False)

        logger.debug(f"Saved fundamentals for {ticker} to {output_file}")

        return True

    except Exception as e:
        logger.error(f"Error collecting fundamentals for {ticker}: {str(e)}")
        return False


def collect_all_fundamentals(
    tickers: List[str],
    forms: List[str] = ['10-K', '10-Q'],
    after_date: str = "2015-01-01",
    sec_dir: str = "data/raw/sec_filings",
    output_dir: str = "data/raw/fundamentals"
) -> None:
    """
    Collect fundamentals for multiple tickers with progress bar.

    Args:
        tickers: List of stock ticker symbols
        forms: Form types to download
        after_date: Download filings after this date
        sec_dir: Directory for SEC filings
        output_dir: Directory for processed fundamentals
    """
    logger.info(f"Starting fundamentals collection for {len(tickers)} tickers...")
    logger.info(f"Forms: {forms}")
    logger.info(f"After date: {after_date}")

    success_count = 0
    fail_count = 0

    pbar = tqdm(tickers, desc="Collecting fundamentals", unit="ticker")

    for ticker in pbar:
        pbar.set_postfix({'ticker': ticker, 'success': success_count, 'failed': fail_count})

        if collect_fundamentals_for_ticker(ticker, forms, after_date, sec_dir, output_dir):
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"Fundamentals collection complete!")
    logger.info(f"Success: {success_count}, Failed: {fail_count}")


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Download and parse SEC EDGAR filings')

    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Path to ticker list file (one per line)'
    )

    parser.add_argument(
        '--forms',
        type=str,
        nargs='+',
        default=['10-K', '10-Q'],
        help='SEC form types to download (default: 10-K 10-Q)'
    )

    parser.add_argument(
        '--after-date',
        type=str,
        default='2015-01-01',
        help='Download filings after this date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--sec-dir',
        type=str,
        default='data/raw/sec_filings',
        help='Directory for SEC filings'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/fundamentals',
        help='Output directory for processed fundamentals'
    )

    args = parser.parse_args()

    # Load ticker list
    with open(args.tickers, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    logger.info(f"Loaded {len(tickers)} tickers from {args.tickers}")

    # Collect fundamentals
    collect_all_fundamentals(
        tickers=tickers,
        forms=args.forms,
        after_date=args.after_date,
        sec_dir=args.sec_dir,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
