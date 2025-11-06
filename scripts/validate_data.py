#!/usr/bin/env python3
"""
CLI tool for validating downloaded data.
Validates stock prices, crypto data, fundamentals, and sentiment data.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_quality import (
    StockPriceValidator,
    CryptoDataValidator,
    FundamentalDataValidator,
    SentimentDataValidator
)
from src.utils.validation_reports import ValidationReportManager
from src.utils.logging_config import setup_logging

logger = setup_logging('validate_data')


def validate_stock_data(ticker: str, data_dir: str, report_manager: ValidationReportManager) -> bool:
    """
    Validate stock price data for a ticker.

    Args:
        ticker: Ticker symbol
        data_dir: Directory containing data
        report_manager: ValidationReportManager instance

    Returns:
        True if validation passed
    """
    try:
        filepath = Path(data_dir) / 'stocks' / 'raw' / f"{ticker}.csv"

        if not filepath.exists():
            logger.warning(f"Stock data not found for {ticker}: {filepath}")
            return False

        # Load data
        df = pd.read_csv(filepath)

        # Validate
        validator = StockPriceValidator(df, ticker)
        is_valid, issues = validator.run_all_checks()

        # Save report
        report_manager.save_validation_report(
            ticker=ticker,
            data_type='stock_price',
            issues=issues,
            is_valid=is_valid,
            metadata={'file_path': str(filepath), 'num_rows': len(df)}
        )

        return is_valid

    except Exception as e:
        logger.error(f"Error validating stock data for {ticker}: {str(e)}")
        return False


def validate_crypto_data(symbol: str, data_dir: str, report_manager: ValidationReportManager) -> bool:
    """
    Validate crypto price data for a symbol.

    Args:
        symbol: Crypto symbol
        data_dir: Directory containing data
        report_manager: ValidationReportManager instance

    Returns:
        True if validation passed
    """
    try:
        filepath = Path(data_dir) / 'crypto' / 'raw' / f"{symbol}.csv"

        if not filepath.exists():
            logger.warning(f"Crypto data not found for {symbol}: {filepath}")
            return False

        # Load data
        df = pd.read_csv(filepath)

        # Validate
        validator = CryptoDataValidator(df, symbol)
        is_valid, issues = validator.run_all_checks()

        # Save report
        report_manager.save_validation_report(
            ticker=symbol,
            data_type='crypto',
            issues=issues,
            is_valid=is_valid,
            metadata={'file_path': str(filepath), 'num_rows': len(df)}
        )

        return is_valid

    except Exception as e:
        logger.error(f"Error validating crypto data for {symbol}: {str(e)}")
        return False


def validate_fundamental_data(ticker: str, data_dir: str, report_manager: ValidationReportManager) -> bool:
    """
    Validate fundamental data for a ticker.

    Args:
        ticker: Ticker symbol
        data_dir: Directory containing data
        report_manager: ValidationReportManager instance

    Returns:
        True if validation passed
    """
    try:
        filepath = Path(data_dir) / 'fundamentals' / f"{ticker}.csv"

        if not filepath.exists():
            logger.warning(f"Fundamental data not found for {ticker}: {filepath}")
            return False

        # Load data
        df = pd.read_csv(filepath)

        # Validate
        validator = FundamentalDataValidator(df, ticker)
        is_valid, issues = validator.run_all_checks()

        # Save report
        report_manager.save_validation_report(
            ticker=ticker,
            data_type='fundamental',
            issues=issues,
            is_valid=is_valid,
            metadata={'file_path': str(filepath), 'num_rows': len(df)}
        )

        return is_valid

    except Exception as e:
        logger.error(f"Error validating fundamental data for {ticker}: {str(e)}")
        return False


def validate_sentiment_data(ticker: str, data_dir: str, report_manager: ValidationReportManager) -> bool:
    """
    Validate sentiment data for a ticker.

    Args:
        ticker: Ticker symbol
        data_dir: Directory containing data
        report_manager: ValidationReportManager instance

    Returns:
        True if validation passed
    """
    try:
        filepath = Path(data_dir) / 'sentiment' / f"{ticker}.csv"

        if not filepath.exists():
            logger.debug(f"Sentiment data not found for {ticker}: {filepath}")
            return False

        # Load data
        df = pd.read_csv(filepath)

        # Validate
        validator = SentimentDataValidator(df, ticker)
        is_valid, issues = validator.run_all_checks()

        # Save report
        report_manager.save_validation_report(
            ticker=ticker,
            data_type='sentiment',
            issues=issues,
            is_valid=is_valid,
            metadata={'file_path': str(filepath), 'num_rows': len(df)}
        )

        return is_valid

    except Exception as e:
        logger.error(f"Error validating sentiment data for {ticker}: {str(e)}")
        return False


def validate_single_ticker(ticker: str, data_type: str, data_dir: str, report_manager: ValidationReportManager):
    """
    Validate a single ticker for specified data type.

    Args:
        ticker: Ticker symbol
        data_type: Type of data to validate
        data_dir: Data directory
        report_manager: ValidationReportManager instance
    """
    logger.info(f"Validating {ticker} ({data_type})...")

    if data_type == 'stock':
        validate_stock_data(ticker, data_dir, report_manager)
    elif data_type == 'crypto':
        validate_crypto_data(ticker, data_dir, report_manager)
    elif data_type == 'fundamental':
        validate_fundamental_data(ticker, data_dir, report_manager)
    elif data_type == 'sentiment':
        validate_sentiment_data(ticker, data_dir, report_manager)
    else:
        # Validate all types
        validate_stock_data(ticker, data_dir, report_manager)
        validate_crypto_data(ticker, data_dir, report_manager)
        validate_fundamental_data(ticker, data_dir, report_manager)
        validate_sentiment_data(ticker, data_dir, report_manager)


def validate_all_data(data_dir: str, data_type: str, report_manager: ValidationReportManager):
    """
    Validate all data in the data directory.

    Args:
        data_dir: Data directory
        data_type: Type of data to validate (or 'all')
        report_manager: ValidationReportManager instance
    """
    data_path = Path(data_dir)

    # Validate stock data
    if data_type in ['stock', 'all']:
        logger.info("Validating stock data...")
        stock_dir = data_path / 'stocks' / 'raw'
        if stock_dir.exists():
            stock_files = list(stock_dir.glob("*.csv"))
            logger.info(f"Found {len(stock_files)} stock files")

            for filepath in stock_files:
                ticker = filepath.stem
                validate_stock_data(ticker, data_dir, report_manager)

    # Validate crypto data
    if data_type in ['crypto', 'all']:
        logger.info("Validating crypto data...")
        crypto_dir = data_path / 'crypto' / 'raw'
        if crypto_dir.exists():
            crypto_files = list(crypto_dir.glob("*.csv"))
            logger.info(f"Found {len(crypto_files)} crypto files")

            for filepath in crypto_files:
                symbol = filepath.stem
                validate_crypto_data(symbol, data_dir, report_manager)

    # Validate fundamental data
    if data_type in ['fundamental', 'all']:
        logger.info("Validating fundamental data...")
        fund_dir = data_path / 'fundamentals'
        if fund_dir.exists():
            fund_files = list(fund_dir.glob("*.csv"))
            logger.info(f"Found {len(fund_files)} fundamental files")

            for filepath in fund_files:
                ticker = filepath.stem
                validate_fundamental_data(ticker, data_dir, report_manager)

    # Validate sentiment data
    if data_type in ['sentiment', 'all']:
        logger.info("Validating sentiment data...")
        sent_dir = data_path / 'sentiment'
        if sent_dir.exists():
            sent_files = list(sent_dir.glob("*.csv"))
            logger.info(f"Found {len(sent_files)} sentiment files")

            for filepath in sent_files:
                ticker = filepath.stem
                validate_sentiment_data(ticker, data_dir, report_manager)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Validate downloaded data for quality issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all data
  python scripts/validate_data.py

  # Validate specific ticker
  python scripts/validate_data.py --ticker AAPL

  # Validate specific data type
  python scripts/validate_data.py --data-type stock

  # Validate specific ticker and data type
  python scripts/validate_data.py --ticker AAPL --data-type stock

  # Show summary of previous validation
  python scripts/validate_data.py --summary

  # Show summary for specific date
  python scripts/validate_data.py --summary --date 2024-01-15
        """
    )

    parser.add_argument(
        '--ticker',
        type=str,
        help='Specific ticker to validate'
    )

    parser.add_argument(
        '--data-type',
        choices=['stock', 'crypto', 'fundamental', 'sentiment', 'all'],
        default='all',
        help='Data type to validate (default: all)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory (default: data/)'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Date to filter reports (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary of validation reports only (do not run new validation)'
    )

    parser.add_argument(
        '--export-csv',
        type=str,
        help='Export summary to CSV file'
    )

    args = parser.parse_args()

    # Initialize report manager
    report_manager = ValidationReportManager()

    # Show summary only
    if args.summary:
        logger.info("Loading validation summary...")
        report_manager.print_summary(date_filter=args.date)

        if args.export_csv:
            report_manager.export_summary_csv(args.export_csv, date_filter=args.date)

        return

    # Run validation
    logger.info("="*70)
    logger.info("DATA VALIDATION")
    logger.info("="*70)

    if args.ticker:
        # Validate specific ticker
        validate_single_ticker(args.ticker, args.data_type, args.data_dir, report_manager)
    else:
        # Validate all data
        validate_all_data(args.data_dir, args.data_type, report_manager)

    # Print summary
    logger.info("")
    logger.info("Validation complete. Generating summary...")
    report_manager.print_summary()

    # Export if requested
    if args.export_csv:
        report_manager.export_summary_csv(args.export_csv)


if __name__ == '__main__':
    main()
