"""
Crypto data collector using CCXT (Binance) and CoinGecko (FREE).
Downloads OHLCV data for cryptocurrencies with rate limiting.
"""

import pandas as pd
import numpy as np
import ccxt
import requests
from typing import List, Optional, Dict
from pathlib import Path
import time
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
from src.utils.logging_config import setup_logging
from src.utils.data_validation import validate_ohlcv_data, check_data_freshness

logger = setup_logging('data_collection')


def get_top_cryptos_by_mcap(n: int = 50) -> List[str]:
    """
    Get top N cryptocurrencies by market cap from CoinGecko (FREE).

    Args:
        n: Number of top cryptos to retrieve

    Returns:
        List of crypto symbols (e.g., ['BTC', 'ETH', 'BNB'])
    """
    try:
        logger.info(f"Fetching top {n} cryptocurrencies by market cap from CoinGecko...")

        # CoinGecko free API endpoint
        url = "https://api.coingecko.com/api/v3/coins/markets"

        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': n,
            'page': 1,
            'sparkline': False
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Extract symbols and convert to uppercase
        symbols = [coin['symbol'].upper() for coin in data]

        logger.info(f"Found {len(symbols)} cryptocurrencies")

        return symbols

    except Exception as e:
        logger.error(f"Error fetching top cryptos from CoinGecko: {str(e)}")
        return []


def download_crypto_ohlcv(
    symbol: str,
    exchange_name: str = 'binance',
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    rate_limit_delay: float = 1.0
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a cryptocurrency using CCXT.

    Args:
        symbol: Crypto symbol (e.g., 'BTC')
        exchange_name: Exchange name (default: 'binance')
        timeframe: Timeframe (e.g., '1d', '1h')
        start_date: Start date (YYYY-MM-DD), defaults to earliest available
        rate_limit_delay: Delay between requests (seconds)

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    try:
        # Add delay for rate limiting
        time.sleep(rate_limit_delay)

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        # Construct trading pair (most cryptos trade against USDT)
        pair = f"{symbol}/USDT"

        # Check if pair exists
        exchange.load_markets()
        if pair not in exchange.markets:
            logger.warning(f"Trading pair {pair} not found on {exchange_name}")
            return None

        # Convert start date to timestamp
        if start_date:
            since = exchange.parse8601(f"{start_date}T00:00:00Z")
        else:
            # Default to 3 years ago
            since = exchange.parse8601((datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%dT00:00:00Z'))

        # Fetch OHLCV data
        all_ohlcv = []
        limit = 1000  # Max per request

        logger.debug(f"Downloading {pair} from {exchange_name}...")

        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Update since to last timestamp
                since = ohlcv[-1][0] + 1

                # Check if we've reached the end
                if len(ohlcv) < limit:
                    break

                # Rate limiting
                time.sleep(rate_limit_delay)

            except ccxt.NetworkError as e:
                logger.warning(f"Network error for {pair}: {str(e)}")
                time.sleep(5)  # Wait before retry
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error for {pair}: {str(e)}")
                break

        if not all_ohlcv:
            logger.warning(f"No data retrieved for {pair}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)

        # Reorder columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        # Add symbol
        df['symbol'] = symbol

        # Validate data
        df_validate = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        is_valid, issues = validate_ohlcv_data(df_validate, symbol)

        if not is_valid:
            logger.warning(f"Data validation failed for {symbol}: {issues}")

        logger.debug(f"Downloaded {len(df)} rows for {symbol}")

        return df

    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {str(e)}")
        return None


def save_crypto_to_parquet(df: pd.DataFrame, symbol: str, output_dir: str = "data/raw/crypto") -> bool:
    """
    Save cryptocurrency DataFrame to parquet file.

    Args:
        df: DataFrame to save
        symbol: Crypto symbol
        output_dir: Output directory path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save to parquet (filename includes USDT pair)
        file_path = output_path / f"{symbol}_USDT.parquet"
        df.to_parquet(file_path, index=False, compression='snappy')

        logger.debug(f"Saved {symbol} to {file_path}")

        return True

    except Exception as e:
        logger.error(f"Error saving {symbol} to parquet: {str(e)}")
        return False


def download_all_cryptos(
    symbols: List[str],
    exchange_name: str = 'binance',
    start_date: Optional[str] = None,
    output_dir: str = "data/raw/crypto",
    rate_limit_delay: float = 1.0,
    skip_existing: bool = True,
    max_age_days: int = 7
) -> None:
    """
    Download data for multiple cryptocurrencies with progress bar.

    Args:
        symbols: List of crypto symbols
        exchange_name: Exchange to use
        start_date: Start date (YYYY-MM-DD)
        output_dir: Output directory
        rate_limit_delay: Delay between requests (seconds)
        skip_existing: Skip cryptos with recent data files
        max_age_days: Max age of existing files to skip (days)
    """
    logger.info(f"Starting download for {len(symbols)} cryptocurrencies...")
    logger.info(f"Exchange: {exchange_name}")
    logger.info(f"Start date: {start_date or 'earliest available'}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Progress bar
    pbar = tqdm(symbols, desc="Downloading crypto", unit="symbol")

    for symbol in pbar:
        pbar.set_postfix({'symbol': symbol, 'success': success_count, 'skipped': skip_count, 'failed': fail_count})

        # Check if file exists and is recent
        file_path = output_path / f"{symbol}_USDT.parquet"
        if skip_existing and file_path.exists():
            if check_data_freshness(str(file_path), max_age_days):
                logger.debug(f"Skipping {symbol} - recent data exists")
                skip_count += 1
                continue

        # Download data
        df = download_crypto_ohlcv(symbol, exchange_name, start_date=start_date, rate_limit_delay=rate_limit_delay)

        if df is not None and len(df) > 0:
            # Save to parquet
            if save_crypto_to_parquet(df, symbol, output_dir):
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1

    logger.info(f"Download complete!")
    logger.info(f"Success: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Download crypto data using CCXT')

    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        help='Exchange name (default: binance)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD), defaults to earliest available'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='Path to symbols list file (one per line)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/crypto',
        help='Output directory'
    )

    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Delay between requests (seconds)'
    )

    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-download even if recent data exists'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=50,
        help='Download top N cryptos by market cap'
    )

    args = parser.parse_args()

    # Get symbol list
    if args.symbols:
        # Load from file
        with open(args.symbols, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {args.symbols}")
    else:
        # Get top N by market cap
        symbols = get_top_cryptos_by_mcap(args.top_n)
        # Save to file
        crypto_file = Path('data/universe/top50_crypto.txt')
        crypto_file.parent.mkdir(parents=True, exist_ok=True)
        with open(crypto_file, 'w') as f:
            f.write('\n'.join(symbols))
        logger.info(f"Saved top {args.top_n} crypto list to {crypto_file}")

    if not symbols:
        logger.error("No symbols to download")
        return

    # Download data
    download_all_cryptos(
        symbols=symbols,
        exchange_name=args.exchange,
        start_date=args.start_date,
        output_dir=args.output,
        rate_limit_delay=args.rate_limit,
        skip_existing=not args.no_skip_existing
    )


if __name__ == '__main__':
    main()
