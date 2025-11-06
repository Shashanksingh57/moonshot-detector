"""
FINRA short interest data parser.
Parses publicly available FINRA short interest reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FINRAParser:
    """Parser for FINRA short interest data."""

    def __init__(self):
        """Initialize the parser."""
        self.base_url = "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data"

    def parse_short_interest_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse a FINRA short interest data file.

        FINRA short interest files are typically pipe-delimited (|) or comma-delimited.
        Format: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

        Args:
            file_path: Path to the FINRA data file

        Returns:
            DataFrame with columns: date, symbol, short_volume, short_exempt_volume,
                                   total_volume, market
        """
        try:
            # Try different delimiters
            for delimiter in ['|', ',', '\t']:
                try:
                    df = pd.read_csv(
                        file_path,
                        delimiter=delimiter,
                        parse_dates=['Date'] if 'Date' in pd.read_csv(file_path, delimiter=delimiter, nrows=1).columns else False
                    )
                    if len(df.columns) >= 5:  # Need at least 5 columns
                        break
                except:
                    continue

            if df is None or len(df) == 0:
                logger.error(f"Could not parse file: {file_path}")
                return pd.DataFrame()

            # Standardize column names
            column_mapping = {
                'Date': 'date',
                'Symbol': 'symbol',
                'ShortVolume': 'short_volume',
                'ShortExemptVolume': 'short_exempt_volume',
                'TotalVolume': 'total_volume',
                'Market': 'market'
            }

            df = df.rename(columns=column_mapping)

            # Ensure required columns exist
            required_cols = ['date', 'symbol', 'short_volume', 'total_volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()

            # Convert date to datetime
            if df['date'].dtype != 'datetime64[ns]':
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Convert volume columns to numeric
            for col in ['short_volume', 'total_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Calculate short interest percentage
            df['short_interest_pct'] = (df['short_volume'] / df['total_volume']) * 100

            # Remove rows with invalid data
            df = df.dropna(subset=['date', 'symbol', 'short_volume', 'total_volume'])

            logger.info(f"Parsed {len(df)} records from {file_path}")

            return df

        except Exception as e:
            logger.error(f"Error parsing FINRA file {file_path}: {str(e)}")
            return pd.DataFrame()

    def aggregate_short_interest(
        self,
        df: pd.DataFrame,
        ticker: str,
        period_days: int = 14
    ) -> pd.DataFrame:
        """
        Aggregate short interest data for a specific ticker.

        FINRA reports are published twice monthly (roughly every 2 weeks).

        Args:
            df: DataFrame from parse_short_interest_file
            ticker: Stock ticker symbol
            period_days: Number of days to aggregate (default 14 for bi-monthly)

        Returns:
            DataFrame with aggregated short interest metrics
        """
        try:
            # Filter for specific ticker
            ticker_df = df[df['symbol'] == ticker.upper()].copy()

            if len(ticker_df) == 0:
                logger.warning(f"No data found for ticker {ticker}")
                return pd.DataFrame()

            # Sort by date
            ticker_df = ticker_df.sort_values('date')

            # Group by settlement date (FINRA reports by settlement date)
            # Calculate average short interest percentage
            aggregated = ticker_df.groupby('date').agg({
                'short_volume': 'sum',
                'total_volume': 'sum',
                'short_interest_pct': 'mean'
            }).reset_index()

            # Set date as index
            aggregated = aggregated.set_index('date')

            return aggregated

        except Exception as e:
            logger.error(f"Error aggregating short interest for {ticker}: {str(e)}")
            return pd.DataFrame()

    def calculate_short_metrics(
        self,
        short_interest_df: pd.DataFrame,
        shares_outstanding: float,
        avg_daily_volume: float
    ) -> pd.DataFrame:
        """
        Calculate derived short interest metrics.

        Args:
            short_interest_df: DataFrame with short interest data
            shares_outstanding: Number of shares outstanding
            avg_daily_volume: Average daily trading volume

        Returns:
            DataFrame with additional metrics:
                - short_interest_ratio: short volume / shares outstanding
                - days_to_cover: short volume / avg daily volume
                - short_volume_ratio: short volume / total volume
        """
        try:
            df = short_interest_df.copy()

            # Short Interest as % of shares outstanding
            if shares_outstanding > 0:
                df['short_interest_ratio'] = (df['short_volume'] / shares_outstanding) * 100

            # Days to Cover (how many days of average volume to cover all short positions)
            if avg_daily_volume > 0:
                df['days_to_cover'] = df['short_volume'] / avg_daily_volume

            # Short Volume Ratio (already calculated as short_interest_pct, but ensure it's there)
            if 'short_interest_pct' not in df.columns:
                df['short_interest_pct'] = (df['short_volume'] / df['total_volume']) * 100

            # Calculate change in short interest (period over period)
            df['short_interest_change_pct'] = df['short_interest_pct'].pct_change() * 100

            return df

        except Exception as e:
            logger.error(f"Error calculating short metrics: {str(e)}")
            return short_interest_df

    def get_latest_short_interest(self, ticker: str, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get the most recent short interest data for a ticker.

        Args:
            ticker: Stock ticker symbol
            df: DataFrame with short interest data

        Returns:
            Dictionary with latest short interest metrics
        """
        try:
            ticker_df = df[df['symbol'] == ticker.upper()].copy()

            if len(ticker_df) == 0:
                return {}

            # Get most recent record
            latest = ticker_df.sort_values('date').iloc[-1]

            return {
                'date': latest['date'],
                'short_volume': latest['short_volume'],
                'total_volume': latest['total_volume'],
                'short_interest_pct': latest['short_interest_pct'],
                'days_to_cover': latest.get('days_to_cover', np.nan),
                'short_interest_ratio': latest.get('short_interest_ratio', np.nan)
            }

        except Exception as e:
            logger.error(f"Error getting latest short interest for {ticker}: {str(e)}")
            return {}

    def detect_short_squeeze_signals(
        self,
        df: pd.DataFrame,
        high_short_interest_threshold: float = 20.0,
        high_days_to_cover_threshold: float = 5.0
    ) -> pd.DataFrame:
        """
        Detect potential short squeeze signals based on short interest metrics.

        A potential short squeeze candidate has:
        - High short interest (>20% of shares outstanding)
        - High days to cover (>5 days)
        - Increasing short interest

        Args:
            df: DataFrame with short interest metrics
            high_short_interest_threshold: Minimum short interest % for signal
            high_days_to_cover_threshold: Minimum days to cover for signal

        Returns:
            DataFrame with short squeeze signal column (True/False)
        """
        try:
            df = df.copy()

            # Initialize signal column
            df['short_squeeze_signal'] = False

            # Criteria for short squeeze potential
            if 'short_interest_ratio' in df.columns and 'days_to_cover' in df.columns:
                df['short_squeeze_signal'] = (
                    (df['short_interest_ratio'] >= high_short_interest_threshold) &
                    (df['days_to_cover'] >= high_days_to_cover_threshold)
                )

            # Log number of signals
            signal_count = df['short_squeeze_signal'].sum()
            logger.info(f"Detected {signal_count} potential short squeeze signals")

            return df

        except Exception as e:
            logger.error(f"Error detecting short squeeze signals: {str(e)}")
            return df
