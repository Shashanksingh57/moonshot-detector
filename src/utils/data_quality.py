"""
Data quality validators for moonshot-detector.
Validates OHLCV, fundamental, crypto, and sentiment data.

CRITICAL: Bad data = bad models = lost money
This module is the first line of defense against data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import json
from pathlib import Path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class StockPriceValidator:
    """Validates stock OHLCV data quality."""

    def __init__(self, df: pd.DataFrame, ticker: str):
        """
        Initialize validator.

        Args:
            df: DataFrame with stock OHLCV data
            ticker: Stock ticker symbol
        """
        self.df = df.copy()
        self.ticker = ticker
        self.issues = []

    def run_all_checks(self) -> Tuple[bool, List[str]]:
        """
        Run all validation checks.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        self.check_required_columns()
        self.check_data_types()
        self.check_missing_values()
        self.check_negative_values()
        self.check_zero_values()
        self.check_price_consistency()
        self.check_volume_spikes()
        self.check_date_gaps()
        self.check_date_order()
        self.check_duplicates()
        self.check_outliers()
        self.check_splits_dividends()

        is_valid = len(self.issues) == 0
        return is_valid, self.issues

    def check_required_columns(self):
        """Ensure all required columns exist."""
        # Handle both lowercase and title case column names
        df_cols_lower = [col.lower() for col in self.df.columns]
        required_lower = ['open', 'high', 'low', 'close', 'volume', 'date']

        missing = [col for col in required_lower if col not in df_cols_lower]
        if missing:
            self.issues.append(f"Missing columns: {missing}")

    def check_data_types(self):
        """Verify data types are correct."""
        # Map to actual column names (handle case insensitivity)
        col_map = {col.lower(): col for col in self.df.columns}

        for col_lower in ['open', 'high', 'low', 'close']:
            if col_lower in col_map:
                col = col_map[col_lower]
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    self.issues.append(f"{col} is not numeric")

        if 'date' in col_map:
            date_col = col_map['date']
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                self.issues.append("Date column is not datetime")

    def check_missing_values(self):
        """Check for missing values."""
        col_map = {col.lower(): col for col in self.df.columns}
        check_cols = [col_map.get(c.lower()) for c in ['open', 'high', 'low', 'close', 'volume'] if c.lower() in col_map]

        if not check_cols:
            return

        missing_counts = self.df[check_cols].isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            missing_pct = (total_missing / (len(self.df) * len(check_cols))) * 100
            self.issues.append(
                f"Missing values: {total_missing} ({missing_pct:.2f}%). "
                f"Details: {missing_counts[missing_counts > 0].to_dict()}"
            )

        # Check for date gaps
        if 'date' in col_map:
            date_col = col_map['date']
            if self.df[date_col].isnull().sum() == 0:
                date_diffs = self.df[date_col].diff()
                large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]
                if len(large_gaps) > 0:
                    self.issues.append(
                        f"Found {len(large_gaps)} date gaps > 7 days. "
                        f"Largest gap: {large_gaps.max()}"
                    )

    def check_negative_values(self):
        """Prices and volume cannot be negative."""
        col_map = {col.lower(): col for col in self.df.columns}

        for col_lower in ['open', 'high', 'low', 'close', 'volume']:
            if col_lower in col_map:
                col = col_map[col_lower]
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    self.issues.append(
                        f"{col} has {negative_count} negative values - IMPOSSIBLE"
                    )

    def check_zero_values(self):
        """Check for suspicious zero values."""
        col_map = {col.lower(): col for col in self.df.columns}

        # Zero prices are impossible
        for col_lower in ['open', 'high', 'low', 'close']:
            if col_lower in col_map:
                col = col_map[col_lower]
                zero_count = (self.df[col] == 0).sum()
                if zero_count > 0:
                    self.issues.append(
                        f"{col} has {zero_count} zero values - suspicious"
                    )

        # Zero volume is suspicious but possible
        if 'volume' in col_map:
            volume_col = col_map['volume']
            zero_volume = (self.df[volume_col] == 0).sum()
            if zero_volume > len(self.df) * 0.05:
                self.issues.append(
                    f"Volume is zero for {zero_volume} days "
                    f"({zero_volume/len(self.df)*100:.1f}%) - suspicious"
                )

    def check_price_consistency(self):
        """High >= Low, Close between High and Low."""
        col_map = {col.lower(): col for col in self.df.columns}

        if not all(k in col_map for k in ['high', 'low', 'close']):
            return

        high_col = col_map['high']
        low_col = col_map['low']
        close_col = col_map['close']

        # High should always be >= Low
        inconsistent_hl = (self.df[high_col] < self.df[low_col]).sum()
        if inconsistent_hl > 0:
            self.issues.append(
                f"{inconsistent_hl} days where High < Low - DATA ERROR"
            )

        # Close should be between High and Low
        close_above_high = (self.df[close_col] > self.df[high_col]).sum()
        close_below_low = (self.df[close_col] < self.df[low_col]).sum()

        if close_above_high > 0:
            self.issues.append(
                f"{close_above_high} days where Close > High - DATA ERROR"
            )
        if close_below_low > 0:
            self.issues.append(
                f"{close_below_low} days where Close < Low - DATA ERROR"
            )

    def check_volume_spikes(self):
        """Detect extreme volume anomalies."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'volume' not in col_map:
            return

        volume_col = col_map['volume']
        volume = self.df[volume_col].replace(0, np.nan)

        # Calculate rolling statistics
        volume_mean = volume.rolling(60, min_periods=20).mean()
        volume_std = volume.rolling(60, min_periods=20).std()

        # Flag volumes > 10 standard deviations
        extreme_spikes = volume > (volume_mean + 10 * volume_std)
        spike_count = extreme_spikes.sum()

        if spike_count > 0:
            self.issues.append(
                f"Found {spike_count} extreme volume spikes (>10 std) - possible data errors"
            )

    def check_date_gaps(self):
        """Check for unexpected date gaps."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'date' not in col_map:
            return

        date_col = col_map['date']

        # Calculate expected vs actual trading days
        date_range = (self.df[date_col].max() - self.df[date_col].min()).days
        expected_trading_days = date_range * 5 / 7  # Rough estimate
        actual_days = len(self.df)

        missing_days = expected_trading_days - actual_days
        missing_pct = (missing_days / expected_trading_days) * 100 if expected_trading_days > 0 else 0

        if missing_pct > 10:
            self.issues.append(
                f"Missing {missing_pct:.1f}% of expected trading days. "
                f"Expected ~{expected_trading_days:.0f}, got {actual_days}"
            )

    def check_date_order(self):
        """Ensure dates are in ascending order."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'date' not in col_map:
            return

        date_col = col_map['date']

        if not self.df[date_col].is_monotonic_increasing:
            self.issues.append("Dates are not in ascending order - SORT ERROR")

    def check_duplicates(self):
        """Check for duplicate dates."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'date' not in col_map:
            return

        date_col = col_map['date']
        duplicates = self.df[date_col].duplicated().sum()

        if duplicates > 0:
            self.issues.append(
                f"Found {duplicates} duplicate dates - DATA ERROR"
            )

    def check_outliers(self):
        """Detect price outliers (possible data errors)."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'close' not in col_map:
            return

        close_col = col_map['close']
        returns = self.df[close_col].pct_change()

        # Flag returns > 50% in a single day
        extreme_returns = (returns.abs() > 0.5).sum()
        if extreme_returns > 0:
            if 'date' in col_map:
                date_col = col_map['date']
                dates_with_extreme = self.df.loc[returns.abs() > 0.5, date_col].tolist()
                self.issues.append(
                    f"Found {extreme_returns} days with >50% returns. "
                    f"Possible unadjusted splits on: {dates_with_extreme[:5]}"
                )
            else:
                self.issues.append(
                    f"Found {extreme_returns} days with >50% returns - possible unadjusted splits"
                )

    def check_splits_dividends(self):
        """Verify splits/dividends are properly adjusted."""
        # Check if adjusted close exists
        adj_close_cols = [col for col in self.df.columns if 'adj' in col.lower() and 'close' in col.lower()]
        close_cols = [col for col in self.df.columns if col.lower() == 'close']

        if adj_close_cols and close_cols:
            adj_close = adj_close_cols[0]
            close = close_cols[0]

            adjustment_ratio = self.df[adj_close] / self.df[close]

            # If ratio is not close to 1.0, adjustments were made
            significant_adjustments = (adjustment_ratio < 0.95) | (adjustment_ratio > 1.05)
            adjustment_count = significant_adjustments.sum()

            if adjustment_count > len(self.df) * 0.1:
                self.issues.append(
                    f"{adjustment_count} days with significant price adjustments. "
                    f"Verify splits/dividends are correct."
                )


class FundamentalDataValidator:
    """Validates fundamental data from SEC filings."""

    def __init__(self, df: pd.DataFrame, ticker: str):
        """
        Initialize validator.

        Args:
            df: DataFrame with fundamental data
            ticker: Stock ticker symbol
        """
        self.df = df.copy()
        self.ticker = ticker
        self.issues = []

    def run_all_checks(self) -> Tuple[bool, List[str]]:
        """Run all validation checks."""
        self.check_required_fields()
        self.check_date_alignment()
        self.check_point_in_time_correctness()
        self.check_ratio_sanity()
        self.check_growth_rates()
        self.check_negative_revenues()
        self.check_missing_quarters()

        is_valid = len(self.issues) == 0
        return is_valid, self.issues

    def check_required_fields(self):
        """Ensure key fundamental fields exist."""
        col_map = {col.lower(): col for col in self.df.columns}
        required_lower = ['filing_date', 'revenue']

        missing = [f for f in required_lower if f not in col_map]
        if missing:
            self.issues.append(f"Missing fundamental fields: {missing}")

    def check_date_alignment(self):
        """Verify filing_date > period_end_date."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'filing_date' in col_map and 'period_end_date' in col_map:
            filing_col = col_map['filing_date']
            period_col = col_map['period_end_date']

            misaligned = (pd.to_datetime(self.df[filing_col]) < pd.to_datetime(self.df[period_col])).sum()
            if misaligned > 0:
                self.issues.append(
                    f"{misaligned} records where filing_date < period_end_date - IMPOSSIBLE"
                )

    def check_point_in_time_correctness(self):
        """CRITICAL: Ensure no look-ahead bias."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'period_end_date' not in col_map or 'filing_date' not in col_map:
            return

        filing_col = col_map['filing_date']
        period_col = col_map['period_end_date']

        # Filing typically happens 45-90 days after quarter end
        filing_delay = (pd.to_datetime(self.df[filing_col]) - pd.to_datetime(self.df[period_col])).dt.days

        # Flag if delay is < 30 days
        too_fast = (filing_delay < 30).sum()
        if too_fast > 0:
            self.issues.append(
                f"{too_fast} filings appear < 30 days after period end - suspicious"
            )

        # Flag if delay is > 120 days
        too_slow = (filing_delay > 120).sum()
        if too_slow > len(self.df) * 0.1:
            self.issues.append(
                f"{too_slow} filings are > 120 days late"
            )

    def check_ratio_sanity(self):
        """Check if calculated ratios make sense."""
        col_map = {col.lower(): col for col in self.df.columns}

        # P/E ratio checks
        if 'pe_ratio' in col_map:
            pe_col = col_map['pe_ratio']
            negative_pe = (self.df[pe_col] < 0).sum()
            if negative_pe > len(self.df) * 0.5:
                self.issues.append(
                    f"{negative_pe} quarters with negative P/E - company may be unprofitable"
                )

            extreme_pe = (self.df[pe_col] > 1000).sum()
            if extreme_pe > 0:
                self.issues.append(
                    f"{extreme_pe} quarters with P/E > 1000 - possible data errors"
                )

        # Debt/Equity checks
        if 'debt_to_equity' in col_map:
            debt_col = col_map['debt_to_equity']
            extreme_leverage = (self.df[debt_col] > 10).sum()
            if extreme_leverage > 0:
                self.issues.append(
                    f"{extreme_leverage} quarters with Debt/Equity > 10 - high leverage or data error"
                )

    def check_growth_rates(self):
        """Check if growth rates are reasonable."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'revenue' in col_map:
            revenue_col = col_map['revenue']
            revenue_growth = self.df[revenue_col].pct_change()

            extreme_growth = (revenue_growth > 5.0).sum()
            if extreme_growth > 0:
                self.issues.append(
                    f"{extreme_growth} quarters with >500% revenue growth - possible M&A or data error"
                )

            extreme_decline = (revenue_growth < -0.8).sum()
            if extreme_decline > 0:
                self.issues.append(
                    f"{extreme_decline} quarters with >80% revenue decline - possible spin-off or data error"
                )

    def check_negative_revenues(self):
        """Revenue should generally be positive."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'revenue' in col_map:
            revenue_col = col_map['revenue']
            negative_revenue = (self.df[revenue_col] < 0).sum()
            if negative_revenue > 0:
                self.issues.append(
                    f"{negative_revenue} quarters with negative revenue - unusual or data error"
                )

    def check_missing_quarters(self):
        """Check for missing quarterly data."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'period_end_date' in col_map:
            period_col = col_map['period_end_date']
            date_diffs = pd.to_datetime(self.df[period_col]).diff()
            large_gaps = (date_diffs > pd.Timedelta(days=120)).sum()

            if large_gaps > 0:
                self.issues.append(
                    f"{large_gaps} gaps > 120 days between quarters - missing filings?"
                )


class CryptoDataValidator:
    """Validates crypto OHLCV data."""

    def __init__(self, df: pd.DataFrame, symbol: str):
        """
        Initialize validator.

        Args:
            df: DataFrame with crypto OHLCV data
            symbol: Crypto symbol
        """
        self.df = df.copy()
        self.symbol = symbol
        self.issues = []

    def run_all_checks(self) -> Tuple[bool, List[str]]:
        """Run crypto-specific validation checks."""
        self.check_required_columns()
        self.check_missing_values()
        self.check_negative_values()
        self.check_price_consistency()
        self.check_extreme_volatility()
        self.check_exchange_issues()
        self.check_stablecoin_peg()

        is_valid = len(self.issues) == 0
        return is_valid, self.issues

    def check_required_columns(self):
        """Ensure required columns exist."""
        df_cols_lower = [col.lower() for col in self.df.columns]
        required_lower = ['open', 'high', 'low', 'close', 'volume']

        missing = [col for col in required_lower if col not in df_cols_lower]
        if missing:
            self.issues.append(f"Missing columns: {missing}")

    def check_missing_values(self):
        """Check for missing values."""
        col_map = {col.lower(): col for col in self.df.columns}
        check_cols = [col_map.get(c) for c in ['open', 'high', 'low', 'close', 'volume'] if c in col_map]

        if not check_cols:
            return

        missing_counts = self.df[check_cols].isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            missing_pct = (total_missing / (len(self.df) * len(check_cols))) * 100
            self.issues.append(
                f"Missing values: {total_missing} ({missing_pct:.2f}%)"
            )

    def check_negative_values(self):
        """Prices and volume cannot be negative."""
        col_map = {col.lower(): col for col in self.df.columns}

        for col_lower in ['open', 'high', 'low', 'close', 'volume']:
            if col_lower in col_map:
                col = col_map[col_lower]
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    self.issues.append(
                        f"{col} has {negative_count} negative values - IMPOSSIBLE"
                    )

    def check_price_consistency(self):
        """High >= Low, Close between High and Low."""
        col_map = {col.lower(): col for col in self.df.columns}

        if not all(k in col_map for k in ['high', 'low', 'close']):
            return

        high_col = col_map['high']
        low_col = col_map['low']
        close_col = col_map['close']

        inconsistent = (self.df[high_col] < self.df[low_col]).sum()
        if inconsistent > 0:
            self.issues.append(
                f"{inconsistent} periods where High < Low - DATA ERROR"
            )

    def check_extreme_volatility(self):
        """Crypto can have extreme moves, but >90% daily change is suspicious."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'close' not in col_map:
            return

        close_col = col_map['close']
        returns = self.df[close_col].pct_change()
        extreme_moves = (returns.abs() > 0.9).sum()

        if extreme_moves > len(self.df) * 0.01:
            self.issues.append(
                f"{extreme_moves} periods with >90% moves - flash crash or data error?"
            )

    def check_exchange_issues(self):
        """Check for exchange downtime or manipulation."""
        col_map = {col.lower(): col for col in self.df.columns}

        if 'volume' not in col_map:
            return

        volume_col = col_map['volume']
        volume = self.df[volume_col].replace(0, np.nan)
        volume_mean = volume.rolling(60, min_periods=20).mean()

        low_volume_days = (volume < volume_mean * 0.05).sum()
        if low_volume_days > 0:
            self.issues.append(
                f"{low_volume_days} periods with extremely low volume (<5% of avg) - exchange issues?"
            )

    def check_stablecoin_peg(self):
        """If this is a stablecoin, check if it maintains peg."""
        stablecoins = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD']
        if not any(stable in self.symbol.upper() for stable in stablecoins):
            return

        col_map = {col.lower(): col for col in self.df.columns}
        if 'close' not in col_map:
            return

        close_col = col_map['close']
        price_deviation = (self.df[close_col] - 1.0).abs()
        depeg_events = (price_deviation > 0.05).sum()

        if depeg_events > 0:
            max_deviation = price_deviation.max()
            self.issues.append(
                f"Stablecoin {self.symbol}: {depeg_events} periods with >5% depeg. "
                f"Max deviation: ${max_deviation:.3f}"
            )


class SentimentDataValidator:
    """Validates sentiment data from Reddit/Trends."""

    def __init__(self, df: pd.DataFrame, ticker: str):
        """
        Initialize validator.

        Args:
            df: DataFrame with sentiment data
            ticker: Ticker symbol
        """
        self.df = df.copy()
        self.ticker = ticker
        self.issues = []

    def run_all_checks(self) -> Tuple[bool, List[str]]:
        """Run sentiment validation checks."""
        self.check_mention_counts()
        self.check_sentiment_range()
        self.check_sudden_spikes()
        self.check_data_freshness()

        is_valid = len(self.issues) == 0
        return is_valid, self.issues

    def check_mention_counts(self):
        """Mentions should be non-negative."""
        mention_cols = [col for col in self.df.columns if 'mention' in col.lower()]

        for col in mention_cols:
            negative_mentions = (self.df[col] < 0).sum()
            if negative_mentions > 0:
                self.issues.append(
                    f"{negative_mentions} records with negative {col} - DATA ERROR"
                )

    def check_sentiment_range(self):
        """VADER sentiment should be in range [-1, 1]."""
        sentiment_cols = [col for col in self.df.columns if 'sentiment' in col.lower()]

        for col in sentiment_cols:
            out_of_range = (
                (self.df[col] < -1) |
                (self.df[col] > 1)
            ).sum()

            if out_of_range > 0:
                self.issues.append(
                    f"{out_of_range} records with {col} outside [-1, 1] range - DATA ERROR"
                )

    def check_sudden_spikes(self):
        """Flag sudden 100x mention spikes (possible bot activity)."""
        mention_cols = [col for col in self.df.columns if 'mention' in col.lower() and 'reddit' in col.lower()]

        for col in mention_cols:
            mentions = self.df[col].replace(0, 1)
            mention_ratio = mentions / mentions.shift(1)

            extreme_spikes = (mention_ratio > 100).sum()
            if extreme_spikes > 0:
                self.issues.append(
                    f"{extreme_spikes} periods with >100x {col} spikes - possible bot campaign"
                )

    def check_data_freshness(self):
        """Ensure sentiment data is recent."""
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]

        if date_cols:
            date_col = date_cols[0]
            latest_date = pd.to_datetime(self.df[date_col]).max()
            days_old = (pd.Timestamp.now() - latest_date).days

            if days_old > 7:
                self.issues.append(
                    f"Sentiment data is {days_old} days old - needs update"
                )
