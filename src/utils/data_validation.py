"""
Data validation utilities for moonshot-detector.
Validates data quality, checks for anomalies, and ensures data integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def validate_ohlcv_data(df: pd.DataFrame, ticker: str) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV (Open, High, Low, Close, Volume) data.

    Args:
        df: DataFrame with OHLCV columns
        ticker: Ticker symbol for logging

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        return False, issues

    # Check for empty dataframe
    if len(df) == 0:
        issues.append("DataFrame is empty")
        return False, issues

    # Check for missing values
    for col in required_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            null_pct = (null_count / len(df)) * 100
            issues.append(f"Column '{col}' has {null_count} null values ({null_pct:.2f}%)")

    # Check for negative values
    for col in required_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            issues.append(f"Column '{col}' has {negative_count} negative values")

    # Check OHLC relationships: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
    invalid_high_low = (df['High'] < df['Low']).sum()
    if invalid_high_low > 0:
        issues.append(f"Found {invalid_high_low} rows where High < Low")

    invalid_high = ((df['High'] < df['Open']) | (df['High'] < df['Close'])).sum()
    if invalid_high > 0:
        issues.append(f"Found {invalid_high} rows where High is not the highest price")

    invalid_low = ((df['Low'] > df['Open']) | (df['Low'] > df['Close'])).sum()
    if invalid_low > 0:
        issues.append(f"Found {invalid_low} rows where Low is not the lowest price")

    # Check for zero prices
    zero_prices = ((df['Open'] == 0) | (df['High'] == 0) | (df['Low'] == 0) | (df['Close'] == 0)).sum()
    if zero_prices > 0:
        issues.append(f"Found {zero_prices} rows with zero prices")

    # Check for zero volume (warning only, not critical)
    zero_volume = (df['Volume'] == 0).sum()
    if zero_volume > 0:
        zero_vol_pct = (zero_volume / len(df)) * 100
        if zero_vol_pct > 5:  # Only flag if more than 5%
            issues.append(f"Found {zero_volume} rows with zero volume ({zero_vol_pct:.2f}%)")

    # Check for data gaps (missing dates)
    if isinstance(df.index, pd.DatetimeIndex):
        date_diff = df.index.to_series().diff()
        # For daily data, gaps > 7 days might be suspicious (excluding weekends/holidays)
        large_gaps = date_diff[date_diff > timedelta(days=7)]
        if len(large_gaps) > 0:
            issues.append(f"Found {len(large_gaps)} date gaps larger than 7 days")

    # Check for duplicate dates
    if isinstance(df.index, pd.DatetimeIndex):
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate dates")

    # Check for unrealistic price changes (>50% in one day)
    if 'Close' in df.columns:
        returns = df['Close'].pct_change()
        extreme_moves = (abs(returns) > 0.5).sum()
        if extreme_moves > 0:
            extreme_pct = (extreme_moves / len(df)) * 100
            if extreme_pct > 1:  # Only flag if more than 1%
                issues.append(f"Found {extreme_moves} days with >50% price change ({extreme_pct:.2f}%)")

    is_valid = len(issues) == 0

    if not is_valid:
        logger.warning(f"Validation issues for {ticker}: {'; '.join(issues)}")
    else:
        logger.debug(f"Data validation passed for {ticker}")

    return is_valid, issues


def validate_fundamental_data(df: pd.DataFrame, ticker: str) -> Tuple[bool, List[str]]:
    """
    Validate fundamental data.

    Args:
        df: DataFrame with fundamental data
        ticker: Ticker symbol

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    if len(df) == 0:
        issues.append("DataFrame is empty")
        return False, issues

    # Check for common fundamental columns
    common_cols = ['EPS', 'Revenue', 'Total_Assets', 'Total_Liabilities']
    present_cols = [col for col in common_cols if col in df.columns]

    if len(present_cols) == 0:
        issues.append(f"No common fundamental columns found. Expected: {common_cols}")
        return False, issues

    # Check for excessive missing values
    for col in present_cols:
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        if null_pct > 50:
            issues.append(f"Column '{col}' has {null_pct:.2f}% missing values")

    # Check for unrealistic values
    if 'Total_Assets' in df.columns and 'Total_Liabilities' in df.columns:
        # Liabilities shouldn't be more than 10x assets (very high leverage)
        high_leverage = (df['Total_Liabilities'] > df['Total_Assets'] * 10).sum()
        if high_leverage > 0:
            issues.append(f"Found {high_leverage} periods with unrealistic leverage (Liab > 10x Assets)")

    is_valid = len(issues) == 0

    if not is_valid:
        logger.warning(f"Fundamental validation issues for {ticker}: {'; '.join(issues)}")

    return is_valid, issues


def check_data_freshness(file_path: str, max_age_days: int = 7) -> bool:
    """
    Check if a data file is fresh (recently updated).

    Args:
        file_path: Path to the data file
        max_age_days: Maximum acceptable age in days

    Returns:
        True if data is fresh, False otherwise
    """
    import os
    from pathlib import Path

    path = Path(file_path)

    if not path.exists():
        logger.debug(f"File does not exist: {file_path}")
        return False

    # Get file modification time
    mod_time = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - mod_time

    is_fresh = age.days <= max_age_days

    if not is_fresh:
        logger.info(f"File {file_path} is {age.days} days old (max {max_age_days} days)")

    return is_fresh


def detect_outliers(series: pd.Series, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a series.

    Args:
        series: Pandas Series
        method: 'zscore' or 'iqr'
        threshold: Z-score threshold (for zscore method) or IQR multiplier (for iqr method)

    Returns:
        Boolean series where True indicates an outlier
    """
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold

    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'")


def validate_date_alignment(
    dfs: Dict[str, pd.DataFrame],
    ticker: str
) -> Tuple[bool, List[str]]:
    """
    Validate that multiple dataframes for the same ticker have aligned dates.

    Args:
        dfs: Dictionary of {data_type: dataframe}
        ticker: Ticker symbol

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check that all dataframes have DatetimeIndex
    for name, df in dfs.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append(f"{name} does not have a DatetimeIndex")

    if issues:
        return False, issues

    # Check date ranges overlap
    date_ranges = {}
    for name, df in dfs.items():
        date_ranges[name] = (df.index.min(), df.index.max())

    # Find common date range
    min_start = max(start for start, _ in date_ranges.values())
    max_end = min(end for _, end in date_ranges.values())

    if min_start >= max_end:
        issues.append(f"No overlapping date range found across datasets")
        return False, issues

    # Log date range info
    logger.debug(f"{ticker} - Common date range: {min_start} to {max_end}")

    return len(issues) == 0, issues
