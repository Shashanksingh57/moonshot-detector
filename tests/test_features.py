"""
Unit tests for feature engineering modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.technical_features import compute_technical_features


def test_compute_technical_features():
    """Test technical feature computation."""
    # Create sample OHLCV data
    np.random.seed(42)
    n = 300

    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(n) * 2),
        'high': 105 + np.cumsum(np.random.randn(n) * 2),
        'low': 95 + np.cumsum(np.random.randn(n) * 2),
        'close': 100 + np.cumsum(np.random.randn(n) * 2),
        'volume': np.random.randint(1000000, 10000000, n)
    })

    # Ensure OHLC relationships are valid
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + 1
    df['low'] = df[['open', 'low', 'close']].min(axis=1) - 1

    # Compute features
    df_features = compute_technical_features(df)

    # Check that features were added
    assert 'rsi_14' in df_features.columns, "RSI feature should be present"
    assert 'macd' in df_features.columns, "MACD feature should be present"
    assert 'atr_14' in df_features.columns, "ATR feature should be present"
    assert 'obv' in df_features.columns, "OBV feature should be present"

    # Check for NaN values (some are expected at the beginning due to rolling windows)
    assert df_features['rsi_14'].notna().sum() > 0, "RSI should have some valid values"

    # Check value ranges
    rsi_values = df_features['rsi_14'].dropna()
    assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), "RSI should be between 0 and 100"

    print("✓ Technical features test passed")


def test_feature_no_lookahead():
    """Test that features don't use future information."""
    # Create data with a known pattern
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Price jumps up on day 50
    prices = np.ones(n) * 100
    prices[50:] = 150

    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': np.ones(n) * 1000000
    })

    df_features = compute_technical_features(df)

    # SMA at day 49 should not know about the jump at day 50
    if 'sma_20' in df_features.columns and len(df_features) > 49:
        sma_before_jump = df_features.loc[49, 'sma_20']
        # SMA should still be around 100, not influenced by 150
        assert sma_before_jump < 110, "SMA should not use future information"

    print("✓ No look-ahead test passed")


if __name__ == '__main__':
    test_compute_technical_features()
    test_feature_no_lookahead()
    print("\nAll feature tests passed! ✓")
