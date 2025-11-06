"""
Technical feature engineering using TA-Lib.
Computes momentum, trend, volatility, and volume indicators.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a ticker.

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

    Returns:
        DataFrame with added technical indicator columns
    """
    try:
        df = df.copy()

        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return df

        # Convert to uppercase for pandas_ta compatibility
        df_ta = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # MOMENTUM INDICATORS
        # RSI
        df['rsi_14'] = ta.rsi(df_ta['Close'], length=14)

        # MACD
        macd = ta.macd(df_ta['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_histogram'] = macd['MACDh_12_26_9']

        # Stochastic Oscillator
        stoch = ta.stoch(df_ta['High'], df_ta['Low'], df_ta['Close'])
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']

        # ROC (Rate of Change)
        df['roc_1'] = ta.roc(df_ta['Close'], length=1)
        df['roc_5'] = ta.roc(df_ta['Close'], length=5)
        df['roc_10'] = ta.roc(df_ta['Close'], length=10)
        df['roc_21'] = ta.roc(df_ta['Close'], length=21)

        # TREND INDICATORS
        # Moving Averages
        df['sma_20'] = ta.sma(df_ta['Close'], length=20)
        df['sma_50'] = ta.sma(df_ta['Close'], length=50)
        df['sma_200'] = ta.sma(df_ta['Close'], length=200)
        df['ema_20'] = ta.ema(df_ta['Close'], length=20)
        df['ema_50'] = ta.ema(df_ta['Close'], length=50)

        # ADX (Average Directional Index)
        adx = ta.adx(df_ta['High'], df_ta['Low'], df_ta['Close'], length=20)
        if adx is not None:
            df['adx_20'] = adx['ADX_20']

        # VOLATILITY INDICATORS
        # Bollinger Bands
        bbands = ta.bbands(df_ta['Close'], length=20, std=2)
        if bbands is not None:
            df['bb_upper'] = bbands['BBU_20_2.0']
            df['bb_middle'] = bbands['BBM_20_2.0']
            df['bb_lower'] = bbands['BBL_20_2.0']
            df['bbw'] = bbands['BBB_20_2.0']  # Bandwidth

        # ATR (Average True Range)
        df['atr_14'] = ta.atr(df_ta['High'], df_ta['Low'], df_ta['Close'], length=14)

        # VOLUME INDICATORS
        # OBV (On-Balance Volume)
        df['obv'] = ta.obv(df_ta['Close'], df_ta['Volume'])

        # Volume SMA
        df['volume_sma_20'] = ta.sma(df_ta['Volume'], length=20)

        # ENGINEERED FEATURES
        # Volume spike ratio
        df['volume_spike_ratio'] = df['volume'] / df['volume_sma_20']

        # RVOL (Relative Volume)
        volume_sma_200 = ta.sma(df_ta['Volume'], length=200)
        df['rvol'] = df['volume'] / volume_sma_200

        # BBW Percentile (100 days)
        if 'bbw' in df.columns:
            df['bbw_percentile_100d'] = df['bbw'].rolling(100).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else np.nan
            )

        # ATR Percentile (100 days)
        if 'atr_14' in df.columns:
            df['atr_percentile_100d'] = df['atr_14'].rolling(100).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else np.nan
            )

        # Price to 52-week high
        df['price_to_52w_high'] = df['close'] / df['high'].rolling(252).max()

        # EMA crossovers
        df['ema_20_cross'] = ((df['close'] > df['ema_20']) &
                              (df['close'].shift(1) <= df['ema_20'].shift(1))).astype(int)

        df['ema_50_cross'] = ((df['close'] > df['ema_50']) &
                              (df['close'].shift(1) <= df['ema_50'].shift(1))).astype(int)

        # Price range percentile
        price_range = df['high'] - df['low']
        df['price_range_percentile'] = price_range.rolling(100).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else np.nan
        )

        logger.debug(f"Computed technical features. Shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error computing technical features: {str(e)}")
        return df
