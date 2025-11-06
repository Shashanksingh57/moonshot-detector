"""
Transaction cost modeling - applies costs at 2X expected levels.
Models bid-ask spread, slippage, and fees.
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def apply_transaction_costs(
    trades_df: pd.DataFrame,
    asset_type: str,
    config: Dict,
    cost_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Apply transaction costs to trades.

    CRITICAL: Test at 2X expected costs to ensure robustness.

    Args:
        trades_df: DataFrame with trades (must have 'entry_price', 'exit_price', 'volume' columns)
        asset_type: 'stocks' or 'crypto'
        config: Configuration dictionary
        cost_multiplier: Multiplier for costs (default 2.0 for conservative testing)

    Returns:
        DataFrame with costs applied and net returns
    """
    try:
        df = trades_df.copy()

        # Get cost configuration
        cost_config = config.get('backtesting', {}).get('transaction_costs', {}).get(asset_type, {})

        # Base costs (in basis points)
        expected_bps = cost_config.get('expected_bps', 10)
        test_bps = expected_bps * cost_multiplier

        # Fixed commission
        fixed_commission = cost_config.get('fixed_commission_usd', 0)

        # Exchange fees (for crypto)
        exchange_fee_pct = cost_config.get('exchange_fee_pct', 0)

        logger.info(f"Applying transaction costs ({asset_type}):")
        logger.info(f"  Base cost: {test_bps} bps (2x {expected_bps} bps)")
        logger.info(f"  Fixed commission: ${fixed_commission}")
        if exchange_fee_pct > 0:
            logger.info(f"  Exchange fee: {exchange_fee_pct}%")

        # Calculate costs per trade
        # Entry costs
        df['entry_cost_bps'] = test_bps
        df['entry_cost_pct'] = test_bps / 10000  # Convert bps to percentage

        # Exit costs
        df['exit_cost_bps'] = test_bps
        df['exit_cost_pct'] = test_bps / 10000

        # Total cost percentage (entry + exit)
        df['total_cost_pct'] = df['entry_cost_pct'] + df['exit_cost_pct']

        # Add exchange fees (if crypto)
        if exchange_fee_pct > 0:
            df['exchange_fee_pct'] = exchange_fee_pct / 100 * 2  # Entry + Exit
            df['total_cost_pct'] += df['exchange_fee_pct']

        # Calculate net return after costs
        if 'gross_return' in df.columns:
            df['net_return'] = df['gross_return'] - df['total_cost_pct']
        elif 'return' in df.columns:
            df['gross_return'] = df['return']
            df['net_return'] = df['gross_return'] - df['total_cost_pct']

        # Log average cost impact
        avg_cost_impact = df['total_cost_pct'].mean()
        logger.info(f"Average total cost per trade: {avg_cost_impact:.2%}")

        return df

    except Exception as e:
        logger.error(f"Error applying transaction costs: {str(e)}")
        return trades_df


def model_slippage(
    trades_df: pd.DataFrame,
    asset_type: str,
    config: Dict
) -> pd.DataFrame:
    """
    Model slippage based on liquidity tiers.

    Args:
        trades_df: DataFrame with trades (must have 'volume', 'avg_daily_volume' columns)
        asset_type: 'stocks' or 'crypto'
        config: Configuration dictionary

    Returns:
        DataFrame with slippage estimates
    """
    try:
        df = trades_df.copy()

        # Get slippage configuration
        slippage_config = config.get('backtesting', {}).get('slippage_model', {}).get(asset_type, {})

        # Determine liquidity tier based on market cap / volume
        # For simplicity, use default mid-tier slippage
        # In production, would categorize each stock/crypto

        if asset_type == 'stocks':
            default_slippage_bps = slippage_config.get('mid_cap_bps', 10)
        else:  # crypto
            default_slippage_bps = slippage_config.get('top50_bps', 25)

        df['slippage_bps'] = default_slippage_bps
        df['slippage_pct'] = default_slippage_bps / 10000

        logger.debug(f"Applied slippage: {default_slippage_bps} bps")

        return df

    except Exception as e:
        logger.error(f"Error modeling slippage: {str(e)}")
        return trades_df


def calculate_total_costs(
    trades_df: pd.DataFrame,
    asset_type: str,
    config: Dict,
    include_slippage: bool = True
) -> pd.DataFrame:
    """
    Calculate total costs including transaction costs and slippage.

    Args:
        trades_df: DataFrame with trades
        asset_type: 'stocks' or 'crypto'
        config: Configuration dictionary
        include_slippage: Whether to include slippage modeling

    Returns:
        DataFrame with all costs calculated
    """
    try:
        # Apply transaction costs (at 2X)
        df = apply_transaction_costs(trades_df, asset_type, config, cost_multiplier=2.0)

        # Add slippage
        if include_slippage:
            df = model_slippage(df, asset_type, config)

            # Add slippage to total costs
            if 'slippage_pct' in df.columns:
                df['total_cost_pct'] += df['slippage_pct']

                # Recalculate net return
                df['net_return'] = df['gross_return'] - df['total_cost_pct']

        return df

    except Exception as e:
        logger.error(f"Error calculating total costs: {str(e)}")
        return trades_df


def analyze_cost_impact(trades_df: pd.DataFrame) -> Dict:
    """
    Analyze the impact of transaction costs on returns.

    Args:
        trades_df: DataFrame with trades (must have 'gross_return', 'net_return')

    Returns:
        Dictionary with cost impact metrics
    """
    try:
        metrics = {}

        if 'gross_return' not in trades_df.columns or 'net_return' not in trades_df.columns:
            return metrics

        # Total returns
        gross_total = trades_df['gross_return'].sum()
        net_total = trades_df['net_return'].sum()

        metrics['gross_return_total'] = gross_total
        metrics['net_return_total'] = net_total
        metrics['cost_impact_total'] = gross_total - net_total
        metrics['cost_impact_pct'] = (gross_total - net_total) / abs(gross_total) if gross_total != 0 else 0

        # Average per trade
        metrics['avg_gross_return'] = trades_df['gross_return'].mean()
        metrics['avg_net_return'] = trades_df['net_return'].mean()
        metrics['avg_cost_per_trade'] = metrics['avg_gross_return'] - metrics['avg_net_return']

        # Check if still profitable
        metrics['profitable_after_costs'] = net_total > 0

        logger.info("="*70)
        logger.info("COST IMPACT ANALYSIS")
        logger.info("="*70)
        logger.info(f"Gross Return: {gross_total:.2%}")
        logger.info(f"Net Return (after costs): {net_total:.2%}")
        logger.info(f"Cost Impact: {metrics['cost_impact_total']:.2%} ({metrics['cost_impact_pct']:.1%} of gross)")
        logger.info(f"Avg Cost per Trade: {metrics['avg_cost_per_trade']:.2%}")

        if not metrics['profitable_after_costs']:
            logger.warning("⚠️  STRATEGY NOT PROFITABLE AFTER COSTS")

        logger.info("="*70)

        return metrics

    except Exception as e:
        logger.error(f"Error analyzing cost impact: {str(e)}")
        return {}
