"""
Statistical validation for backtests.
Implements t-statistic, Probability of Backtest Overfitting (PBO), and Deflated Sharpe Ratio.
"""

import numpy as np
import pandas as pd
from typing import List
from scipy import stats
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def calculate_t_statistic(returns: pd.Series) -> float:
    """
    Calculate t-statistic for Sharpe ratio.

    CRITICAL: Require t > 3.0 (not standard 1.96) due to multiple hypothesis testing.

    Reference: Bailey & Lopez de Prado (2014)

    Args:
        returns: Series of returns

    Returns:
        t-statistic
    """
    try:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        n = len(returns)
        mean_return = returns.mean()
        std_return = returns.std()

        # t-stat for Sharpe ratio
        t_stat = (mean_return / std_return) * np.sqrt(n)

        return t_stat

    except Exception as e:
        logger.error(f"Error calculating t-statistic: {str(e)}")
        return 0.0


def calculate_deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    skew: float = None,
    kurtosis: float = None,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Deflated Sharpe Ratio.

    Adjusts Sharpe ratio for multiple testing and non-Gaussianity.

    Reference: Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"

    Args:
        returns: Series of returns
        n_trials: Number of trials/strategies tested
        skew: Skewness of returns (calculated if not provided)
        kurtosis: Excess kurtosis (calculated if not provided)
        risk_free_rate: Annual risk-free rate

    Returns:
        Deflated Sharpe Ratio
    """
    try:
        if len(returns) == 0:
            return 0.0

        # Calculate observed Sharpe ratio
        mean_return = returns.mean() * 252  # Annualize
        std_return = returns.std() * np.sqrt(252)

        if std_return == 0:
            return 0.0

        observed_sharpe = (mean_return - risk_free_rate) / std_return

        # Calculate skew and kurtosis if not provided
        if skew is None:
            skew = stats.skew(returns)

        if kurtosis is None:
            kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis

        # Number of observations
        n = len(returns)

        # Expected maximum Sharpe ratio under null hypothesis (multiple testing adjustment)
        # This is the Sharpe ratio you'd expect to see by chance given n_trials
        expected_max_sharpe = np.sqrt(2 * np.log(n_trials))

        # Standard error of Sharpe ratio
        # Adjusted for non-Gaussianity
        sharpe_std = np.sqrt((1 + 0.5 * observed_sharpe**2 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2) / n)

        # Deflated Sharpe Ratio
        # This is a z-score: how many standard errors is observed Sharpe above expected max
        deflated_sharpe = (observed_sharpe - expected_max_sharpe) / sharpe_std

        logger.debug(f"Observed Sharpe: {observed_sharpe:.2f}")
        logger.debug(f"Expected Max Sharpe (n_trials={n_trials}): {expected_max_sharpe:.2f}")
        logger.debug(f"Deflated Sharpe: {deflated_sharpe:.2f}")

        return deflated_sharpe

    except Exception as e:
        logger.error(f"Error calculating Deflated Sharpe Ratio: {str(e)}")
        return 0.0


def calculate_pbo(
    in_sample_returns: List[pd.Series],
    out_sample_returns: List[pd.Series]
) -> float:
    """
    Calculate Probability of Backtest Overfitting (PBO).

    Measures the likelihood that the best in-sample strategy was selected by chance
    and will underperform out-of-sample.

    Reference: Bailey et al. (2015) "The Probability of Backtest Overfitting"

    Args:
        in_sample_returns: List of return series from in-sample periods
        out_sample_returns: List of return series from out-of-sample periods

    Returns:
        PBO (0-1, where >0.5 indicates likely overfitting)
    """
    try:
        if len(in_sample_returns) != len(out_sample_returns):
            logger.error("In-sample and out-of-sample returns must have same length")
            return 1.0

        n_runs = len(in_sample_returns)

        if n_runs < 2:
            logger.warning("Need at least 2 runs for PBO calculation")
            return 1.0

        # Calculate Sharpe ratio for each run
        sharpe_is = []
        sharpe_oos = []

        for returns_is, returns_oos in zip(in_sample_returns, out_sample_returns):
            # In-sample Sharpe
            if len(returns_is) > 0 and returns_is.std() > 0:
                sharpe_is.append(returns_is.mean() / returns_is.std() * np.sqrt(252))
            else:
                sharpe_is.append(0)

            # Out-of-sample Sharpe
            if len(returns_oos) > 0 and returns_oos.std() > 0:
                sharpe_oos.append(returns_oos.mean() / returns_oos.std() * np.sqrt(252))
            else:
                sharpe_oos.append(0)

        # Rank correlation between IS and OOS performance
        # If strategies that perform well IS also perform well OOS, correlation is high
        # Low/negative correlation indicates overfitting

        # Count how many times best IS strategy underperforms median OOS
        best_is_idx = np.argmax(sharpe_is)
        median_sharpe_oos = np.median(sharpe_oos)

        # PBO: probability that best IS strategy has OOS Sharpe < median OOS Sharpe
        # Simplified calculation: if best IS underperforms median OOS, likely overfit

        count_underperform = 0
        for i in range(n_runs):
            if sharpe_is[i] >= sharpe_is[best_is_idx] * 0.9:  # Top performing IS strategies
                if sharpe_oos[i] < median_sharpe_oos:
                    count_underperform += 1

        pbo = count_underperform / n_runs if n_runs > 0 else 1.0

        logger.debug(f"IS Sharpe ratios: {sharpe_is}")
        logger.debug(f"OOS Sharpe ratios: {sharpe_oos}")
        logger.debug(f"Best IS index: {best_is_idx}, Sharpe: {sharpe_is[best_is_idx]:.2f}")
        logger.debug(f"Median OOS Sharpe: {median_sharpe_oos:.2f}")
        logger.debug(f"PBO: {pbo:.2%}")

        return pbo

    except Exception as e:
        logger.error(f"Error calculating PBO: {str(e)}")
        return 1.0


def validate_backtest(
    returns: pd.Series,
    config: Dict,
    n_trials: int = 100,
    in_sample_returns: List[pd.Series] = None,
    out_sample_returns: List[pd.Series] = None
) -> Dict:
    """
    Complete backtest validation.

    Checks all statistical requirements:
    - t-statistic > 3.0
    - PBO < 0.30
    - Deflated Sharpe > 1.0

    Args:
        returns: Series of returns
        config: Configuration dictionary
        n_trials: Number of strategies tested
        in_sample_returns: List of IS returns (for PBO)
        out_sample_returns: List of OOS returns (for PBO)

    Returns:
        Dictionary with validation results
    """
    try:
        logger.info("="*70)
        logger.info("STATISTICAL VALIDATION")
        logger.info("="*70)

        validation = {}

        # Get requirements from config
        requirements = config.get('backtesting', {}).get('validation_requirements', {})
        min_t_stat = requirements.get('min_t_statistic', 3.0)
        max_pbo = requirements.get('max_pbo', 0.30)
        min_trades = requirements.get('min_trades', 50)

        # 1. t-statistic
        t_stat = calculate_t_statistic(returns)
        validation['t_statistic'] = t_stat
        validation['t_stat_pass'] = t_stat >= min_t_stat

        logger.info(f"t-statistic: {t_stat:.2f} (require > {min_t_stat})")
        if not validation['t_stat_pass']:
            logger.warning(f"  ⚠️  FAILED - High risk of overfitting")

        # 2. Deflated Sharpe Ratio
        deflated_sharpe = calculate_deflated_sharpe(returns, n_trials)
        validation['deflated_sharpe'] = deflated_sharpe
        validation['deflated_sharpe_pass'] = deflated_sharpe >= 1.0

        logger.info(f"Deflated Sharpe Ratio: {deflated_sharpe:.2f} (require > 1.0)")
        if not validation['deflated_sharpe_pass']:
            logger.warning(f"  ⚠️  FAILED - Performance likely due to chance given {n_trials} trials")

        # 3. PBO (if data provided)
        if in_sample_returns and out_sample_returns:
            pbo = calculate_pbo(in_sample_returns, out_sample_returns)
            validation['pbo'] = pbo
            validation['pbo_pass'] = pbo <= max_pbo

            logger.info(f"Probability of Backtest Overfitting: {pbo:.2%} (require < {max_pbo:.0%})")
            if not validation['pbo_pass']:
                logger.warning(f"  ⚠️  FAILED - High probability of overfitting")
        else:
            validation['pbo'] = None
            validation['pbo_pass'] = None
            logger.info("PBO: Not calculated (need multiple IS/OOS splits)")

        # 4. Number of trades
        n_trades = len(returns)
        validation['n_trades'] = n_trades
        validation['n_trades_pass'] = n_trades >= min_trades

        logger.info(f"Number of trades: {n_trades} (require > {min_trades})")
        if not validation['n_trades_pass']:
            logger.warning(f"  ⚠️  FAILED - Insufficient trades for statistical significance")

        # Overall validation
        required_checks = ['t_stat_pass', 'deflated_sharpe_pass', 'n_trades_pass']
        if validation['pbo_pass'] is not None:
            required_checks.append('pbo_pass')

        validation['all_checks_pass'] = all(validation[check] for check in required_checks)

        logger.info("="*70)
        if validation['all_checks_pass']:
            logger.info("✅ VALIDATION PASSED - Strategy meets all requirements")
        else:
            logger.warning("❌ VALIDATION FAILED - Strategy does not meet requirements")
            logger.warning("DO NOT TRADE THIS STRATEGY WITH REAL MONEY")

        logger.info("="*70)

        return validation

    except Exception as e:
        logger.error(f"Error in backtest validation: {str(e)}")
        return {'all_checks_pass': False}
