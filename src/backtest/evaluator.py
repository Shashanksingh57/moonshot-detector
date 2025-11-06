"""
Performance evaluator - calculates all metrics for backtest validation.
Includes classification metrics, trading metrics, and statistical validation.
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    try:
        metrics = {}

        # Basic metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = tp
        metrics['false_positives'] = fp
        metrics['true_negatives'] = tn
        metrics['false_negatives'] = fn

        # Win rate
        metrics['win_rate'] = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Probability-based metrics (if available)
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            except:
                pass

        return metrics

    except Exception as e:
        logger.error(f"Error calculating classification metrics: {str(e)}")
        return {}


def calculate_tstatistic(returns: pd.Series) -> float:
    """
    Calculate t-statistic for Sharpe ratio.

    CRITICAL: Require t > 3.0 (not standard 1.96) due to multiple hypothesis testing.

    Args:
        returns: Series of returns

    Returns:
        t-statistic
    """
    try:
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()
        n = len(returns)

        if std_return == 0:
            return 0.0

        t_stat = (mean_return / std_return) * np.sqrt(n)

        return t_stat

    except Exception as e:
        logger.error(f"Error calculating t-statistic: {str(e)}")
        return 0.0


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Sharpe ratio
    """
    try:
        if len(returns) == 0:
            return 0.0

        # Annualize
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - risk_free_rate) / std_return

        return sharpe

    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return 0.0


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        cumulative_returns: Cumulative returns series

    Returns:
        Maximum drawdown (as positive percentage)
    """
    try:
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        max_dd = abs(drawdown.min())

        return max_dd

    except Exception as e:
        logger.error(f"Error calculating max drawdown: {str(e)}")
        return 0.0


def calculate_trading_metrics(
    signals_df: pd.DataFrame,
    returns_df: pd.DataFrame
) -> Dict:
    """
    Calculate trading-specific metrics.

    Args:
        signals_df: DataFrame with signals (ticker, date, signal)
        returns_df: DataFrame with actual returns (ticker, date, return)

    Returns:
        Dictionary of trading metrics
    """
    try:
        metrics = {}

        # Merge signals with returns
        df = pd.merge(
            signals_df,
            returns_df,
            on=['ticker', 'date'],
            how='inner'
        )

        if len(df) == 0:
            return metrics

        # Basic stats
        metrics['total_trades'] = len(df)
        metrics['avg_return'] = df['return'].mean()
        metrics['median_return'] = df['return'].median()

        # Win/Loss stats
        wins = df[df['return'] > 0]
        losses = df[df['return'] <= 0]

        metrics['num_wins'] = len(wins)
        metrics['num_losses'] = len(losses)
        metrics['win_rate'] = len(wins) / len(df) if len(df) > 0 else 0

        metrics['avg_win'] = wins['return'].mean() if len(wins) > 0 else 0
        metrics['avg_loss'] = losses['return'].mean() if len(losses) > 0 else 0

        # Risk-adjusted metrics
        if len(df) > 0:
            metrics['sharpe_ratio'] = calculate_sharpe_ratio(df['return'])

            # Calculate cumulative returns
            cumulative = (1 + df['return']).cumprod()
            metrics['max_drawdown'] = calculate_max_drawdown(cumulative)

            metrics['total_return'] = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0

        # Statistical significance
        metrics['t_statistic'] = calculate_tstatistic(df['return'])

        return metrics

    except Exception as e:
        logger.error(f"Error calculating trading metrics: {str(e)}")
        return {}


def evaluate_backtest(
    predictions_df: pd.DataFrame,
    actual_returns_df: pd.DataFrame
) -> Dict:
    """
    Complete backtest evaluation.

    Args:
        predictions_df: DataFrame with predictions (ticker, date, prediction, proba)
        actual_returns_df: DataFrame with actual returns (ticker, date, return, label)

    Returns:
        Dictionary of all metrics
    """
    try:
        logger.info("Evaluating backtest performance...")

        # Merge predictions with actual results
        df = pd.merge(
            predictions_df,
            actual_returns_df,
            on=['ticker', 'date'],
            how='inner'
        )

        if len(df) == 0:
            logger.error("No data to evaluate")
            return {}

        # Classification metrics
        y_true = df['label'].values
        y_pred = df['prediction'].values
        y_proba = df['proba'].values if 'proba' in df.columns else None

        classification_metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        # Trading metrics
        # Create signals DataFrame
        signals_df = df[df['prediction'] == 1][['ticker', 'date']].copy()
        signals_df['signal'] = 'BUY'

        returns_df = df[['ticker', 'date', 'return']].copy()

        trading_metrics = calculate_trading_metrics(signals_df, returns_df)

        # Combine all metrics
        all_metrics = {
            **classification_metrics,
            **trading_metrics
        }

        # Log key metrics
        logger.info("="*70)
        logger.info("BACKTEST EVALUATION RESULTS")
        logger.info("="*70)
        logger.info(f"Classification Metrics:")
        logger.info(f"  Precision: {all_metrics.get('precision', 0):.2%}")
        logger.info(f"  Recall: {all_metrics.get('recall', 0):.2%}")
        logger.info(f"  F1 Score: {all_metrics.get('f1', 0):.3f}")
        logger.info(f"")
        logger.info(f"Trading Metrics:")
        logger.info(f"  Total Trades: {all_metrics.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {all_metrics.get('win_rate', 0):.2%}")
        logger.info(f"  Avg Win: {all_metrics.get('avg_win', 0):.2%}")
        logger.info(f"  Avg Loss: {all_metrics.get('avg_loss', 0):.2%}")
        logger.info(f"  Total Return: {all_metrics.get('total_return', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {all_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Max Drawdown: {all_metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"")
        logger.info(f"Statistical Validation:")
        logger.info(f"  t-statistic: {all_metrics.get('t_statistic', 0):.2f} (require > 3.0)")
        logger.info("="*70)

        # Validation checks
        t_stat = all_metrics.get('t_statistic', 0)
        if t_stat < 3.0:
            logger.warning(f"⚠️  t-statistic {t_stat:.2f} < 3.0 - HIGH RISK OF OVERFITTING")

        return all_metrics

    except Exception as e:
        logger.error(f"Error evaluating backtest: {str(e)}")
        return {}
