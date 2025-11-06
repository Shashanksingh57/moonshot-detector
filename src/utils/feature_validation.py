"""
Feature validation - checks for look-ahead bias, feature leakage, and data quality issues.

CRITICAL: Feature validation is the last line of defense before training.
Look-ahead bias or feature leakage = overfitted models = disaster in production.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureValidator:
    """Validates engineered features before training."""

    def __init__(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, ticker: str):
        """
        Initialize feature validator.

        Args:
            features_df: DataFrame with engineered features
            labels_df: DataFrame with labels
            ticker: Ticker symbol
        """
        self.features = features_df.copy()
        self.labels = labels_df.copy()
        self.ticker = ticker
        self.issues = []

    def run_all_checks(self) -> Tuple[bool, List[str]]:
        """
        Run all validation checks.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        self.check_feature_alignment()
        self.check_look_ahead_bias()
        self.check_feature_leakage()
        self.check_inf_nan()
        self.check_feature_drift()
        self.check_target_leakage()

        is_valid = len(self.issues) == 0
        return is_valid, self.issues

    def check_feature_alignment(self):
        """Features and labels must have same dates."""
        # Check if indices match
        if len(self.features) != len(self.labels):
            self.issues.append(
                f"CRITICAL: Features ({len(self.features)}) and labels ({len(self.labels)}) have different lengths"
            )
            return

        # If both have date columns, check alignment
        feature_dates = None
        label_dates = None

        date_cols_features = [col for col in self.features.columns if 'date' in col.lower()]
        date_cols_labels = [col for col in self.labels.columns if 'date' in col.lower()]

        if date_cols_features and date_cols_labels:
            feature_dates = pd.to_datetime(self.features[date_cols_features[0]])
            label_dates = pd.to_datetime(self.labels[date_cols_labels[0]])

            if not feature_dates.equals(label_dates):
                self.issues.append(
                    "CRITICAL: Features and labels have misaligned dates"
                )

    def check_look_ahead_bias(self):
        """
        CRITICAL: Verify no future information in features.

        Check for features that suspiciously correlate with future targets.
        """
        # Find target column
        target_col = None
        for col in self.labels.columns:
            if 'label' in col.lower() or 'target' in col.lower():
                target_col = col
                break

        if target_col is None:
            logger.warning("No target column found for look-ahead bias check")
            return

        future_target = self.labels[target_col]

        # Check correlation with each feature
        numeric_features = self.features.select_dtypes(include=[np.number])

        suspicious_features = []

        for col in numeric_features.columns:
            try:
                corr = numeric_features[col].corr(future_target)

                # Correlation > 0.7 is very suspicious
                if abs(corr) > 0.7:
                    suspicious_features.append((col, corr))
            except:
                continue

        if suspicious_features:
            features_str = ', '.join([f"{name} ({corr:.3f})" for name, corr in suspicious_features[:5]])
            self.issues.append(
                f"CRITICAL: {len(suspicious_features)} features have high correlation (>0.7) with future target. "
                f"Top: {features_str}. Possible look-ahead bias!"
            )

    def check_feature_leakage(self):
        """Check for features that perfectly predict target (leakage)."""
        # Find target column
        target_col = None
        for col in self.labels.columns:
            if 'label' in col.lower() or 'target' in col.lower():
                target_col = col
                break

        if target_col is None:
            return

        try:
            # Quick RF to detect leakage
            X = self.features.select_dtypes(include=[np.number]).fillna(0)
            y = self.labels[target_col]

            if len(X) < 100:  # Need enough samples
                return

            # Limit features to prevent memory issues
            if len(X.columns) > 50:
                X = X.iloc[:, :50]

            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            scores = cross_val_score(rf, X, y, cv=min(3, len(X)//50), scoring='roc_auc')

            # AUC > 0.95 suggests possible leakage
            if scores.mean() > 0.95:
                self.issues.append(
                    f"WARNING: Extremely high AUC ({scores.mean():.3f}) suggests possible "
                    f"feature leakage - investigate top features"
                )
        except Exception as e:
            logger.debug(f"Could not run feature leakage check: {str(e)}")

    def check_inf_nan(self):
        """Check for infinite or NaN values."""
        numeric_features = self.features.select_dtypes(include=[np.number])

        # Check for infinite values
        inf_counts = np.isinf(numeric_features).sum()
        total_inf = inf_counts.sum()

        if total_inf > 0:
            cols_with_inf = inf_counts[inf_counts > 0]
            top_inf_cols = cols_with_inf.head(10).index.tolist()
            self.issues.append(
                f"{total_inf} infinite values found in {len(cols_with_inf)} columns. "
                f"Top columns: {top_inf_cols}"
            )

        # Check for NaN values
        nan_counts = numeric_features.isnull().sum()
        total_nan = nan_counts.sum()

        total_cells = len(numeric_features) * len(numeric_features.columns)
        nan_pct = (total_nan / total_cells * 100) if total_cells > 0 else 0

        if nan_pct > 5:  # More than 5% missing
            cols_with_many_nan = nan_counts[nan_counts > len(numeric_features) * 0.1]
            if len(cols_with_many_nan) > 0:
                top_nan_cols = cols_with_many_nan.head(10).index.tolist()
                self.issues.append(
                    f"{total_nan} NaN values ({nan_pct:.1f}% of all cells). "
                    f"Columns with >10% missing: {top_nan_cols}"
                )

    def check_feature_drift(self):
        """Check if feature distributions change over time (non-stationarity)."""
        numeric_features = self.features.select_dtypes(include=[np.number])

        if len(numeric_features) < 100:
            return

        # Split into first half and second half
        split_point = len(numeric_features) // 2
        first_half = numeric_features.iloc[:split_point]
        second_half = numeric_features.iloc[split_point:]

        # Check if means shifted significantly
        first_mean = first_half.mean()
        first_std = first_half.std()

        # Avoid division by zero
        first_std = first_std.replace(0, 1)

        mean_shift = (second_half.mean() - first_mean) / first_std

        # Flag features with >2 std shift
        significant_shifts = mean_shift[abs(mean_shift) > 2]

        if len(significant_shifts) > 0:
            top_shifts = significant_shifts.abs().nlargest(10).index.tolist()
            self.issues.append(
                f"WARNING: {len(significant_shifts)} features show significant distribution shift (>2 std). "
                f"Top drifted: {top_shifts}"
            )

    def check_target_leakage(self):
        """Verify target labels don't leak into features."""
        # Check for suspicious column names
        suspicious_keywords = ['target', 'label', 'future', 'forward', 'next', 'tomorrow']

        suspicious_names = [
            col for col in self.features.columns
            if any(word in col.lower() for word in suspicious_keywords)
        ]

        if suspicious_names:
            self.issues.append(
                f"CRITICAL: Suspicious feature names (possible target leakage): {suspicious_names}"
            )


def validate_features_before_training(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ticker: str
) -> bool:
    """
    Convenience function to validate features before training.

    Args:
        features_df: Features DataFrame
        labels_df: Labels DataFrame
        ticker: Ticker symbol

    Returns:
        True if validation passes, False otherwise
    """
    validator = FeatureValidator(features_df, labels_df, ticker)
    is_valid, issues = validator.run_all_checks()

    if not is_valid:
        logger.error(f"{ticker}: Feature validation failed with {len(issues)} issues:")
        for issue in issues:
            if 'CRITICAL' in issue:
                logger.error(f"  ❌ {issue}")
            else:
                logger.warning(f"  ⚠️  {issue}")

        # Check for critical issues
        critical_issues = [i for i in issues if 'CRITICAL' in i]
        if critical_issues:
            logger.error(f"{ticker}: Found {len(critical_issues)} CRITICAL issues - DO NOT TRAIN")
            return False
        else:
            logger.warning(f"{ticker}: Found warnings but no critical issues - proceed with caution")
            return True
    else:
        logger.info(f"{ticker}: Feature validation passed ✓")
        return True
