"""
Continuous data quality monitoring - tracks feature drift and prediction quality degradation.
Uses Population Stability Index (PSI) to detect when retraining is needed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataQualityMonitor:
    """Continuous monitoring of data quality and feature drift."""

    def __init__(self, reference_data: pd.DataFrame, n_bins: int = 10):
        """
        Initialize data quality monitor with reference distribution.

        Args:
            reference_data: Reference dataset (typically training data)
            n_bins: Number of bins for PSI calculation
        """
        self.n_bins = n_bins
        self.reference_distributions = self._calculate_distributions(reference_data)
        self.feature_names = list(reference_data.select_dtypes(include=[np.number]).columns)
        logger.info(f"DataQualityMonitor initialized with {len(self.feature_names)} features")

    def _calculate_distributions(self, data: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate binned distributions for all numeric features.

        Args:
            data: DataFrame with features

        Returns:
            Dictionary mapping feature name to (bin_edges, bin_counts)
        """
        distributions = {}

        numeric_data = data.select_dtypes(include=[np.number])

        for col in numeric_data.columns:
            try:
                # Remove NaN values
                values = numeric_data[col].dropna()

                if len(values) == 0:
                    continue

                # Create bins using quantiles to ensure balanced bins
                try:
                    _, bin_edges = pd.qcut(values, q=self.n_bins, retbins=True, duplicates='drop')
                except:
                    # If qcut fails (e.g., too many duplicates), use cut instead
                    _, bin_edges = pd.cut(values, bins=self.n_bins, retbins=True)

                # Calculate counts in each bin
                counts, _ = np.histogram(values, bins=bin_edges)

                # Store normalized counts (percentages)
                percentages = counts / counts.sum()

                distributions[col] = (bin_edges, percentages)

            except Exception as e:
                logger.warning(f"Could not calculate distribution for {col}: {str(e)}")

        return distributions

    def calculate_psi(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Population Stability Index for each feature.

        PSI measures distribution shift between reference and current data:
        - PSI < 0.10: No significant change
        - PSI 0.10-0.25: Moderate change - monitor closely
        - PSI > 0.25: Significant change - RETRAIN REQUIRED

        Args:
            current_data: Current dataset to compare against reference

        Returns:
            Dictionary mapping feature name to PSI value
        """
        psi_values = {}

        numeric_data = current_data.select_dtypes(include=[np.number])

        for col in self.feature_names:
            if col not in numeric_data.columns:
                logger.warning(f"Feature {col} not found in current data")
                continue

            try:
                # Get reference distribution
                if col not in self.reference_distributions:
                    continue

                bin_edges, reference_pct = self.reference_distributions[col]

                # Calculate current distribution using same bins
                values = numeric_data[col].dropna()

                if len(values) == 0:
                    continue

                current_counts, _ = np.histogram(values, bins=bin_edges)
                current_pct = current_counts / current_counts.sum()

                # Calculate PSI
                # PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
                psi = 0.0

                for i in range(len(reference_pct)):
                    # Avoid division by zero and log(0)
                    ref_val = reference_pct[i] if reference_pct[i] > 0 else 0.0001
                    cur_val = current_pct[i] if current_pct[i] > 0 else 0.0001

                    psi += (cur_val - ref_val) * np.log(cur_val / ref_val)

                psi_values[col] = psi

            except Exception as e:
                logger.warning(f"Could not calculate PSI for {col}: {str(e)}")

        return psi_values

    def monitor_prediction_quality(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        reference_metrics: Dict[str, float] = None
    ) -> Dict:
        """
        Monitor if prediction quality is degrading over time.

        Args:
            predictions: Model predictions
            actuals: Actual outcomes
            reference_metrics: Optional reference metrics from training/validation

        Returns:
            Dictionary with monitoring results
        """
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

            results = {}

            # Calculate current metrics
            results['accuracy'] = accuracy_score(actuals, predictions)
            results['precision'] = precision_score(actuals, predictions, zero_division=0)
            results['recall'] = recall_score(actuals, predictions, zero_division=0)

            # Try to calculate AUC if we have probabilities
            try:
                if hasattr(predictions, 'values'):
                    # If predictions are probabilities
                    if predictions.max() <= 1.0 and predictions.min() >= 0.0:
                        results['roc_auc'] = roc_auc_score(actuals, predictions)
            except:
                pass

            # Compare to reference metrics if provided
            if reference_metrics:
                results['accuracy_change'] = results['accuracy'] - reference_metrics.get('accuracy', 0)
                results['precision_change'] = results['precision'] - reference_metrics.get('precision', 0)
                results['recall_change'] = results['recall'] - reference_metrics.get('recall', 0)

                # Flag significant degradation (>10% drop)
                results['quality_degraded'] = (
                    results['accuracy_change'] < -0.10 or
                    results['precision_change'] < -0.10 or
                    results['recall_change'] < -0.10
                )
            else:
                results['quality_degraded'] = False

            # Calculate prediction distribution
            pred_positive_rate = predictions.mean() if len(predictions) > 0 else 0
            actual_positive_rate = actuals.mean() if len(actuals) > 0 else 0

            results['pred_positive_rate'] = pred_positive_rate
            results['actual_positive_rate'] = actual_positive_rate
            results['prediction_bias'] = pred_positive_rate - actual_positive_rate

            return results

        except Exception as e:
            logger.error(f"Error monitoring prediction quality: {str(e)}")
            return {}

    def generate_monitoring_report(
        self,
        current_data: pd.DataFrame,
        predictions: pd.Series = None,
        actuals: pd.Series = None,
        reference_metrics: Dict = None
    ) -> Dict:
        """
        Generate comprehensive monitoring report with alerts.

        Args:
            current_data: Current feature data
            predictions: Optional predictions to monitor
            actuals: Optional actual outcomes
            reference_metrics: Optional reference performance metrics

        Returns:
            Dictionary with monitoring results and alerts
        """
        try:
            logger.info("="*70)
            logger.info("DATA QUALITY MONITORING REPORT")
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*70)

            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'alerts': []
            }

            # 1. Calculate PSI for all features
            logger.info("1. Feature Distribution Stability (PSI)")
            logger.info("-" * 70)

            psi_values = self.calculate_psi(current_data)
            report['psi_values'] = psi_values

            # Classify features by PSI
            stable_features = {k: v for k, v in psi_values.items() if v < 0.10}
            moderate_drift = {k: v for k, v in psi_values.items() if 0.10 <= v <= 0.25}
            significant_drift = {k: v for k, v in psi_values.items() if v > 0.25}

            logger.info(f"Stable features (PSI < 0.10): {len(stable_features)}")
            logger.info(f"Moderate drift (PSI 0.10-0.25): {len(moderate_drift)}")
            logger.info(f"Significant drift (PSI > 0.25): {len(significant_drift)}")

            if len(significant_drift) > 0:
                logger.warning(f"⚠️  {len(significant_drift)} features have SIGNIFICANT drift:")
                # Show top 10 drifted features
                sorted_drift = sorted(significant_drift.items(), key=lambda x: x[1], reverse=True)
                for feat, psi in sorted_drift[:10]:
                    logger.warning(f"  {feat}: PSI = {psi:.3f}")
                    report['alerts'].append(f"SIGNIFICANT DRIFT: {feat} (PSI={psi:.3f})")

                if len(significant_drift) >= 5:
                    logger.error("🚨 ALERT: 5+ features with significant drift - RETRAIN REQUIRED")
                    report['alerts'].append("CRITICAL: Retrain required due to feature drift")

            elif len(moderate_drift) > 0:
                logger.warning(f"⚠️  {len(moderate_drift)} features have moderate drift (monitor):")
                sorted_drift = sorted(moderate_drift.items(), key=lambda x: x[1], reverse=True)
                for feat, psi in sorted_drift[:5]:
                    logger.warning(f"  {feat}: PSI = {psi:.3f}")
            else:
                logger.info("✅ All features stable")

            report['num_stable_features'] = len(stable_features)
            report['num_moderate_drift'] = len(moderate_drift)
            report['num_significant_drift'] = len(significant_drift)
            report['retrain_required'] = len(significant_drift) >= 5

            logger.info("")

            # 2. Monitor prediction quality if provided
            if predictions is not None and actuals is not None:
                logger.info("2. Prediction Quality Monitoring")
                logger.info("-" * 70)

                quality_results = self.monitor_prediction_quality(
                    predictions, actuals, reference_metrics
                )
                report['prediction_quality'] = quality_results

                logger.info(f"Accuracy: {quality_results.get('accuracy', 0):.2%}")
                logger.info(f"Precision: {quality_results.get('precision', 0):.2%}")
                logger.info(f"Recall: {quality_results.get('recall', 0):.2%}")

                if 'roc_auc' in quality_results:
                    logger.info(f"ROC AUC: {quality_results.get('roc_auc', 0):.3f}")

                # Check for degradation
                if reference_metrics:
                    logger.info("")
                    logger.info("Change from reference:")
                    logger.info(f"  Accuracy: {quality_results.get('accuracy_change', 0):+.2%}")
                    logger.info(f"  Precision: {quality_results.get('precision_change', 0):+.2%}")
                    logger.info(f"  Recall: {quality_results.get('recall_change', 0):+.2%}")

                    if quality_results.get('quality_degraded', False):
                        logger.warning("⚠️  ALERT: Prediction quality has degraded >10%")
                        report['alerts'].append("WARNING: Prediction quality degraded")

                # Check prediction bias
                bias = quality_results.get('prediction_bias', 0)
                if abs(bias) > 0.15:
                    logger.warning(f"⚠️  Large prediction bias: {bias:+.2%}")
                    report['alerts'].append(f"WARNING: Prediction bias {bias:+.2%}")

                logger.info("")

            # 3. Summary
            logger.info("="*70)
            logger.info("SUMMARY")
            logger.info("="*70)

            if len(report['alerts']) == 0:
                logger.info("✅ No issues detected - system healthy")
            else:
                logger.warning(f"⚠️  {len(report['alerts'])} alerts:")
                for alert in report['alerts']:
                    logger.warning(f"  - {alert}")

            if report['retrain_required']:
                logger.error("")
                logger.error("🚨 ACTION REQUIRED: RETRAIN MODELS")
                logger.error("Feature drift has exceeded acceptable thresholds")

            logger.info("="*70)

            return report

        except Exception as e:
            logger.error(f"Error generating monitoring report: {str(e)}")
            return {'alerts': ['ERROR: Could not generate report']}

    def save_monitoring_history(self, report: Dict, output_path: str):
        """
        Save monitoring report to disk for historical tracking.

        Args:
            report: Monitoring report dictionary
            output_path: Path to save report (will append timestamp)
        """
        try:
            import json
            from pathlib import Path

            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monitoring_report_{timestamp}.json"
            filepath = output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Monitoring report saved to: {filepath}")

        except Exception as e:
            logger.error(f"Error saving monitoring history: {str(e)}")


def monitor_production_data(
    reference_data_path: str,
    current_data_path: str,
    config: Dict = None
) -> bool:
    """
    Convenience function for monitoring production data.

    Args:
        reference_data_path: Path to reference (training) data
        current_data_path: Path to current production data
        config: Optional configuration dictionary

    Returns:
        True if no retraining required, False if retraining needed
    """
    try:
        # Load data
        reference_data = pd.read_csv(reference_data_path)
        current_data = pd.read_csv(current_data_path)

        logger.info(f"Reference data: {len(reference_data)} samples")
        logger.info(f"Current data: {len(current_data)} samples")

        # Initialize monitor
        monitor = DataQualityMonitor(reference_data)

        # Generate report
        report = monitor.generate_monitoring_report(current_data)

        # Save report
        if config and 'monitoring' in config:
            output_path = config['monitoring'].get('report_dir', 'data/monitoring')
            monitor.save_monitoring_history(report, output_path)

        return not report['retrain_required']

    except Exception as e:
        logger.error(f"Error monitoring production data: {str(e)}")
        return False
