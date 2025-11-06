"""
Model 6: Anomaly / Regime Detector
Uses Isolation Forest to detect unusual market conditions that may precede explosive moves.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Model6Anomaly(BaseModel):
    """Anomaly detection model using Isolation Forest (unsupervised)."""

    def __init__(self, config: dict):
        """
        Initialize Model 6.

        Args:
            config: Model configuration from config.yaml
        """
        super().__init__(config)

        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})

        self.model = IsolationForest(
            contamination=hyperparams.get('contamination', 0.1),
            n_estimators=hyperparams.get('n_estimators', 100),
            random_state=42,
            n_jobs=-1
        )

        # Get feature list from config
        self.feature_names = config.get('features', [])

        # This is unsupervised - no labels needed
        self.unsupervised = True

    def train(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> None:
        """
        Train the Isolation Forest (unsupervised).

        Note: y_train is ignored for unsupervised learning.

        Args:
            X_train: Training features
            y_train: Ignored (unsupervised learning)
        """
        try:
            logger.info(f"Training {self.model_name} (unsupervised)...")

            # Check available features
            available_features = [f for f in self.feature_names if f in X_train.columns]

            if len(available_features) == 0:
                logger.warning("No configured features found in training data")
                # Fall back to all numeric features
                available_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

            logger.info(f"Using {len(available_features)} features for anomaly detection")

            # Select features
            X_train_selected = X_train[available_features]
            self.feature_names = available_features

            # Fit model (no labels needed)
            self.model.fit(X_train_selected)

            self.is_trained = True
            logger.info(f"{self.model_name} training complete")

        except Exception as e:
            logger.error(f"Error training {self.model_name}: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            X: Features

        Returns:
            Array of predictions (1 = normal, -1 = anomaly in sklearn format)
            Converted to (0 = normal, 1 = anomaly) for consistency
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X_selected = X[self.feature_names]
        predictions = self.model.predict(X_selected)

        # Convert from sklearn format (-1 = anomaly, 1 = normal)
        # to our format (1 = anomaly, 0 = normal)
        return (predictions == -1).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores (0-1, higher = more anomalous).

        Args:
            X: Features

        Returns:
            Array of anomaly probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X_selected = X[self.feature_names]

        # Get anomaly scores (more negative = more anomalous)
        scores = self.model.score_samples(X_selected)

        # Convert to probability-like scores (0-1 range, higher = more anomalous)
        # Use sigmoid transformation
        probas = 1 / (1 + np.exp(scores))

        return probas

    def detect_anomalies(self, X: pd.DataFrame, threshold: float = 0.7) -> np.ndarray:
        """
        Detect anomalies using probability threshold.

        Args:
            X: Features
            threshold: Probability threshold (default 0.7)

        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        probas = self.predict_proba(X)
        return (probas > threshold).astype(int)

    def get_anomaly_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed anomaly analysis.

        Args:
            X: Features

        Returns:
            DataFrame with anomaly scores and classifications
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        probas = self.predict_proba(X)
        predictions = self.predict(X)

        results = pd.DataFrame({
            'anomaly_score': probas,
            'is_anomaly': predictions,
            'percentile': pd.Series(probas).rank(pct=True) * 100
        })

        return results
