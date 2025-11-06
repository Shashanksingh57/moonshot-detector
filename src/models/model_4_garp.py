"""
Model 4: GARP Fundamental Filter
Growth At Reasonable Price - filters for quality stocks before trading signals.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict
from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Model4GARP(BaseModel):
    """GARP fundamental filter using Random Forest regression."""

    def __init__(self, config: dict):
        """
        Initialize Model 4.

        Args:
            config: Model configuration from config.yaml
        """
        super().__init__(config)

        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})

        self.model = RandomForestRegressor(
            n_estimators=hyperparams.get('n_estimators', 200),
            max_depth=hyperparams.get('max_depth', 10),
            min_samples_split=hyperparams.get('min_samples_split', 20),
            random_state=42,
            n_jobs=-1
        )

        # Get feature list from config
        self.feature_names = config.get('features', [])

        # This is a filter, not a predictor
        self.role = config.get('role', 'filter')
        self.threshold_score = config.get('threshold_score', 80)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Random Forest regressor.

        Note: For GARP filter, y_train should be forward returns (not binary labels).
        If binary labels are provided, we'll still train but results may not be optimal.

        Args:
            X_train: Training features (fundamental ratios)
            y_train: Target (forward returns or quality score)
        """
        try:
            logger.info(f"Training {self.model_name}...")

            # Check available features
            available_features = [f for f in self.feature_names if f in X_train.columns]

            if len(available_features) == 0:
                logger.warning("No configured features found in training data")
                # Fall back to all numeric features
                available_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

            logger.info(f"Using {len(available_features)} fundamental features")

            # Select features
            X_train_selected = X_train[available_features]
            self.feature_names = available_features

            # Fit model
            self.model.fit(X_train_selected, y_train)

            self.is_trained = True
            logger.info(f"{self.model_name} training complete")

        except Exception as e:
            logger.error(f"Error training {self.model_name}: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict quality scores (not used for filtering).

        Args:
            X: Features

        Returns:
            Array of predicted scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X_selected = X[self.feature_names]
        return self.model.predict(X_selected)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Not applicable for regression model.
        Returns normalized scores between 0 and 1.

        Args:
            X: Features

        Returns:
            Array of normalized scores
        """
        scores = self.predict(X)

        # Normalize to 0-1 range using sigmoid
        # This makes it compatible with ensemble voting
        normalized = 1 / (1 + np.exp(-scores))

        return normalized

    def score_stock(self, X: pd.DataFrame) -> float:
        """
        Score a stock on 0-100 scale.

        Args:
            X: Features (single row)

        Returns:
            Quality score (0-100)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        prediction = self.predict(X)

        # Convert to 0-100 scale
        # Assuming predictions are in reasonable range (-5 to +5 for returns)
        # Map to percentile-like score
        score = min(100, max(0, (prediction[0] + 5) / 10 * 100))

        return score

    def filter_stocks(self, tickers: List[str], features_dict: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Filter stocks based on GARP score threshold.

        Args:
            tickers: List of ticker symbols
            features_dict: Dictionary mapping tickers to feature DataFrames

        Returns:
            List of tickers that pass the filter (score >= threshold)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        passed_tickers = []

        for ticker in tickers:
            if ticker not in features_dict:
                logger.warning(f"Features not found for {ticker}")
                continue

            X = features_dict[ticker]

            if len(X) == 0:
                continue

            # Score the most recent data point
            score = self.score_stock(X.tail(1))

            if score >= self.threshold_score:
                passed_tickers.append(ticker)
                logger.debug(f"{ticker}: score={score:.1f} - PASS")
            else:
                logger.debug(f"{ticker}: score={score:.1f} - FILTERED OUT")

        logger.info(f"GARP filter: {len(passed_tickers)}/{len(tickers)} passed (threshold={self.threshold_score})")

        return passed_tickers

    def analyze_quality_factors(self) -> pd.DataFrame:
        """
        Analyze which fundamental factors are most important for quality.

        Returns:
            DataFrame with feature importance
        """
        return self.get_feature_importance()
