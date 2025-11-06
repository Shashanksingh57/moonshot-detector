"""
Base model class - abstract interface for all trading models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all trading models."""

    def __init__(self, config: Dict):
        """
        Initialize model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.model_name = config.get('name', 'UnnamedModel')

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Features

        Returns:
            Array of predictions (0 or 1)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Features

        Returns:
            Array of probabilities for positive class
        """
        pass

    def save(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save model
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            model_dict = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.config,
                'is_trained': self.is_trained
            }

            joblib.dump(model_dict, filepath)
            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load(self, filepath: str) -> None:
        """
        Load model from disk.

        Args:
            filepath: Path to load model from
        """
        try:
            model_dict = joblib.load(filepath)

            self.model = model_dict['model']
            self.scaler = model_dict.get('scaler')
            self.feature_names = model_dict['feature_names']
            self.config = model_dict['config']
            self.is_trained = model_dict['is_trained']

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if available).

        Returns:
            DataFrame with feature names and importance scores
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained yet")
                return None

            if not hasattr(self.model, 'feature_importances_'):
                logger.warning("Model does not support feature importance")
                return None

            importance = self.model.feature_importances_

            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            return df

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None

    def validate_features(self, X: pd.DataFrame) -> bool:
        """
        Validate that input features match expected features.

        Args:
            X: Features to validate

        Returns:
            True if valid
        """
        if not self.feature_names:
            logger.warning("No feature names defined")
            return True

        missing_features = set(self.feature_names) - set(X.columns)

        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False

        return True
