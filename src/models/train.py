"""
Training pipeline - trains all active models.
Loads features and labels, performs train/validation split, and saves trained models.
"""

import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from src.models.model_1_breakout import Model1Breakout
from src.models.model_2_volatility import Model2Volatility
from src.models.model_3_pead_squeeze import Model3PEADSqueeze
from src.models.model_4_garp import Model4GARP
from src.models.model_6_anomaly import Model6Anomaly
from src.utils.logging_config import setup_logging

logger = setup_logging('training')


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_features_and_labels(
    ticker: str,
    model_name: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load features and labels for a ticker.

    Args:
        ticker: Ticker symbol
        model_name: Model name (for loading correct labels)

    Returns:
        Tuple of (features, labels)
    """
    try:
        # Load features
        features_file = Path(f'data/processed/features/{ticker}_features.parquet')
        if not features_file.exists():
            return None, None

        df_features = pd.read_parquet(features_file)

        # Load labels
        label_name = model_name.replace(' ', '_').lower()
        labels_file = Path(f'data/processed/labels/{ticker}_{label_name}_labels.parquet')

        if not labels_file.exists():
            # Try generic label file
            labels_file = Path(f'data/processed/labels/{ticker}_labels.parquet')
            if not labels_file.exists():
                return None, None

        df_labels = pd.read_parquet(labels_file)

        # Merge on date
        df = pd.merge(df_features, df_labels[['date', 'label']], on='date', how='inner')

        if len(df) == 0:
            return None, None

        # Separate features and labels
        X = df.drop(['label', 'date'], axis=1, errors='ignore')
        y = df['label']

        return X, y

    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return None, None


def load_all_data(
    ticker_file: str,
    model_name: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and combine data from multiple tickers.

    Args:
        ticker_file: Path to ticker list file
        model_name: Model name

    Returns:
        Combined features and labels
    """
    # Load ticker list
    with open(ticker_file, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    logger.info(f"Loading data for {len(tickers)} tickers...")

    all_X = []
    all_y = []

    for ticker in tqdm(tickers, desc="Loading data"):
        X, y = load_features_and_labels(ticker, model_name)

        if X is not None and y is not None:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        raise ValueError("No data loaded")

    # Combine all data
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)

    logger.info(f"Loaded {len(X_combined)} samples")
    logger.info(f"Positive samples: {y_combined.sum()} ({y_combined.mean():.1%})")

    return X_combined, y_combined


def train_model(
    model_id: int,
    model_class,
    model_config: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: str = 'models/trained'
) -> bool:
    """
    Train a single model.

    Args:
        model_id: Model ID number
        model_class: Model class
        model_config: Model configuration
        X_train: Training features
        y_train: Training labels
        output_dir: Directory to save trained model

    Returns:
        True if successful
    """
    try:
        logger.info(f"Training Model {model_id}: {model_config['name']}")

        # Initialize model
        model = model_class(model_config)

        # Train model
        model.train(X_train, y_train)

        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_name = model_config['name'].replace(' ', '_').lower()
        model_file = output_path / f"model_{model_id}_{model_name}.pkl"

        model.save(str(model_file))

        logger.info(f"Model {model_id} saved to {model_file}")

        # Log feature importance if available
        importance = model.get_feature_importance()
        if importance is not None:
            logger.info(f"Top 10 features for Model {model_id}:")
            for idx, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return True

    except Exception as e:
        logger.error(f"Error training Model {model_id}: {str(e)}")
        return False


def train_all_models(
    ticker_file: str,
    config_path: str = 'config.yaml'
) -> None:
    """
    Train all enabled models.

    Args:
        ticker_file: Path to ticker list file
        config_path: Path to config file
    """
    logger.info("Starting model training pipeline...")

    # Load config
    config = load_config(config_path)
    models_config = config.get('models', {})

    # Model classes
    model_classes = {
        1: Model1Breakout,
        2: Model2Volatility,
        3: Model3PEADSqueeze,
        4: Model4GARP,
        6: Model6Anomaly
    }

    # Train each enabled model
    results = {}

    for model_id, model_class in model_classes.items():
        model_key = f"model_{model_id}_" + {
            1: "breakout",
            2: "volatility",
            3: "pead_squeeze",
            4: "garp",
            6: "anomaly"
        }[model_id]

        model_config = models_config.get(model_key, {})

        if not model_config.get('enabled', False):
            logger.info(f"Skipping Model {model_id} (disabled)")
            continue

        # Load data
        try:
            logger.info(f"Loading data for Model {model_id}...")
            X, y = load_all_data(ticker_file, model_config['name'])

            # Train model
            success = train_model(model_id, model_class, model_config, X, y)
            results[model_id] = success

        except Exception as e:
            logger.error(f"Failed to train Model {model_id}: {str(e)}")
            results[model_id] = False

    # Summary
    logger.info("="*70)
    logger.info("Training Summary:")
    for model_id, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  Model {model_id}: {status}")

    successful = sum(1 for s in results.values() if s)
    logger.info(f"Total: {successful}/{len(results)} models trained successfully")
    logger.info("="*70)


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Train all models')

    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Path to ticker list file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    train_all_models(args.tickers, args.config)


if __name__ == '__main__':
    main()
