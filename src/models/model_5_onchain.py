"""
Model 5: Crypto On-Chain Analyst (DISABLED - Requires Glassnode)
Placeholder for on-chain analysis - add after profitability.
"""

from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Model5OnChain(BaseModel):
    """
    On-Chain analysis model for cryptocurrencies.

    DISABLED: Requires Glassnode subscription ($29-799/month).
    Add this model after achieving first $5k profit.

    Features (when enabled):
    - Exchange netflow (tokens moving to/from exchanges)
    - Whale transfers (large transactions)
    - Active addresses (network usage)
    - MVRV Z-Score (market value vs realized value)
    - NVT Ratio (network value to transactions)
    - Funding rates (perpetual futures)
    """

    def __init__(self, config: dict):
        """
        Initialize Model 5 (placeholder).

        Args:
            config: Model configuration from config.yaml
        """
        super().__init__(config)

        # Check if model is enabled
        if config.get('enabled', False):
            raise NotImplementedError(
                "\n" + "="*70 + "\n"
                "Model 5 (On-Chain Analyst) requires Glassnode subscription.\n"
                "\n"
                "Cost: $29-799/month depending on tier\n"
                "  - Starter: $29/mo (basic on-chain metrics)\n"
                "  - Advanced: $299/mo (full metrics + alerts)\n"
                "  - Professional: $799/mo (institutional data)\n"
                "\n"
                "RECOMMENDATION: Add this model after achieving first $5k profit.\n"
                "\n"
                "To enable:\n"
                "  1. Sign up at https://glassnode.com/\n"
                "  2. Get API key\n"
                "  3. Add to .env: GLASSNODE_API_KEY=your_key\n"
                "  4. Implement on-chain data collection in src/data_collection/\n"
                "  5. Implement this model\n"
                "  6. Set enabled: true in config.yaml\n"
                "\n"
                "For now, set enabled: false in config.yaml\n"
                "="*70
            )

        logger.info("Model 5 (On-Chain Analyst) is DISABLED - set enabled: false in config")

    def train(self, X_train, y_train):
        """Not implemented."""
        raise NotImplementedError("Model 5 is disabled - requires Glassnode subscription")

    def predict(self, X):
        """Not implemented."""
        raise NotImplementedError("Model 5 is disabled - requires Glassnode subscription")

    def predict_proba(self, X):
        """Not implemented."""
        raise NotImplementedError("Model 5 is disabled - requires Glassnode subscription")
