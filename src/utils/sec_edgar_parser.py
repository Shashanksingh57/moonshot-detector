"""
SEC EDGAR filing parser for extracting fundamental data.
Parses 10-K, 10-Q, and 8-K filings to extract financial metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import xml.etree.ElementTree as ET
import json
from datetime import datetime
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SECEDGARParser:
    """Parser for SEC EDGAR XBRL filings."""

    def __init__(self):
        """Initialize the parser."""
        # Common XBRL tags for financial metrics
        self.tag_mappings = {
            # Income Statement
            'Revenue': [
                'Revenues',
                'SalesRevenueNet',
                'RevenueFromContractWithCustomerExcludingAssessedTax',
                'SalesRevenueGoodsNet'
            ],
            'Net_Income': [
                'NetIncomeLoss',
                'ProfitLoss',
                'NetIncomeLossAvailableToCommonStockholdersBasic'
            ],
            'Operating_Income': [
                'OperatingIncomeLoss',
                'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
            ],
            'EPS_Basic': [
                'EarningsPerShareBasic',
                'IncomeLossFromContinuingOperationsPerBasicShare'
            ],
            'EPS_Diluted': [
                'EarningsPerShareDiluted',
                'IncomeLossFromContinuingOperationsPerDilutedShare'
            ],
            # Balance Sheet
            'Total_Assets': [
                'Assets',
                'AssetsCurrent'
            ],
            'Total_Liabilities': [
                'Liabilities',
                'LiabilitiesCurrent'
            ],
            'Stockholders_Equity': [
                'StockholdersEquity',
                'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
            ],
            'Cash': [
                'Cash',
                'CashAndCashEquivalentsAtCarryingValue'
            ],
            # Cash Flow
            'Operating_Cash_Flow': [
                'NetCashProvidedByUsedInOperatingActivities',
                'CashProvidedByUsedInOperatingActivities'
            ],
            'Free_Cash_Flow': [
                'FreeCashFlow'
            ],
        }

    def parse_xbrl_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse an XBRL filing and extract financial metrics.

        Args:
            file_path: Path to the XBRL file (XML or JSON)

        Returns:
            Dictionary of financial metrics
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return {}

        try:
            if path.suffix == '.json':
                return self._parse_json_filing(file_path)
            elif path.suffix in ['.xml', '.xbrl']:
                return self._parse_xml_filing(file_path)
            else:
                logger.warning(f"Unsupported file type: {path.suffix}")
                return {}
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return {}

    def _parse_json_filing(self, file_path: str) -> Dict[str, Any]:
        """Parse JSON-format XBRL filing."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            metrics = {}
            facts = data.get('facts', {}).get('us-gaap', {})

            for metric_name, tag_options in self.tag_mappings.items():
                value = None
                for tag in tag_options:
                    if tag in facts:
                        # Get the most recent value
                        units = facts[tag].get('units', {})
                        for unit_type, values in units.items():
                            if values:
                                # Sort by filing date and get most recent
                                sorted_values = sorted(
                                    values,
                                    key=lambda x: x.get('end', ''),
                                    reverse=True
                                )
                                if sorted_values:
                                    value = sorted_values[0].get('val')
                                    break
                        if value is not None:
                            break

                if value is not None:
                    metrics[metric_name] = float(value)

            return metrics

        except Exception as e:
            logger.error(f"Error parsing JSON filing: {str(e)}")
            return {}

    def _parse_xml_filing(self, file_path: str) -> Dict[str, Any]:
        """Parse XML-format XBRL filing."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract namespace
            namespace = {}
            for elem in root.iter():
                if '}' in elem.tag:
                    ns = elem.tag.split('}')[0] + '}'
                    namespace['gaap'] = ns
                    break

            metrics = {}

            for metric_name, tag_options in self.tag_mappings.items():
                value = None
                for tag in tag_options:
                    # Try to find the tag
                    xpath = f".//{namespace.get('gaap', '')}{tag}"
                    elements = root.findall(xpath)

                    if elements:
                        # Get the first element's value
                        try:
                            value = float(elements[0].text)
                            break
                        except (ValueError, AttributeError):
                            continue

                if value is not None:
                    metrics[metric_name] = value

            return metrics

        except Exception as e:
            logger.error(f"Error parsing XML filing: {str(e)}")
            return {}

    def extract_filing_date(self, file_path: str) -> Optional[datetime]:
        """
        Extract the filing date from a filing.

        Args:
            file_path: Path to the filing

        Returns:
            Filing date as datetime, or None if not found
        """
        path = Path(file_path)

        try:
            if path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Look for filing date in various locations
                    filing_date_str = data.get('filingDate') or data.get('filed')
                    if filing_date_str:
                        return datetime.strptime(filing_date_str, '%Y-%m-%d')

            elif path.suffix in ['.xml', '.xbrl']:
                tree = ET.parse(file_path)
                root = tree.getroot()
                # Try to find filing date element
                for elem in root.iter():
                    if 'filingDate' in elem.tag.lower() or 'documentPeriodEndDate' in elem.tag:
                        try:
                            return datetime.strptime(elem.text, '%Y-%m-%d')
                        except:
                            continue

        except Exception as e:
            logger.error(f"Error extracting filing date from {file_path}: {str(e)}")

        return None

    def calculate_derived_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate derived financial ratios from raw metrics.

        Args:
            metrics: Dictionary of raw financial metrics

        Returns:
            Dictionary of derived metrics
        """
        derived = {}

        try:
            # Profit Margin
            if 'Net_Income' in metrics and 'Revenue' in metrics and metrics['Revenue'] != 0:
                derived['Profit_Margin'] = metrics['Net_Income'] / metrics['Revenue']

            # Operating Margin
            if 'Operating_Income' in metrics and 'Revenue' in metrics and metrics['Revenue'] != 0:
                derived['Operating_Margin'] = metrics['Operating_Income'] / metrics['Revenue']

            # ROE (Return on Equity)
            if 'Net_Income' in metrics and 'Stockholders_Equity' in metrics and metrics['Stockholders_Equity'] != 0:
                derived['ROE'] = metrics['Net_Income'] / metrics['Stockholders_Equity']

            # ROA (Return on Assets)
            if 'Net_Income' in metrics and 'Total_Assets' in metrics and metrics['Total_Assets'] != 0:
                derived['ROA'] = metrics['Net_Income'] / metrics['Total_Assets']

            # Debt to Equity
            if 'Total_Liabilities' in metrics and 'Stockholders_Equity' in metrics and metrics['Stockholders_Equity'] != 0:
                derived['Debt_to_Equity'] = metrics['Total_Liabilities'] / metrics['Stockholders_Equity']

            # Current Ratio (if we have current assets/liabilities)
            # This would need current_assets and current_liabilities specifically

        except Exception as e:
            logger.error(f"Error calculating derived metrics: {str(e)}")

        return derived

    def parse_earnings_announcement(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse an 8-K filing for earnings announcement details.

        Args:
            file_path: Path to 8-K filing

        Returns:
            Dictionary with earnings announcement details
        """
        try:
            metrics = self.parse_xbrl_file(file_path)
            filing_date = self.extract_filing_date(file_path)

            if not metrics and not filing_date:
                return None

            return {
                'filing_date': filing_date,
                'metrics': metrics,
                'is_earnings_announcement': True
            }

        except Exception as e:
            logger.error(f"Error parsing earnings announcement: {str(e)}")
            return None
