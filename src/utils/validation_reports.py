"""
Validation report management - saves and summarizes data quality validation results.
Provides JSON-based reporting and aggregation for tracking data quality over time.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationReportManager:
    """Manages data quality validation reports."""

    def __init__(self, report_dir: str = "data/validation_reports"):
        """
        Initialize validation report manager.

        Args:
            report_dir: Directory to save validation reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Validation reports will be saved to: {self.report_dir}")

    def save_validation_report(
        self,
        ticker: str,
        data_type: str,
        issues: List[str],
        is_valid: bool,
        metadata: Dict = None
    ):
        """
        Save validation issues to file.

        Args:
            ticker: Ticker symbol
            data_type: Type of data ('stock_price', 'fundamental', 'crypto', 'sentiment')
            issues: List of validation issues
            is_valid: Whether validation passed
            metadata: Optional additional metadata
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            date_str = datetime.now().strftime('%Y-%m-%d')

            # Create report structure
            report = {
                'ticker': ticker,
                'data_type': data_type,
                'timestamp': timestamp,
                'date': date_str,
                'is_valid': is_valid,
                'num_issues': len(issues),
                'issues': issues,
                'metadata': metadata or {}
            }

            # Classify issues by severity
            critical_issues = [i for i in issues if 'CRITICAL' in i or 'DATA ERROR' in i or 'IMPOSSIBLE' in i]
            warning_issues = [i for i in issues if 'WARNING' in i]

            report['num_critical_issues'] = len(critical_issues)
            report['num_warnings'] = len(warning_issues)
            report['has_critical_issues'] = len(critical_issues) > 0

            # Save to file (organized by data type)
            data_type_dir = self.report_dir / data_type
            data_type_dir.mkdir(exist_ok=True)

            filename = f"{ticker}_{timestamp}.json"
            filepath = data_type_dir / filename

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            if is_valid:
                logger.debug(f"Validation passed for {ticker} ({data_type})")
            else:
                if len(critical_issues) > 0:
                    logger.error(f"Validation FAILED for {ticker} ({data_type}) - {len(critical_issues)} critical issues")
                else:
                    logger.warning(f"Validation passed with warnings for {ticker} ({data_type}) - {len(warning_issues)} warnings")

        except Exception as e:
            logger.error(f"Error saving validation report for {ticker}: {str(e)}")

    def load_all_reports(self) -> List[Dict]:
        """
        Load all validation reports from disk.

        Returns:
            List of report dictionaries
        """
        try:
            all_reports = []

            # Search all subdirectories
            for json_file in self.report_dir.rglob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        report = json.load(f)
                        all_reports.append(report)
                except Exception as e:
                    logger.warning(f"Could not load report {json_file}: {str(e)}")

            logger.info(f"Loaded {len(all_reports)} validation reports")
            return all_reports

        except Exception as e:
            logger.error(f"Error loading validation reports: {str(e)}")
            return []

    def generate_summary_report(self, date_filter: str = None) -> pd.DataFrame:
        """
        Generate summary of all validation reports.

        Args:
            date_filter: Optional date to filter by (YYYY-MM-DD)

        Returns:
            DataFrame with summary statistics
        """
        try:
            reports = self.load_all_reports()

            if len(reports) == 0:
                logger.warning("No validation reports found")
                return pd.DataFrame()

            # Filter by date if requested
            if date_filter:
                reports = [r for r in reports if r.get('date') == date_filter]
                logger.info(f"Filtered to {len(reports)} reports from {date_filter}")

            # Convert to DataFrame
            df = pd.DataFrame(reports)

            # Summary statistics
            summary = df.groupby(['data_type', 'ticker']).agg({
                'is_valid': 'last',  # Most recent validation result
                'num_issues': 'last',
                'num_critical_issues': 'last',
                'num_warnings': 'last',
                'timestamp': 'max'  # Most recent timestamp
            }).reset_index()

            summary.columns = [
                'data_type', 'ticker', 'is_valid', 'num_issues',
                'num_critical_issues', 'num_warnings', 'last_checked'
            ]

            return summary

        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return pd.DataFrame()

    def print_summary(self, date_filter: str = None):
        """
        Print validation summary to console.

        Args:
            date_filter: Optional date to filter by (YYYY-MM-DD)
        """
        try:
            summary_df = self.generate_summary_report(date_filter)

            if len(summary_df) == 0:
                logger.warning("No validation data to summarize")
                return

            logger.info("="*70)
            logger.info("DATA QUALITY VALIDATION SUMMARY")
            if date_filter:
                logger.info(f"Date: {date_filter}")
            logger.info("="*70)

            # Overall stats
            total_tickers = len(summary_df)
            valid_tickers = len(summary_df[summary_df['is_valid'] == True])
            critical_issues_count = summary_df['num_critical_issues'].sum()
            warnings_count = summary_df['num_warnings'].sum()

            logger.info(f"Total Assets Validated: {total_tickers}")
            logger.info(f"Passed Validation: {valid_tickers} ({valid_tickers/total_tickers*100:.1f}%)")
            logger.info(f"Total Critical Issues: {critical_issues_count}")
            logger.info(f"Total Warnings: {warnings_count}")
            logger.info("")

            # By data type
            logger.info("By Data Type:")
            by_type = summary_df.groupby('data_type').agg({
                'ticker': 'count',
                'is_valid': lambda x: (x == True).sum(),
                'num_critical_issues': 'sum',
                'num_warnings': 'sum'
            })
            by_type.columns = ['Total', 'Valid', 'Critical Issues', 'Warnings']

            for data_type, row in by_type.iterrows():
                logger.info(f"  {data_type}:")
                logger.info(f"    Total: {int(row['Total'])}, Valid: {int(row['Valid'])}, "
                          f"Critical: {int(row['Critical Issues'])}, Warnings: {int(row['Warnings'])}")

            logger.info("")

            # Assets with critical issues
            critical_assets = summary_df[summary_df['num_critical_issues'] > 0]
            if len(critical_assets) > 0:
                logger.warning(f"⚠️  {len(critical_assets)} assets have CRITICAL issues:")
                for _, row in critical_assets.iterrows():
                    logger.warning(f"  {row['ticker']} ({row['data_type']}): {int(row['num_critical_issues'])} critical issues")
                logger.warning("  → DO NOT USE THESE ASSETS FOR TRAINING")
            else:
                logger.info("✅ No critical issues found")

            logger.info("")

            # Assets with warnings only
            warning_assets = summary_df[(summary_df['num_warnings'] > 0) & (summary_df['num_critical_issues'] == 0)]
            if len(warning_assets) > 0:
                logger.info(f"⚠️  {len(warning_assets)} assets have warnings (no critical issues):")
                for _, row in warning_assets.head(10).iterrows():  # Show first 10
                    logger.info(f"  {row['ticker']} ({row['data_type']}): {int(row['num_warnings'])} warnings")
                if len(warning_assets) > 10:
                    logger.info(f"  ... and {len(warning_assets) - 10} more")

            logger.info("="*70)

        except Exception as e:
            logger.error(f"Error printing summary: {str(e)}")

    def get_issues_for_ticker(self, ticker: str, data_type: str = None) -> List[str]:
        """
        Get all validation issues for a specific ticker.

        Args:
            ticker: Ticker symbol
            data_type: Optional data type filter

        Returns:
            List of all issues for this ticker
        """
        try:
            reports = self.load_all_reports()

            # Filter by ticker
            ticker_reports = [r for r in reports if r['ticker'] == ticker]

            # Filter by data type if specified
            if data_type:
                ticker_reports = [r for r in ticker_reports if r['data_type'] == data_type]

            # Get most recent report for each data type
            if len(ticker_reports) == 0:
                return []

            # Sort by timestamp and get latest
            ticker_reports.sort(key=lambda x: x['timestamp'], reverse=True)

            # If data_type specified, return issues from latest report
            if data_type:
                return ticker_reports[0]['issues'] if ticker_reports else []

            # Otherwise, combine issues from all data types (latest of each)
            all_issues = []
            seen_types = set()
            for report in ticker_reports:
                dt = report['data_type']
                if dt not in seen_types:
                    all_issues.extend(report['issues'])
                    seen_types.add(dt)

            return all_issues

        except Exception as e:
            logger.error(f"Error getting issues for {ticker}: {str(e)}")
            return []

    def export_summary_csv(self, output_path: str, date_filter: str = None):
        """
        Export summary report to CSV.

        Args:
            output_path: Path to save CSV
            date_filter: Optional date filter
        """
        try:
            summary_df = self.generate_summary_report(date_filter)

            if len(summary_df) == 0:
                logger.warning("No data to export")
                return

            summary_df.to_csv(output_path, index=False)
            logger.info(f"Summary report exported to: {output_path}")

        except Exception as e:
            logger.error(f"Error exporting summary: {str(e)}")

    def clean_old_reports(self, days_to_keep: int = 30):
        """
        Clean up old validation reports.

        Args:
            days_to_keep: Number of days of reports to keep
        """
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.strftime('%Y%m%d')

            deleted_count = 0

            for json_file in self.report_dir.rglob("*.json"):
                try:
                    # Extract date from filename (format: TICKER_YYYYMMDD_HHMMSS.json)
                    parts = json_file.stem.split('_')
                    if len(parts) >= 2:
                        date_str = parts[1]  # YYYYMMDD
                        if date_str < cutoff_str:
                            json_file.unlink()
                            deleted_count += 1
                except Exception as e:
                    logger.debug(f"Could not check age of {json_file}: {str(e)}")

            logger.info(f"Cleaned up {deleted_count} old validation reports (kept last {days_to_keep} days)")

        except Exception as e:
            logger.error(f"Error cleaning old reports: {str(e)}")
