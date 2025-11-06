"""
Sentiment data collector using Reddit (PRAW) and Google Trends (FREE).
Collects mentions and sentiment for stocks and crypto.
"""

import pandas as pd
import praw
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Optional
from pathlib import Path
import time
import argparse
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from tqdm import tqdm
from src.utils.logging_config import setup_logging

logger = setup_logging('data_collection')
load_dotenv()


def init_reddit() -> Optional[praw.Reddit]:
    """Initialize Reddit API client using PRAW."""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'moonshot-detector:v0.1')
        )
        logger.info("Reddit API initialized")
        return reddit
    except Exception as e:
        logger.error(f"Error initializing Reddit API: {str(e)}")
        return None


def collect_reddit_mentions(
    ticker: str,
    subreddits: List[str],
    days_lookback: int = 30,
    reddit: Optional[praw.Reddit] = None
) -> Dict:
    """
    Collect Reddit mentions and sentiment for a ticker.

    Args:
        ticker: Stock/crypto ticker or symbol
        subreddits: List of subreddits to search
        days_lookback: Number of days to look back
        reddit: PRAW Reddit instance

    Returns:
        Dictionary with mention counts and sentiment
    """
    try:
        if reddit is None:
            reddit = init_reddit()
            if reddit is None:
                return {}

        analyzer = SentimentIntensityAnalyzer()

        mentions = []
        sentiment_scores = []

        # Search each subreddit
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)

                # Search for ticker mentions
                for submission in subreddit.search(ticker, limit=100):
                    # Check if within time window
                    post_time = datetime.fromtimestamp(submission.created_utc)
                    if (datetime.now() - post_time).days > days_lookback:
                        continue

                    # Analyze sentiment
                    text = f"{submission.title} {submission.selftext}"
                    sentiment = analyzer.polarity_scores(text)

                    mentions.append({
                        'date': post_time,
                        'subreddit': subreddit_name,
                        'score': submission.score,
                        'num_comments': submission.num_comments
                    })

                    sentiment_scores.append(sentiment['compound'])

            except Exception as e:
                logger.warning(f"Error searching r/{subreddit_name}: {str(e)}")
                continue

        # Aggregate results
        result = {
            'ticker': ticker,
            'mentions_total': len(mentions),
            'mentions_7d': sum(1 for m in mentions if (datetime.now() - m['date']).days <= 7),
            'mentions_30d': len(mentions),
            'avg_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            'total_score': sum(m['score'] for m in mentions),
            'total_comments': sum(m['num_comments'] for m in mentions)
        }

        return result

    except Exception as e:
        logger.error(f"Error collecting Reddit mentions for {ticker}: {str(e)}")
        return {}


def collect_google_trends(ticker: str, timeframe: str = 'today 3-m') -> Dict:
    """
    Collect Google Trends search interest for a ticker.

    Args:
        ticker: Stock/crypto ticker or symbol
        timeframe: Timeframe string (e.g., 'today 3-m', 'today 12-m')

    Returns:
        Dictionary with trends data
    """
    try:
        pytrends = TrendReq(hl='en-US', tz=360)

        # Build payload
        pytrends.build_payload([ticker], cat=0, timeframe=timeframe, geo='US', gprop='')

        # Get interest over time
        df = pytrends.interest_over_time()

        if df is None or len(df) == 0:
            return {}

        # Calculate metrics
        result = {
            'ticker': ticker,
            'current_interest': df[ticker].iloc[-1] if len(df) > 0 else 0,
            'avg_interest': df[ticker].mean(),
            'max_interest': df[ticker].max(),
            'trend_momentum': (df[ticker].iloc[-1] - df[ticker].iloc[-30]) / df[ticker].iloc[-30] if len(df) >= 30 else 0
        }

        return result

    except Exception as e:
        logger.error(f"Error collecting Google Trends for {ticker}: {str(e)}")
        return {}


def collect_sentiment_for_ticker(
    ticker: str,
    subreddits: List[str],
    output_dir: str = "data/raw/sentiment"
) -> bool:
    """
    Collect all sentiment data for a ticker and save to parquet.

    Args:
        ticker: Stock/crypto ticker
        subreddits: List of subreddits to monitor
        output_dir: Output directory

    Returns:
        True if successful
    """
    try:
        # Collect Reddit data
        reddit_data = collect_reddit_mentions(ticker, subreddits)

        # Collect Google Trends data
        time.sleep(2)  # Rate limiting
        trends_data = collect_google_trends(ticker)

        # Combine data
        sentiment_data = {
            'ticker': ticker,
            'date': datetime.now(),
            **reddit_data,
            **trends_data
        }

        # Convert to DataFrame
        df = pd.DataFrame([sentiment_data])

        # Save to parquet
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"{ticker}_sentiment.parquet"

        # Append if file exists
        if output_file.exists():
            existing_df = pd.read_parquet(output_file)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_parquet(output_file, index=False)

        logger.debug(f"Saved sentiment data for {ticker}")

        return True

    except Exception as e:
        logger.error(f"Error collecting sentiment for {ticker}: {str(e)}")
        return False


def collect_all_sentiment(
    tickers: List[str],
    subreddits: List[str],
    output_dir: str = "data/raw/sentiment"
) -> None:
    """
    Collect sentiment data for multiple tickers.

    Args:
        tickers: List of tickers
        subreddits: List of subreddits to monitor
        output_dir: Output directory
    """
    logger.info(f"Starting sentiment collection for {len(tickers)} tickers...")
    logger.info(f"Subreddits: {subreddits}")

    success_count = 0
    fail_count = 0

    # Initialize Reddit once
    reddit = init_reddit()

    pbar = tqdm(tickers, desc="Collecting sentiment", unit="ticker")

    for ticker in pbar:
        pbar.set_postfix({'ticker': ticker, 'success': success_count, 'failed': fail_count})

        if collect_sentiment_for_ticker(ticker, subreddits, output_dir):
            success_count += 1
        else:
            fail_count += 1

        # Rate limiting
        time.sleep(1)

    logger.info(f"Sentiment collection complete!")
    logger.info(f"Success: {success_count}, Failed: {fail_count}")


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Collect sentiment data from Reddit and Google Trends')

    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Path to ticker list file (one per line)'
    )

    parser.add_argument(
        '--subreddits',
        type=str,
        nargs='+',
        default=['wallstreetbets', 'stocks', 'investing'],
        help='Subreddits to monitor'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/sentiment',
        help='Output directory'
    )

    args = parser.parse_args()

    # Load ticker list
    with open(args.tickers, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    logger.info(f"Loaded {len(tickers)} tickers from {args.tickers}")

    # Collect sentiment
    collect_all_sentiment(tickers, args.subreddits, args.output)


if __name__ == '__main__':
    main()
