"""
Sentiment Features for generating ML features from news sentiment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available. Using simplified sentiment analysis.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER sentiment not available.")


class SentimentFeatures:
    """
    Generates sentiment-based features from news data for ML models
    """
    
    def __init__(self, news_data: Optional[pd.DataFrame] = None):
        """
        Initialize the sentiment feature generator
        
        Args:
            news_data: DataFrame with news data (title, content, published_date)
        """
        self.news_data = news_data
        self.feature_names = []
        
        # Initialize sentiment analyzers
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        
        # Precompute sentiment scores if news data provided
        if news_data is not None:
            self.sentiment_df = self._compute_sentiment_scores(news_data)
        else:
            self.sentiment_df = pd.DataFrame()
    
    def generate_features(
        self,
        data: pd.DataFrame,
        ticker: str = None,
        feature_config: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Generate sentiment-based features
        
        Args:
            data: DataFrame with market data (must have timestamp/date column)
            ticker: Stock ticker to filter news for (optional)
            feature_config: Configuration for features to generate
            
        Returns:
            DataFrame with original data plus sentiment features
        """
        
        if data.empty:
            return data
        
        logger.info(f"Generating sentiment features for {ticker or 'all tickers'}")
        
        # Use default config if none provided
        config = feature_config or self._get_default_config()
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Prepare datetime index
        df = self._prepare_datetime_index(df)
        
        # Filter sentiment data for ticker if specified
        ticker_sentiment = self._get_ticker_sentiment(ticker) if ticker else self.sentiment_df
        
        if ticker_sentiment.empty:
            logger.warning(f"No sentiment data found for {ticker or 'any ticker'}")
            return self._add_zero_features(df, config)
        
        logger.info(f"Found {len(ticker_sentiment)} sentiment records")
        
        # Generate different types of sentiment features
        if config.get('basic_sentiment_features', True):
            df = self._add_basic_sentiment_features(df, ticker_sentiment, config)
        
        if config.get('rolling_sentiment_features', True):
            df = self._add_rolling_sentiment_features(df, ticker_sentiment, config)
        
        if config.get('decay_weighted_features', True):
            df = self._add_decay_weighted_features(df, ticker_sentiment, config)
        
        if config.get('sentiment_momentum_features', True):
            df = self._add_sentiment_momentum_features(df, ticker_sentiment, config)
        
        if config.get('sentiment_volatility_features', True):
            df = self._add_sentiment_volatility_features(df, ticker_sentiment, config)
        
        if config.get('news_volume_features', True):
            df = self._add_news_volume_features(df, ticker_sentiment, config)
        
        # Store feature names
        new_features = [col for col in df.columns if col not in data.columns]
        self.feature_names.extend(new_features)
        
        logger.info(f"Generated {len(new_features)} sentiment features")
        return df
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature generation configuration"""
        return {
            'basic_sentiment_features': True,
            'rolling_sentiment_features': True,
            'decay_weighted_features': True,
            'sentiment_momentum_features': True,
            'sentiment_volatility_features': True,
            'news_volume_features': True,
            'rolling_windows': [1, 3, 7, 14, 30],
            'decay_halflife_days': [1, 3, 7],
            'sentiment_thresholds': [-0.5, -0.1, 0.1, 0.5]
        }
    
    def _prepare_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has datetime index"""
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("No datetime column found, using integer index")
        
        return df
    
    def _compute_sentiment_scores(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Compute sentiment scores for news data"""
        
        if news_data.empty:
            return pd.DataFrame()
        
        logger.info(f"Computing sentiment scores for {len(news_data)} news articles")
        
        sentiment_records = []
        
        for idx, row in news_data.iterrows():
            try:
                # Combine title and content
                text = str(row.get('title', '')) + ' ' + str(row.get('content', ''))
                
                if not text.strip():
                    continue
                
                # Compute different sentiment scores
                sentiment_record = {
                    'timestamp': pd.to_datetime(row.get('published_date', datetime.utcnow())),
                    'source': row.get('source', 'unknown'),
                    'ticker': row.get('ticker', None)
                }
                
                # TextBlob sentiment
                if TEXTBLOB_AVAILABLE:
                    blob = TextBlob(text)
                    sentiment_record['textblob_polarity'] = blob.sentiment.polarity
                    sentiment_record['textblob_subjectivity'] = blob.sentiment.subjectivity
                else:
                    sentiment_record['textblob_polarity'] = 0.0
                    sentiment_record['textblob_subjectivity'] = 0.5
                
                # VADER sentiment
                if VADER_AVAILABLE:
                    scores = self.vader.polarity_scores(text)
                    sentiment_record['vader_positive'] = scores['pos']
                    sentiment_record['vader_neutral'] = scores['neu']
                    sentiment_record['vader_negative'] = scores['neg']
                    sentiment_record['vader_compound'] = scores['compound']
                else:
                    sentiment_record.update({
                        'vader_positive': 0.33,
                        'vader_neutral': 0.34,
                        'vader_negative': 0.33,
                        'vader_compound': 0.0
                    })
                
                # Simple keyword-based sentiment (fallback)
                sentiment_record['keyword_sentiment'] = self._simple_keyword_sentiment(text)
                
                # Create composite sentiment score
                sentiment_record['composite_sentiment'] = np.mean([
                    sentiment_record['textblob_polarity'],
                    sentiment_record['vader_compound'],
                    sentiment_record['keyword_sentiment']
                ])
                
                # Article metadata
                sentiment_record['text_length'] = len(text)
                sentiment_record['title_length'] = len(str(row.get('title', '')))
                
                sentiment_records.append(sentiment_record)
                
            except Exception as e:
                logger.error(f"Error computing sentiment for row {idx}: {e}")
                continue
        
        if not sentiment_records:
            return pd.DataFrame()
        
        sentiment_df = pd.DataFrame(sentiment_records)
        sentiment_df = sentiment_df.sort_values('timestamp')
        
        logger.info(f"Computed sentiment for {len(sentiment_df)} articles")
        return sentiment_df
    
    def _simple_keyword_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment analysis"""
        
        positive_words = [
            'bullish', 'positive', 'up', 'gain', 'growth', 'profit', 'beat', 'strong',
            'outperform', 'buy', 'upgrade', 'bull', 'rise', 'surge', 'rally', 'boom',
            'success', 'wins', 'breakthrough', 'record', 'high', 'soar'
        ]
        
        negative_words = [
            'bearish', 'negative', 'down', 'loss', 'decline', 'miss', 'weak',
            'underperform', 'sell', 'downgrade', 'bear', 'fall', 'drop', 'crash', 'bust',
            'failure', 'loses', 'problem', 'low', 'plunge', 'collapse'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count
    
    def _get_ticker_sentiment(self, ticker: str) -> pd.DataFrame:
        """Get sentiment data filtered for a specific ticker"""
        
        if self.sentiment_df.empty or not ticker:
            return self.sentiment_df
        
        # Filter by ticker if ticker column exists
        if 'ticker' in self.sentiment_df.columns:
            ticker_sentiment = self.sentiment_df[
                (self.sentiment_df['ticker'] == ticker) |
                (self.sentiment_df['ticker'].isna())  # Include general market news
            ]
        else:
            # If no ticker column, assume all news is relevant
            ticker_sentiment = self.sentiment_df
        
        return ticker_sentiment.copy()
    
    def _add_zero_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add zero-valued features when no sentiment data is found"""
        
        # Basic sentiment features
        df['sentiment_score'] = 0.0
        df['sentiment_positive'] = 0.0
        df['sentiment_negative'] = 0.0
        df['news_count_1d'] = 0
        
        # Rolling features
        for window in config.get('rolling_windows', [1, 3, 7, 14, 30]):
            df[f'sentiment_avg_{window}d'] = 0.0
            df[f'news_count_{window}d'] = 0
        
        # Decay weighted features
        for halflife in config.get('decay_halflife_days', [1, 3, 7]):
            df[f'sentiment_decay_{halflife}d'] = 0.0
        
        return df
    
    def _add_basic_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add basic sentiment features"""
        
        # Initialize features
        df['sentiment_score'] = 0.0
        df['sentiment_positive'] = 0.0
        df['sentiment_negative'] = 0.0
        df['sentiment_neutral'] = 0.0
        df['sentiment_compound'] = 0.0
        df['sentiment_subjectivity'] = 0.5
        df['last_sentiment'] = 0.0
        
        for idx, row in df.iterrows():
            current_time = idx
            
            # Get sentiment for current day
            day_sentiment = sentiment_df[
                sentiment_df['timestamp'].dt.date == current_time.date()
            ]
            
            if not day_sentiment.empty:
                # Aggregate daily sentiment
                df.at[idx, 'sentiment_score'] = day_sentiment['composite_sentiment'].mean()
                df.at[idx, 'sentiment_positive'] = day_sentiment['vader_positive'].mean()
                df.at[idx, 'sentiment_negative'] = day_sentiment['vader_negative'].mean()
                df.at[idx, 'sentiment_neutral'] = day_sentiment['vader_neutral'].mean()
                df.at[idx, 'sentiment_compound'] = day_sentiment['vader_compound'].mean()
                
                if 'textblob_subjectivity' in day_sentiment.columns:
                    df.at[idx, 'sentiment_subjectivity'] = day_sentiment['textblob_subjectivity'].mean()
            
            # Get most recent sentiment (within last 3 days)
            recent_sentiment = sentiment_df[
                sentiment_df['timestamp'] <= current_time
            ].tail(1)
            
            if not recent_sentiment.empty:
                df.at[idx, 'last_sentiment'] = recent_sentiment['composite_sentiment'].iloc[0]
        
        return df
    
    def _add_rolling_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add rolling sentiment features"""
        
        rolling_windows = config.get('rolling_windows', [1, 3, 7, 14, 30])
        
        for window in rolling_windows:
            df[f'sentiment_avg_{window}d'] = 0.0
            df[f'sentiment_std_{window}d'] = 0.0
            df[f'sentiment_min_{window}d'] = 0.0
            df[f'sentiment_max_{window}d'] = 0.0
            df[f'sentiment_trend_{window}d'] = 0.0
            
            for idx, row in df.iterrows():
                current_time = idx
                window_start = current_time - timedelta(days=window)
                
                window_sentiment = sentiment_df[
                    (sentiment_df['timestamp'] >= window_start) &
                    (sentiment_df['timestamp'] <= current_time)
                ]
                
                if not window_sentiment.empty:
                    sentiments = window_sentiment['composite_sentiment']
                    
                    df.at[idx, f'sentiment_avg_{window}d'] = sentiments.mean()
                    df.at[idx, f'sentiment_std_{window}d'] = sentiments.std()
                    df.at[idx, f'sentiment_min_{window}d'] = sentiments.min()
                    df.at[idx, f'sentiment_max_{window}d'] = sentiments.max()
                    
                    # Calculate trend (correlation with time)
                    if len(sentiments) >= 3:
                        time_numeric = (window_sentiment['timestamp'] - window_sentiment['timestamp'].min()).dt.total_seconds()
                        trend = np.corrcoef(time_numeric, sentiments)[0, 1]
                        if not np.isnan(trend):
                            df.at[idx, f'sentiment_trend_{window}d'] = trend
        
        return df
    
    def _add_decay_weighted_features(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add decay-weighted sentiment features"""
        
        decay_halflife_days = config.get('decay_halflife_days', [1, 3, 7])
        
        for halflife in decay_halflife_days:
            df[f'sentiment_decay_{halflife}d'] = 0.0
            
            for idx, row in df.iterrows():
                current_time = idx
                
                # Get historical sentiment (last 30 days)
                hist_sentiment = sentiment_df[
                    sentiment_df['timestamp'] <= current_time
                ].tail(100)  # Limit for performance
                
                if not hist_sentiment.empty:
                    # Calculate time differences in days
                    time_diffs = (current_time - hist_sentiment['timestamp']).dt.total_seconds() / (24 * 3600)
                    
                    # Apply exponential decay
                    decay_weights = np.exp(-np.log(2) * time_diffs / halflife)
                    
                    # Calculate weighted average
                    weighted_sentiment = (hist_sentiment['composite_sentiment'] * decay_weights).sum() / decay_weights.sum()
                    df.at[idx, f'sentiment_decay_{halflife}d'] = weighted_sentiment
        
        return df
    
    def _add_sentiment_momentum_features(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add sentiment momentum features"""
        
        # Sentiment change features
        df['sentiment_change_1d'] = 0.0
        df['sentiment_change_3d'] = 0.0
        df['sentiment_acceleration'] = 0.0
        
        for idx, row in df.iterrows():
            current_time = idx
            
            # 1-day sentiment change
            yesterday = current_time - timedelta(days=1)
            current_sentiment = sentiment_df[
                sentiment_df['timestamp'].dt.date == current_time.date()
            ]['composite_sentiment'].mean() if not sentiment_df.empty else 0
            
            yesterday_sentiment = sentiment_df[
                sentiment_df['timestamp'].dt.date == yesterday.date()
            ]['composite_sentiment'].mean() if not sentiment_df.empty else 0
            
            if not pd.isna(current_sentiment) and not pd.isna(yesterday_sentiment):
                df.at[idx, 'sentiment_change_1d'] = current_sentiment - yesterday_sentiment
            
            # 3-day sentiment change
            three_days_ago = current_time - timedelta(days=3)
            three_days_sentiment = sentiment_df[
                sentiment_df['timestamp'].dt.date == three_days_ago.date()
            ]['composite_sentiment'].mean() if not sentiment_df.empty else 0
            
            if not pd.isna(current_sentiment) and not pd.isna(three_days_sentiment):
                df.at[idx, 'sentiment_change_3d'] = current_sentiment - three_days_sentiment
        
        # Calculate acceleration (second derivative)
        if 'sentiment_change_1d' in df.columns:
            df['sentiment_acceleration'] = df['sentiment_change_1d'].diff()
        
        # Sentiment momentum (moving average of changes)
        if 'sentiment_change_1d' in df.columns:
            df['sentiment_momentum_5d'] = df['sentiment_change_1d'].rolling(window=5).mean()
        
        return df
    
    def _add_sentiment_volatility_features(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add sentiment volatility features"""
        
        for window in [7, 14, 30]:
            df[f'sentiment_volatility_{window}d'] = 0.0
            
            for idx, row in df.iterrows():
                current_time = idx
                window_start = current_time - timedelta(days=window)
                
                window_sentiment = sentiment_df[
                    (sentiment_df['timestamp'] >= window_start) &
                    (sentiment_df['timestamp'] <= current_time)
                ]
                
                if len(window_sentiment) >= 3:
                    # Group by day and calculate daily average sentiment
                    daily_sentiment = window_sentiment.groupby(
                        window_sentiment['timestamp'].dt.date
                    )['composite_sentiment'].mean()
                    
                    # Calculate volatility as standard deviation of daily sentiments
                    sentiment_vol = daily_sentiment.std()
                    if not pd.isna(sentiment_vol):
                        df.at[idx, f'sentiment_volatility_{window}d'] = sentiment_vol
        
        return df
    
    def _add_news_volume_features(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add news volume features"""
        
        rolling_windows = config.get('rolling_windows', [1, 3, 7, 14, 30])
        
        for window in rolling_windows:
            df[f'news_count_{window}d'] = 0
            df[f'positive_news_count_{window}d'] = 0
            df[f'negative_news_count_{window}d'] = 0
            
            for idx, row in df.iterrows():
                current_time = idx
                window_start = current_time - timedelta(days=window)
                
                window_sentiment = sentiment_df[
                    (sentiment_df['timestamp'] >= window_start) &
                    (sentiment_df['timestamp'] <= current_time)
                ]
                
                df.at[idx, f'news_count_{window}d'] = len(window_sentiment)
                
                if not window_sentiment.empty:
                    positive_news = len(window_sentiment[window_sentiment['composite_sentiment'] > 0.1])
                    negative_news = len(window_sentiment[window_sentiment['composite_sentiment'] < -0.1])
                    
                    df.at[idx, f'positive_news_count_{window}d'] = positive_news
                    df.at[idx, f'negative_news_count_{window}d'] = negative_news
        
        # News frequency and sentiment ratios
        if 'news_count_7d' in df.columns:
            df['news_frequency'] = df['news_count_7d'] / 7
            
            # Sentiment ratios
            df['positive_news_ratio'] = (
                df['positive_news_count_7d'] / (df['news_count_7d'] + 1)
            )
            df['negative_news_ratio'] = (
                df['negative_news_count_7d'] / (df['news_count_7d'] + 1)
            )
        
        return df
    
    def get_feature_importance_proxy(
        self,
        df: pd.DataFrame,
        target_col: str = 'price_change_pct'
    ) -> Dict[str, float]:
        """Calculate feature importance proxy based on correlation with target"""
        
        if target_col not in df.columns or len(self.feature_names) == 0:
            return {}
        
        importance = {}
        target = df[target_col].shift(-1)  # Next period target
        
        for feature in self.feature_names:
            if feature in df.columns and df[feature].dtype in ['int64', 'float64', 'int32', 'float32']:
                try:
                    corr = df[feature].corr(target)
                    importance[feature] = abs(corr) if not pd.isna(corr) else 0.0
                except Exception:
                    importance[feature] = 0.0
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def generate_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for sentiment features"""
        
        feature_cols = [col for col in df.columns if col in self.feature_names]
        
        if not feature_cols:
            return {}
        
        summary = {
            'total_features': len(feature_cols),
            'feature_types': {
                'basic_sentiment': len([f for f in feature_cols if 'sentiment_' in f and not any(x in f for x in ['avg', 'std', 'decay', 'change'])]),
                'rolling_sentiment': len([f for f in feature_cols if any(x in f for x in ['avg', 'std', 'min', 'max', 'trend'])]),
                'decay_weighted': len([f for f in feature_cols if 'decay' in f]),
                'momentum_features': len([f for f in feature_cols if any(x in f for x in ['change', 'momentum', 'acceleration'])]),
                'volatility_features': len([f for f in feature_cols if 'volatility' in f]),
                'volume_features': len([f for f in feature_cols if 'news_count' in f or 'news_frequency' in f])
            },
            'sentiment_ranges': {
                col: {
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None
                }
                for col in feature_cols[:10] if 'sentiment' in col
            }
        }
        
        return summary 