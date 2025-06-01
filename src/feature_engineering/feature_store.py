"""
Feature Store for managing and orchestrating all feature generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
import pickle
import os

from .timeseries_features import TimeseriesFeatures
from .event_features import EventFeatures
from .sentiment_features import SentimentFeatures
from .market_context_features import MarketContextFeatures
from ..event_detection.event_store import EventStore


class FeatureStore:
    """
    Central feature store for managing all ML features
    """
    
    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        news_data: Optional[pd.DataFrame] = None,
        cache_dir: str = "data/features"
    ):
        """
        Initialize the feature store
        
        Args:
            event_store: EventStore instance for event features
            news_data: News data for sentiment features
            cache_dir: Directory for caching computed features
        """
        
        # Initialize feature generators
        self.timeseries_generator = TimeseriesFeatures()
        self.event_generator = EventFeatures(event_store)
        self.sentiment_generator = SentimentFeatures(news_data)
        self.market_context_generator = MarketContextFeatures()
        
        # Cache management
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Feature metadata
        self.feature_metadata = {}
        self.all_feature_names = []
        
        logger.info("FeatureStore initialized with all feature generators")
    
    def generate_complete_features(
        self,
        data: pd.DataFrame,
        ticker: str,
        include_timeseries: bool = True,
        include_events: bool = True,
        include_sentiment: bool = True,
        include_market_context: bool = True,
        feature_config: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete feature set for a ticker
        
        Args:
            data: Market data DataFrame with OHLCV columns
            ticker: Stock ticker symbol
            include_timeseries: Include technical indicators
            include_events: Include event-based features
            include_sentiment: Include sentiment features
            include_market_context: Include market context features
            feature_config: Configuration for feature generation
            use_cache: Whether to use cached features
            
        Returns:
            DataFrame with all features
        """
        
        if data.empty:
            logger.warning("Empty data provided to feature store")
            return data
        
        logger.info(f"Generating complete feature set for {ticker}")
        
        # Check cache first
        cache_key = self._get_cache_key(ticker, data.index.min(), data.index.max())
        if use_cache:
            cached_features = self._load_from_cache(cache_key)
            if cached_features is not None:
                logger.info(f"Loaded features from cache for {ticker}")
                return cached_features
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Use default config if none provided
        config = feature_config or self._get_default_config()
        
        # Generate features step by step
        feature_count_start = len(df.columns)
        
        # 1. Timeseries features (technical indicators)
        if include_timeseries:
            logger.info(f"Generating timeseries features for {ticker}")
            df = self.timeseries_generator.generate_features(
                df, ticker, config.get('timeseries_config')
            )
            timeseries_count = len(df.columns) - feature_count_start
            logger.info(f"Added {timeseries_count} timeseries features")
        
        # 2. Event features
        if include_events:
            logger.info(f"Generating event features for {ticker}")
            df = self.event_generator.generate_features(
                df, ticker, config.get('event_config')
            )
            event_count = len(df.columns) - feature_count_start - timeseries_count if include_timeseries else len(df.columns) - feature_count_start
            logger.info(f"Added {event_count} event features")
        
        # 3. Sentiment features
        if include_sentiment:
            logger.info(f"Generating sentiment features for {ticker}")
            df = self.sentiment_generator.generate_features(
                df, ticker, config.get('sentiment_config')
            )
            sentiment_count = len(df.columns) - (feature_count_start + (timeseries_count if include_timeseries else 0) + (event_count if include_events else 0))
            logger.info(f"Added {sentiment_count} sentiment features")
        
        # 4. Market context features
        if include_market_context:
            logger.info(f"Generating market context features for {ticker}")
            df = self.market_context_generator.generate_features(
                df, ticker, config.get('market_context_config')
            )
            context_count = len(df.columns) - feature_count_start - sum([
                timeseries_count if include_timeseries else 0,
                event_count if include_events else 0,
                sentiment_count if include_sentiment else 0
            ])
            logger.info(f"Added {context_count} market context features")
        
        # Store feature metadata
        self._update_feature_metadata(df, ticker, {
            'timeseries_included': include_timeseries,
            'events_included': include_events,
            'sentiment_included': include_sentiment,
            'market_context_included': include_market_context,
            'generation_time': datetime.utcnow(),
            'data_range': (data.index.min(), data.index.max()),
            'feature_count': len(df.columns) - feature_count_start
        })
        
        # Cache the results
        if use_cache:
            self._save_to_cache(cache_key, df)
        
        total_features = len(df.columns) - feature_count_start
        logger.info(f"Generated {total_features} total features for {ticker}")
        
        return df
    
    def generate_batch_features(
        self,
        tickers: List[str],
        data_dict: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate features for multiple tickers in batch
        
        Args:
            tickers: List of ticker symbols
            data_dict: Dictionary mapping tickers to their market data
            **kwargs: Arguments passed to generate_complete_features
            
        Returns:
            Dictionary mapping tickers to their feature DataFrames
        """
        
        logger.info(f"Generating features for {len(tickers)} tickers in batch")
        
        results = {}
        
        for ticker in tickers:
            if ticker in data_dict:
                try:
                    logger.info(f"Processing {ticker}")
                    features_df = self.generate_complete_features(
                        data_dict[ticker], ticker, **kwargs
                    )
                    results[ticker] = features_df
                    logger.info(f"Completed {ticker}")
                except Exception as e:
                    logger.error(f"Error generating features for {ticker}: {e}")
                    continue
            else:
                logger.warning(f"No data found for {ticker}")
        
        logger.info(f"Batch feature generation complete. Processed {len(results)} tickers")
        return results
    
    def create_ml_dataset(
        self,
        features_dict: Dict[str, pd.DataFrame],
        target_column: str = 'price_change_pct',
        lookforward_periods: int = 1,
        min_samples: int = 100,
        feature_selection: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create ML-ready dataset from features
        
        Args:
            features_dict: Dictionary of ticker -> features DataFrame
            target_column: Column to use as target variable
            lookforward_periods: Periods to shift target forward
            min_samples: Minimum samples required per ticker
            feature_selection: Specific features to include (None for all)
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        
        logger.info("Creating ML dataset from features")
        
        # Combine all ticker data
        combined_data = []
        
        for ticker, df in features_dict.items():
            if len(df) < min_samples:
                logger.warning(f"Skipping {ticker} - insufficient samples ({len(df)} < {min_samples})")
                continue
            
            # Add ticker identifier
            ticker_df = df.copy()
            ticker_df['ticker'] = ticker
            
            # Create target variable (future returns)
            if target_column in ticker_df.columns:
                ticker_df['target'] = ticker_df[target_column].shift(-lookforward_periods)
            else:
                logger.warning(f"Target column {target_column} not found in {ticker}")
                continue
            
            combined_data.append(ticker_df)
        
        if not combined_data:
            logger.error("No valid data found for ML dataset creation")
            return pd.DataFrame(), pd.Series()
        
        # Combine all data
        full_df = pd.concat(combined_data, ignore_index=False)
        logger.info(f"Combined dataset shape: {full_df.shape}")
        
        # Select features
        if feature_selection:
            # Use specified features
            available_features = [f for f in feature_selection if f in full_df.columns]
            if len(available_features) != len(feature_selection):
                missing = set(feature_selection) - set(available_features)
                logger.warning(f"Missing features: {missing}")
        else:
            # Use all non-target, non-identifier columns
            exclude_cols = ['target', 'ticker', target_column] + ['open', 'high', 'low', 'close', 'volume']
            available_features = [col for col in full_df.columns if col not in exclude_cols]
        
        logger.info(f"Using {len(available_features)} features")
        
        # Create feature matrix and target vector
        X = full_df[available_features].copy()
        y = full_df['target'].copy()
        
        # Remove rows with NaN targets
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Handle remaining NaN values in features
        X = X.fillna(X.median())
        
        logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def get_feature_importance(
        self,
        features_df: pd.DataFrame,
        target_col: str = 'price_change_pct'
    ) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from all generators
        
        Args:
            features_df: DataFrame with all features
            target_col: Target column for importance calculation
            
        Returns:
            Dictionary with importance scores by feature type
        """
        
        importance_results = {}
        
        # Get importance from each generator
        if hasattr(self.timeseries_generator, 'get_feature_importance_proxy'):
            importance_results['timeseries'] = self.timeseries_generator.get_feature_importance_proxy(
                features_df, target_col
            )
        
        if hasattr(self.event_generator, 'get_feature_importance_proxy'):
            importance_results['events'] = self.event_generator.get_feature_importance_proxy(
                features_df, target_col
            )
        
        if hasattr(self.sentiment_generator, 'get_feature_importance_proxy'):
            importance_results['sentiment'] = self.sentiment_generator.get_feature_importance_proxy(
                features_df, target_col
            )
        
        if hasattr(self.market_context_generator, 'get_feature_importance_proxy'):
            importance_results['market_context'] = self.market_context_generator.get_feature_importance_proxy(
                features_df, target_col
            )
        
        return importance_results
    
    def generate_feature_summary(
        self,
        features_df: pd.DataFrame,
        ticker: str = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive feature summary
        
        Args:
            features_df: DataFrame with all features
            ticker: Ticker symbol (optional)
            
        Returns:
            Summary statistics and metadata
        """
        
        summary = {
            'ticker': ticker,
            'total_features': len(features_df.columns),
            'total_samples': len(features_df),
            'date_range': (features_df.index.min(), features_df.index.max()) if isinstance(features_df.index, pd.DatetimeIndex) else None,
            'feature_types': {}
        }
        
        # Get summaries from each generator
        if hasattr(self.timeseries_generator, 'generate_feature_summary'):
            summary['feature_types']['timeseries'] = self.timeseries_generator.generate_feature_summary(features_df)
        
        if hasattr(self.event_generator, 'generate_feature_summary'):
            summary['feature_types']['events'] = self.event_generator.generate_feature_summary(features_df)
        
        if hasattr(self.sentiment_generator, 'generate_feature_summary'):
            summary['feature_types']['sentiment'] = self.sentiment_generator.generate_feature_summary(features_df)
        
        if hasattr(self.market_context_generator, 'generate_feature_summary'):
            summary['feature_types']['market_context'] = self.market_context_generator.generate_feature_summary(features_df)
        
        # Calculate missing value statistics
        summary['missing_values'] = {
            'total_missing': features_df.isnull().sum().sum(),
            'columns_with_missing': features_df.isnull().any().sum(),
            'missing_percentage': (features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))) * 100
        }
        
        return summary
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for all feature generators"""
        return {
            'timeseries_config': {
                'price_features': True,
                'moving_averages': True,
                'volatility_features': True,
                'volume_features': True,
                'momentum_features': True,
                'technical_indicators': True,
                'pattern_features': True,
                'time_features': True
            },
            'event_config': {
                'time_to_event_features': True,
                'event_count_features': True,
                'event_impact_features': True,
                'event_type_features': True,
                'event_clustering_features': True,
                'event_momentum_features': True
            },
            'sentiment_config': {
                'basic_sentiment_features': True,
                'rolling_sentiment_features': True,
                'decay_weighted_features': True,
                'sentiment_momentum_features': True,
                'sentiment_volatility_features': True,
                'news_volume_features': True
            },
            'market_context_config': {
                'market_regime_features': True,
                'sector_performance_features': True,
                'macro_indicator_features': True,
                'volatility_regime_features': True,
                'crypto_market_features': True,
                'correlation_features': True,
                'calendar_features': True
            }
        }
    
    def _get_cache_key(self, ticker: str, start_date: datetime, end_date: datetime) -> str:
        """Generate cache key for features"""
        return f"{ticker}_{start_date.date()}_{end_date.date()}"
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save features to cache"""
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved features to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load features from cache"""
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Loaded features from cache: {cache_path}")
                return data
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
        return None
    
    def _update_feature_metadata(self, df: pd.DataFrame, ticker: str, metadata: Dict[str, Any]) -> None:
        """Update feature metadata"""
        self.feature_metadata[ticker] = metadata
        
        # Update global feature names list
        current_features = set(df.columns)
        self.all_feature_names = list(set(self.all_feature_names) | current_features)
    
    def clear_cache(self) -> None:
        """Clear all cached features"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info("Feature cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_available_features(self) -> List[str]:
        """Get list of all available feature names"""
        return self.all_feature_names.copy()
    
    def get_metadata(self, ticker: str = None) -> Dict[str, Any]:
        """Get feature metadata"""
        if ticker:
            return self.feature_metadata.get(ticker, {})
        return self.feature_metadata.copy() 