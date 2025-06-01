"""
Main Feature Engineering Module

Orchestrates all feature generation components to create a comprehensive feature matrix
for machine learning models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from .timeseries_features import TimeseriesFeatures
from .event_features import EventFeatures
from .sentiment_features import SentimentFeatures
from .market_context_features import MarketContextFeatures
from .feature_store import FeatureStore
from ..event_detection.event_classifier import DetectedEvent


@dataclass
class FeatureConfig:
    """Configuration for feature generation"""
    # Timeseries features
    enable_timeseries_features: bool = True
    rolling_windows: List[int] = None
    
    # Event features
    enable_event_features: bool = True
    event_lookback_days: List[int] = None
    event_lookahead_days: List[int] = None
    
    # Sentiment features
    enable_sentiment_features: bool = True
    sentiment_decay_halflife: List[int] = None
    
    # Market context features
    enable_market_context_features: bool = True
    include_macro_features: bool = True
    include_sector_features: bool = True
    
    # General settings
    feature_selection_method: Optional[str] = None  # 'correlation', 'mutual_info', None
    max_features: Optional[int] = None
    handle_missing: str = 'fill'  # 'fill', 'drop', 'interpolate'
    
    def __post_init__(self):
        # Set defaults
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50]
        if self.event_lookback_days is None:
            self.event_lookback_days = [1, 3, 7, 14, 30]
        if self.event_lookahead_days is None:
            self.event_lookahead_days = [1, 3, 7]
        if self.sentiment_decay_halflife is None:
            self.sentiment_decay_halflife = [1, 3, 7]


class FeatureEngineer:
    """
    Main feature engineering orchestrator that combines all feature types
    """
    
    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        feature_store: Optional[FeatureStore] = None
    ):
        self.config = config or FeatureConfig()
        self.feature_store = feature_store or FeatureStore()
        
        # Initialize feature generators
        self.timeseries_features = TimeseriesFeatures()
        self.event_features = EventFeatures()
        self.sentiment_features = SentimentFeatures()
        self.market_context_features = MarketContextFeatures()
        
        # Track generated features
        self.feature_names: List[str] = []
        self.feature_metadata: Dict[str, Any] = {}
        
        logger.info("Initialized FeatureEngineer with all feature generators")
    
    def generate_features(
        self,
        market_data: Dict[str, pd.DataFrame],
        news_data: Optional[pd.DataFrame] = None,
        events: Optional[List[DetectedEvent]] = None,
        symbols: Optional[List[str]] = None,
        current_time: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive feature matrix for all symbols
        
        Args:
            market_data: Dict of symbol -> OHLCV DataFrame
            news_data: DataFrame with news and sentiment data
            events: List of detected events
            symbols: List of symbols to generate features for
            current_time: Current timestamp for real-time features
            
        Returns:
            Dict of symbol -> feature DataFrame
        """
        
        logger.info("Starting comprehensive feature generation...")
        
        # Determine symbols to process
        if symbols is None:
            symbols = list(market_data.keys())
        
        # Validate inputs
        if not market_data:
            logger.warning("No market data provided")
            return {}
        
        feature_results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Generating features for {symbol}")
                
                if symbol not in market_data or market_data[symbol].empty:
                    logger.warning(f"No market data for {symbol}")
                    continue
                
                # Generate features for this symbol
                symbol_features = self._generate_symbol_features(
                    symbol=symbol,
                    market_data=market_data[symbol],
                    news_data=news_data,
                    events=events,
                    current_time=current_time
                )
                
                feature_results[symbol] = symbol_features
                
            except Exception as e:
                logger.error(f"Failed to generate features for {symbol}: {e}")
                continue
        
        logger.info(f"Generated features for {len(feature_results)} symbols")
        return feature_results
    
    def _generate_symbol_features(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame],
        events: Optional[List[DetectedEvent]],
        current_time: Optional[datetime]
    ) -> pd.DataFrame:
        """Generate features for a single symbol"""
        
        # Start with market data
        df = market_data.copy()
        
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df.set_index('date', inplace=True)
            else:
                logger.warning(f"No datetime index for {symbol}, using integer index")
        
        # Generate timeseries features
        if self.config.enable_timeseries_features:
            try:
                logger.debug(f"Generating timeseries features for {symbol}")
                df = self.timeseries_features.generate_features(
                    data=df,
                    feature_config={
                        'rolling_windows': self.config.rolling_windows
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to generate timeseries features for {symbol}: {e}")
        
        # Generate event features
        if self.config.enable_event_features and events:
            try:
                logger.debug(f"Generating event features for {symbol}")
                df = self.event_features.generate_features(
                    data=df,
                    ticker=symbol,
                    feature_config={
                        'lookback_days': self.config.event_lookback_days,
                        'lookahead_days': self.config.event_lookahead_days
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to generate event features for {symbol}: {e}")
        
        # Generate sentiment features
        if self.config.enable_sentiment_features and news_data is not None:
            try:
                logger.debug(f"Generating sentiment features for {symbol}")
                df = self.sentiment_features.generate_features(
                    data=df,
                    ticker=symbol,
                    feature_config={
                        'decay_halflife_days': self.config.sentiment_decay_halflife
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to generate sentiment features for {symbol}: {e}")
        
        # Generate market context features
        if self.config.enable_market_context_features:
            try:
                logger.debug(f"Generating market context features for {symbol}")
                
                start_date = df.index.min() if not df.empty else datetime.now() - timedelta(days=30)
                end_date = df.index.max() if not df.empty else datetime.now()
                
                df = self.market_context_features.generate_features(
                    data=df,
                    start_date=start_date,
                    end_date=end_date,
                    feature_config={
                        'include_macro_features': self.config.include_macro_features,
                        'include_sector_features': self.config.include_sector_features
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to generate market context features for {symbol}: {e}")
        
        # Add symbol identifier
        df['symbol'] = symbol
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Apply feature selection if configured
        if self.config.feature_selection_method or self.config.max_features:
            df = self._apply_feature_selection(df, symbol)
        
        # Update feature tracking
        self._update_feature_tracking(df, symbol)
        
        logger.debug(f"Generated {len(df.columns)} features for {symbol}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration"""
        
        if self.config.handle_missing == 'fill':
            # Forward fill, then backward fill, then fill with 0
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
        elif self.config.handle_missing == 'drop':
            # Drop rows with any missing values
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows due to missing values")
                
        elif self.config.handle_missing == 'interpolate':
            # Interpolate numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate()
            df = df.fillna(0)
        
        return df
    
    def _apply_feature_selection(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply feature selection if configured"""
        
        try:
            # Exclude non-feature columns
            exclude_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                return df
            
            # Apply maximum features limit
            if self.config.max_features and len(feature_cols) > self.config.max_features:
                
                if self.config.feature_selection_method == 'correlation':
                    # Select features with highest correlation to price change
                    if 'price_change_pct' in df.columns:
                        correlations = {}
                        target = df['price_change_pct'].shift(-1)  # Next period return
                        
                        for col in feature_cols:
                            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                                try:
                                    corr = abs(df[col].corr(target))
                                    correlations[col] = corr if not pd.isna(corr) else 0.0
                                except Exception:
                                    correlations[col] = 0.0
                        
                        # Select top features
                        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                        selected_features = [f[0] for f in top_features[:self.config.max_features]]
                        
                        # Keep selected features plus non-feature columns
                        keep_cols = selected_features + exclude_cols
                        df = df[[col for col in keep_cols if col in df.columns]]
                        
                        logger.info(f"Selected {len(selected_features)} features for {symbol} using correlation")
                
                else:
                    # Simple selection: keep first N features
                    selected_features = feature_cols[:self.config.max_features]
                    keep_cols = selected_features + exclude_cols
                    df = df[[col for col in keep_cols if col in df.columns]]
                    
                    logger.info(f"Selected first {len(selected_features)} features for {symbol}")
        
        except Exception as e:
            logger.warning(f"Feature selection failed for {symbol}: {e}")
        
        return df
    
    def _update_feature_tracking(self, df: pd.DataFrame, symbol: str) -> None:
        """Update feature tracking metadata"""
        
        # Track feature names
        exclude_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
        symbol_features = [col for col in df.columns if col not in exclude_cols]
        
        # Update master feature list
        for feature in symbol_features:
            if feature not in self.feature_names:
                self.feature_names.append(feature)
        
        # Update metadata
        self.feature_metadata[symbol] = {
            'feature_count': len(symbol_features),
            'timeseries_features': len([f for f in symbol_features if any(ts in f for ts in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'atr'])]),
            'event_features': len([f for f in symbol_features if any(ev in f for ev in ['event_', 'days_since', 'days_to'])]),
            'sentiment_features': len([f for f in symbol_features if 'sentiment' in f or 'news_' in f]),
            'market_context_features': len([f for f in symbol_features if any(mc in f for mc in ['vix', 'regime', 'sector', 'calendar'])])
        }
    
    def create_training_dataset(
        self,
        feature_data: Dict[str, pd.DataFrame],
        target_column: str = 'price_change_pct',
        lookforward_periods: int = 1,
        min_samples: int = 100
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create ML-ready dataset from feature data
        
        Args:
            feature_data: Dict of symbol -> feature DataFrame
            target_column: Column to use as prediction target
            lookforward_periods: How many periods ahead to predict
            min_samples: Minimum samples required per symbol
            
        Returns:
            Tuple of (features_df, target_series)
        """
        
        logger.info("Creating training dataset from feature data...")
        
        # Use FeatureStore to create the dataset
        return self.feature_store.create_ml_dataset(
            features_dict=feature_data,
            target_column=target_column,
            lookforward_periods=lookforward_periods,
            min_samples=min_samples
        )
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of generated features"""
        
        total_features = len(self.feature_names)
        
        # Categorize features
        feature_categories = {
            'timeseries': len([f for f in self.feature_names if any(ts in f for ts in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'atr', 'roc', 'momentum'])]),
            'event': len([f for f in self.feature_names if any(ev in f for ev in ['event_', 'days_since', 'days_to', 'impact_'])]),
            'sentiment': len([f for f in self.feature_names if any(s in f for s in ['sentiment', 'news_', 'positive_', 'negative_'])]),
            'market_context': len([f for f in self.feature_names if any(mc in f for mc in ['vix', 'regime', 'sector', 'calendar', 'macro'])])
        }
        
        summary = {
            'total_features': total_features,
            'feature_categories': feature_categories,
            'symbols_processed': len(self.feature_metadata),
            'symbol_metadata': self.feature_metadata,
            'config': {
                'timeseries_enabled': self.config.enable_timeseries_features,
                'event_enabled': self.config.enable_event_features,
                'sentiment_enabled': self.config.enable_sentiment_features,
                'market_context_enabled': self.config.enable_market_context_features,
                'feature_selection': self.config.feature_selection_method,
                'max_features': self.config.max_features
            }
        }
        
        return summary
    
    def save_features(self, feature_data: Dict[str, pd.DataFrame], filename: str) -> None:
        """Save generated features to storage"""
        
        self.feature_store.save_features(feature_data, filename)
        logger.info(f"Saved features to {filename}")
    
    def load_features(self, filename: str) -> Dict[str, pd.DataFrame]:
        """Load previously generated features"""
        
        feature_data = self.feature_store.load_features(filename)
        logger.info(f"Loaded features from {filename}")
        return feature_data
    
    def get_feature_importance(
        self,
        feature_data: Dict[str, pd.DataFrame],
        target_column: str = 'price_change_pct'
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature importance for each symbol
        
        Returns:
            Dict of symbol -> feature importance scores
        """
        
        importance_results = {}
        
        for symbol, df in feature_data.items():
            try:
                # Calculate importance using each feature generator
                importance_scores = {}
                
                if self.config.enable_timeseries_features:
                    ts_importance = self.timeseries_features.get_feature_importance_proxy(df, target_column)
                    importance_scores.update(ts_importance)
                
                if self.config.enable_event_features:
                    event_importance = self.event_features.get_feature_importance_proxy(df, target_column)
                    importance_scores.update(event_importance)
                
                if self.config.enable_sentiment_features:
                    sent_importance = self.sentiment_features.get_feature_importance_proxy(df, target_column)
                    importance_scores.update(sent_importance)
                
                if self.config.enable_market_context_features:
                    mc_importance = self.market_context_features.get_feature_importance_proxy(df, target_column)
                    importance_scores.update(mc_importance)
                
                importance_results[symbol] = importance_scores
                
            except Exception as e:
                logger.warning(f"Failed to calculate feature importance for {symbol}: {e}")
        
        return importance_results
    
    def reset(self) -> None:
        """Reset the feature engineer state"""
        
        self.feature_names = []
        self.feature_metadata = {}
        
        # Reset individual feature generators
        self.timeseries_features.feature_names = []
        self.event_features.feature_names = []
        self.sentiment_features.feature_names = []
        self.market_context_features.feature_names = []
        
        logger.info("FeatureEngineer reset to initial state") 