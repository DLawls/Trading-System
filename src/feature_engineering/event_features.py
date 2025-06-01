"""
Event Features for generating ML features from detected events
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger

from ..event_detection.event_store import EventStore
from ..event_detection.event_classifier import EventType


class EventFeatures:
    """
    Generates features based on detected events for ML models
    """
    
    def __init__(self, event_store: Optional[EventStore] = None):
        """Initialize the event feature generator"""
        self.event_store = event_store
        self.feature_names = []
    
    def generate_features(
        self,
        data: pd.DataFrame,
        ticker: str,
        feature_config: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Generate event-based features for a specific ticker
        
        Args:
            data: DataFrame with market data (must have timestamp/date column)
            ticker: Stock ticker to generate features for
            feature_config: Configuration for features to generate
            
        Returns:
            DataFrame with original data plus event features
        """
        
        if data.empty or not self.event_store:
            logger.warning("No data or event store provided")
            return data
        
        if ticker is None:
            logger.warning("No ticker provided for event features")
            return data
        
        logger.info(f"Generating event features for {ticker}")
        
        # Use default config if none provided
        config = feature_config or self._get_default_config()
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Ensure we have a datetime column
        df = self._prepare_datetime_index(df)
        
        # Get relevant events for this ticker
        events_df = self._get_ticker_events(ticker, df.index.min(), df.index.max())
        
        if events_df.empty:
            logger.warning(f"No events found for {ticker}")
            return self._add_zero_features(df, config)
        
        logger.info(f"Found {len(events_df)} events for {ticker}")
        
        # Generate different types of event features
        if config.get('time_to_event_features', True):
            df = self._add_time_to_event_features(df, events_df, config)
        
        if config.get('event_count_features', True):
            df = self._add_event_count_features(df, events_df, config)
        
        if config.get('event_impact_features', True):
            df = self._add_event_impact_features(df, events_df, config)
        
        if config.get('event_type_features', True):
            df = self._add_event_type_features(df, events_df, config)
        
        if config.get('event_clustering_features', True):
            df = self._add_event_clustering_features(df, events_df, config)
        
        if config.get('event_momentum_features', True):
            df = self._add_event_momentum_features(df, events_df, config)
        
        # Store feature names
        new_features = [col for col in df.columns if col not in data.columns]
        self.feature_names.extend(new_features)
        
        logger.info(f"Generated {len(new_features)} event features")
        return df
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature generation configuration"""
        return {
            'time_to_event_features': True,
            'event_count_features': True,
            'event_impact_features': True,
            'event_type_features': True,
            'event_clustering_features': True,
            'event_momentum_features': True,
            'lookback_days': [1, 3, 7, 14, 30],
            'lookahead_days': [1, 3, 7, 14],
            'impact_thresholds': [0.3, 0.5, 0.7],
            'event_types': list(EventType)
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
    
    def _get_ticker_events(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get events for a specific ticker within date range"""
        
        # Expand date range to capture events outside the window
        extended_start = start_date - timedelta(days=60)
        extended_end = end_date + timedelta(days=30)
        
        events = self.event_store.get_events(
            ticker=ticker,
            start_date=extended_start,
            end_date=extended_end,
            limit=1000
        )
        
        if not events:
            return pd.DataFrame()
        
        # Convert to DataFrame
        events_df = pd.DataFrame(events)
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        
        return events_df.sort_values('timestamp')
    
    def _add_zero_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add zero-valued features when no events are found"""
        
        # Add basic zero features
        df['event_count_total'] = 0
        df['days_since_last_event'] = np.inf
        df['days_to_next_event'] = np.inf
        df['recent_event_impact'] = 0
        
        # Add zero features for different lookback periods
        for days in config.get('lookback_days', [1, 3, 7, 14, 30]):
            df[f'event_count_{days}d'] = 0
            df[f'max_impact_{days}d'] = 0
            df[f'avg_impact_{days}d'] = 0
        
        return df
    
    def _add_time_to_event_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add time-to-event features"""
        
        # Initialize time-based features
        df['days_since_last_event'] = np.nan
        df['days_to_next_event'] = np.nan
        df['hours_since_last_event'] = np.nan
        df['hours_to_next_event'] = np.nan
        
        for idx, row in df.iterrows():
            current_time = idx
            
            # Find previous events
            prev_events = events_df[events_df['timestamp'] <= current_time]
            if not prev_events.empty:
                last_event_time = prev_events['timestamp'].max()
                days_since = (current_time - last_event_time).total_seconds() / (24 * 3600)
                hours_since = (current_time - last_event_time).total_seconds() / 3600
                df.at[idx, 'days_since_last_event'] = days_since
                df.at[idx, 'hours_since_last_event'] = hours_since
            
            # Find future events
            future_events = events_df[events_df['timestamp'] > current_time]
            if not future_events.empty:
                next_event_time = future_events['timestamp'].min()
                days_to = (next_event_time - current_time).total_seconds() / (24 * 3600)
                hours_to = (next_event_time - current_time).total_seconds() / 3600
                df.at[idx, 'days_to_next_event'] = days_to
                df.at[idx, 'hours_to_next_event'] = hours_to
        
        # Fill infinite values for missing events
        df['days_since_last_event'].fillna(np.inf, inplace=True)
        df['days_to_next_event'].fillna(np.inf, inplace=True)
        df['hours_since_last_event'].fillna(np.inf, inplace=True)
        df['hours_to_next_event'].fillna(np.inf, inplace=True)
        
        # Binary features for recent/upcoming events
        df['has_recent_event_1d'] = (df['days_since_last_event'] <= 1).astype(int)
        df['has_recent_event_3d'] = (df['days_since_last_event'] <= 3).astype(int)
        df['has_upcoming_event_1d'] = (df['days_to_next_event'] <= 1).astype(int)
        df['has_upcoming_event_3d'] = (df['days_to_next_event'] <= 3).astype(int)
        
        return df
    
    def _add_event_count_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add event count features for different time windows"""
        
        lookback_days = config.get('lookback_days', [1, 3, 7, 14, 30])
        
        for days in lookback_days:
            df[f'event_count_{days}d'] = 0
            df[f'high_impact_count_{days}d'] = 0
            
            for idx, row in df.iterrows():
                current_time = idx
                window_start = current_time - timedelta(days=days)
                
                # Count events in window
                window_events = events_df[
                    (events_df['timestamp'] >= window_start) &
                    (events_df['timestamp'] <= current_time)
                ]
                
                df.at[idx, f'event_count_{days}d'] = len(window_events)
                
                # Count high impact events
                high_impact_events = window_events[window_events['impact_score'] >= 0.6]
                df.at[idx, f'high_impact_count_{days}d'] = len(high_impact_events)
        
        # Total cumulative event count
        df['event_count_total'] = 0
        for idx, row in df.iterrows():
            current_time = idx
            total_events = events_df[events_df['timestamp'] <= current_time]
            df.at[idx, 'event_count_total'] = len(total_events)
        
        # Event frequency (events per day in last 30 days)
        if 'event_count_30d' in df.columns:
            df['event_frequency'] = df['event_count_30d'] / 30
        
        return df
    
    def _add_event_impact_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add event impact-based features"""
        
        lookback_days = config.get('lookback_days', [1, 3, 7, 14, 30])
        impact_thresholds = config.get('impact_thresholds', [0.3, 0.5, 0.7])
        
        for days in lookback_days:
            df[f'max_impact_{days}d'] = 0
            df[f'avg_impact_{days}d'] = 0
            df[f'sum_impact_{days}d'] = 0
            
            for threshold in impact_thresholds:
                df[f'impact_above_{threshold}_{days}d'] = 0
            
            for idx, row in df.iterrows():
                current_time = idx
                window_start = current_time - timedelta(days=days)
                
                window_events = events_df[
                    (events_df['timestamp'] >= window_start) &
                    (events_df['timestamp'] <= current_time)
                ]
                
                if not window_events.empty:
                    impacts = window_events['impact_score']
                    df.at[idx, f'max_impact_{days}d'] = impacts.max()
                    df.at[idx, f'avg_impact_{days}d'] = impacts.mean()
                    df.at[idx, f'sum_impact_{days}d'] = impacts.sum()
                    
                    # Count events above impact thresholds
                    for threshold in impact_thresholds:
                        count = len(window_events[window_events['impact_score'] >= threshold])
                        df.at[idx, f'impact_above_{threshold}_{days}d'] = count
        
        # Recent event impact (weighted by recency)
        df['recent_event_impact'] = 0
        for idx, row in df.iterrows():
            current_time = idx
            recent_events = events_df[
                events_df['timestamp'] <= current_time
            ].tail(5)  # Last 5 events
            
            if not recent_events.empty:
                # Weight by recency (more recent = higher weight)
                weights = np.exp(-((current_time - recent_events['timestamp']).dt.total_seconds() / (24 * 3600)) / 7)
                weighted_impact = (recent_events['impact_score'] * weights).sum() / weights.sum()
                df.at[idx, 'recent_event_impact'] = weighted_impact
        
        return df
    
    def _add_event_type_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add features based on event types"""
        
        event_types = config.get('event_types', list(EventType))
        lookback_days = config.get('lookback_days', [7, 14, 30])
        
        # Initialize event type count features
        for event_type in event_types:
            type_name = event_type.value if hasattr(event_type, 'value') else str(event_type)
            
            for days in lookback_days:
                df[f'{type_name}_count_{days}d'] = 0
        
        # Count events by type in different windows
        for days in lookback_days:
            for idx, row in df.iterrows():
                current_time = idx
                window_start = current_time - timedelta(days=days)
                
                window_events = events_df[
                    (events_df['timestamp'] >= window_start) &
                    (events_df['timestamp'] <= current_time)
                ]
                
                if not window_events.empty:
                    type_counts = window_events['event_type'].value_counts()
                    
                    for event_type in event_types:
                        type_name = event_type.value if hasattr(event_type, 'value') else str(event_type)
                        count = type_counts.get(type_name, 0)
                        df.at[idx, f'{type_name}_count_{days}d'] = count
        
        # Most recent event type
        df['last_event_type'] = 'none'
        for idx, row in df.iterrows():
            current_time = idx
            past_events = events_df[events_df['timestamp'] <= current_time]
            if not past_events.empty:
                last_event = past_events.iloc[-1]
                df.at[idx, 'last_event_type'] = last_event['event_type']
        
        # One-hot encode last event type
        event_type_dummies = pd.get_dummies(df['last_event_type'], prefix='last_event')
        df = pd.concat([df, event_type_dummies], axis=1)
        
        return df
    
    def _add_event_clustering_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add features about event clustering/spacing"""
        
        # Event clustering in different windows
        for days in [3, 7, 14]:
            df[f'event_cluster_{days}d'] = 0
            
            for idx, row in df.iterrows():
                current_time = idx
                window_start = current_time - timedelta(days=days)
                
                window_events = events_df[
                    (events_df['timestamp'] >= window_start) &
                    (events_df['timestamp'] <= current_time)
                ]
                
                # Consider it a cluster if 3+ events in the window
                if len(window_events) >= 3:
                    df.at[idx, f'event_cluster_{days}d'] = 1
        
        # Average time between events
        df['avg_event_spacing'] = np.nan
        for idx, row in df.iterrows():
            current_time = idx
            past_events = events_df[events_df['timestamp'] <= current_time].tail(10)
            
            if len(past_events) >= 2:
                time_diffs = past_events['timestamp'].diff().dt.total_seconds() / (24 * 3600)
                avg_spacing = time_diffs.mean()
                df.at[idx, 'avg_event_spacing'] = avg_spacing
        
        df['avg_event_spacing'].fillna(np.inf, inplace=True)
        
        return df
    
    def _add_event_momentum_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add features about event momentum/trends"""
        
        # Impact momentum (trend in event impact over time)
        df['impact_momentum_7d'] = 0
        df['impact_momentum_14d'] = 0
        
        for window_days in [7, 14]:
            for idx, row in df.iterrows():
                current_time = idx
                window_start = current_time - timedelta(days=window_days)
                
                window_events = events_df[
                    (events_df['timestamp'] >= window_start) &
                    (events_df['timestamp'] <= current_time)
                ]
                
                if len(window_events) >= 3:
                    # Calculate momentum as correlation with time
                    window_events = window_events.sort_values('timestamp')
                    time_numeric = (window_events['timestamp'] - window_events['timestamp'].min()).dt.total_seconds()
                    momentum = np.corrcoef(time_numeric, window_events['impact_score'])[0, 1]
                    
                    if not np.isnan(momentum):
                        df.at[idx, f'impact_momentum_{window_days}d'] = momentum
        
        # Event acceleration (change in event frequency)
        df['event_acceleration'] = 0
        for idx, row in df.iterrows():
            if 'event_count_7d' in df.columns and 'event_count_14d' in df.columns:
                recent_freq = df.at[idx, 'event_count_7d'] / 7
                older_freq = (df.at[idx, 'event_count_14d'] - df.at[idx, 'event_count_7d']) / 7
                
                if older_freq > 0:
                    acceleration = (recent_freq - older_freq) / older_freq
                    df.at[idx, 'event_acceleration'] = acceleration
        
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
        """Generate summary statistics for event features"""
        
        feature_cols = [col for col in df.columns if col in self.feature_names]
        
        if not feature_cols:
            return {}
        
        summary = {
            'total_features': len(feature_cols),
            'feature_types': {
                'time_features': len([f for f in feature_cols if 'days_' in f or 'hours_' in f]),
                'count_features': len([f for f in feature_cols if 'count' in f]),
                'impact_features': len([f for f in feature_cols if 'impact' in f]),
                'type_features': len([f for f in feature_cols if any(t.value in f for t in EventType)]),
                'clustering_features': len([f for f in feature_cols if 'cluster' in f or 'spacing' in f]),
                'momentum_features': len([f for f in feature_cols if 'momentum' in f or 'acceleration' in f])
            },
            'value_ranges': {
                col: {
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None
                }
                for col in feature_cols[:10]  # Show first 10
            }
        }
        
        return summary 