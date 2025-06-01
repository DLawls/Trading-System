"""
Target Builder for creating ML target variables from market data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum


class TargetType(Enum):
    """Types of prediction targets"""
    BINARY_RETURN = "binary_return"          # Up/down classification
    REGRESSION_RETURN = "regression_return"  # Continuous return prediction
    VOLATILITY = "volatility"                # Volatility prediction
    BINARY_BREAKOUT = "binary_breakout"      # Breakout detection
    MULTI_CLASS_RETURN = "multi_class_return" # Multiple return bins


class TargetBuilder:
    """
    Builds various types of ML target variables from market data
    """
    
    def __init__(self):
        """Initialize the target builder"""
        self.target_metadata = {}
    
    def create_targets(
        self,
        data: pd.DataFrame,
        target_config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create target variables based on configuration
        
        Args:
            data: DataFrame with OHLCV data (timestamp indexed)
            target_config: Configuration for target creation
            
        Returns:
            Tuple of (data with targets, target metadata)
        """
        
        if data.empty:
            return data, {}
        
        logger.info(f"Creating targets with config: {target_config}")
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        target_type = TargetType(target_config.get('type', 'binary_return'))
        lookforward_periods = target_config.get('lookforward_periods', 1)
        
        metadata = {
            'target_type': target_type.value,
            'lookforward_periods': lookforward_periods,
            'created_at': datetime.utcnow(),
            'feature_count': 0
        }
        
        # Create targets based on type
        if target_type == TargetType.BINARY_RETURN:
            df, target_meta = self._create_binary_return_targets(df, target_config)
        elif target_type == TargetType.REGRESSION_RETURN:
            df, target_meta = self._create_regression_return_targets(df, target_config)
        elif target_type == TargetType.VOLATILITY:
            df, target_meta = self._create_volatility_targets(df, target_config)
        elif target_type == TargetType.BINARY_BREAKOUT:
            df, target_meta = self._create_breakout_targets(df, target_config)
        elif target_type == TargetType.MULTI_CLASS_RETURN:
            df, target_meta = self._create_multiclass_return_targets(df, target_config)
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        # Update metadata
        metadata.update(target_meta)
        metadata['feature_count'] = len([col for col in df.columns if col.startswith('target_')])
        
        # Store metadata
        self.target_metadata = metadata
        
        logger.info(f"Created {metadata['feature_count']} target features")
        return df, metadata
    
    def _create_binary_return_targets(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create binary return classification targets"""
        
        lookforward_periods = config.get('lookforward_periods', 1)
        threshold = config.get('threshold', 0.01)  # 1% threshold
        
        # Calculate future returns
        future_returns = df['close'].pct_change(periods=lookforward_periods).shift(-lookforward_periods)
        
        # Create binary target (1 = up, 0 = down)
        df['target_binary_return'] = (future_returns > threshold).astype(int)
        
        # Additional targets with different thresholds
        thresholds = config.get('additional_thresholds', [0.005, 0.02, 0.05])
        for thresh in thresholds:
            col_name = f'target_binary_return_{int(thresh*1000):03d}bp'
            df[col_name] = (future_returns > thresh).astype(int)
        
        # Create balanced target (ignore small moves)
        neutral_threshold = config.get('neutral_threshold', 0.005)
        df['target_binary_strong'] = 0  # Default to neutral
        df.loc[future_returns > threshold, 'target_binary_strong'] = 1  # Up
        df.loc[future_returns < -threshold, 'target_binary_strong'] = -1  # Down
        
        metadata = {
            'threshold': threshold,
            'additional_thresholds': thresholds,
            'neutral_threshold': neutral_threshold,
            'class_distribution': df['target_binary_return'].value_counts().to_dict()
        }
        
        return df, metadata
    
    def _create_regression_return_targets(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create regression return targets"""
        
        lookforward_periods = config.get('lookforward_periods', 1)
        clip_outliers = config.get('clip_outliers', True)
        outlier_threshold = config.get('outlier_threshold', 0.1)  # 10%
        
        # Calculate future returns
        future_returns = df['close'].pct_change(periods=lookforward_periods).shift(-lookforward_periods)
        
        # Clip outliers if requested
        if clip_outliers:
            future_returns = future_returns.clip(
                lower=-outlier_threshold,
                upper=outlier_threshold
            )
        
        df['target_return'] = future_returns
        
        # Log returns for better distribution
        df['target_log_return'] = np.log1p(future_returns)
        
        # Multiple horizons
        horizons = config.get('horizons', [3, 5, 10])
        for horizon in horizons:
            col_name = f'target_return_{horizon}p'
            horizon_returns = df['close'].pct_change(periods=horizon).shift(-horizon)
            if clip_outliers:
                horizon_returns = horizon_returns.clip(
                    lower=-outlier_threshold,
                    upper=outlier_threshold
                )
            df[col_name] = horizon_returns
        
        metadata = {
            'clip_outliers': clip_outliers,
            'outlier_threshold': outlier_threshold,
            'horizons': horizons,
            'return_stats': {
                'mean': float(future_returns.mean()),
                'std': float(future_returns.std()),
                'min': float(future_returns.min()),
                'max': float(future_returns.max())
            }
        }
        
        return df, metadata
    
    def _create_volatility_targets(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create volatility prediction targets"""
        
        lookforward_periods = config.get('lookforward_periods', 1)
        vol_window = config.get('vol_window', 5)
        
        # Calculate returns
        returns = df['close'].pct_change()
        
        # Future realized volatility
        future_vol = returns.rolling(window=vol_window).std().shift(-lookforward_periods)
        df['target_volatility'] = future_vol
        
        # Annualized volatility
        df['target_volatility_annualized'] = future_vol * np.sqrt(252)
        
        # High/low volatility binary target
        vol_threshold = config.get('vol_threshold', future_vol.quantile(0.7))
        df['target_high_volatility'] = (future_vol > vol_threshold).astype(int)
        
        # Intraday volatility
        intraday_vol = (df['high'] - df['low']) / df['close']
        future_intraday_vol = intraday_vol.rolling(window=vol_window).mean().shift(-lookforward_periods)
        df['target_intraday_volatility'] = future_intraday_vol
        
        metadata = {
            'vol_window': vol_window,
            'vol_threshold': float(vol_threshold),
            'vol_stats': {
                'mean': float(future_vol.mean()),
                'std': float(future_vol.std()),
                'quantiles': future_vol.quantile([0.25, 0.5, 0.75]).to_dict()
            }
        }
        
        return df, metadata
    
    def _create_breakout_targets(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create breakout detection targets"""
        
        lookforward_periods = config.get('lookforward_periods', 1)
        lookback_window = config.get('lookback_window', 20)
        breakout_threshold = config.get('breakout_threshold', 0.02)  # 2%
        
        # Rolling highs and lows
        rolling_high = df['high'].rolling(window=lookback_window).max()
        rolling_low = df['low'].rolling(window=lookback_window).min()
        
        # Future prices
        future_high = df['high'].shift(-lookforward_periods)
        future_low = df['low'].shift(-lookforward_periods)
        
        # Upward breakout
        upward_breakout = (future_high > rolling_high * (1 + breakout_threshold))
        df['target_upward_breakout'] = upward_breakout.astype(int)
        
        # Downward breakout
        downward_breakout = (future_low < rolling_low * (1 - breakout_threshold))
        df['target_downward_breakout'] = downward_breakout.astype(int)
        
        # Any breakout
        df['target_any_breakout'] = (upward_breakout | downward_breakout).astype(int)
        
        # Strong breakouts (larger threshold)
        strong_threshold = config.get('strong_threshold', 0.05)  # 5%
        strong_up = (future_high > rolling_high * (1 + strong_threshold))
        strong_down = (future_low < rolling_low * (1 - strong_threshold))
        df['target_strong_breakout'] = (strong_up | strong_down).astype(int)
        
        metadata = {
            'lookback_window': lookback_window,
            'breakout_threshold': breakout_threshold,
            'strong_threshold': strong_threshold,
            'breakout_stats': {
                'upward_rate': float(upward_breakout.mean()),
                'downward_rate': float(downward_breakout.mean()),
                'any_breakout_rate': float((upward_breakout | downward_breakout).mean())
            }
        }
        
        return df, metadata
    
    def _create_multiclass_return_targets(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create multi-class return targets"""
        
        lookforward_periods = config.get('lookforward_periods', 1)
        bins = config.get('bins', [-np.inf, -0.02, -0.01, 0.01, 0.02, np.inf])
        labels = config.get('labels', ['strong_down', 'down', 'neutral', 'up', 'strong_up'])
        
        # Calculate future returns
        future_returns = df['close'].pct_change(periods=lookforward_periods).shift(-lookforward_periods)
        
        # Create categorical target
        df['target_return_class'] = pd.cut(
            future_returns,
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Convert to numeric
        df['target_return_class_numeric'] = df['target_return_class'].cat.codes
        
        # One-hot encode
        class_dummies = pd.get_dummies(df['target_return_class'], prefix='target_class')
        df = pd.concat([df, class_dummies], axis=1)
        
        metadata = {
            'bins': bins,
            'labels': labels,
            'class_distribution': df['target_return_class'].value_counts().to_dict()
        }
        
        return df, metadata
    
    def create_event_driven_targets(
        self,
        data: pd.DataFrame,
        events_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create targets specifically around events
        
        Args:
            data: Market data DataFrame
            events_df: Events DataFrame with timestamp column
            config: Target configuration
            
        Returns:
            Tuple of (data with event targets, metadata)
        """
        
        if data.empty or events_df.empty:
            return data, {}
        
        logger.info("Creating event-driven targets")
        
        df = data.copy()
        
        # Parameters
        pre_event_hours = config.get('pre_event_hours', 24)
        post_event_hours = config.get('post_event_hours', 24) 
        impact_threshold = config.get('impact_threshold', 0.01)
        
        # Initialize event targets
        df['target_pre_event_return'] = np.nan
        df['target_post_event_return'] = np.nan
        df['target_event_impact'] = 0
        df['target_has_event_soon'] = 0
        
        for idx, event in events_df.iterrows():
            event_time = pd.to_datetime(event['timestamp'])
            
            # Pre-event window
            pre_start = event_time - timedelta(hours=pre_event_hours)
            pre_data = df[(df.index >= pre_start) & (df.index < event_time)]
            
            if not pre_data.empty:
                pre_return = (pre_data['close'].iloc[-1] - pre_data['close'].iloc[0]) / pre_data['close'].iloc[0]
                
                # Mark period before event
                pre_mask = (df.index >= pre_start) & (df.index < event_time)
                df.loc[pre_mask, 'target_has_event_soon'] = 1
                df.loc[pre_mask, 'target_pre_event_return'] = pre_return
            
            # Post-event window
            post_end = event_time + timedelta(hours=post_event_hours)
            post_data = df[(df.index > event_time) & (df.index <= post_end)]
            
            if not post_data.empty:
                post_return = (post_data['close'].iloc[-1] - post_data['close'].iloc[0]) / post_data['close'].iloc[0]
                
                # Mark post-event impact
                post_mask = (df.index > event_time) & (df.index <= post_end)
                df.loc[post_mask, 'target_post_event_return'] = post_return
                
                # Binary impact target
                if abs(post_return) > impact_threshold:
                    df.loc[post_mask, 'target_event_impact'] = 1
        
        # Fill remaining NaN values
        df['target_pre_event_return'].fillna(0, inplace=True)
        df['target_post_event_return'].fillna(0, inplace=True)
        
        metadata = {
            'pre_event_hours': pre_event_hours,
            'post_event_hours': post_event_hours,
            'impact_threshold': impact_threshold,
            'events_processed': len(events_df),
            'event_impact_rate': float(df['target_event_impact'].mean())
        }
        
        return df, metadata
    
    def get_target_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for all targets in DataFrame"""
        
        target_cols = [col for col in df.columns if col.startswith('target_')]
        
        if not target_cols:
            return {}
        
        summary = {
            'target_count': len(target_cols),
            'targets': {}
        }
        
        for col in target_cols:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Numeric targets
                summary['targets'][col] = {
                    'type': 'numeric',
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'null_count': int(df[col].isnull().sum())
                }
            else:
                # Categorical targets
                summary['targets'][col] = {
                    'type': 'categorical',
                    'unique_values': df[col].nunique(),
                    'value_counts': df[col].value_counts().to_dict(),
                    'null_count': int(df[col].isnull().sum())
                }
        
        return summary 