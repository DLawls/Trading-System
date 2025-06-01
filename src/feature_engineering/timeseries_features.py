"""
Timeseries Features for generating technical indicators and market features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from loguru import logger

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Using manual implementations for technical indicators.")


class TimeseriesFeatures:
    """
    Generates technical indicators and time-series features from OHLCV data
    """
    
    def __init__(self):
        """Initialize the timeseries feature generator"""
        self.feature_names = []
        
    def generate_features(
        self,
        data: pd.DataFrame,
        ticker: str = None,
        feature_config: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive timeseries features from OHLCV data
        
        Args:
            data: DataFrame with OHLCV columns
            ticker: Stock ticker (optional, for naming)
            feature_config: Configuration for features to generate
            
        Returns:
            DataFrame with original data plus generated features
        """
        
        if data.empty:
            return data
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return data
        
        logger.info(f"Generating timeseries features for {ticker or 'unknown ticker'}")
        
        # Use default config if none provided
        config = feature_config or self._get_default_config()
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Generate basic price features
        if config.get('price_features', True):
            df = self._add_price_features(df)
        
        # Generate moving averages
        if config.get('moving_averages', True):
            df = self._add_moving_averages(df, config.get('ma_windows', [5, 10, 20, 50, 200]))
        
        # Generate volatility features
        if config.get('volatility_features', True):
            df = self._add_volatility_features(df, config.get('vol_windows', [5, 10, 20]))
        
        # Generate volume features
        if config.get('volume_features', True):
            df = self._add_volume_features(df, config.get('vol_ma_windows', [5, 10, 20]))
        
        # Generate momentum indicators
        if config.get('momentum_features', True):
            df = self._add_momentum_features(df)
        
        # Generate technical indicators
        if config.get('technical_indicators', True):
            df = self._add_technical_indicators(df)
        
        # Generate pattern features
        if config.get('pattern_features', True):
            df = self._add_pattern_features(df)
        
        # Generate time-based features
        if config.get('time_features', True):
            df = self._add_time_features(df)
        
        # Store feature names for later use
        new_features = [col for col in df.columns if col not in data.columns]
        self.feature_names.extend(new_features)
        
        logger.info(f"Generated {len(new_features)} timeseries features")
        return df
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature generation configuration"""
        return {
            'price_features': True,
            'moving_averages': True,
            'volatility_features': True,
            'volume_features': True,
            'momentum_features': True,
            'technical_indicators': True,
            'pattern_features': True,
            'time_features': True,
            'ma_windows': [5, 10, 20, 50, 200],
            'vol_windows': [5, 10, 20, 50],
            'vol_ma_windows': [5, 10, 20]
        }
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        
        # Price changes
        df['price_change'] = df['close'] - df['close'].shift(1)
        df['price_change_pct'] = df['close'].pct_change()
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Intraday features
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['open_close_pct'] = (df['close'] - df['open']) / df['open']
        
        # Price gaps
        df['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).clip(lower=0)
        df['gap_down'] = ((df['close'].shift(1) - df['open']) / df['close'].shift(1)).clip(lower=0)
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Weighted close
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add simple and exponential moving averages"""
        
        for window in windows:
            # Simple moving average
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            
            # Exponential moving average
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # Price vs MA ratios
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
            
            # MA slope (momentum)
            df[f'sma_{window}_slope'] = (df[f'sma_{window}'] - df[f'sma_{window}'].shift(5)) / 5
        
        # MA crossovers (using common pairs)
        if 10 in windows and 20 in windows:
            df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        
        if 20 in windows and 50 in windows:
            df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        if 50 in windows and 200 in windows:
            df['sma_50_200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)  # Golden/Death cross
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add volatility-based features"""
        
        for window in windows:
            # Rolling standard deviation
            df[f'volatility_{window}'] = df['log_return'].rolling(window=window).std()
            
            # Rolling range (high-low volatility)
            df[f'range_volatility_{window}'] = (df['high'] - df['low']).rolling(window=window).mean() / df['close']
            
            # Parkinson volatility (more efficient than close-to-close)
            df[f'parkinson_vol_{window}'] = np.sqrt(
                0.361 * (np.log(df['high'] / df['low'])).rolling(window=window).mean()
            )
        
        # Volatility regimes
        df['vol_regime'] = pd.cut(
            df['volatility_20'], 
            bins=[-np.inf, df['volatility_20'].quantile(0.33), df['volatility_20'].quantile(0.67), np.inf],
            labels=['low', 'medium', 'high']
        )
        
        # VIX-like fear index (simplified)
        df['fear_index'] = df['volatility_20'] * 100
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add volume-based features"""
        
        for window in windows:
            # Volume moving averages
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            
            # Volume ratio
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['price_change']) * df['volume']).cumsum()
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['price_change_pct'] * df['volume']).cumsum()
        
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        # Volume spikes
        if 'volume_sma_20' in df.columns:
            df['volume_spike'] = (df['volume'] > 2 * df['volume_sma_20']).astype(int)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        
        # Rate of Change (ROC)
        for period in [1, 5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # Williams %R
        for period in [14, 20]:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            df[f'williams_r_{period}'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators (using TA-Lib if available, else manual)"""
        
        if TALIB_AVAILABLE:
            df = self._add_talib_indicators(df)
        else:
            df = self._add_manual_indicators(df)
        
        return df
    
    def _add_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators using TA-Lib"""
        
        high, low, close, volume = df['high'].values, df['low'].values, df['close'].values, df['volume'].values
        
        # RSI
        df['rsi_14'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        df['bb_position'] = (close - lower) / (upper - lower)
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Commodity Channel Index
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)
        
        return df
    
    def _add_manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators using manual calculations"""
        
        # RSI (manual implementation)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Simple MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        
        # Doji patterns (open ≈ close)
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        df['is_doji'] = (body_size / total_range < 0.1).astype(int)
        
        # Hammer/Hanging man patterns
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        df['is_hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
        
        # Engulfing patterns
        bullish_engulfing = (
            (df['open'] < df['close']) &  # Current candle is green
            (df['open'].shift(1) > df['close'].shift(1)) &  # Previous candle is red
            (df['open'] < df['close'].shift(1)) &  # Current open < previous close
            (df['close'] > df['open'].shift(1))  # Current close > previous open
        )
        df['bullish_engulfing'] = bullish_engulfing.astype(int)
        
        # Gap patterns
        df['gap_up_pattern'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down_pattern'] = (df['high'] < df['low'].shift(1)).astype(int)
        
        # Higher highs, higher lows (uptrend)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['uptrend_signal'] = (df['higher_high'] & df['higher_low']).astype(int)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        if df.index.name == 'timestamp' or 'timestamp' in df.columns:
            # If timestamp is available
            if 'timestamp' in df.columns:
                ts = pd.to_datetime(df['timestamp'])
            else:
                ts = df.index
            
            df['hour'] = ts.hour
            df['day_of_week'] = ts.dayofweek  # Monday=0, Sunday=6
            df['day_of_month'] = ts.day
            df['month'] = ts.month
            df['quarter'] = ts.quarter
            
            # Market session features
            df['is_market_open'] = ((ts.hour >= 9) & (ts.hour < 16)).astype(int)
            df['is_pre_market'] = ((ts.hour >= 4) & (ts.hour < 9)).astype(int)
            df['is_after_market'] = ((ts.hour >= 16) & (ts.hour < 20)).astype(int)
            
            # Weekend effect
            df['is_weekend'] = (ts.dayofweek >= 5).astype(int)
            df['is_monday'] = (ts.dayofweek == 0).astype(int)
            df['is_friday'] = (ts.dayofweek == 4).astype(int)
            
            # End of month/quarter effects - Fixed calculation
            month_end = pd.to_datetime(ts).to_series().apply(lambda x: x + pd.offsets.MonthEnd(0))
            df['days_to_month_end'] = (month_end - pd.to_datetime(ts).to_series()).dt.days
            df['is_month_end'] = (df['days_to_month_end'] <= 3).astype(int)
        
        return df
    
    def get_feature_importance_proxy(self, df: pd.DataFrame, target_col: str = 'price_change_pct') -> Dict[str, float]:
        """
        Calculate simple feature importance proxy based on correlation with returns
        """
        
        if target_col not in df.columns or len(self.feature_names) == 0:
            return {}
        
        importance = {}
        target = df[target_col].shift(-1)  # Next period return
        
        for feature in self.feature_names:
            if feature in df.columns:
                # Only calculate correlation for numeric features
                if df[feature].dtype in ['int64', 'float64', 'int32', 'float32']:
                    try:
                        corr = df[feature].corr(target)
                        importance[feature] = abs(corr) if not pd.isna(corr) else 0.0
                    except Exception:
                        importance[feature] = 0.0
                else:
                    # Skip categorical features
                    importance[feature] = 0.0
        
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def generate_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for generated features"""
        
        feature_cols = [col for col in df.columns if col in self.feature_names]
        
        if not feature_cols:
            return {}
        
        summary = {
            'total_features': len(feature_cols),
            'feature_types': {
                'price_features': len([f for f in feature_cols if 'price' in f or 'return' in f]),
                'ma_features': len([f for f in feature_cols if 'sma' in f or 'ema' in f]),
                'volatility_features': len([f for f in feature_cols if 'vol' in f or 'std' in f]),
                'volume_features': len([f for f in feature_cols if 'volume' in f or 'obv' in f]),
                'technical_features': len([f for f in feature_cols if any(x in f for x in ['rsi', 'macd', 'bb_', 'stoch'])]),
                'pattern_features': len([f for f in feature_cols if any(x in f for x in ['doji', 'hammer', 'engulf'])]),
                'time_features': len([f for f in feature_cols if any(x in f for x in ['hour', 'day', 'month', 'is_'])])
            },
            'missing_values': df[feature_cols].isnull().sum().to_dict(),
            'feature_ranges': {
                col: {
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None
                }
                for col in feature_cols[:10]  # Show first 10 for brevity
            }
        }
        
        return summary 