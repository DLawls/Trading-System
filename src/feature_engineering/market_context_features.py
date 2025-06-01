"""
Market Context Features for generating ML features from market environment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Market context features will be limited.")


class MarketContextFeatures:
    """
    Generates market context features including macro indicators and crypto-specific metrics
    """
    
    def __init__(self, market_data_cache: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the market context feature generator
        
        Args:
            market_data_cache: Pre-loaded market data for common indicators
        """
        self.market_data_cache = market_data_cache or {}
        self.feature_names = []
        
        # Common market indicators to track
        self.market_indicators = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ 100 ETF', 
            'IWM': 'Russell 2000 ETF',
            'TLT': '20+ Year Treasury ETF',
            'GLD': 'Gold ETF',
            'VIX': 'Volatility Index',
            'DXY': 'US Dollar Index',
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum'
        }
        
        # Crypto-specific indicators
        self.crypto_indicators = {
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum',
            'BNB-USD': 'Binance Coin',
            'ADA-USD': 'Cardano',
            'SOL-USD': 'Solana'
        }
    
    def generate_features(
        self,
        data: pd.DataFrame,
        ticker: str = None,
        feature_config: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Generate market context features
        
        Args:
            data: DataFrame with market data (must have timestamp/date column)
            ticker: Stock ticker for context (optional)
            feature_config: Configuration for features to generate
            
        Returns:
            DataFrame with original data plus market context features
        """
        
        if data.empty:
            return data
        
        logger.info(f"Generating market context features for {ticker or 'general market'}")
        
        # Use default config if none provided
        config = feature_config or self._get_default_config()
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Prepare datetime index
        df = self._prepare_datetime_index(df)
        
        # Get date range for fetching market data
        start_date = df.index.min() - timedelta(days=60)  # Extra buffer for indicators
        end_date = df.index.max() + timedelta(days=1)
        
        # Generate different types of market context features
        if config.get('market_regime_features', True):
            df = self._add_market_regime_features(df, start_date, end_date, config)
        
        if config.get('sector_performance_features', True):
            df = self._add_sector_performance_features(df, start_date, end_date, config)
        
        if config.get('macro_indicator_features', True):
            df = self._add_macro_indicator_features(df, start_date, end_date, config)
        
        if config.get('volatility_regime_features', True):
            df = self._add_volatility_regime_features(df, start_date, end_date, config)
        
        if config.get('crypto_market_features', True):
            df = self._add_crypto_market_features(df, start_date, end_date, config)
        
        if config.get('correlation_features', True):
            df = self._add_correlation_features(df, start_date, end_date, config, ticker)
        
        if config.get('calendar_features', True):
            df = self._add_calendar_features(df, config)
        
        # Store feature names
        new_features = [col for col in df.columns if col not in data.columns]
        self.feature_names.extend(new_features)
        
        logger.info(f"Generated {len(new_features)} market context features")
        return df
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature generation configuration"""
        return {
            'market_regime_features': True,
            'sector_performance_features': True,
            'macro_indicator_features': True,
            'volatility_regime_features': True,
            'crypto_market_features': True,
            'correlation_features': True,
            'calendar_features': True,
            'lookback_periods': [5, 10, 20, 50],
            'regime_windows': [20, 50, 200],
            'correlation_windows': [20, 60]
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
    
    def _get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get market data for a symbol"""
        
        # Check cache first
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]
        
        if not YFINANCE_AVAILABLE:
            logger.warning(f"Cannot fetch {symbol} data - yfinance not available")
            return None
        
        try:
            ticker_obj = yf.Ticker(symbol)
            data = ticker_obj.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Cache the data
            self.market_data_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _add_market_regime_features(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add market regime features"""
        
        # Get S&P 500 data for market regime
        spy_data = self._get_market_data('SPY', start_date, end_date)
        
        if spy_data is None:
            # Add zero features if no data
            df['market_regime'] = 'neutral'
            df['market_trend_20d'] = 0
            df['market_trend_50d'] = 0
            df['market_above_ma200'] = 0
            return df
        
        regime_windows = config.get('regime_windows', [20, 50, 200])
        
        # Calculate moving averages for SPY
        for window in regime_windows:
            spy_data[f'ma_{window}'] = spy_data['close'].rolling(window=window).mean()
        
        # Align with our dataframe dates
        market_features = pd.DataFrame(index=df.index)
        
        for idx in df.index:
            if idx in spy_data.index:
                spy_row = spy_data.loc[idx]
                
                # Market regime based on price vs moving averages
                if spy_row['close'] > spy_row.get('ma_200', spy_row['close']):
                    if spy_row['close'] > spy_row.get('ma_50', spy_row['close']):
                        regime = 'bull'
                    else:
                        regime = 'neutral'
                else:
                    regime = 'bear'
                
                market_features.at[idx, 'market_regime'] = regime
                
                # Market trend strength
                for window in [20, 50]:
                    if f'ma_{window}' in spy_data.columns:
                        ma_val = spy_row.get(f'ma_{window}')
                        if pd.notna(ma_val) and ma_val > 0:
                            trend = (spy_row['close'] - ma_val) / ma_val * 100
                            market_features.at[idx, f'market_trend_{window}d'] = trend
                
                # Binary: Above 200-day MA
                if 'ma_200' in spy_data.columns:
                    ma_200 = spy_row.get('ma_200')
                    if pd.notna(ma_200):
                        market_features.at[idx, 'market_above_ma200'] = int(spy_row['close'] > ma_200)
            else:
                # Fill with neutral values for missing dates
                market_features.at[idx, 'market_regime'] = 'neutral'
                market_features.at[idx, 'market_trend_20d'] = 0
                market_features.at[idx, 'market_trend_50d'] = 0
                market_features.at[idx, 'market_above_ma200'] = 0
        
        # Fill any remaining NaN values
        market_features.fillna(method='ffill', inplace=True)
        market_features.fillna(0, inplace=True)
        
        # Add to main dataframe
        df = df.join(market_features, how='left')
        
        # One-hot encode market regime
        regime_dummies = pd.get_dummies(df['market_regime'], prefix='regime')
        df = pd.concat([df, regime_dummies], axis=1)
        
        return df
    
    def _add_sector_performance_features(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add sector performance features"""
        
        # Major sector ETFs
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare', 
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        sector_data = {}
        for etf, sector in sector_etfs.items():
            data = self._get_market_data(etf, start_date, end_date)
            if data is not None:
                sector_data[sector.lower().replace(' ', '_')] = data
        
        if not sector_data:
            # Add zero features if no sector data
            df['best_sector_performance'] = 0
            df['worst_sector_performance'] = 0
            df['sector_dispersion'] = 0
            return df
        
        # Calculate sector performance features
        lookback_periods = config.get('lookback_periods', [5, 10, 20])
        
        for period in lookback_periods:
            sector_returns = {}
            
            for idx in df.index:
                daily_returns = {}
                
                for sector, data in sector_data.items():
                    if idx in data.index:
                        current_price = data.loc[idx, 'close']
                        past_date = idx - timedelta(days=period)
                        
                        # Find closest past date in data
                        past_prices = data[data.index <= past_date]
                        if not past_prices.empty:
                            past_price = past_prices['close'].iloc[-1]
                            if past_price > 0:
                                ret = (current_price - past_price) / past_price * 100
                                daily_returns[sector] = ret
                
                if daily_returns:
                    returns_list = list(daily_returns.values())
                    df.at[idx, f'best_sector_performance_{period}d'] = max(returns_list)
                    df.at[idx, f'worst_sector_performance_{period}d'] = min(returns_list)
                    df.at[idx, f'sector_dispersion_{period}d'] = np.std(returns_list)
                else:
                    df.at[idx, f'best_sector_performance_{period}d'] = 0
                    df.at[idx, f'worst_sector_performance_{period}d'] = 0
                    df.at[idx, f'sector_dispersion_{period}d'] = 0
        
        return df
    
    def _add_macro_indicator_features(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add macro economic indicator features"""
        
        macro_indicators = {
            'TLT': 'treasury_20y',  # 20+ Year Treasury ETF
            'GLD': 'gold',          # Gold ETF
            'DXY': 'dollar_index',  # US Dollar Index (if available)
            'VIX': 'volatility'     # Volatility Index
        }
        
        macro_data = {}
        for symbol, name in macro_indicators.items():
            data = self._get_market_data(symbol, start_date, end_date)
            if data is not None:
                macro_data[name] = data
        
        # Add macro features
        for name, data in macro_data.items():
            for idx in df.index:
                if idx in data.index:
                    current_price = data.loc[idx, 'close']
                    
                    # Price level
                    df.at[idx, f'{name}_price'] = current_price
                    
                    # 1-day return
                    prev_date = idx - timedelta(days=1)
                    prev_prices = data[data.index <= prev_date]
                    if not prev_prices.empty:
                        prev_price = prev_prices['close'].iloc[-1]
                        if prev_price > 0:
                            ret = (current_price - prev_price) / prev_price * 100
                            df.at[idx, f'{name}_return_1d'] = ret
                    
                    # 5-day return
                    past_date = idx - timedelta(days=5)
                    past_prices = data[data.index <= past_date]
                    if not past_prices.empty:
                        past_price = past_prices['close'].iloc[-1]
                        if past_price > 0:
                            ret = (current_price - past_price) / past_price * 100
                            df.at[idx, f'{name}_return_5d'] = ret
                else:
                    # Fill with zeros for missing data
                    df.at[idx, f'{name}_price'] = 0
                    df.at[idx, f'{name}_return_1d'] = 0
                    df.at[idx, f'{name}_return_5d'] = 0
        
        # Fill missing values
        macro_cols = [col for col in df.columns if any(indicator in col for indicator in macro_indicators.values())]
        df[macro_cols] = df[macro_cols].fillna(method='ffill').fillna(0)
        
        return df
    
    def _add_volatility_regime_features(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add volatility regime features"""
        
        # Get VIX data
        vix_data = self._get_market_data('VIX', start_date, end_date)
        
        if vix_data is None:
            df['volatility_regime'] = 'medium'
            df['vix_level'] = 20  # Default VIX level
            df['vix_spike'] = 0
            return df
        
        # Calculate VIX features
        for idx in df.index:
            if idx in vix_data.index:
                vix_level = vix_data.loc[idx, 'close']
                df.at[idx, 'vix_level'] = vix_level
                
                # Volatility regime based on VIX levels
                if vix_level < 15:
                    regime = 'low'
                elif vix_level < 25:
                    regime = 'medium'
                elif vix_level < 35:
                    regime = 'high'
                else:
                    regime = 'extreme'
                
                df.at[idx, 'volatility_regime'] = regime
                
                # VIX spike detection (>20% increase in 1 day)
                prev_date = idx - timedelta(days=1)
                prev_vix = vix_data[vix_data.index <= prev_date]
                if not prev_vix.empty:
                    prev_vix_val = prev_vix['close'].iloc[-1]
                    if prev_vix_val > 0:
                        vix_change = (vix_level - prev_vix_val) / prev_vix_val
                        df.at[idx, 'vix_spike'] = int(vix_change > 0.2)
                    else:
                        df.at[idx, 'vix_spike'] = 0
                else:
                    df.at[idx, 'vix_spike'] = 0
            else:
                df.at[idx, 'volatility_regime'] = 'medium'
                df.at[idx, 'vix_level'] = 20
                df.at[idx, 'vix_spike'] = 0
        
        # One-hot encode volatility regime
        vol_regime_dummies = pd.get_dummies(df['volatility_regime'], prefix='vol_regime')
        df = pd.concat([df, vol_regime_dummies], axis=1)
        
        return df
    
    def _add_crypto_market_features(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add crypto market features"""
        
        # Get Bitcoin data as crypto market proxy
        btc_data = self._get_market_data('BTC-USD', start_date, end_date)
        
        if btc_data is None:
            # Add default crypto features
            df['crypto_market_trend'] = 0
            df['crypto_volatility'] = 0
            df['crypto_dominance_signal'] = 0
            return df
        
        # Calculate BTC features
        btc_data['returns'] = btc_data['close'].pct_change()
        btc_data['volatility_20d'] = btc_data['returns'].rolling(window=20).std() * np.sqrt(365) * 100
        
        # Get Ethereum data for dominance calculation
        eth_data = self._get_market_data('ETH-USD', start_date, end_date)
        
        for idx in df.index:
            if idx in btc_data.index:
                btc_row = btc_data.loc[idx]
                
                # Crypto market trend (20-day momentum)
                past_date = idx - timedelta(days=20)
                past_btc = btc_data[btc_data.index <= past_date]
                if not past_btc.empty:
                    past_price = past_btc['close'].iloc[-1]
                    if past_price > 0:
                        trend = (btc_row['close'] - past_price) / past_price * 100
                        df.at[idx, 'crypto_market_trend'] = trend
                    else:
                        df.at[idx, 'crypto_market_trend'] = 0
                else:
                    df.at[idx, 'crypto_market_trend'] = 0
                
                # Crypto volatility
                vol = btc_row.get('volatility_20d', 0)
                df.at[idx, 'crypto_volatility'] = vol if pd.notna(vol) else 0
                
                # BTC/ETH dominance signal
                if eth_data is not None and idx in eth_data.index:
                    eth_row = eth_data.loc[idx]
                    
                    # Calculate relative performance
                    btc_20d_ret = 0
                    eth_20d_ret = 0
                    
                    past_date = idx - timedelta(days=20)
                    past_btc = btc_data[btc_data.index <= past_date]
                    past_eth = eth_data[eth_data.index <= past_date]
                    
                    if not past_btc.empty and not past_eth.empty:
                        btc_past = past_btc['close'].iloc[-1]
                        eth_past = past_eth['close'].iloc[-1]
                        
                        if btc_past > 0 and eth_past > 0:
                            btc_20d_ret = (btc_row['close'] - btc_past) / btc_past
                            eth_20d_ret = (eth_row['close'] - eth_past) / eth_past
                    
                    # Dominance signal: BTC outperforming ETH
                    dominance_signal = btc_20d_ret - eth_20d_ret
                    df.at[idx, 'crypto_dominance_signal'] = dominance_signal
                else:
                    df.at[idx, 'crypto_dominance_signal'] = 0
            else:
                df.at[idx, 'crypto_market_trend'] = 0
                df.at[idx, 'crypto_volatility'] = 0
                df.at[idx, 'crypto_dominance_signal'] = 0
        
        return df
    
    def _add_correlation_features(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any],
        ticker: str = None
    ) -> pd.DataFrame:
        """Add correlation features with market indices"""
        
        if not ticker:
            # Add default correlation features
            df['spy_correlation_20d'] = 0
            df['qqq_correlation_20d'] = 0
            return df
        
        # Get ticker data
        ticker_data = self._get_market_data(ticker, start_date, end_date)
        if ticker_data is None:
            df['spy_correlation_20d'] = 0
            df['qqq_correlation_20d'] = 0
            return df
        
        # Get market index data
        spy_data = self._get_market_data('SPY', start_date, end_date)
        qqq_data = self._get_market_data('QQQ', start_date, end_date)
        
        correlation_windows = config.get('correlation_windows', [20, 60])
        
        # Calculate returns
        ticker_data['returns'] = ticker_data['close'].pct_change()
        if spy_data is not None:
            spy_data['returns'] = spy_data['close'].pct_change()
        if qqq_data is not None:
            qqq_data['returns'] = qqq_data['close'].pct_change()
        
        for window in correlation_windows:
            for idx in df.index:
                # SPY correlation
                if spy_data is not None and idx in ticker_data.index and idx in spy_data.index:
                    end_date_window = idx
                    start_date_window = idx - timedelta(days=window)
                    
                    ticker_window = ticker_data[
                        (ticker_data.index >= start_date_window) & 
                        (ticker_data.index <= end_date_window)
                    ]['returns'].dropna()
                    
                    spy_window = spy_data[
                        (spy_data.index >= start_date_window) & 
                        (spy_data.index <= end_date_window)
                    ]['returns'].dropna()
                    
                    # Align dates
                    common_dates = ticker_window.index.intersection(spy_window.index)
                    if len(common_dates) >= 10:  # Minimum observations
                        ticker_aligned = ticker_window.loc[common_dates]
                        spy_aligned = spy_window.loc[common_dates]
                        
                        correlation = ticker_aligned.corr(spy_aligned)
                        df.at[idx, f'spy_correlation_{window}d'] = correlation if pd.notna(correlation) else 0
                    else:
                        df.at[idx, f'spy_correlation_{window}d'] = 0
                else:
                    df.at[idx, f'spy_correlation_{window}d'] = 0
                
                # QQQ correlation
                if qqq_data is not None and idx in ticker_data.index and idx in qqq_data.index:
                    end_date_window = idx
                    start_date_window = idx - timedelta(days=window)
                    
                    ticker_window = ticker_data[
                        (ticker_data.index >= start_date_window) & 
                        (ticker_data.index <= end_date_window)
                    ]['returns'].dropna()
                    
                    qqq_window = qqq_data[
                        (qqq_data.index >= start_date_window) & 
                        (qqq_data.index <= end_date_window)
                    ]['returns'].dropna()
                    
                    # Align dates
                    common_dates = ticker_window.index.intersection(qqq_window.index)
                    if len(common_dates) >= 10:  # Minimum observations
                        ticker_aligned = ticker_window.loc[common_dates]
                        qqq_aligned = qqq_window.loc[common_dates]
                        
                        correlation = ticker_aligned.corr(qqq_aligned)
                        df.at[idx, f'qqq_correlation_{window}d'] = correlation if pd.notna(correlation) else 0
                    else:
                        df.at[idx, f'qqq_correlation_{window}d'] = 0
                else:
                    df.at[idx, f'qqq_correlation_{window}d'] = 0
        
        return df
    
    def _add_calendar_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add calendar-based features"""
        
        # Market calendar effects
        df['is_month_end'] = 0
        df['is_quarter_end'] = 0
        df['is_year_end'] = 0
        df['days_to_month_end'] = 0
        df['days_to_quarter_end'] = 0
        df['is_options_expiry_week'] = 0
        df['is_earnings_season'] = 0
        
        for idx in df.index:
            current_date = idx.date() if hasattr(idx, 'date') else idx
            
            # Month end effects (last 3 trading days)
            month_end = pd.Timestamp(current_date).to_period('M').end_time.date()
            days_to_month_end = (month_end - current_date).days
            df.at[idx, 'days_to_month_end'] = days_to_month_end
            df.at[idx, 'is_month_end'] = int(days_to_month_end <= 3)
            
            # Quarter end effects
            quarter_end = pd.Timestamp(current_date).to_period('Q').end_time.date()
            days_to_quarter_end = (quarter_end - current_date).days
            df.at[idx, 'days_to_quarter_end'] = days_to_quarter_end
            df.at[idx, 'is_quarter_end'] = int(days_to_quarter_end <= 5)
            
            # Year end effects
            year_end = pd.Timestamp(current_date).to_period('Y').end_time.date()
            df.at[idx, 'is_year_end'] = int((year_end - current_date).days <= 10)
            
            # Options expiry (3rd Friday of month - approximate)
            third_friday = pd.Timestamp(current_date.year, current_date.month, 15)
            while third_friday.weekday() != 4:  # Friday is 4
                third_friday += timedelta(days=1)
                if third_friday.day > 21:  # Safeguard
                    break
            
            days_to_expiry = abs((third_friday.date() - current_date).days)
            df.at[idx, 'is_options_expiry_week'] = int(days_to_expiry <= 3)
            
            # Earnings season (roughly Jan, Apr, Jul, Oct)
            earnings_months = [1, 4, 7, 10]
            is_earnings_month = current_date.month in earnings_months
            # Focus on mid-month when most earnings come out
            is_earnings_period = 10 <= current_date.day <= 25
            df.at[idx, 'is_earnings_season'] = int(is_earnings_month and is_earnings_period)
        
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
        """Generate summary statistics for market context features"""
        
        feature_cols = [col for col in df.columns if col in self.feature_names]
        
        if not feature_cols:
            return {}
        
        summary = {
            'total_features': len(feature_cols),
            'feature_types': {
                'market_regime': len([f for f in feature_cols if 'regime' in f or 'trend' in f]),
                'sector_features': len([f for f in feature_cols if 'sector' in f]),
                'macro_features': len([f for f in feature_cols if any(x in f for x in ['treasury', 'gold', 'dollar', 'volatility'])]),
                'crypto_features': len([f for f in feature_cols if 'crypto' in f]),
                'correlation_features': len([f for f in feature_cols if 'correlation' in f]),
                'calendar_features': len([f for f in feature_cols if any(x in f for x in ['month', 'quarter', 'year', 'expiry', 'earnings'])])
            },
            'value_ranges': {
                col: {
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None
                }
                for col in feature_cols[:10] if df[col].dtype in ['int64', 'float64']
            }
        }
        
        return summary 