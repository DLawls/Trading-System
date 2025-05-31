"""
Market Data Ingestion Module
Handles real-time and historical market data from Alpaca
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)

class MarketDataIngestor:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        """
        Initialize the market data ingestor.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: Alpaca API base URL
        """
        self.client = StockHistoricalDataClient(api_key, api_secret)
        self.base_url = base_url
        
    async def get_historical_bars(
        self,
        symbols: List[str],
        timeframe: TimeFrame,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical bar data for given symbols.
        
        Args:
            symbols: List of stock symbols
            timeframe: Bar timeframe (e.g., 1m, 5m, 1h)
            start: Start time
            end: End time (defaults to now)
            
        Returns:
            Dictionary mapping symbols to DataFrames of OHLCV data
        """
        if end is None:
            end = datetime.now()
            
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start,
                end=end
            )
            
            bars = self.client.get_stock_bars(request)
            
            # Convert to DataFrames
            result = {}
            for symbol in symbols:
                symbol_bars = bars[symbol]
                df = pd.DataFrame({
                    'timestamp': [bar.timestamp for bar in symbol_bars],
                    'open': [bar.open for bar in symbol_bars],
                    'high': [bar.high for bar in symbol_bars],
                    'low': [bar.low for bar in symbol_bars],
                    'close': [bar.close for bar in symbol_bars],
                    'volume': [bar.volume for bar in symbol_bars]
                })
                df.set_index('timestamp', inplace=True)
                result[symbol] = df
                
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical bars: {str(e)}")
            raise
            
    async def get_latest_bars(
        self,
        symbols: List[str],
        timeframe: TimeFrame,
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch the latest bar data for given symbols.
        
        Args:
            symbols: List of stock symbols
            timeframe: Bar timeframe
            limit: Number of bars to fetch
            
        Returns:
            Dictionary mapping symbols to DataFrames of OHLCV data
        """
        end = datetime.now()
        start = end - timedelta(days=1)  # Fetch last day of data
        
        return await self.get_historical_bars(symbols, timeframe, start, end) 