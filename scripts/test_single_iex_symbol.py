from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')

client = StockHistoricalDataClient(API_KEY, API_SECRET)

symbol = 'AAPL'  # Try 'MSFT', 'F', or another large-cap if this fails

request = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=10),
    end=datetime.now(),
    feed='iex'
)

try:
    bars = client.get_stock_bars(request)
    print(f"Bars for {symbol}:")
    print(bars)
except Exception as e:
    print(f"Error fetching bars for {symbol}: {e}") 