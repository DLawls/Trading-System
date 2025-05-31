import os
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import time

load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')

client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Load symbols from the docs file (skip comments and blank lines)
symbols = []
with open('docs/alpaca_tradable_symbols_and_timeframes.md', 'r') as f:
    for line in f:
        if line.startswith('- '):
            symbols.append(line.strip().replace('- ', ''))

BATCH_SIZE = 200

daily_supported = []
minute_supported = []

for i in range(0, len(symbols), BATCH_SIZE):
    batch = symbols[i:i+BATCH_SIZE]
    print(f"Checking batch {i//BATCH_SIZE+1}: {batch}")
    # Check daily bars
    try:
        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=10),
            end=datetime.now(),
            feed='iex'
        )
        bars = client.get_stock_bars(request)
        for symbol in batch:
            if symbol in bars and len(bars[symbol]) > 0:
                daily_supported.append(symbol)
    except Exception as e:
        print(f"Error checking daily bars for batch {batch}: {e}")
    time.sleep(0.1)  # faster, but still avoid hammering the API

# Now check minute bars for those with daily bars
for i in range(0, len(daily_supported), BATCH_SIZE):
    batch = daily_supported[i:i+BATCH_SIZE]
    print(f"Checking minute bars for batch {i//BATCH_SIZE+1}: {batch}")
    try:
        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(days=1),
            end=datetime.now(),
            feed='iex'
        )
        bars = client.get_stock_bars(request)
        for symbol in batch:
            if symbol in bars and len(bars[symbol]) > 0:
                minute_supported.append(symbol)
    except Exception as e:
        print(f"Error checking minute bars for batch {batch}: {e}")
    time.sleep(0.1)

with open('docs/iex_symbol_frequencies.txt', 'w') as f:
    f.write(f"Symbols with IEX daily bars: {len(daily_supported)}\n")
    for symbol in daily_supported:
        f.write(f"DAILY: {symbol}\n")
    f.write(f"\nSymbols with IEX minute bars: {len(minute_supported)}\n")
    for symbol in minute_supported:
        f.write(f"MINUTE: {symbol}\n")

print(f"Done. See docs/iex_symbol_frequencies.txt for results.") 