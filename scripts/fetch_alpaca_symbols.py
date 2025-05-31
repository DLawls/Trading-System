import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
search_params = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
assets = trading_client.get_all_assets(search_params)
tradable_assets = [asset for asset in assets if asset.tradable]

symbols = sorted([asset.symbol for asset in tradable_assets])

output_file = 'docs/alpaca_tradable_symbols_and_timeframes.md'

with open(output_file, 'a') as f:
    f.write('\n---\n')
    f.write(f"\n### Tradable Symbols (Total: {len(symbols)})\n\n")
    for symbol in symbols:
        f.write(f"- {symbol}\n") 