"""
Test script for the data ingestion system
"""

import asyncio
import os
from dotenv import load_dotenv
from loguru import logger

from .main import DataIngestionManager

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_api_secret = os.getenv('ALPACA_API_SECRET')
    alpaca_base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    news_api_key = os.getenv('NEWS_API_KEY')
    
    if not all([alpaca_api_key, alpaca_api_secret, news_api_key]):
        logger.error("Missing required API keys in .env file")
        return
        
    # Initialize data ingestion manager
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    manager = DataIngestionManager(
        alpaca_api_key=alpaca_api_key,
        alpaca_api_secret=alpaca_api_secret,
        alpaca_base_url=alpaca_base_url,
        news_api_key=news_api_key,
        symbols=symbols,
        update_interval=300  # 5 minutes
    )
    
    try:
        # Start the data ingestion
        manager.start()
        
        # Run for a few minutes
        logger.info("Running data ingestion for 5 minutes...")
        await asyncio.sleep(300)
        
        # Get the latest data
        data = manager.get_latest_data()
        
        # Print some statistics
        logger.info("\nData Statistics:")
        logger.info(f"Market Data Symbols: {list(data['market_data'].keys())}")
        logger.info(f"News Categories: {list(data['news'].keys())}")
        
        # Print some sample data
        for symbol in symbols:
            if symbol in data['market_data']:
                df = data['market_data'][symbol]
                logger.info(f"\nLatest market data for {symbol}:")
                logger.info(df.tail(1))
                
            if symbol in data['news']:
                df = data['news'][symbol]
                logger.info(f"\nLatest news sentiment for {symbol}:")
                logger.info(df[['title', 'sentiment_label', 'sentiment_score']].head(3))
                
    except KeyboardInterrupt:
        logger.info("Stopping data ingestion...")
    finally:
        manager.stop()
        
if __name__ == "__main__":
    asyncio.run(main()) 