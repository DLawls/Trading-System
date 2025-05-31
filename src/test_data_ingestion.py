"""
Test script to verify data ingestion functionality
"""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
from alpaca.data.timeframe import TimeFrame

from src.data_ingestion.main import DataIngestionManager

async def test_market_data():
    """Test market data ingestion."""
    load_dotenv()
    
    print("ALPACA_API_KEY:", os.getenv('ALPACA_API_KEY'))
    print("ALPACA_API_SECRET:", os.getenv('ALPACA_API_SECRET'))
    print("ALPACA_BASE_URL:", os.getenv('ALPACA_BASE_URL'))

    manager = DataIngestionManager(
        alpaca_api_key=os.getenv('ALPACA_API_KEY'),
        alpaca_api_secret=os.getenv('ALPACA_API_SECRET'),
        alpaca_base_url=os.getenv('ALPACA_BASE_URL'),
        news_api_key=os.getenv('NEWS_API_KEY'),
        symbols=['TSLA', 'IBM', 'GE'],
        update_interval=60  # 1 minute for testing
    )
    
    try:
        # Start the data ingestion
        manager.start()
        logger.info("Started data ingestion...")
        
        # Run for 2 minutes
        for _ in range(2):
            await asyncio.sleep(60)
            data = manager.get_latest_data()
            
            # Print market data
            logger.info("\nMarket Data:")
            for symbol, df in data['market_data'].items():
                logger.info(f"\n{symbol} latest data:")
                logger.info(df.tail(1))
            
            # Print news data
            logger.info("\nNews Data:")
            for category, df in data['news'].items():
                logger.info(f"\n{category} latest news:")
                if not df.empty:
                    logger.info(df[['title', 'sentiment_label', 'sentiment_score']].head(2))
            
            # Print market data sample
            logger.info("\nSample Market Data (Daily Bars):")
            for symbol, df in data['market_data'].items():
                logger.info(f"\n{symbol} daily data sample:")
                print(f"\n{symbol} daily data sample:")
                print(df.head(3))
                logger.info(df.head(3))
            if not data['market_data']:
                print("No market data was returned for the selected symbols.")
                logger.info("No market data was returned for the selected symbols.")
                    
    except KeyboardInterrupt:
        logger.info("Stopping data ingestion...")
    finally:
        manager.stop()
        logger.info("Data ingestion stopped")

if __name__ == "__main__":
    # Configure logging
    logger.add(
        "logs/data_ingestion_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    
    # Run the test
    asyncio.run(test_market_data()) 