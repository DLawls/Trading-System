"""
Main Data Ingestion Module
Coordinates data ingestion from multiple sources
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from alpaca.data.timeframe import TimeFrame

from .market_data import MarketDataIngestor
from .news import NewsIngestor
from .sentiment import SentimentAnalyzer
from .events import EventScheduler

class DataIngestionManager:
    def __init__(
        self,
        alpaca_api_key: str,
        alpaca_api_secret: str,
        alpaca_base_url: str,
        news_api_key: str,
        symbols: List[str],
        update_interval: int = 300  # 5 minutes
    ):
        """
        Initialize the data ingestion manager.
        
        Args:
            alpaca_api_key: Alpaca API key
            alpaca_api_secret: Alpaca API secret
            alpaca_base_url: Alpaca API base URL
            news_api_key: NewsAPI key
            symbols: List of symbols to track
            update_interval: Update interval in seconds
        """
        self.symbols = symbols
        self.update_interval = update_interval
        
        # Initialize components
        self.market_data = MarketDataIngestor(
            api_key=alpaca_api_key,
            api_secret=alpaca_api_secret,
            base_url=alpaca_base_url
        )
        self.news = NewsIngestor(api_key=news_api_key)
        self.sentiment = SentimentAnalyzer()
        self.events = EventScheduler()
        
        # Initialize scheduler
        self.scheduler = AsyncIOScheduler()
        self.data_store = {
            'market_data': {},
            'news': {},
            'sentiment': {},
            'events': {}
        }
        
    async def update_market_data(self):
        """Update market data for all symbols."""
        try:
            data = await self.market_data.get_latest_bars(
                symbols=self.symbols,
                timeframe=TimeFrame.Day,
                limit=100
            )
            self.data_store['market_data'] = data
            logger.info(f"Updated market data for {len(data)} symbols")
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            
    async def update_news(self):
        """Update news and sentiment for all symbols."""
        try:
            # Get market news
            market_news = self.news.get_market_news(days_back=1)
            if not market_news.empty:
                market_news = self.sentiment.analyze_news_df(market_news)
                self.data_store['news']['market'] = market_news
                
            # Get company-specific news
            for symbol in self.symbols:
                company_news = self.news.get_company_news(symbol, days_back=1)
                if not company_news.empty:
                    company_news = self.sentiment.analyze_news_df(company_news)
                    self.data_store['news'][symbol] = company_news
                    
            logger.info("Updated news and sentiment data")
        except Exception as e:
            logger.error(f"Error updating news: {str(e)}")
            
    async def update_events(self):
        """Update event calendar."""
        try:
            await self.events.update_events(self.symbols)
            self.data_store['events'] = {
                'upcoming': self.events.get_upcoming_events(hours_ahead=48),
                'high_impact': self.events.get_high_impact_events(),
                'all_events': self.events.events
            }
            logger.info(f"Updated events calendar: {len(self.events.events)} events")
        except Exception as e:
            logger.error(f"Error updating events: {str(e)}")
            
    async def update_all(self):
        """Update all data sources."""
        await asyncio.gather(
            self.update_market_data(),
            self.update_news(),
            self.update_events()
        )
        
    def start(self):
        """Start the data ingestion scheduler."""
        self.scheduler.add_job(
            self.update_all,
            trigger=IntervalTrigger(seconds=self.update_interval),
            id='update_all',
            replace_existing=True
        )
        self.scheduler.start()
        logger.info(f"Started data ingestion scheduler (interval: {self.update_interval}s)")
        
    def stop(self):
        """Stop the data ingestion scheduler."""
        self.scheduler.shutdown()
        logger.info("Stopped data ingestion scheduler")
        
    def get_latest_data(self) -> Dict:
        """Get the latest data from all sources."""
        return self.data_store 