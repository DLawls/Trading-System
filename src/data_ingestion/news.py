"""
News Ingestion Module
Handles fetching and processing news from various sources
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
from newsapi import NewsApiClient
from loguru import logger

class NewsIngestor:
    def __init__(self, api_key: str):
        """
        Initialize the news ingestor.
        
        Args:
            api_key: NewsAPI key
        """
        self.client = NewsApiClient(api_key=api_key)
        
    def get_news(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = 'en',
        sort_by: str = 'publishedAt',
        page_size: int = 100
    ) -> pd.DataFrame:
        """
        Fetch news articles from NewsAPI.
        
        Args:
            query: Search query
            from_date: Start date for news search
            to_date: End date for news search
            language: Language of articles
            sort_by: Sort order ('relevancy', 'popularity', 'publishedAt')
            page_size: Number of articles per page
            
        Returns:
            DataFrame containing news articles
        """
        if from_date is None:
            from_date = datetime.now() - timedelta(days=1)
        if to_date is None:
            to_date = datetime.now()
            
        try:
            response = self.client.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language=language,
                sort_by=sort_by,
                page_size=page_size
            )
            
            if response['status'] != 'ok':
                logger.error(f"NewsAPI error: {response.get('message', 'Unknown error')}")
                return pd.DataFrame()
                
            articles = response['articles']
            
            # Convert to DataFrame
            df = pd.DataFrame(articles)
            if not df.empty:
                df['publishedAt'] = pd.to_datetime(df['publishedAt'])
                df.set_index('publishedAt', inplace=True)
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return pd.DataFrame()
            
    def get_company_news(
        self,
        company_name: str,
        days_back: int = 1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch news articles for a specific company.
        
        Args:
            company_name: Name of the company
            days_back: Number of days to look back
            **kwargs: Additional arguments for get_news()
            
        Returns:
            DataFrame containing company news articles
        """
        from_date = datetime.now() - timedelta(days=days_back)
        return self.get_news(
            query=company_name,
            from_date=from_date,
            **kwargs
        )
        
    def get_market_news(
        self,
        days_back: int = 1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch general market news.
        
        Args:
            days_back: Number of days to look back
            **kwargs: Additional arguments for get_news()
            
        Returns:
            DataFrame containing market news articles
        """
        from_date = datetime.now() - timedelta(days=days_back)
        return self.get_news(
            query="stock market OR trading OR finance",
            from_date=from_date,
            **kwargs
        ) 