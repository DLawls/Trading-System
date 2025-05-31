"""
Sentiment Analysis Module
Processes text data to extract sentiment scores
"""

import logging
from typing import List, Dict, Union
import numpy as np
import pandas as pd
from transformers import pipeline
from loguru import logger

class SentimentAnalyzer:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        batch_size: int = 32
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the transformer model to use
            batch_size: Batch size for processing
        """
        self.analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1  # Use CPU. Set to 0 for GPU
        )
        self.batch_size = batch_size
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment label and score
        """
        try:
            result = self.analyzer(text)[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {'label': 'ERROR', 'score': 0.0}
            
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment labels and scores
        """
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                batch_results = self.analyzer(batch)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error analyzing batch: {str(e)}")
                # Add error results for failed batch
                results.extend([{'label': 'ERROR', 'score': 0.0}] * len(batch))
        return results
        
    def analyze_news_df(self, df: pd.DataFrame, text_column: str = 'title') -> pd.DataFrame:
        """
        Analyze sentiment of news articles in a DataFrame.
        
        Args:
            df: DataFrame containing news articles
            text_column: Column name containing the text to analyze
            
        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty:
            return df
            
        texts = df[text_column].tolist()
        results = self.analyze_batch(texts)
        
        # Add sentiment columns to DataFrame
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        
        # Convert sentiment to numeric score (-1 to 1)
        df['sentiment_numeric'] = df.apply(
            lambda x: x['sentiment_score'] if x['sentiment_label'] == 'POSITIVE' else -x['sentiment_score'],
            axis=1
        )
        
        return df
        
    def get_aggregate_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate aggregate sentiment metrics from a DataFrame.
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with aggregate sentiment metrics
        """
        if df.empty:
            return {
                'mean_sentiment': 0.0,
                'std_sentiment': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0
            }
            
        return {
            'mean_sentiment': df['sentiment_numeric'].mean(),
            'std_sentiment': df['sentiment_numeric'].std(),
            'positive_ratio': (df['sentiment_label'] == 'POSITIVE').mean(),
            'negative_ratio': (df['sentiment_label'] == 'NEGATIVE').mean()
        } 