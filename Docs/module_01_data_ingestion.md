# Module 01 – Data Ingestion

## Purpose
Ingest and normalize data from various sources (market, news, sentiment, events) to feed downstream modules.

---

## Responsibilities

- Connect to Alpaca for real-time and historical price data
- Collect financial news and social media sentiment
- Schedule macroeconomic and event-based calendar updates

---

## Submodules

### 1. `MarketDataIngestor` (Python)
**Source:** Alpaca API (REST + WebSocket)  
**Data:** OHLCV, trades, quotes

- Daily/hourly ingestion
- WebSocket stream for live updates
- Saves to structured format (Parquet/Postgres)

---

### 2. `NewsIngestor` (Python)
**Source:** NewsAPI, RSS feeds, Twitter API (or scraper)

- Preprocesses and timestamps articles/posts
- Stores raw + parsed versions
- Optional: Keyword flagging (e.g. "hacked", "bankruptcy")

---

### 3. `SentimentIngestor` (Python)
**Source:** Social and news data

- Pretrained or fine-tuned sentiment model (HuggingFace, VADER)
- Entity detection and per-asset aggregation
- Stores rolling sentiment scores

---

### 4. `EventScheduler` (Python)
**Source:** Economic calendars, earnings APIs, token unlock schedules

- Stores forward-looking event data
- Flags time-to-event, market open/close context
- Example APIs: FRED, Yahoo Finance, Coingecko, CryptoRank

---

## Output

All ingested data is:
- Time-indexed
- Asset-mapped
- Stored in versioned parquet or SQL tables
- Ready for tagging, feature engineering, and modeling

---

## Next Steps

- Set up Alpaca key management
- Implement logging and retry wrappers
- Build dev/test dataset (e.g. BTC, ETH, TSLA)
