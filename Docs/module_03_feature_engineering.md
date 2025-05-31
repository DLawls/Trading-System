# Module 03 – Feature Engineering

## Purpose
Transform raw market, event, and sentiment data into model-ready features.

---

## Responsibilities

- Generate rolling statistics and technical indicators
- Encode event timing and frequency
- Aggregate sentiment scores
- Contextualize features with macro and market regime data

---

## Submodules

### 1. `TimeseriesFeatures` (Python)
**Input:** OHLCV data from Alpaca  
**Output:** Rolling indicators

- Technical indicators (SMA, EMA, RSI, MACD)
- Volatility metrics (ATR, realized volatility)
- Volume trends (VWAP, volume spikes)
- Lag features and change rates

---

### 2. `EventFeatures` (Python)
**Input:** Structured event data  
**Output:** Event-related numerical features

- Time-since-event
- Frequency of past events per type
- Cumulative event scores
- Event window dummy variables

---

### 3. `SentimentFeatures` (Python)
**Input:** NLP scores from news and social  
**Output:** Aggregated, asset-level sentiment scores

- Rolling mean/variance of sentiment
- Entity sentiment over past `N` periods
- Momentum or decay-weighted sentiment
- Separate by source type (news vs social)

---

### 4. `MarketContextFeatures` (Python)
**Input:** Global market/macroeconomic data  
**Output:** Contextual features

- Market regime indicators (bull/bear/neutral)
- Macro (CPI, Fed rate, unemployment)
- Crypto-specific (BTC dominance, stablecoin flows)

---

## Output

Each feature set is:
- Aligned by timestamp and asset
- Versioned and cached (Parquet/Feather/SQL)
- Used for model training and live inference

---

## Next Steps

- Define target prediction window (e.g. 1h forward return)
- Implement feature generation pipeline
- Store intermediate features in a separate layer
