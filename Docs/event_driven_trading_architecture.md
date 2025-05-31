# Event-Driven ML Trading System – High-Level Architecture (Alpaca Edition)

## Overview

This system is a high-performance, event-driven trading platform focused on integrating machine learning with fast execution. It is designed to ingest news, sentiment, and fundamental events, generate predictions, and act quickly through Alpaca's trading API.

---

## 🧱 System Modules

### 1. Data Ingestion Module (Python)
Ingests historical and live data.

- `MarketDataIngestor`: Alpaca OHLCV, trades, quotes
- `NewsIngestor`: NewsAPI, RSS feeds, Twitter
- `SentimentIngestor`: NLP scores from headlines/posts
- `EventScheduler`: Tracks economic calendars, earnings, token unlocks

---

### 2. Event Detection & Tagging Module (Python)
Extracts and classifies actionable events.

- `EventClassifier`: NLP-based classification (earnings, FUD, etc.)
- `ImpactScorer`: ML/rule-based impact estimation
- `EntityLinker`: Maps news/events to assets
- `EventStore`: Structured historical events database

---

### 3. Feature Engineering Module (Python)
Builds features for ML from events and prices.

- `TimeseriesFeatures`: Momentum, volatility, volume
- `EventFeatures`: Event frequency, decay functions
- `SentimentFeatures`: Rolling sentiment indicators
- `MarketContextFeatures`: VIX, macro, BTC dominance

---

### 4. ML Prediction Module (Python)
Applies ML models to predict price movement.

- `ModelTrainer`: Train models offline
- `ModelPredictor`: Real-time inference
- `ModelStore`: Versioned model saving/loading
- `EnsembleManager`: Combine different models' outputs

---

### 5. Signal Generation Module (Python)
Translates ML predictions into trade signals.

- `SignalEvaluator`: Confidence thresholds, filters
- `PositionSizer`: Volatility/risk-adjusted sizing
- `PortfolioAllocator`: Diversification, exposure management

---

### 6. Execution Engine (C++)
Handles fast order execution with Alpaca.

- `LatencySensitiveRouter` (C++): Fast routing logic
- `OrderManager` (C++): Tracks state of orders
- `BrokerAdapter` (C++): REST/WebSocket interface to Alpaca
- Python bindings for all C++ modules using `pybind11`

---

### 7. Backtesting & Simulation Module (Python)
Tests and validates strategies on historical data.

- `HistoricalEventSimulator`
- `PortfolioSimulator`
- `MetricsLogger`

---

### 8. Infrastructure & Monitoring (Python)
Scheduling, logging, dashboards, alerts.

- `Scheduler`
- `Logger`
- `MonitoringDashboard`
- `AlertManager`

---

## ⚙️ C++ Components Summary

| Component              | Purpose                   |
|------------------------|---------------------------|
| `LatencySensitiveRouter` | Low-latency order execution |
| `OrderManager`         | Fast state management     |
| `BrokerAdapter`        | Alpaca API interface      |

---

## Deployment Strategy

1. **Phase 1 (MVP)** – On-premise, Python-dominant
2. **Phase 2** – Add event-based prediction and paper trading
3. **Phase 3** – Cloud deployment, ML microservices, monitoring

