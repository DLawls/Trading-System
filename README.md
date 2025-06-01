# 🚀 Event-Driven ML Trading System

A comprehensive event-driven machine learning trading system that ingests market data, analyzes news sentiment, detects trading events, and provides the foundation for automated trading strategies.

## 📊 **Current Implementation Status**

### ✅ **Phase 0: Environment & Tooling Setup** (100% Complete)
- Project structure with organized modules
- Poetry dependency management
- Docker containerization
- Git version control with GitHub integration
- Comprehensive logging with Loguru
- Environment configuration

### ✅ **Phase 1: Data Ingestion** (100% Complete)
- **MarketDataIngestor**: Real-time Alpaca market data
- **NewsIngestor**: News articles from NewsAPI
- **SentimentAnalyzer**: NLP sentiment analysis using Transformers
- **EventScheduler**: Earnings, macro events, and token unlocks
- **DataIngestionManager**: Automated orchestration and scheduling

### 🚧 **Phase 2: Event Detection** (Partially Complete)
- **EventClassifier**: Rule-based event detection with 9 event types
- **ImpactScorer**: Coming next
- **EntityLinker**: Coming next
- **EventStore**: Coming next

---

## 🛠️ **Setup Instructions**

### **Prerequisites**
- Python 3.10+
- Docker (optional but recommended)
- API Keys (see Environment Setup)

### **1. Clone Repository**
```bash
git clone https://github.com/DLawls/Trading-System.git
cd "Trading System"
```

### **2. Install Dependencies**
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### **3. Environment Setup**
Create a `.env` file in the root directory:
```env
# Alpaca API (Required for market data)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
# ALPACA_BASE_URL=https://api.alpaca.markets      # Live trading

# NewsAPI (Required for news data)
NEWS_API_KEY=your_newsapi_key

# Optional: Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_system
DB_USER=trader
DB_PASSWORD=secure_password_123
```

### **4. Docker Setup (Recommended)**
```bash
# Build development image
docker build -f Dockerfile.dev -t trading-system:dev .

# Or use our helper script (Windows)
scripts/docker-run.bat

# Run with docker-compose (includes Redis + PostgreSQL)
docker-compose up -d
```

---

## 🔧 **How to Use Each Component**

### **1. Market Data Ingestion**

```python
from src.data_ingestion.market_data import MarketDataIngestor
from alpaca.data.timeframe import TimeFrame

# Initialize
market_data = MarketDataIngestor(
    api_key="your_key",
    api_secret="your_secret",
    base_url="https://paper-api.alpaca.markets"
)

# Get latest market data
data = await market_data.get_latest_bars(
    symbols=['TSLA', 'AAPL'], 
    timeframe=TimeFrame.Day, 
    limit=100
)
# Returns: Dict of DataFrames with OHLCV data
```

### **2. News Ingestion & Sentiment Analysis**

```python
from src.data_ingestion.news import NewsIngestor
from src.data_ingestion.sentiment import SentimentAnalyzer

# Get news
news = NewsIngestor(api_key="your_newsapi_key")
tesla_news = news.get_company_news('TSLA', days_back=1)
market_news = news.get_market_news(days_back=1)

# Analyze sentiment
sentiment = SentimentAnalyzer()
analyzed_news = sentiment.analyze_news_df(tesla_news)
# Adds: sentiment_label, sentiment_score columns
```

### **3. Event Scheduling**

```python
from src.data_ingestion.events import EventScheduler

# Track upcoming events
scheduler = EventScheduler()
await scheduler.update_events(['TSLA', 'AAPL', 'ETHUSD'])

# Get events
upcoming = scheduler.get_upcoming_events(hours_ahead=24)
high_impact = scheduler.get_high_impact_events()
tesla_events = scheduler.get_events_by_symbol('TSLA')

# Convert to DataFrame for analysis
events_df = scheduler.to_dataframe()
```

### **4. Event Classification**

```python
from src.event_detection.event_classifier import EventClassifier

classifier = EventClassifier()

# Classify single news article
events = classifier.classify_text(
    "Tesla reports record Q4 earnings, beats revenue estimates",
    "Tesla Inc announced quarterly results showing..."
)
# Returns: List of DetectedEvent objects with confidence scores

# Process entire news DataFrame
classified_news = classifier.classify_news_df(news_df)
# Adds: event_type, event_confidence, event_entity, keywords_matched
```

### **5. Complete Data Pipeline (Automated)**

```python
from src.data_ingestion.main import DataIngestionManager

# Initialize with your symbols and API keys
manager = DataIngestionManager(
    alpaca_api_key="your_key",
    alpaca_api_secret="your_secret", 
    alpaca_base_url="https://paper-api.alpaca.markets",
    news_api_key="your_newsapi_key",
    symbols=['TSLA', 'AAPL', 'MSFT'],
    update_interval=300  # 5 minutes
)

# Start automated data collection
manager.start()

# Access unified data
data = manager.get_latest_data()
print(data.keys())  # ['market_data', 'news', 'events']

# Stop when done
manager.stop()
```

---

## 🎯 **What You Can Build With This**

### **1. Real-time Trading Dashboard**
```python
# Get live data every 5 minutes
data = manager.get_latest_data()

# Display:
# - Current prices: data['market_data']['TSLA']
# - Latest news: data['news']['TSLA'] 
# - Sentiment scores: data['news']['market']['sentiment_score']
# - Upcoming events: data['events']['upcoming']
```

### **2. Event-driven Alerts**
```python
# Monitor for high-impact events
events = scheduler.get_high_impact_events()
for event in events:
    if event.impact_level == 'high':
        send_alert(f"High impact event: {event.description}")
```

### **3. News Sentiment Analysis**
```python
# Track sentiment changes
news_df = news.get_company_news('TSLA', days_back=7)
sentiment_df = sentiment.analyze_news_df(news_df)

# Calculate daily sentiment averages
daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean()
```

### **4. Event Classification Pipeline**
```python
# Automatically classify all incoming news
classified = classifier.classify_news_df(news_df)

# Filter for specific event types
earnings_news = classified[classified['event_type'] == 'earnings']
merger_news = classified[classified['event_type'] == 'merger_acquisition']
```

---

## 🧪 **Testing the System**

### **Test Individual Components**
```bash
# Test event scheduler
python src/test_events.py

# Test data ingestion
python src/test_data_ingestion.py

# Test event classification
python -c "
from src.event_detection.event_classifier import EventClassifier
classifier = EventClassifier()
events = classifier.classify_text('Apple reports strong quarterly earnings')
print([e.event_type.value for e in events])
"
```

### **Test Docker Setup**
```bash
# Test Docker image
docker run --rm trading-system:dev python -c "print('System ready!')"

# Test full stack
docker-compose up -d
docker-compose logs -f trading-system
```

---

## 📁 **Project Structure**

```
Trading System/
├── 🐳 Docker files
│   ├── Dockerfile              # Production container
│   ├── Dockerfile.dev          # Development container  
│   ├── docker-compose.yml      # Full stack (app + Redis + PostgreSQL)
│   └── docker-compose.dev.yml  # Development stack
│
├── 📋 Configuration
│   ├── .env                    # Environment variables
│   ├── pyproject.toml          # Dependencies & project config
│   └── config/                 # Configuration files
│
├── 📊 Source Code
│   ├── src/
│   │   ├── data_ingestion/     # ✅ Phase 1 (Complete)
│   │   │   ├── main.py         # DataIngestionManager
│   │   │   ├── market_data.py  # MarketDataIngestor
│   │   │   ├── news.py         # NewsIngestor
│   │   │   ├── sentiment.py    # SentimentAnalyzer
│   │   │   └── events.py       # EventScheduler
│   │   │
│   │   └── event_detection/    # 🚧 Phase 2 (Partial)
│   │       ├── event_classifier.py  # EventClassifier
│   │       ├── impact_scorer.py     # Coming next
│   │       └── entity_linker.py     # Coming next
│   │
├── 📚 Documentation
│   ├── docs/
│   │   └── project_development_checklist.md
│   └── README.md               # This file
│
├── 🔧 Scripts
│   ├── scripts/
│   │   ├── docker-build.sh     # Docker build helper
│   │   └── docker-run.bat      # Windows Docker helper
│
└── 📝 Logs & Data
    ├── logs/                   # Application logs
    └── data/                   # Data storage (gitignored)
```

---

## 🔍 **Event Detection Capabilities**

The **EventClassifier** can detect and classify 9 types of trading events:

| Event Type | Examples | Impact Level |
|------------|----------|--------------|
| **Earnings** | "Q4 earnings beat estimates" | High |
| **M&A** | "Company X acquires Company Y" | High |
| **Regulatory** | "FDA approval granted" | High |
| **Leadership** | "New CEO appointed" | Medium |
| **Legal Issues** | "Lawsuit filed against company" | Medium |
| **Product Launch** | "New iPhone announced" | Medium |
| **Partnership** | "Strategic alliance formed" | Low |
| **Market Movement** | "Stock price target raised" | Low |
| **Economic Data** | "Fed raises interest rates" | High |

---

## 🐳 **Docker Usage**

### **Development Workflow**
```bash
# Build and run development environment
docker build -f Dockerfile.dev -t trading-system:dev .
docker run --rm -it trading-system:dev bash

# Or use docker-compose for full stack
docker-compose -f docker-compose.dev.yml up -d
```

### **Production Deployment**
```bash
# Build production image
docker build -t trading-system:prod .

# Run full stack with Redis + PostgreSQL
docker-compose up -d

# Check logs
docker-compose logs -f
```

---

## 🛣️ **Development Roadmap**

### **Immediate Next Steps (Phase 2 completion):**
- [ ] **ImpactScorer**: Predict market impact of detected events
- [ ] **EntityLinker**: Advanced NER using spaCy/Transformers
- [ ] **EventStore**: Database storage for historical events
- [ ] **Historical Analysis**: Backtest event detection on past news

### **Phase 3: Feature Engineering**
- [ ] Technical indicators (SMA, RSI, MACD)
- [ ] Event-based features
- [ ] Sentiment rolling averages
- [ ] Market context features

### **Phase 4: ML Modeling**
- [ ] Target label definition
- [ ] Model training pipeline
- [ ] Model evaluation and validation
- [ ] Ensemble methods

---

## 🚀 **Getting Started Quickly**

1. **Set up API keys** in `.env` file
2. **Install dependencies**: `poetry install`
3. **Test event scheduling**: `python src/test_events.py`
4. **Start data ingestion**: `python src/test_data_ingestion.py`
5. **Explore event classification** with your own news headlines

## 📞 **Support**

- **Issues**: Open GitHub issues for bugs or feature requests
- **Documentation**: Check `docs/` folder for detailed guides
- **Progress**: See `docs/project_development_checklist.md` for development status

---

**🎯 Current Status: 1.5/10 phases complete | Ready for event-driven trading research and development!** 