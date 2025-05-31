# Event-Driven ML Trading System

A high-performance, event-driven trading platform focused on integrating machine learning with fast execution through Alpaca's trading API.

## Features

- Real-time market data ingestion
- News and sentiment analysis
- ML-based prediction system
- Low-latency execution engine
- Comprehensive backtesting framework
- Robust monitoring and infrastructure

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd event-driven-trading
   ```

2. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Environment Setup**
   Create a `.env` file in the root directory with the following variables:
   ```
   # Alpaca API Credentials
   ALPACA_API_KEY=your_api_key_here
   ALPACA_API_SECRET=your_api_secret_here

   # News API Credentials
   NEWS_API_KEY=your_news_api_key_here

   # Twitter API Credentials (if using)
   TWITTER_API_KEY=your_twitter_api_key_here
   TWITTER_API_SECRET=your_twitter_api_secret_here
   TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
   TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret_here

   # Database Configuration (if using)
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=trading_db
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password

   # Logging Configuration
   LOG_LEVEL=INFO
   LOG_FILE=logs/trading_system.log
   ```

5. **Configuration**
   - Review and modify `config/config.yaml` for your specific needs
   - Update trading symbols, timeframes, and other parameters

## Project Structure

```
event-driven-trading/
├── config/
│   └── config.yaml
├── data/
│   ├── market_data/
│   ├── news/
│   └── models/
├── src/
│   ├── data_ingestion/
│   ├── event_detection/
│   ├── feature_engineering/
│   ├── ml_prediction/
│   ├── signal_generation/
│   ├── execution_engine/
│   ├── backtesting/
│   └── infrastructure/
├── tests/
├── logs/
├── pyproject.toml
└── README.md
```

## Development

1. **Create a virtual environment**
   ```bash
   poetry shell
   ```

2. **Run tests**
   ```bash
   poetry run pytest
   ```

3. **Format code**
   ```bash
   poetry run black .
   poetry run isort .
   ```

## Usage

1. **Start the data ingestion service**
   ```bash
   poetry run python -m src.data_ingestion.main
   ```

2. **Start the trading system**
   ```bash
   poetry run python -m src.main
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 