# Module 02 – Event Detection & Tagging

## Purpose
Convert raw ingested news, social, and fundamental data into structured, actionable events.

---

## Responsibilities

- Detect event types (e.g. earnings beat, hack, delisting, whale transfer)
- Tag events with associated assets and metadata
- Score potential impact for prioritization

---

## Submodules

### 1. `EventClassifier` (Python)
**Inputs:** Preprocessed text/news/sentiment data  
**Output:** Structured event labels

- NLP classification (e.g. rule-based, zero-shot models, LLMs)
- Custom labels: “Hacked Exchange”, “Bullish Tweet”, “Earnings Miss”
- Can use spaCy, transformers, or keyword models initially

---

### 2. `ImpactScorer` (Python)
**Inputs:** Event + context (volume, volatility, sentiment)  
**Output:** Numerical or categorical impact score

- Scores based on historical movement
- Can evolve into ML classifier (target = price change)
- Optional: Confidence interval estimation

---

### 3. `EntityLinker` (Python)
**Inputs:** Raw news, tweet, or filing  
**Output:** Tagged tickers, symbols, coins

- Named Entity Recognition (NER)
- Maps entities to internal asset IDs
- Handles synonyms, tickers, aliases

---

### 4. `EventStore` (Python/Postgres)
**Purpose:** Central repository of detected events

- Fields: timestamp, ticker, type, score, raw source
- Used for ML training, backtests, and live trading
- Indexed for time and symbol search

---

## Output

Each event includes:
- `timestamp`
- `asset`
- `event_type`
- `confidence_score`
- `source` (news URL, tweet ID, etc.)

---

## Next Steps

- Define event taxonomies (types + subtypes)
- Create sample event templates
- Build basic rule-based or keyword classifier for initial version
