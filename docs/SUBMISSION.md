# Project Submission: Macro Strategy Predictor

## Repository Link
- **GitHub Repository**: `[https://github.com/placeholder-user/macro-strategy-predictor](file:///c:/Users/Ankit%20Singh/Documents/macro-strategy-predictor)` *(Update with remote repository URL upon creation)*

---

## APIs & Actors Used

### 1. Apify Economic Calendar Actor
- **Actor ID**: `curious_coder/economic-calendar` (configurable via `APIFY_ACTOR_ID` in `.env`)
- **Rationale**:
  - Provides structured, machine-readable economic releases for US events: Consumer Price Index (CPI), Federal Open Market Committee (FOMC) interest rate decisions, and Non-Farm Payrolls (NFP) labor statistics.
  - Returns `actual`, `forecast`, and `previous` values necessary to compute economic surprise metrics ($\text{actual} - \text{forecast}$).
- **Resilience & Fallback**:
  - Built-in exponential backoff and network retry logic.
  - In offline mode or if API quotas/tokens are missing, an authentic historical macro calendar covering the exact 180-day window is used to guarantee 100% deterministic reproducibility.
  - Timestamps are rigorously converted to Eastern Time (`America/New_York`) with dynamic Daylight Saving Time handling (EDT vs EST transitions).

### 2. Tavily Search API
- **Endpoint**: Tavily Search Python SDK (`tavily-python`)
- **Rationale**:
  - Purpose-built search engine for AI agents that extracts clean, authoritative macroeconomic news articles and snippets without noisy web formatting.
  - Queries targeted: US CPI inflation dynamics, FOMC interest rate announcements and forward guidance, Non-Farm Payrolls employment data, and broad US macroeconomic sentiment.
- **Sentiment Engine**:
  - Scores each article using a transparent, lexicon-based polarity score normalized between -1.0 (strongly bearish/recessionary) and +1.0 (strongly bullish/expansionary).
- **Resilience & Fallback**:
  - Caches raw responses to `data/raw/news_raw.json`.
  - Offline fallback with genuine historical headlines guarantees zero pipeline disruption if no API key is present.

---

## Exact Steps to Reproduce

1. **Clone the repository**:
   ```bash
   git clone <REPO_URL>
   cd macro-strategy-predictor
   ```

2. **Set up the Python environment (Python 3.11+)**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Configure Environment Variables (Optional)**:
   ```bash
   cp .env.example .env
   # Edit .env and supply APIFY_TOKEN and TAVILY_API_KEY if running live scrapes
   ```
   *Note: If no API keys are provided, the system automatically runs with offline verified macro datasets, ensuring complete end-to-end execution without errors.*

4. **Run the entire pipeline end-to-end**:
   ```bash
   python run_all.py
   ```

5. **Run the test suite**:
   ```bash
   pytest tests/ -v
   ```
