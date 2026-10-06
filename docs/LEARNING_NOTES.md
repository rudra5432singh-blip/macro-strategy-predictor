# Quantitative Learning Notes

Documenting key quantitative finance and machine learning concepts in plain language as the pipeline is developed stage by stage.

---

## Stage 0: Environment Setup, Secrets Hygiene & Pipeline Architecture

### What I Learned
In quantitative finance and machine learning workflows, reproducibility and strict security hygiene are foundational. Financial data systems must never expose API keys (e.g., Apify or Tavily credentials) in code or git commits, because compromised keys can lead to service exhaustion or data tampering. Structuring the repository with explicit separation between raw immutable data (`data/raw/`), intermediate storage (`data/macro.db`), source logic (`src/`), verification tests (`tests/`), and visual/reporting outputs (`outputs/`, `reports/`) ensures that every stage of data transformation can be audited and reproduced deterministically without leakage.

---

## Stage 1: Macro Data Ingestion, Timezone Mechanics & Synthetic Regime Generation

### What I Learned
1. **Timezone Pitfalls in Macro Trading**: High-impact US macroeconomic figures (CPI, Non-Farm Payrolls) are released at 08:30 AM Eastern Time, and FOMC rate decisions occur at 02:00 PM Eastern Time. Due to Daylight Saving Time (EDT = UTC-4 in summer, EST = UTC-5 in winter), the exact UTC release time shifts by 60 minutes twice a year. Without dynamic IANA timezone resolution (`America/New_York`), algorithms suffer either a 60-minute lookahead leak (trading before numbers are published) or an execution lag error.
2. **Economic Surprise vs. Raw Levels**: Efficient market hypothesis posits that anticipated economic levels are already priced into assets. Market returns correlate with the surprise element: $\text{Surprise} = \text{Actual} - \text{Forecast}$.
3. **Lexicon Polarity in Quantitative NLP**: Generic language models or sentiment analyzers often stumble on financial jargon. Words like "cut" or "cooling" can carry negative linguistic sentiment, but in a macro cycle, a "rate cut" or "cooling inflation" is bullish for asset prices and duration assets. Combining financial lexicons with VADER compound scores ensures proper economic context.
4. **Regime-Conditional Strategy Permutations**: Trading strategies do not exist in a vacuum. A tight stop-loss (1%) offers tight risk containment in calm markets, but gets whipsawed repeatedly during high-volatility macro event windows. A wider stop-loss (3%) survives volatility shocks and allows the strategy to capture larger trend extensions.

