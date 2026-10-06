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

---

## Stage 2: Database Schema Architecture & Point-in-Time Temporal Joins

### What I Learned
1. **The Critical Difference Between Relational Joins and As-Of Joins**: In traditional databases, records match on exact keys (`id == id`). In quantitative financial pipelines, trade timestamps and event release times almost never coincide down to the second. Attempting standard equi-joins drops data or requires arbitrary time buckets.
2. **Point-in-Time Integrity via `merge_asof`**: Using pandas `merge_asof` with `direction='backward'` guarantees that every trade is joined strictly to the most recent historical event whose publication time $\le T_{\text{trade}}$. Conversely, an upcoming event uses `direction='forward'` where $T_{\text{trade}} < T_{\text{event}}$, allowing the model to know an announcement is approaching without seeing its surprise outcome ahead of time.
3. **Structured Indexing in SQLite**: Storing time series in SQLite requires indexing timestamp columns (`idx_trades_time`, `idx_events_time`) to prevent quadratic $O(N \times M)$ scan degradation as historical trade volume grows.

---

## Stage 3: Feature Engineering Without Lookahead Bias & Temporal Assertions

### What I Learned
1. **Expanding vs. Full-Sample Standardization**: Normalizing macroeconomic surprise values ($z = \frac{x - \mu}{\sigma}$) using the mean and standard deviation of the full historical sample introduces lookahead bias. The model learns parameters using information from future releases that had not yet occurred. Computing expanding-window statistics strictly up to $T$ guarantees that future volatility shocks do not leak into past representations.
2. **Period-Aligned Target Formulation**: In quantitative strategy optimization, predicting the tick-level return of a single millisecond trade is noisy. Structuring observations into $( \text{permutation\_id}, \text{period} )$ maps the macro environment (e.g. Fed week, CPI release, disinflationary sentiment) to the aggregate efficacy (expected P&L and win rate) of a trading strategy configuration over that period.
3. **Automated Causality Tests**: Writing automated tests that modify future data and verify that historical features remain 100% bitwise identical is the gold standard for verifying zero lookahead bias before training any predictive model.

---

## Stage 4: Quantitative Risk Metrics & Ranking Evaluation

### What I Learned
1. **Asymmetric Risk in Sharpe vs. Sortino**: The standard Sharpe ratio assumes returns are normally distributed and penalizes upside volatility identical to downside drawdowns. In quantitative trading, upside variance represents profitable right-tail alpha. The Sortino ratio isolates downside semi-deviation below the risk-free hurdle rate, ensuring strategies with positive skew are not unfairly penalized.
2. **Annualization Mechanics**: Annualizing Sharpe and Sortino ratios requires multiplying by the square root of the number of compounding periods per year ($\sqrt{N}$). For our 7-day weekly evaluation blocks, the factor is $\sqrt{52} \approx 7.211$. Stating this parameter explicitly avoids the common pitfall of mixing daily ($\sqrt{252}$) and monthly ($\sqrt{12}$) assumptions.
3. **Why Spearman Rank Correlation Trumps MSE in Meta-Modeling**: When predicting strategy parameters, our operational objective is selecting the Top-K permutations for the next period. A model could have high Mean Squared Error (e.g. predicting a return of +2% when actual was +6%), but if it correctly ranks the superior parameter permutation higher than mediocre ones (high Spearman $\rho$), the resulting portfolio allocation will be optimal.




