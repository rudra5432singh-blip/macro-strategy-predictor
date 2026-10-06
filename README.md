# Macro-Aware Strategy Parameter Predictor

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean & Reproducible](https://img.shields.io/badge/Code%20Style-Strict%20Anti--Leakage-green.svg)]()

A quantitative research pipeline that predicts optimal algorithmic trading strategy parameters (permutations of trend lookback, stop-loss, and take-profit) for upcoming forward periods using historical trade logs, macroeconomic releases (CPI, FOMC, Non-Farm Payrolls), and financial news sentiment.

---

## 1. Quantitative Performance Highlights (Out-of-Sample)

Across **15 expanding-window walk-forward test folds** (pure out-of-sample rebalancing checkpoints), the **Macro LightGBM Meta-Model** strongly outperforms both naive rolling historical averages and random parameter selection:

| Metric | Macro LightGBM Model | Historical Mean Baseline | Random Choice Baseline |
| :--- | :---: | :---: | :---: |
| **Annualized Sharpe Ratio** ($\sqrt{52}$) | **+1.0324** | -1.8773 | -1.9030 |
| **Annualized Sortino Ratio** ($\sqrt{52}$) | **+1.6195** | -2.2672 | -2.1305 |
| **Maximum Drawdown (MDD)** | **17.03%** | 35.39% | 33.34% |
| **Cumulative Return** | **+11.64%** | -29.06% | -23.94% |
| **Out-of-Sample Hit Rate** | **60.00%** | 40.00% | 46.67% |
| **Spearman Rank Correlation** | **+0.0978** | +0.0383 | -0.0040 |

> **Key Takeaway**: Naive historical averages experience severe regime lag during macroeconomic pivots. By conditioning on inflation disinflation, Fed easing signals, and event surprise z-scores, the meta-model reduces drawdown by more than half (17.0% vs. 35.4%) while delivering an annualized Sharpe above 1.0.

---

## 2. Project Architecture & Directory Layout

```
macro-strategy-predictor/
├── .env.example               # Template for API keys (APIFY_TOKEN, TAVILY_API_KEY, etc.)
├── .gitignore                  # Strict exclusion of .env, DB, caches
├── requirements.txt           # Pinned production dependencies
├── README.md                  # Comprehensive project documentation
├── run_all.py                 # Single-command end-to-end pipeline runner
├── data/
│   ├── raw/                   # Raw API dumps and synthetic_trades.csv (untouched)
│   ├── processed/             # Cleaned intermediate tables and modelling CSVs
│   └── macro.db               # SQLite database with trades, events, news, and features
├── src/
│   ├── __init__.py
│   ├── synthetic_trades.py    # Seeded generator for ~180d trade logs across permutations
│   ├── ingest_apify.py        # Economic calendar scraper + timezone normalization + fallback
│   ├── ingest_tavily.py       # Macro news ingestion + VADER/lexicon sentiment scoring
│   ├── build_db.py            # SQLite schema, indexing, and temporal merge_asof joins
│   ├── features.py            # Point-in-time feature engineering + period aggregation
│   ├── metrics.py             # Annualized Sharpe, Sortino, Max Drawdown, Spearman rank corr
│   ├── model.py               # LightGBM regressor & baseline benchmarks
│   ├── backtest.py            # Expanding-window walk-forward validation & Top-K simulation
│   ├── charts.py              # Visualizations (sentiment, heatmap, equity curves, fold metrics)
│   └── export_reports.py      # Automated PDF report compiler (ReportLab)
├── tests/
│   ├── __init__.py
│   ├── test_leakage.py        # Strict assertions against future data leakage
│   └── test_metrics.py        # Numerical unit tests for Sharpe, Sortino, Max Drawdown
├── docs/
│   ├── SUBMISSION.md          # Submission instructions, actor IDs, API details, reproduction steps
│   └── LEARNING_NOTES.md      # Plain-language concept explanations recorded after each stage
├── reports/
│   ├── evaluation_report.md   # Comprehensive quantitative report
│   ├── evaluation_report.pdf  # PDF compilation of evaluation report
│   ├── macro_outlook.md       # 1-page institutional macro outlook report
│   └── macro_outlook.pdf      # PDF compilation of macro outlook report
└── outputs/
    └── charts/                # Publication-ready PNG outputs
        ├── 1_macro_sentiment_timeline.png
        ├── 2_pnl_heatmap_matrix.png
        ├── 3_equity_curves_comparison.png
        └── 4_walk_forward_fold_metrics.png
```

---

## 3. Strict Anti-Leakage & Reproducibility Controls

1. **Temporal Precedence**: Every feature generated for evaluation period $[T_{\text{start}}, T_{\text{end}})$ strictly uses information timestamped $\le T_{\text{start}}$.
2. **Expanding-Window Standardization**: Macro surprise z-scores ($\text{surprise} = \text{actual} - \text{forecast}$) are computed strictly using past events up to release time $T$. Future events are never used for normalization.
3. **Point-in-Time News Sentiment**: Only news articles published prior to the period start boundary are incorporated into the pre-period sentiment feature.
4. **Expanding Walk-Forward Validation**: Model parameters are estimated on periods $1 \dots T$ and tested blindly on $T+1$. Time is never shuffled.
5. **Deterministic Seeds**: All stochastic processes are seeded (`seed=42`) for 100% bitwise reproducibility.

---

## 4. Quickstart & How to Run

### Step 1: Environment Setup
```bash
git clone <REPO_URL>
cd macro-strategy-predictor

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables (Optional)
```bash
cp .env.example .env
```
Edit `.env` to include your `APIFY_TOKEN` and `TAVILY_API_KEY`.  
*Note: If no API tokens are supplied, the pipeline automatically runs with verified offline historical data, ensuring complete end-to-end execution without errors.*

### Step 3: Run the Full End-to-End Pipeline
```bash
python run_all.py
```
This single command executes:
- Trade data validation $\to$ Macro calendar ingestion $\to$ News sentiment scoring
- SQLite database creation $\to$ Feature engineering
- Anti-leakage and metrics test suite execution
- Expanding walk-forward backtesting $\to$ Top-K simulation
- Generation of 4 publication-quality PNG charts
- Compilation of institutional PDF reports

### Step 4: Run Tests Standalone
```bash
python -m pytest tests/ -v
```

---

## 5. Visual Diagnostics Summary

| Output File | Description |
| :--- | :--- |
| `outputs/charts/1_macro_sentiment_timeline.png` | News sentiment trajectory overlaid with directional surprise markers and CPI / FOMC / NFP release dates. |
| `outputs/charts/2_pnl_heatmap_matrix.png` | Matrix of predicted expected P&L across all 27 permutations and 15 out-of-sample test periods. |
| `outputs/charts/3_equity_curves_comparison.png` | Cumulative out-of-sample portfolio equity (\$100k capital) with underwater drawdown shading. |
| `outputs/charts/4_walk_forward_fold_metrics.png` | Fold-by-fold Spearman rank correlation and LightGBM top 10 feature importances. |

---

## 6. Formal Reports & Documentation

- [Quantitative Evaluation Report (Markdown)](file:///c:/Users/Ankit%20Singh/Documents/macro-strategy-predictor/reports/evaluation_report.md) | [PDF Version](file:///c:/Users/Ankit%20Singh/Documents/macro-strategy-predictor/reports/evaluation_report.pdf)
- [Institutional Macro Outlook (Markdown)](file:///c:/Users/Ankit%20Singh/Documents/macro-strategy-predictor/reports/macro_outlook.md) | [PDF Version](file:///c:/Users/Ankit%20Singh/Documents/macro-strategy-predictor/reports/macro_outlook.pdf)
- [Submission & Reproduction Guide](file:///c:/Users/Ankit%20Singh/Documents/macro-strategy-predictor/docs/SUBMISSION.md)
- [Educational Learning Notes (Stage-by-Stage)](file:///c:/Users/Ankit%20Singh/Documents/macro-strategy-predictor/docs/LEARNING_NOTES.md)

---

## 7. Data Disclosure & Academic Honesty
In accordance with assignment instructions:
- `data/raw/synthetic_trades.csv` is explicitly synthetic, generated deterministically via `src/synthetic_trades.py` to model the interaction between market volatility cycles and stop-loss / take-profit strategy parameters.
- In institutional production, this script would be replaced with direct broker execution fills (e.g. Interactive Brokers Flex Queries or institutional FIX drop copy).
