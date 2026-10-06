# Quantitative Learning Notes

Documenting key quantitative finance and machine learning concepts in plain language as the pipeline is developed stage by stage.

---

## Stage 0: Environment Setup, Secrets Hygiene & Pipeline Architecture

### What I Learned
In quantitative finance and machine learning workflows, reproducibility and strict security hygiene are foundational. Financial data systems must never expose API keys (e.g., Apify or Tavily credentials) in code or git commits, because compromised keys can lead to service exhaustion or data tampering. Structuring the repository with explicit separation between raw immutable data (`data/raw/`), intermediate storage (`data/macro.db`), source logic (`src/`), verification tests (`tests/`), and visual/reporting outputs (`outputs/`, `reports/`) ensures that every stage of data transformation can be audited and reproduced deterministically without leakage.
