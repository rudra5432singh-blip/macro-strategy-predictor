"""End-to-End Pipeline Orchestrator for Macro Strategy Predictor.

Executes all pipeline stages deterministically in sequence:
1. Synthetic / User Trade Verification (src/synthetic_trades.py)
2. Macro Calendar Ingestion & Timezone Standardization (src/ingest_apify.py)
3. Macro News Ingestion & Sentiment Scoring (src/ingest_tavily.py)
4. SQLite Database Loading & Temporal Event Alignment (src/build_db.py)
5. Point-in-Time Leakage-Free Feature Engineering (src/features.py)
6. Automated Unit & Anti-Leakage Test Suite (pytest tests/)
7. Expanding Walk-Forward Validation & Top-K Backtesting (src/backtest.py)
8. Publication-Quality Visualization Generation (src/charts.py)
9. Institutional PDF Report Compilation (src/export_reports.py)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def print_banner(step_num: int, title: str) -> None:
    print("\n" + "=" * 78)
    print(f" [STAGE {step_num}] {title.upper()}")
    print("=" * 78)


def run_command(cmd_args: list[str], desc: str) -> None:
    print(f"--> Executing: {' '.join(cmd_args)}")
    start_time = time.time()
    res = subprocess.run(cmd_args)
    elapsed = time.time() - start_time
    if res.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at step: {desc} (exit code {res.returncode})")
        sys.exit(res.returncode)
    print(f"--> [SUCCESS] {desc} finished in {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete Macro Strategy Predictor pipeline.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest test suite.")
    args = parser.parse_args()

    overall_start = time.time()
    py = sys.executable

    print("\n" + "#" * 78)
    print("   MACRO-AWARE STRATEGY PARAMETER PREDICTOR: END-TO-END PIPELINE RUNNER   ")
    print("#" * 78)

    # 1. Trade Data
    print_banner(1, "Verifying Trade Execution Logs")
    run_command([py, "-m", "src.synthetic_trades"], "Trade Log Verification")

    # 2. Economic Calendar
    print_banner(2, "Macro Calendar Ingestion & DST Normalization")
    run_command([py, "-m", "src.ingest_apify"], "Apify Macro Calendar Ingestion")

    # 3. Macro News & Sentiment
    print_banner(3, "Macro News Retrieval & Sentiment Scoring")
    run_command([py, "-m", "src.ingest_tavily"], "Tavily Macro News Ingestion")

    # 4. Database Loading & Temporal Alignment
    print_banner(4, "SQLite Database & Point-in-Time Temporal Joins")
    run_command([py, "-m", "src.build_db"], "SQLite Database Builder")

    # 5. Feature Engineering
    print_banner(5, "Leakage-Free Feature Engineering & Modelling Dataset")
    run_command([py, "-m", "src.features"], "Point-in-Time Feature Engineering")

    # 6. Test Suite
    if not args.skip_tests:
        print_banner(6, "Executing Anti-Leakage & Metrics Test Suite")
        run_command([py, "-m", "pytest", "tests/", "-v"], "Pytest Test Suite")
    else:
        print("\n--> [SKIPPED] Test suite skipped per command-line flag.")

    # 7. Model Training & Walk-Forward Backtesting
    print_banner(7, "Expanding Walk-Forward Validation & Top-K Simulation")
    run_command([py, "-m", "src.backtest"], "Walk-Forward Backtesting Engine")

    # 8. Visualizations
    print_banner(8, "Generating Publication-Quality Visualizations")
    run_command([py, "-m", "src.charts"], "Chart Generation Suite")

    # 9. PDF Compilation
    print_banner(9, "Compiling Institutional PDF Reports")
    run_command([py, "-m", "src.export_reports"], "PDF Report Compilation")

    total_time = time.time() - overall_start
    print("\n" + "#" * 78)
    print(f"   PIPELINE COMPLETE: ALL DELIVERABLES GENERATED IN {total_time:.2f}s   ")
    print("#" * 78)
    print("\nDeliverables Directory Summary:")
    print("  • Database:        data/macro.db")
    print("  • Modelling Table: data/processed/modelling_table.csv")
    print("  • OOS Predictions: data/processed/oos_predictions.csv")
    print("  • Charts (PNG):    outputs/charts/ (4 charts)")
    print("  • Reports (MD):    reports/evaluation_report.md, reports/macro_outlook.md")
    print("  • Reports (PDF):   reports/evaluation_report.pdf, reports/macro_outlook.pdf")
    print("  • Submission Doc:  docs/SUBMISSION.md")
    print("  • Learning Notes:  docs/LEARNING_NOTES.md\n")


if __name__ == "__main__":
    main()
