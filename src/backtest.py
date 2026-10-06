"""Walk-Forward Out-of-Sample Backtesting & Strategy Parameter Simulation.

This module executes strict expanding-window walk-forward validation:
1. Training Window: Expands from period 1..T
2. Test Window: Exactly period T+1
3. Pure Out-of-Sample: No future data or labels ever touch parameter estimation.
4. Top-K Portfolio Simulation: In each test period, the model and baselines select
   the Top-K permutations by predicted P&L, realize the actual returns, and build
   uninterrupted out-of-sample equity curves.

KEY CONCEPTS IN WALK-FORWARD QUANT RESEARCH:
1. Expanding vs Rolling Window: An expanding window provides increasing statistical sample
   size while strictly respecting time ordering, preventing standard k-fold cross-validation
   leakage (where future trades train past predictions).
2. Top-K Selection: Rather than betting 100% of capital on a single permutation, selecting
   Top-K (e.g. Top 3) diversifies parameter idiosyncratic noise while concentrating capital
   in top-ranked parameter space.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.metrics import (
    calculate_hit_rate,
    calculate_mae,
    calculate_max_drawdown,
    calculate_rmse,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_spearman_rank_correlation,
    calculate_underwater_series,
    compute_comprehensive_metrics_report,
)
from src.model import (
    FEATURE_COLS,
    TARGET_COL,
    HistoricalMeanBaseline,
    MacroLightGBMModel,
    RandomChoiceBaseline,
)


def run_walk_forward_validation(
    modelling_path: str = "data/processed/modelling_table.csv",
    min_train_periods: int = 10,
    top_k: int = 3,
    initial_capital: float = 100000.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes expanding-window walk-forward validation across all test periods.

    Args:
        modelling_path: CSV containing (permutation_id, period_id) features and targets.
        min_train_periods: Number of initial periods reserved for training warm-up.
        top_k: Number of highest-predicted permutations to execute in each period.
        initial_capital: Portfolio baseline dollars ($100k).
        seed: Random seed for reproducibility.

    Returns:
        Dict containing simulation results, fold metrics, out-of-sample predictions,
        and equity curves for Model, Historical Mean Baseline, and Random Baseline.
    """
    df = pd.read_csv(modelling_path)
    periods = sorted(df["period_id"].unique())
    total_periods = len(periods)

    if total_periods <= min_train_periods:
        raise ValueError(
            f"Insufficient periods ({total_periods}) for warm-up of {min_train_periods}."
        )

    test_periods = periods[min_train_periods:]
    print(f"[backtest] Total periods: {total_periods}. Training warm-up: {min_train_periods}. Test folds: {len(test_periods)}.")

    oos_records: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []

    # Track period returns for portfolio simulation
    model_period_returns: List[float] = []
    hist_period_returns: List[float] = []
    rand_period_returns: List[float] = []
    test_period_dates: List[str] = []

    last_model = None

    for fold_idx, test_p in enumerate(test_periods, 1):
        train_p_set = periods[:periods.index(test_p)]
        
        train_df = df[df["period_id"].isin(train_p_set)].copy()
        test_df = df[df["period_id"] == test_p].copy()

        y_train = train_df[TARGET_COL]
        y_test = test_df[TARGET_COL].values

        # 1. Fit LightGBM Model
        lgb_model = MacroLightGBMModel(random_state=seed + fold_idx)
        lgb_model.fit(train_df, y_train)
        pred_lgb = lgb_model.predict(test_df)
        last_model = lgb_model

        # 2. Fit Historical Mean Baseline
        hist_base = HistoricalMeanBaseline()
        hist_base.fit(train_df, y_train)
        pred_hist = hist_base.predict(test_df)

        # 3. Fit Random Baseline
        rand_base = RandomChoiceBaseline(seed=seed + fold_idx * 7)
        pred_rand = rand_base.predict(test_df)

        # Record out-of-sample predictions per permutation
        test_df_eval = test_df.copy()
        test_df_eval["pred_lgb"] = pred_lgb
        test_df_eval["pred_hist"] = pred_hist
        test_df_eval["pred_rand"] = pred_rand

        for _, r in test_df_eval.iterrows():
            oos_records.append({
                "fold": fold_idx,
                "period_id": test_p,
                "period_start_utc": r["period_start_utc"],
                "period_end_utc": r["period_end_utc"],
                "permutation_id": r["permutation_id"],
                "actual_pnl": r[TARGET_COL],
                "actual_win_rate": r["target_win_rate"],
                "pred_lgb": r["pred_lgb"],
                "pred_hist": r["pred_hist"],
                "pred_rand": r["pred_rand"],
            })

        # Calculate fold-level predictive metrics
        fold_spearman_lgb = calculate_spearman_rank_correlation(y_test, pred_lgb)
        fold_mae_lgb = calculate_mae(y_test, pred_lgb)
        fold_rmse_lgb = calculate_rmse(y_test, pred_lgb)

        fold_metrics.append({
            "fold": fold_idx,
            "test_period": test_p,
            "train_size": len(train_df),
            "spearman_lgb": fold_spearman_lgb,
            "mae_lgb": fold_mae_lgb,
            "rmse_lgb": fold_rmse_lgb,
        })

        # Simulation: Select Top-K permutations
        # Approximate period return as: (selected mean P&L / standard risk unit $500) * 0.01
        # To maintain clean economic realism, we scale average dollar PnL per trade to a % return
        scale_pnl_to_return = 0.00015

        top_lgb = test_df_eval.nlargest(top_k, "pred_lgb")
        top_hist = test_df_eval.nlargest(top_k, "pred_hist")
        top_rand = test_df_eval.nlargest(top_k, "pred_rand")

        ret_lgb = float(top_lgb[TARGET_COL].mean()) * scale_pnl_to_return
        ret_hist = float(top_hist[TARGET_COL].mean()) * scale_pnl_to_return
        ret_rand = float(top_rand[TARGET_COL].mean()) * scale_pnl_to_return

        model_period_returns.append(ret_lgb)
        hist_period_returns.append(ret_hist)
        rand_period_returns.append(ret_rand)
        test_period_dates.append(str(test_df["period_start_utc"].iloc[0])[:10])

    # Build Cumulative Equity Curves
    equity_lgb = initial_capital * np.cumprod(1.0 + np.array(model_period_returns))
    equity_hist = initial_capital * np.cumprod(1.0 + np.array(hist_period_returns))
    equity_rand = initial_capital * np.cumprod(1.0 + np.array(rand_period_returns))

    oos_df = pd.DataFrame(oos_records)
    fold_df = pd.DataFrame(fold_metrics)

    # Compute overall summary metrics
    y_true_all = oos_df["actual_pnl"].values
    y_pred_all_lgb = oos_df["pred_lgb"].values
    y_pred_all_hist = oos_df["pred_hist"].values

    metrics_lgb = compute_comprehensive_metrics_report(
        returns=model_period_returns,
        y_true=y_true_all,
        y_pred=y_pred_all_lgb,
        initial_capital=initial_capital,
    )
    metrics_hist = compute_comprehensive_metrics_report(
        returns=hist_period_returns,
        y_true=y_true_all,
        y_pred=y_pred_all_hist,
        initial_capital=initial_capital,
    )
    metrics_rand = compute_comprehensive_metrics_report(
        returns=rand_period_returns,
        initial_capital=initial_capital,
    )

    feature_importances = last_model.get_feature_importances() if last_model else pd.DataFrame()

    results = {
        "metrics_summary": {
            "Macro LightGBM": metrics_lgb,
            "Historical Mean Baseline": metrics_hist,
            "Random Selection Baseline": metrics_rand,
        },
        "period_dates": test_period_dates,
        "equity_curves": {
            "dates": test_period_dates,
            "LightGBM": [round(float(v), 2) for v in equity_lgb],
            "Historical Mean": [round(float(v), 2) for v in equity_hist],
            "Random Baseline": [round(float(v), 2) for v in equity_rand],
        },
        "drawdowns": {
            "LightGBM": [round(float(v), 4) for v in calculate_underwater_series(equity_lgb)],
            "Historical Mean": [round(float(v), 4) for v in calculate_underwater_series(equity_hist)],
            "Random Baseline": [round(float(v), 4) for v in calculate_underwater_series(equity_rand)],
        },
        "returns": {
            "LightGBM": [round(float(v), 4) for v in model_period_returns],
            "Historical Mean": [round(float(v), 4) for v in hist_period_returns],
            "Random Baseline": [round(float(v), 4) for v in rand_period_returns],
        },
        "feature_importances": feature_importances.to_dict(orient="records"),
    }

    # Save to disk
    oos_df.to_csv("data/processed/oos_predictions.csv", index=False)
    fold_df.to_csv("data/processed/fold_metrics.csv", index=False)
    feature_importances.to_csv("data/processed/feature_importances.csv", index=False)
    
    with open("data/processed/backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[backtest] Walk-Forward Simulation Complete!")
    print(f"[backtest] Saved out-of-sample predictions ({len(oos_df)} rows) to data/processed/oos_predictions.csv")
    print(f"[backtest] Saved backtest results to data/processed/backtest_results.json")
    
    return results


if __name__ == "__main__":
    res = run_walk_forward_validation()
    print("\n=== WALK-FORWARD OUT-OF-SAMPLE PERFORMANCE SUMMARY ===")
    summary_df = pd.DataFrame(res["metrics_summary"]).T
    print(summary_df[["Annualized Sharpe", "Annualized Sortino", "Max Drawdown", "Cumulative Return", "Hit Rate"]])
