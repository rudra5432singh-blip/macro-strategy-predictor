"""Visualizations Suite for Macro-Aware Strategy Predictor.

Generates 4 publication-quality quantitative charts saved to outputs/charts/:
1. 1_macro_sentiment_timeline.png: Macro news sentiment timeline, event surprise dots,
   and CPI/FOMC/NFP release markers.
2. 2_pnl_heatmap_matrix.png: Heatmap matrix of predicted P&L by permutation and period.
3. 3_equity_curves_comparison.png: Cumulative out-of-sample equity curves with underwater
   drawdown shading comparing LightGBM vs Baselines.
4. 4_walk_forward_fold_metrics.png: Fold-by-fold Spearman rank correlation and LightGBM
   feature importances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set publication aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

OUTPUTS_DIR = Path("outputs/charts")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_macro_sentiment_timeline(
    events_path: str = "data/processed/macro_calendar.csv",
    news_path: str = "data/processed/macro_news.csv",
    output_file: str = "outputs/charts/1_macro_sentiment_timeline.png",
) -> None:
    """Chart 1: Macro sentiment over time with surprise direction and event markers."""
    events_df = pd.read_csv(events_path)
    news_df = pd.read_csv(news_path)

    events_df["event_time_utc"] = pd.to_datetime(events_df["event_time_utc"])
    events_df["surprise"] = events_df["actual"] - events_df["forecast"]
    news_df["date"] = pd.to_datetime(news_df["date"])

    fig, ax1 = plt.subplots(figsize=(13, 6), dpi=300)

    # Plot news sentiment series
    ax1.plot(
        news_df["date"],
        news_df["sentiment_score"],
        color="#2b5c8f",
        linewidth=2.0,
        marker="o",
        markersize=5,
        label="News Sentiment Score",
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("Macro News Sentiment Score (-1.0 to +1.0)", color="#2b5c8f", fontweight="bold")
    ax1.set_ylim(-1.05, 1.05)

    # Event markers with vertical lines
    marker_colors = {
        "CPI YoY": "#e74c3c",           # Red
        "FOMC Rate Decision": "#8e44ad", # Purple
        "Non-Farm Payrolls": "#27ae60",  # Green
    }

    used_labels = set()
    for _, event in events_df.iterrows():
        ename = event["event_name"]
        etime = event["event_time_utc"]
        color = marker_colors.get(ename, "#f39c12")
        lbl = ename if ename not in used_labels else ""
        used_labels.add(ename)

        ax1.axvline(etime, color=color, linestyle=":", alpha=0.6, linewidth=1.5, label=lbl)

        # Plot surprise marker
        surprise_val = event["surprise"]
        marker_symbol = "^" if surprise_val >= 0 else "v"
        ax1.scatter(
            etime,
            0.85 if surprise_val >= 0 else -0.85,
            color=color,
            s=80,
            marker=marker_symbol,
            edgecolors="black",
            zorder=5,
        )

    ax1.set_title(
        "Macro Sentiment Timeline: News Sentiment & High-Impact Event Releases (CPI / FOMC / NFP)",
        fontweight="bold",
        pad=14,
    )
    ax1.set_xlabel("Date (2026)", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"[charts] Saved Chart 1 to: {output_file}")


def plot_pnl_heatmap_matrix(
    oos_path: str = "data/processed/oos_predictions.csv",
    output_file: str = "outputs/charts/2_pnl_heatmap_matrix.png",
) -> None:
    """Chart 2: Heatmap matrix of predicted P&L by permutation and walk-forward period."""
    oos_df = pd.read_csv(oos_path)

    # Pivot: Index = Permutation ID, Columns = Period ID, Values = Predicted LGB PnL
    pivot_df = oos_df.pivot(index="permutation_id", columns="period_id", values="pred_lgb")

    plt.figure(figsize=(14, 8), dpi=300)
    cmap = sns.diverging_palette(15, 140, as_cmap=True)

    ax = sns.heatmap(
        pivot_df,
        cmap=cmap,
        center=0.0,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Predicted Expected P&L ($)"},
        linewidths=0.5,
        linecolor="#f0f0f0",
    )

    plt.title(
        "Strategy Permutation Optimization Heatmap: Predicted P&L across Walk-Forward Periods",
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Walk-Forward Test Period (Out-of-Sample)", fontweight="bold")
    plt.ylabel("Strategy Permutation ID", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"[charts] Saved Chart 2 to: {output_file}")


def plot_equity_curves_comparison(
    backtest_results_path: str = "data/processed/backtest_results.json",
    output_file: str = "outputs/charts/3_equity_curves_comparison.png",
) -> None:
    """Chart 3: Equity curves comparison (Model vs Baselines) with underwater drawdown shading."""
    with open(backtest_results_path, "r") as f:
        data = json.load(f)

    dates = data["equity_curves"]["dates"]
    eq_lgb = data["equity_curves"]["LightGBM"]
    eq_hist = data["equity_curves"]["Historical Mean"]
    eq_rand = data["equity_curves"]["Random Baseline"]

    dd_lgb = np.array(data["drawdowns"]["LightGBM"]) * 100
    dd_hist = np.array(data["drawdowns"]["Historical Mean"]) * 100
    dd_rand = np.array(data["drawdowns"]["Random Baseline"]) * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    x_indices = range(len(dates))

    # Top panel: Cumulative Equity Curves
    ax1.plot(x_indices, eq_lgb, label=f"Macro LightGBM (Sharpe: {data['metrics_summary']['Macro LightGBM']['Annualized Sharpe']:.2f})", color="#1f77b4", linewidth=2.5)
    ax1.plot(x_indices, eq_hist, label=f"Historical Mean Baseline (Sharpe: {data['metrics_summary']['Historical Mean Baseline']['Annualized Sharpe']:.2f})", color="#e67e22", linestyle="--", linewidth=2.0)
    ax1.plot(x_indices, eq_rand, label=f"Random Selection Baseline (Sharpe: {data['metrics_summary']['Random Selection Baseline']['Annualized Sharpe']:.2f})", color="#7f8c8d", linestyle=":", linewidth=1.8)

    ax1.set_ylabel("Portfolio Value ($)", fontweight="bold")
    ax1.set_title("Walk-Forward Out-of-Sample Cumulative Equity: Model vs Benchmarks", fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True, facecolor="white")
    ax1.yaxis.set_major_formatter("${x:,.0f}")

    # Bottom panel: Underwater Drawdowns
    ax2.fill_between(x_indices, dd_lgb, 0, color="#1f77b4", alpha=0.3, label="LightGBM Drawdown")
    ax2.plot(x_indices, dd_lgb, color="#1f77b4", linewidth=1.5)
    ax2.plot(x_indices, dd_hist, color="#e67e22", linestyle="--", linewidth=1.2, label="Hist Mean Drawdown")
    ax2.plot(x_indices, dd_rand, color="#7f8c8d", linestyle=":", linewidth=1.0, label="Random Drawdown")

    ax2.set_ylabel("Drawdown (%)", fontweight="bold")
    ax2.set_xlabel("Rebalance Period Date", fontweight="bold")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(dates, rotation=45, ha="right")
    ax2.legend(loc="lower left", frameon=True, facecolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"[charts] Saved Chart 3 to: {output_file}")


def plot_walk_forward_metrics_and_importance(
    fold_path: str = "data/processed/fold_metrics.csv",
    feat_path: str = "data/processed/feature_importances.csv",
    output_file: str = "outputs/charts/4_walk_forward_fold_metrics.png",
) -> None:
    """Chart 4: Walk-forward metrics by fold and global feature importance."""
    fold_df = pd.read_csv(fold_path)
    feat_df = pd.read_csv(feat_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Panel 1: Spearman Rank Correlation by Fold
    ax1.plot(
        fold_df["fold"],
        fold_df["spearman_lgb"],
        color="#2980b9",
        marker="s",
        linewidth=2.0,
        markersize=6,
        label="Spearman Rank Corr",
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axhline(
        fold_df["spearman_lgb"].mean(),
        color="#27ae60",
        linestyle="-.",
        linewidth=1.5,
        label=f"Mean Rank Corr ({fold_df['spearman_lgb'].mean():.3f})",
    )
    ax1.set_title("Out-of-Sample Spearman Rank Correlation by Fold", fontweight="bold")
    ax1.set_xlabel("Walk-Forward Fold Number", fontweight="bold")
    ax1.set_ylabel("Rank Correlation Coefficient", fontweight="bold")
    ax1.set_ylim(-0.6, 0.8)
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Feature Importance Bar Chart
    feat_sorted = feat_df.sort_values("importance", ascending=True).tail(10)
    y_pos = range(len(feat_sorted))
    ax2.barh(y_pos, feat_sorted["importance"], color="#34495e", edgecolor="#2c3e50")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feat_sorted["feature"])
    ax2.set_title("LightGBM Feature Importance (Top 10 Drivers)", fontweight="bold")
    ax2.set_xlabel("Importance Score (Splits / Information Gain)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"[charts] Saved Chart 4 to: {output_file}")


def generate_all_charts() -> None:
    """Executes all visualization routines."""
    plot_macro_sentiment_timeline()
    plot_pnl_heatmap_matrix()
    plot_equity_curves_comparison()
    plot_walk_forward_metrics_and_importance()
    print("[charts] All 4 publication-quality charts successfully generated!")


if __name__ == "__main__":
    generate_all_charts()
