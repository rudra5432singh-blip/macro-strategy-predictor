"""Quantitative Performance Metrics & Predictive Evaluation Library.

This module provides transparent, mathematically rigorous metric functions for
evaluating both strategy return streams (Sharpe, Sortino, Max Drawdown) and
predictive model rankings (MAE, RMSE, Spearman Rank Correlation, Hit Rate).

KEY CONCEPTS IN QUANTITATIVE PERFORMANCE EVALUATION:
1. Sharpe vs. Sortino: The Sharpe ratio penalizes both upside and downside volatility
   equally. However, an investor welcomes upside volatility (large windfalls).
   The Sortino ratio calculates the Semi-Deviation (downside variance below the target
   or risk-free rate), providing a truer measure of risk-adjusted excess return.
2. Annualization Factor: If periods are weekly (7-day intervals), the annualization
   scalar is sqrt(52). If returns are daily, sqrt(252). This module makes the annualization
   factor an explicit, documented parameter.
3. Spearman Rank Correlation vs. MSE: In parameter selection, we don't care if the model
   underpredicts exact dollar amounts as long as it ranks the best permutations at the top.
   Spearman correlation measures monotonic ranking quality rather than linear fit.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats


def calculate_sharpe_ratio(
    returns: Union[Sequence[float], np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    annualization_factor: float = np.sqrt(52.0),
) -> float:
    """Computes the annualized Sharpe ratio.

    Args:
        returns: Array of period returns or P&L series.
        risk_free_rate: Risk-free hurdle rate per period (default 0.0).
        annualization_factor: Scaling factor (sqrt(52) for weekly periods, sqrt(252) for daily).

    Returns:
        Annualized Sharpe ratio as float, or 0.0 if standard deviation is zero.
    """
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 2:
        return 0.0

    excess = arr - risk_free_rate
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)

    if std_excess < 1e-8:
        return 0.0

    sharpe = (mean_excess / std_excess) * annualization_factor
    return round(float(sharpe), 4)


def calculate_sortino_ratio(
    returns: Union[Sequence[float], np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    annualization_factor: float = np.sqrt(52.0),
) -> float:
    """Computes the annualized Sortino ratio using downside semi-deviation.

    Args:
        returns: Array of period returns.
        risk_free_rate: Hurdle rate per period.
        annualization_factor: Scaling factor (default sqrt(52)).

    Returns:
        Annualized Sortino ratio as float.
    """
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 2:
        return 0.0

    excess = arr - risk_free_rate
    mean_excess = np.mean(excess)

    # Downside deviation: only consider returns below hurdle rate
    downside_deviations = np.minimum(0.0, excess)
    downside_variance = np.mean(downside_deviations ** 2)
    downside_std = np.sqrt(downside_variance)

    if downside_std < 1e-8:
        # If no downside deviations and mean is positive, strategy has perfect downside preservation
        return 999.0 if mean_excess > 0 else 0.0

    sortino = (mean_excess / downside_std) * annualization_factor
    return round(float(sortino), 4)


def calculate_max_drawdown(
    cumulative_series: Union[Sequence[float], np.ndarray, pd.Series],
) -> float:
    """Computes maximum peak-to-trough drawdown of a cumulative equity or wealth curve.

    Args:
        cumulative_series: Series of cumulative equity or cumulative P&L values.

    Returns:
        Maximum drawdown expressed as a positive percentage/fraction (e.g. 0.15 for 15%).
    """
    equity = np.asarray(cumulative_series, dtype=float)
    if len(equity) < 1:
        return 0.0

    # Running maximum
    peaks = np.maximum.accumulate(equity)
    
    # Avoid zero division if starting equity is zero (for dollar PnL)
    if np.all(peaks <= 0):
        # Dollar drawdown
        drawdowns = peaks - equity
        return round(float(np.max(drawdowns)), 4)

    drawdowns = (peaks - equity) / np.where(peaks > 0, peaks, 1.0)
    mdd = np.max(drawdowns)
    return round(float(max(0.0, mdd)), 4)


def calculate_underwater_series(
    cumulative_series: Union[Sequence[float], np.ndarray, pd.Series],
) -> np.ndarray:
    """Generates the full drawdown series (underwater chart values) over time."""
    equity = np.asarray(cumulative_series, dtype=float)
    peaks = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (equity - peaks) / np.where(peaks > 0, peaks, 1.0)
    return np.nan_to_num(dd, nan=0.0)


def calculate_mae(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
) -> float:
    """Computes Mean Absolute Error."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return round(float(np.mean(np.abs(t - p))), 4)


def calculate_rmse(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
) -> float:
    """Computes Root Mean Squared Error."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return round(float(np.sqrt(np.mean((t - p) ** 2))), 4)


def calculate_spearman_rank_correlation(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
) -> float:
    """Computes Spearman Rank Correlation between predicted and realized values.

    Returns:
        Rank correlation coefficient in range [-1.0, 1.0].
    """
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if len(t) < 2 or np.all(t == t[0]) or np.all(p == p[0]):
        return 0.0

    res = stats.spearmanr(t, p)
    corr = res.statistic if hasattr(res, "statistic") else res[0]
    return round(float(corr), 4)


def calculate_hit_rate(returns: Union[Sequence[float], np.ndarray]) -> float:
    """Computes directional hit rate (proportion of periods with return > 0)."""
    arr = np.asarray(returns, dtype=float)
    if len(arr) == 0:
        return 0.0
    return round(float(np.mean(arr > 0)), 4)


def compute_comprehensive_metrics_report(
    returns: Union[Sequence[float], np.ndarray],
    y_true: Optional[Union[Sequence[float], np.ndarray]] = None,
    y_pred: Optional[Union[Sequence[float], np.ndarray]] = None,
    initial_capital: float = 100000.0,
    annualization_factor: float = np.sqrt(52.0),
) -> Dict[str, float]:
    """Generates a complete dictionary of strategy performance and model metrics."""
    ret_arr = np.asarray(returns, dtype=float)
    
    # Cumulative equity curve
    equity_curve = initial_capital * np.cumprod(1.0 + ret_arr)

    metrics = {
        "Annualized Sharpe": calculate_sharpe_ratio(ret_arr, annualization_factor=annualization_factor),
        "Annualized Sortino": calculate_sortino_ratio(ret_arr, annualization_factor=annualization_factor),
        "Max Drawdown": calculate_max_drawdown(equity_curve),
        "Cumulative Return": round(float((equity_curve[-1] / initial_capital) - 1.0), 4) if len(equity_curve) > 0 else 0.0,
        "Hit Rate": calculate_hit_rate(ret_arr),
        "Average Period Return": round(float(np.mean(ret_arr)), 4) if len(ret_arr) > 0 else 0.0,
    }

    if y_true is not None and y_pred is not None:
        metrics["MAE"] = calculate_mae(y_true, y_pred)
        metrics["RMSE"] = calculate_rmse(y_true, y_pred)
        metrics["Spearman Rank Corr"] = calculate_spearman_rank_correlation(y_true, y_pred)

    return metrics
