"""Unit tests for quantitative metrics library.

Verifies Sharpe ratio, Sortino ratio, Max Drawdown, Spearman rank correlation,
MAE, RMSE, and Hit Rate against exact analytical solutions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    calculate_hit_rate,
    calculate_mae,
    calculate_max_drawdown,
    calculate_rmse,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_spearman_rank_correlation,
)


def test_sharpe_ratio_analytical():
    """Verifies Sharpe ratio against manual analytical calculation."""
    # Returns with mean = 0.02, std = 0.01, annualization = sqrt(52) = 7.2111
    returns = [0.01, 0.03, 0.01, 0.03]  # mean = 0.02, sample std (ddof=1) = 0.011547
    ann_factor = np.sqrt(52.0)
    expected = (0.02 / np.std(returns, ddof=1)) * ann_factor

    calculated = calculate_sharpe_ratio(returns, annualization_factor=ann_factor)
    assert abs(calculated - expected) < 1e-3

    # Zero volatility test
    assert calculate_sharpe_ratio([0.05, 0.05, 0.05]) == 0.0


def test_sortino_ratio_downside_penalty():
    """Verifies Sortino ratio only penalizes downside deviations below hurdle rate."""
    # Series A and Series B have same mean, but Series B has large upside spikes
    # Series A: constant [0.02, 0.02, -0.01, 0.01]
    # Series B: [0.10, 0.02, -0.01, 0.01] -> higher upside, identical downside
    series_a = [0.02, 0.02, -0.01, 0.01]
    series_b = [0.10, 0.02, -0.01, 0.01]

    sortino_a = calculate_sortino_ratio(series_a)
    sortino_b = calculate_sortino_ratio(series_b)

    # Higher upside with same downside must result in strictly higher Sortino
    assert sortino_b > sortino_a


def test_max_drawdown_known_sequence():
    """Verifies Max Drawdown on a known equity series with 25% peak-to-trough drop."""
    # Peak = 120, trough = 90 -> drawdown = (120 - 90) / 120 = 30 / 120 = 0.25 (25%)
    equity = [100.0, 110.0, 120.0, 96.0, 90.0, 105.0, 115.0]
    mdd = calculate_max_drawdown(equity)
    assert round(mdd, 2) == 0.25


def test_spearman_rank_correlation_properties():
    """Verifies Spearman rank correlation is 1.0 for monotonic rank, -1.0 for inverse."""
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred_perfect = [10.0, 20.0, 30.0, 40.0, 50.0]
    y_pred_inverse = [50.0, 40.0, 30.0, 20.0, 10.0]

    assert calculate_spearman_rank_correlation(y_true, y_pred_perfect) == 1.0
    assert calculate_spearman_rank_correlation(y_true, y_pred_inverse) == -1.0


def test_mae_and_rmse():
    """Verifies MAE and RMSE on simple integer vectors."""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 5.0]  # errors: [1, 0, 2] -> MAE = 3/3 = 1.0; squared errors: [1, 0, 4] -> RMSE = sqrt(5/3) = 1.2910

    assert calculate_mae(y_true, y_pred) == 1.0
    assert abs(calculate_rmse(y_true, y_pred) - np.sqrt(5.0 / 3.0)) < 1e-3


def test_hit_rate():
    """Verifies directional hit rate."""
    returns = [0.05, -0.02, 0.01, 0.03, -0.01]  # 3 positive out of 5 -> 60% = 0.60
    assert calculate_hit_rate(returns) == 0.60
