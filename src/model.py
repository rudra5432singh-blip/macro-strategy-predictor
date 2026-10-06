"""Predictive Models and Quantitative Baselines for Strategy Parameter Selection.

This module implements:
1. MacroLightGBMModel: Gradient boosted trees integrating strategy hyperparameter
   permutations with point-in-time macroeconomic state and news sentiment features.
2. HistoricalMeanBaseline: Benchmark predicting future return as historical mean P&L.
3. RandomChoiceBaseline: Uniformly random permutation selection benchmark.

KEY CONCEPTS IN QUANTITATIVE META-LEARNING:
1. Interaction Modeling: Gradient boosted trees excel at discovering non-linear interactions
   between strategy parameters and macro regimes (e.g., tight stops fail when surprise z-scores
   are high; wide stops thrive during trend/easing regimes).
2. The Historical Mean Benchmark: The most critical baseline in quant finance. If an ML model
   cannot outperform simple rolling historical mean ranking, the macro signals are merely noise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


FEATURE_COLS = [
    "lookback",
    "stop_loss",
    "take_profit",
    "hour_of_day",
    "day_of_week",
    "is_month_end",
    "minutes_since_last_event",
    "minutes_to_next_event",
    "is_event_day",
    "prior_event_code",
    "prior_surprise",
    "prior_surprise_zscore",
    "events_last_7d",
    "daily_avg_news_sentiment",
]

TARGET_COL = "target_mean_pnl"


class MacroLightGBMModel:
    """LightGBM regressor predicting strategy permutation performance given macro context."""

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 5,
            "feature_fraction": 0.85,
            "n_estimators": 120,
            "random_state": random_state,
            "verbose": -1,
        }
        self.params.update(kwargs)
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = FEATURE_COLS

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MacroLightGBMModel:
        """Trains LightGBM regressor on feature matrix X and target y."""
        X_feats = X[self.feature_names].copy()
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_feats, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predicted expected P&L for each permutation."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_feats = X[self.feature_names].copy()
        return self.model.predict(X_feats)

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns sorted feature importances."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class HistoricalMeanBaseline(BaseEstimator, RegressorMixin):
    """Predicts future permutation P&L based on its historical historical mean."""

    def __init__(self):
        self.history_means_: Dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HistoricalMeanBaseline:
        df = X[["permutation_id"]].copy()
        df["target"] = y.values
        means = df.groupby("permutation_id")["target"].mean().to_dict()
        self.history_means_ = means
        self.global_mean_ = float(y.mean()) if len(y) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([
            self.history_means_.get(pid, self.global_mean_)
            for pid in X["permutation_id"]
        ])


class RandomChoiceBaseline(BaseEstimator, RegressorMixin):
    """Generates uniform random noise scores to simulate naive unguided selection."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomChoiceBaseline:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        return rng.uniform(-1.0, 1.0, size=len(X))
