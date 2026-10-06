"""Leakage-Free Feature Engineering & Modelling Dataset Builder.

This module constructs point-in-time features from trade logs, macro calendar events,
and financial news sentiment, aggregating into a modelling table where:
- Unit of observation: (permutation_id, period_id)
- Target: Expected / Realized Mean PnL of that permutation during that period
- Auxiliary Target: Realized Win Rate

STRICT ANTI-LEAKAGE CONTROLS:
1. Timestamp Precedence: Features characterizing period [T_start, T_end) are computed
   strictly using information available at or before T_start.
2. Expanding Window Statistics: Macro surprise z-score is computed using an expanding
   window anchored strictly in the past, preventing full-sample lookahead contamination.
3. Pre-Period News Window: Sentiment is aggregated across articles published before T_start.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_expanding_surprise_zscores(events_df: pd.DataFrame) -> pd.DataFrame:
    """Computes macroeconomic surprise z-scores strictly expanding into the past.

    Formula at event index i:
        z_score[i] = (surprise[i] - mean(surprises[0:i])) / std(surprises[0:i])
    
    If fewer than 3 historical events exist for an event type, defaults to 0.0.
    """
    df = events_df.sort_values("event_time_utc").copy()
    df["surprise"] = df["actual"] - df["forecast"]
    zscores = []

    for idx in range(len(df)):
        current_event = df.iloc[idx]
        event_name = current_event["event_name"]
        
        # Look strictly at past events of the same type up to (not including) the current one
        past_events = df.iloc[:idx]
        past_matching = past_events[past_events["event_name"] == event_name]["surprise"].dropna()

        if len(past_matching) >= 3:
            past_mean = past_matching.mean()
            past_std = past_matching.std(ddof=1)
            if past_std > 1e-6:
                z = (current_event["surprise"] - past_mean) / past_std
            else:
                z = 0.0
        else:
            z = 0.0
        zscores.append(round(float(z), 4))

    df["surprise_zscore"] = zscores
    return df


def generate_time_periods(
    start_date: str = "2026-04-01",
    end_date: str = "2026-09-28",
    period_days: int = 7,
) -> List[Dict[str, Any]]:
    """Divides the 180-day span into discrete, non-overlapping evaluation periods.

    Returns:
        List of dicts containing period_id, start_utc, end_utc.
    """
    periods = []
    current_start = pd.to_datetime(start_date, utc=True)
    final_end = pd.to_datetime(end_date, utc=True)
    period_idx = 1

    while current_start + pd.Timedelta(days=period_days) <= final_end:
        current_end = current_start + pd.Timedelta(days=period_days)
        periods.append({
            "period_id": f"PER_{period_idx:02d}",
            "start_utc": current_start,
            "end_utc": current_end,
        })
        current_start = current_end
        period_idx += 1

    return periods


def extract_point_in_time_macro_features(
    period_start: pd.Timestamp,
    events_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Extracts macroeconomic state features strictly available at or before period_start.

    Features engineered:
    - hour_of_day, day_of_week, is_month_end
    - minutes_since_last_event, minutes_to_next_event, is_event_day
    - prior_event_name, prior_surprise, prior_surprise_zscore
    - events_last_7d (count of macro announcements in preceding 7 days)
    - daily_avg_news_sentiment (average news sentiment published before period_start)
    """
    # 1. Calendar Features at Period Start
    hour_of_day = period_start.hour
    day_of_week = period_start.dayofweek
    is_month_end = int(period_start.is_month_end)

    # 2. Event Alignment (Past vs Upcoming)
    past_events = events_df[events_df["event_time_utc"] <= period_start].sort_values("event_time_utc")
    upcoming_events = events_df[events_df["event_time_utc"] > period_start].sort_values("event_time_utc")

    if not past_events.empty:
        latest_event = past_events.iloc[-1]
        minutes_since_last = (period_start - latest_event["event_time_utc"]).total_seconds() / 60.0
        prior_event_name = latest_event["event_name"]
        prior_surprise = float(latest_event["surprise"])
        prior_surprise_zscore = float(latest_event["surprise_zscore"])
        latest_event_time_str = latest_event["event_time_utc"].isoformat()
    else:
        minutes_since_last = 99999.0
        prior_event_name = "NONE"
        prior_surprise = 0.0
        prior_surprise_zscore = 0.0
        latest_event_time_str = ""

    if not upcoming_events.empty:
        next_event = upcoming_events.iloc[0]
        minutes_to_next = (next_event["event_time_utc"] - period_start).total_seconds() / 60.0
        next_event_name = next_event["event_name"]
    else:
        minutes_to_next = 99999.0
        next_event_name = "NONE"

    # Event day flag: event occurring within +/- 24 hours (1440 minutes) of period boundary
    is_event_day = int((minutes_since_last <= 1440.0) or (minutes_to_next <= 1440.0))

    # Events in past 7 days strictly before period_start
    seven_days_prior = period_start - pd.Timedelta(days=7)
    events_last_7d = int(len(events_df[
        (events_df["event_time_utc"] >= seven_days_prior) &
        (events_df["event_time_utc"] <= period_start)
    ]))

    # 3. News Sentiment strictly before period_start
    # Filter news published before period_start (prior 14-day window for freshness)
    fourteen_days_prior = period_start - pd.Timedelta(days=14)
    relevant_news = news_df[
        (news_df["date"] >= fourteen_days_prior) &
        (news_df["date"] <= period_start)
    ]
    if not relevant_news.empty:
        daily_avg_news_sentiment = float(relevant_news["sentiment_score"].mean())
    else:
        # Fallback to all preceding news if recent window empty
        all_prior_news = news_df[news_df["date"] <= period_start]
        daily_avg_news_sentiment = float(all_prior_news["sentiment_score"].mean()) if not all_prior_news.empty else 0.0

    return {
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_month_end": is_month_end,
        "minutes_since_last_event": round(minutes_since_last, 1),
        "minutes_to_next_event": round(minutes_to_next, 1),
        "is_event_day": is_event_day,
        "prior_event_name": prior_event_name,
        "prior_surprise": round(prior_surprise, 4),
        "prior_surprise_zscore": round(prior_surprise_zscore, 4),
        "events_last_7d": events_last_7d,
        "daily_avg_news_sentiment": round(daily_avg_news_sentiment, 4),
        "latest_event_time_utc": latest_event_time_str,
    }


def build_modelling_table(
    trades_path: str = "data/raw/synthetic_trades.csv",
    events_path: str = "data/processed/macro_calendar.csv",
    news_path: str = "data/processed/macro_news.csv",
    db_path: str = "data/macro.db",
) -> pd.DataFrame:
    """Builds the primary modeling dataset where one row = (permutation_id, period_id).

    Target variables:
    - target_mean_pnl: Realized average P&L per trade of that permutation during that period
    - target_win_rate: Proportion of trades with positive PnL during that period
    - total_pnl: Sum of PnL during that period
    - trade_count: Number of trades executed
    """
    trades_df = pd.read_csv(trades_path)
    events_df = pd.read_csv(events_path)
    news_df = pd.read_csv(news_path)

    trades_df["timestamp_utc"] = pd.to_datetime(trades_df["timestamp_utc"], utc=True)
    events_df["event_time_utc"] = pd.to_datetime(events_df["event_time_utc"], utc=True)
    news_df["date"] = pd.to_datetime(news_df["date"], utc=True)

    # Calculate strictly expanding surprise z-scores
    events_enriched = compute_expanding_surprise_zscores(events_df)

    # Get distinct permutations with their hyperparameters
    perm_meta = trades_df[[
        "permutation_id", "lookback", "stop_loss", "take_profit"
    ]].drop_duplicates().sort_values("permutation_id").reset_index(drop=True)

    periods = generate_time_periods()
    rows = []

    for per in periods:
        p_id = per["period_id"]
        p_start = per["start_utc"]
        p_end = per["end_utc"]

        # Extract macro features point-in-time at period_start
        macro_feats = extract_point_in_time_macro_features(p_start, events_enriched, news_df)

        # Filter trades that occurred during this specific period
        period_trades = trades_df[
            (trades_df["timestamp_utc"] >= p_start) &
            (trades_df["timestamp_utc"] < p_end)
        ]

        for _, p_info in perm_meta.iterrows():
            perm_id = p_info["permutation_id"]
            perm_trades = period_trades[period_trades["permutation_id"] == perm_id]

            if len(perm_trades) > 0:
                mean_pnl = float(perm_trades["pnl"].mean())
                total_pnl = float(perm_trades["pnl"].sum())
                win_rate = float((perm_trades["pnl"] > 0).mean())
                trade_count = len(perm_trades)
            else:
                # If no trades triggered, performance is 0
                mean_pnl = 0.0
                total_pnl = 0.0
                win_rate = 0.0
                trade_count = 0

            row = {
                "period_id": p_id,
                "period_start_utc": p_start.isoformat(),
                "period_end_utc": p_end.isoformat(),
                "permutation_id": perm_id,
                "lookback": p_info["lookback"],
                "stop_loss": p_info["stop_loss"],
                "take_profit": p_info["take_profit"],
                # Point-in-time Macro Features (calculated strictly at or before period_start)
                **macro_feats,
                # Realized Targets during [period_start, period_end)
                "target_mean_pnl": round(mean_pnl, 2),
                "target_win_rate": round(win_rate, 4),
                "total_pnl": round(total_pnl, 2),
                "trade_count": trade_count,
            }
            rows.append(row)

    model_df = pd.DataFrame(rows)

    # Encode categorical event name as numeric integer code for LightGBM
    event_mapping = {"NONE": 0, "CPI YoY": 1, "FOMC Rate Decision": 2, "Non-Farm Payrolls": 3}
    model_df["prior_event_code"] = model_df["prior_event_name"].map(lambda x: event_mapping.get(x, 0))

    # Save to SQLite features table
    conn = sqlite3.connect(db_path)
    try:
        model_df.to_sql("features", conn, if_exists="replace", index=False)
        print(f"[features] Successfully created 'features' modelling table ({len(model_df)} rows) in {db_path}")
    finally:
        conn.close()

    # Save processed copy
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    model_df.to_csv("data/processed/modelling_table.csv", index=False)
    return model_df


if __name__ == "__main__":
    df = build_modelling_table()
    print("\nModelling Table Overview:")
    print(df.info())
    print("\nFirst 3 rows:")
    print(df[["period_id", "permutation_id", "lookback", "stop_loss", "prior_event_name", "target_mean_pnl", "target_win_rate"]].head(3))
