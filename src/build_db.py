"""SQLite Database Builder and Point-in-Time Temporal Event Alignment.

This module loads trade logs, macro calendar events, and news articles into
a unified SQLite database (`data/macro.db`). It implements both pandas `merge_asof`
and SQL point-in-time temporal joins to link each trade to its latest prior and
nearest upcoming macroeconomic event.

KEY CONCEPTS IN QUANTITATIVE DATA ENGINEERING:
1. Point-in-Time Temporal Joins: Standard SQL INNER/LEFT JOIN on exact timestamps
   fails because trades occur at discrete seconds/minutes while macro events occur
   intermittently. A point-in-time join (`merge_asof` backward) ensures a trade
   is matched strictly to the most recent information that was already known
   prior to order entry.
2. Avoiding Lookahead Bias: When calculating `minutes_since_prior_event`, the event
   timestamp must be strictly <= trade timestamp. When calculating `minutes_to_next_event`,
   the trade timestamp must be < event timestamp. The model may know an event is scheduled
   (calendar schedule is public knowledge), but CANNOT know the outcome until release.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


DB_PATH = "data/macro.db"


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates a connection to SQLite database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


def load_raw_data_to_tables(
    trades_path: str = "data/raw/synthetic_trades.csv",
    events_path: str = "data/processed/macro_calendar.csv",
    news_path: str = "data/processed/macro_news.csv",
    db_path: str = DB_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads CSV data, writes to SQLite tables, and returns dataframes."""
    trades_df = pd.read_csv(trades_path)
    events_df = pd.read_csv(events_path)
    news_df = pd.read_csv(news_path)

    # Convert timestamps to standard ISO format
    trades_df["timestamp_utc"] = pd.to_datetime(trades_df["timestamp_utc"], utc=True)
    events_df["event_time_utc"] = pd.to_datetime(events_df["event_time_utc"], utc=True)
    news_df["date"] = pd.to_datetime(news_df["date"], utc=True)

    # Calculate raw event surprise
    events_df["surprise"] = events_df["actual"] - events_df["forecast"]

    conn = get_db_connection(db_path)
    try:
        # Write clean tables to SQLite
        trades_df.to_sql("trades", conn, if_exists="replace", index=False)
        events_df.to_sql("events", conn, if_exists="replace", index=False)
        news_df.to_sql("news", conn, if_exists="replace", index=False)

        # Create indexes for fast temporal lookups
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(timestamp_utc);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time_utc);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_time ON news(date);")
        conn.commit()
        print(f"[build_db] Written tables 'trades' ({len(trades_df)}), 'events' ({len(events_df)}), 'news' ({len(news_df)}) to {db_path}")
    finally:
        conn.close()

    return trades_df, events_df, news_df


def perform_temporal_merge_asof(
    trades_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Joins each trade to latest prior macro event and nearest upcoming event using merge_asof.

    Args:
        trades_df: Executed trades with 'timestamp_utc'.
        events_df: Macro events with 'event_time_utc', 'event_name', 'surprise'.

    Returns:
        pd.DataFrame: Trades enriched with prior and next macro event features.
    """
    trades_sorted = trades_df.sort_values("timestamp_utc").copy()
    events_sorted = events_df.sort_values("event_time_utc").copy()

    # 1. Backward merge: Latest PRIOR macro event
    prior_events = events_sorted[[
        "event_time_utc", "event_name", "actual", "forecast", "previous", "surprise"
    ]].rename(columns={
        "event_time_utc": "prior_event_time",
        "event_name": "prior_event_name",
        "actual": "prior_actual",
        "forecast": "prior_forecast",
        "previous": "prior_previous",
        "surprise": "prior_surprise",
    })

    merged_prior = pd.merge_asof(
        trades_sorted,
        prior_events,
        left_on="timestamp_utc",
        right_on="prior_event_time",
        direction="backward",
        allow_exact_matches=True,
    )

    # Compute minutes elapsed since prior event
    time_diff_prior = (merged_prior["timestamp_utc"] - merged_prior["prior_event_time"]).dt.total_seconds() / 60.0
    merged_prior["minutes_since_last_event"] = time_diff_prior.fillna(99999.0)

    # 2. Forward merge: Nearest NEXT macro event
    next_events = events_sorted[[
        "event_time_utc", "event_name"
    ]].rename(columns={
        "event_time_utc": "next_event_time",
        "event_name": "next_event_name",
    })

    merged_all = pd.merge_asof(
        merged_prior,
        next_events,
        left_on="timestamp_utc",
        right_on="next_event_time",
        direction="forward",
        allow_exact_matches=False,
    )

    # Compute minutes until next event
    time_diff_next = (merged_all["next_event_time"] - merged_all["timestamp_utc"]).dt.total_seconds() / 60.0
    merged_all["minutes_to_next_event"] = time_diff_next.fillna(99999.0)

    # Flag if trade occurred on an event day (+/- 12 hours)
    merged_all["is_event_day"] = (
        (merged_all["minutes_since_last_event"] <= 720.0) |
        (merged_all["minutes_to_next_event"] <= 720.0)
    ).astype(int)

    return merged_all


def build_and_save_enriched_trades(db_path: str = DB_PATH) -> pd.DataFrame:
    """Builds the point-in-time aligned trade dataset and stores it in SQLite."""
    trades_df, events_df, news_df = load_raw_data_to_tables(db_path=db_path)
    enriched_trades = perform_temporal_merge_asof(trades_df, events_df)

    conn = get_db_connection(db_path)
    try:
        # Convert datetimes to strings for SQLite persistence
        save_df = enriched_trades.copy()
        for col in ["timestamp_utc", "prior_event_time", "next_event_time"]:
            if col in save_df.columns:
                save_df[col] = save_df[col].astype(str)

        save_df.to_sql("trades_aligned", conn, if_exists="replace", index=False)
        print(f"[build_db] Created 'trades_aligned' table with {len(save_df)} records in {db_path}")
    finally:
        conn.close()

    return enriched_trades


if __name__ == "__main__":
    df = build_and_save_enriched_trades()
    print("\nSample Aligned Trade Records:")
    sample_cols = [
        "trade_id", "timestamp_utc", "symbol", "prior_event_name",
        "minutes_since_last_event", "next_event_name", "minutes_to_next_event", "is_event_day"
    ]
    print(df[sample_cols].head())
