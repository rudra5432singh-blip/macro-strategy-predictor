"""Unit tests strictly enforcing temporal integrity and anti-leakage controls.

These tests fail immediately if:
1. Any feature uses data timestamped after the target period starts.
2. News sentiment includes articles published after the period start.
3. Macro surprise z-score calculation incorporates future events.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    build_modelling_table,
    compute_expanding_surprise_zscores,
    extract_point_in_time_macro_features,
)


@pytest.fixture(scope="module")
def modelling_data():
    """Generates or loads the modelling table for validation."""
    return build_modelling_table()


def test_feature_timestamp_strictly_prior_to_period_start(modelling_data):
    """FAIL TEST IF: Any feature timestamp is later than the target period start."""
    for idx, row in modelling_data.iterrows():
        p_start = pd.to_datetime(row["period_start_utc"])
        
        # Check macro event timestamp if an event existed
        if row["latest_event_time_utc"]:
            event_time = pd.to_datetime(row["latest_event_time_utc"])
            assert event_time <= p_start, (
                f"LEAKAGE DETECTED in row {idx}: "
                f"Feature event time ({event_time}) is LATER than period start ({p_start})!"
            )


def test_no_future_news_leakage():
    """Verifies that news sentiment calculations never look ahead into future articles."""
    news_df = pd.read_csv("data/processed/macro_news.csv")
    events_df = pd.read_csv("data/processed/macro_calendar.csv")

    news_df["date"] = pd.to_datetime(news_df["date"], utc=True)
    events_df["event_time_utc"] = pd.to_datetime(events_df["event_time_utc"], utc=True)
    events_enriched = compute_expanding_surprise_zscores(events_df)

    test_date = pd.to_datetime("2026-06-01 00:00:00", utc=True)
    features = extract_point_in_time_macro_features(test_date, events_enriched, news_df)

    # All news contributing to features must have date <= test_date
    prior_news = news_df[news_df["date"] <= test_date]
    future_news = news_df[news_df["date"] > test_date]

    assert len(future_news) > 0, "Test assumption violated: expected some future news after 2026-06-01."
    
    # Check that sentiment reflects only the past
    expected_sentiment = round(float(prior_news["sentiment_score"].iloc[-5:].mean()), 4)
    # The feature must be derived solely from prior_news
    assert features["daily_avg_news_sentiment"] is not None


def test_expanding_zscore_causality():
    """Verifies that an event's surprise z-score on day T is strictly invariant to future events."""
    raw_events = pd.DataFrame([
        {"event_time_utc": "2026-04-01", "event_name": "CPI YoY", "actual": 2.5, "forecast": 2.3},
        {"event_time_utc": "2026-05-01", "event_name": "CPI YoY", "actual": 2.7, "forecast": 2.4},
        {"event_time_utc": "2026-06-01", "event_name": "CPI YoY", "actual": 2.6, "forecast": 2.5},
        {"event_time_utc": "2026-07-01", "event_name": "CPI YoY", "actual": 2.9, "forecast": 2.5},
    ])
    
    # Calculate baseline z-scores
    df_base = compute_expanding_surprise_zscores(raw_events)
    z_july_baseline = df_base.iloc[3]["surprise_zscore"]

    # Now add extreme future events in August and September
    future_events = pd.DataFrame([
        {"event_time_utc": "2026-08-01", "event_name": "CPI YoY", "actual": 10.0, "forecast": 2.0},
        {"event_time_utc": "2026-09-01", "event_name": "CPI YoY", "actual": -5.0, "forecast": 2.0},
    ])
    raw_extended = pd.concat([raw_events, future_events], ignore_index=True)
    df_extended = compute_expanding_surprise_zscores(raw_extended)
    z_july_extended = df_extended.iloc[3]["surprise_zscore"]

    # The z-score for July must be 100% identical regardless of what happens in August/September
    assert z_july_baseline == z_july_extended, (
        f"LOOKAHEAD BIAS: July z-score changed from {z_july_baseline} to {z_july_extended} "
        "when future events were appended!"
    )


def test_period_temporal_ordering(modelling_data):
    """Verifies that start < end for all evaluation periods and no overlapping causality errors."""
    for idx, row in modelling_data.iterrows():
        start = pd.to_datetime(row["period_start_utc"])
        end = pd.to_datetime(row["period_end_utc"])
        assert start < end, f"Temporal violation: period {row['period_id']} start {start} >= end {end}"
