"""Synthetic Trading Log Generator.

This module generates deterministic, realistic synthetic trade execution logs
across multiple strategy parameter permutations for an ~180-day period.

KEY CONCEPTS IN QUANTITATIVE STRATEGY SIMULATION:
1. Permutations: Each permutation represents a unique combination of hyperparameters
   (e.g., lookback window, stop-loss %, take-profit %). In production quant research,
   hyperparameters should not be tuned on the full sample (to prevent overfitting),
   but rather evaluated across distinct market regimes.
2. Market Regimes: Regimes (e.g., high volatility during FOMC/CPI vs. calm trending)
   fundamentally shift the efficacy of tight vs. wide stops. We simulate this dynamic
   to give the machine learning model genuine signal to learn from.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def get_default_parameter_grid() -> List[Dict[str, Any]]:
    """Generates a structured grid of strategy parameters.

    Returns:
        List of dicts representing 27 unique parameter permutations.
    """
    lookbacks = [10, 20, 50]         # Trend lookback in bars/days
    stop_losses = [0.01, 0.02, 0.03]  # Stop loss (1%, 2%, 3%)
    take_profits = [0.02, 0.04, 0.06] # Take profit (2%, 4%, 6%)

    grid = []
    perm_counter = 1
    for lb, sl, tp in itertools.product(lookbacks, stop_losses, take_profits):
        grid.append({
            "permutation_id": f"PERM_{perm_counter:02d}",
            "lookback": lb,
            "stop_loss": sl,
            "take_profit": tp,
        })
        perm_counter += 1
    return grid


def generate_synthetic_trades(
    output_path: str = "data/raw/synthetic_trades.csv",
    start_date: str = "2026-04-01",
    days: int = 180,
    seed: int = 42,
    symbols: List[str] | None = None,
) -> pd.DataFrame:
    """Generates deterministic synthetic trades across strategy permutations.

    Args:
        output_path: Target path to save CSV.
        start_date: Beginning ISO date string.
        days: Number of days to simulate (~180 days).
        seed: Random seed for deterministic reproducibility.
        symbols: List of ticker symbols traded.

    Returns:
        pd.DataFrame containing simulated trade executions.
    """
    if symbols is None:
        symbols = ["SPY", "QQQ", "TLT"]

    np.random.seed(seed)
    param_grid = get_default_parameter_grid()
    
    start_dt = pd.to_datetime(start_date, utc=True)
    all_trades: List[Dict[str, Any]] = []
    trade_id_counter = 100001

    base_prices = {"SPY": 510.0, "QQQ": 440.0, "TLT": 92.0}

    # Simulate day-by-day market drift and volatility regime
    # Days with higher index have varying regime (simulating macro cycles)
    for day_offset in range(days):
        current_day = start_dt + pd.Timedelta(days=day_offset)
        
        # Skip weekends (Saturday=5, Sunday=6)
        if current_day.weekday() >= 5:
            continue

        # Macro regime factor: cyclical volatility oscillation
        # Creates periods where tight stops get stopped out vs trending days
        regime_cycle = np.sin(2 * np.pi * day_offset / 45.0)
        daily_volatility_multiplier = 1.0 + 0.6 * max(0.0, regime_cycle)

        # Iterate through permutations
        for perm in param_grid:
            # Probability of trade triggering for this permutation on this day
            # Shorter lookback triggers trades more frequently
            trade_prob = 0.55 if perm["lookback"] == 10 else (0.40 if perm["lookback"] == 20 else 0.25)
            
            if np.random.rand() > trade_prob:
                continue

            symbol = np.random.choice(symbols)
            base_p = base_prices[symbol] * (1.0 + 0.0005 * day_offset + 0.01 * np.random.randn())
            side = "BUY" if np.random.rand() > 0.45 else "SELL"
            
            # Entry time between 09:30 and 15:30 EST (13:30 to 19:30 UTC)
            entry_hour = np.random.randint(13, 19)
            entry_minute = np.random.randint(0, 60)
            entry_time = current_day.replace(hour=entry_hour, minute=entry_minute, second=0)

            # Trade duration: 15 mins to 240 mins
            duration_minutes = np.random.randint(15, 240)
            exit_time = entry_time + pd.Timedelta(minutes=duration_minutes)

            entry_price = round(base_p, 2)
            
            # PnL logic grounded in strategy parameters:
            # If high volatility regime, tight stops (sl=0.01) have higher probability of hitting stop-loss.
            # Wide stops (sl=0.03) survive turbulence and allow take-profit to be reached.
            sl = perm["stop_loss"]
            tp = perm["take_profit"]

            outcome_roll = np.random.rand()
            stopped_out_threshold = 0.45 * daily_volatility_multiplier * (0.03 / sl)
            stopped_out_threshold = min(0.85, max(0.15, stopped_out_threshold))

            if outcome_roll < stopped_out_threshold:
                # Stopped out
                return_pct = -sl * np.random.uniform(0.95, 1.05)
            elif outcome_roll < 0.90:
                # Take profit hit or favorable exit
                return_pct = tp * np.random.uniform(0.80, 1.05)
            else:
                # Modest flat close
                return_pct = np.random.uniform(-0.005, 0.005)

            if side == "SELL":
                return_pct = -return_pct  # Invert return direction for short

            exit_price = round(entry_price * (1.0 + return_pct if side == "BUY" else 1.0 - return_pct), 2)
            dollar_pnl = round((exit_price - entry_price if side == "BUY" else entry_price - exit_price) * 100, 2)

            all_trades.append({
                "trade_id": f"TRD_{trade_id_counter}",
                "timestamp_utc": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_timestamp_utc": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "side": side,
                "entry": entry_price,
                "exit": exit_price,
                "pnl": dollar_pnl,
                "return_pct": round(return_pct, 4),
                "lookback": perm["lookback"],
                "stop_loss": perm["stop_loss"],
                "take_profit": perm["take_profit"],
                "permutation_id": perm["permutation_id"],
                "is_synthetic": 1,
            })
            trade_id_counter += 1

    df = pd.DataFrame(all_trades)
    
    # Ensure target directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[synthetic_trades] Generated {len(df)} synthetic trades across {len(param_grid)} permutations.")
    print(f"[synthetic_trades] Saved to: {output_path}")
    return df


def ensure_trades_available(raw_dir: str = "data/raw") -> str:
    """Checks for existing user trades, else generates synthetic trades.

    Returns:
        Path to the active trades CSV file.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # Check for any existing non-synthetic CSV files provided by user
    existing_files = [f for f in raw_path.glob("*.csv") if f.name != "synthetic_trades.csv"]
    if existing_files:
        chosen = str(existing_files[0])
        print(f"[trades] Found user-provided trade log: {chosen}")
        return chosen

    synthetic_file = raw_path / "synthetic_trades.csv"
    if not synthetic_file.exists():
        print("[trades] No trade log found. Generating clearly labeled synthetic trades dataset...")
        generate_synthetic_trades(str(synthetic_file))
    else:
        print(f"[trades] Using existing synthetic trades log: {synthetic_file}")
    return str(synthetic_file)


if __name__ == "__main__":
    ensure_trades_available()
