import pandas as pd

def add_indicators(df: pd.DataFrame, fast=20, slow=60, slope_window=5) -> pd.DataFrame:
    out = df.copy()
    out["ma_fast"] = out["close"].rolling(fast).mean()
    out["ma_slow"] = out["close"].rolling(slow).mean()
    out["slope"]   = out["ma_slow"].diff(slope_window)
    return out

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_indicators(out)
    out = out.dropna().reset_index(drop=True)

    fast = out["ma_fast"]
    slow = out["ma_slow"]
    slope = out["slope"]

    above = fast > slow
    below = fast < slow

    long_now  = (above) & (slope > 0)
    short_now = (below) & (slope < 0)

    long_prev  = long_now.shift(1, fill_value=False)
    short_prev = short_now.shift(1, fill_value=False)

    out["entry_long"]  = long_now & (~long_prev) & (~short_prev)
    out["entry_short"] = short_now & (~short_prev) & (~long_prev)
    return out
