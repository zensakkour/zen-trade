import pandas as pd
from .base import BaseStrategy

class SimpleMAStrategy(BaseStrategy):
    def __init__(self, params: dict):
        super().__init__(params)
        self.fast = self.params.get("fast", 20)
        self.slow = self.params.get("slow", 60)
        self.slope_window = self.params.get("slope_window", 5)

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ma_fast"] = out["close"].rolling(self.fast).mean()
        out["ma_slow"] = out["close"].rolling(self.slow).mean()
        out["slope"]   = out["ma_slow"].diff(self.slope_window)
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self._add_indicators(out)
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
