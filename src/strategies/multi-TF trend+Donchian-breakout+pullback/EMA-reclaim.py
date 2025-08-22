# strategy.py
# ------------------------------------------------------------
# Model-style strategy wrapper + backward-compatible function.
# Keeps the winning logic:
#  - Multi-TF EMA trend alignment (≈5m/1h/4h via longer spans)
#  - Donchian breakout with tiny buffer (trend expansion)
#  - Pullback -> EMA(5m fast) reclaim (trend continuation)
#  - Light ADX/ATR/Volume regime filters
#
# Exposes:
#  - class StrategyModel: .transform(df) -> df with entry_long/entry_short
#  - function generate_signals(df): calls StrategyModel for backward compatibility
# ------------------------------------------------------------

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# =================== Default Params (tuned from your good run) ===================
@dataclass
class StrategyParams:
    # 5m base EMAs
    fast_ema_5m: int = 50
    slow_ema_5m: int = 200

    # Higher TF via longer EMAs on 5m (12*5m≈1h, 48*5m≈4h)
    htf1_mult: int = 12
    htf2_mult: int = 48

    # Trend strength & volatility regime
    adx_n: int = 14
    adx_min_5m: float = 13.0
    adx_min_htf: float = 11.0

    atr_n: int = 14
    min_atr_pct: float = 0.002    # 0.20%
    max_atr_pct: float = 0.030    # 3.0%

    # --- Entry A: Donchian breakout (trend expansion) ---
    donchian_n: int = 40
    breakout_buffer_bps: float = 3.0  # 0.03%

    # --- Entry B: Pullback -> EMA reclaim (continuation) ---
    use_pullback: bool = True
    pullback_atr_mult: float = 0.6    # distance to fast EMA when reclaiming
    rsi_n: int = 14
    rsi_long_max: float = 72.0        # avoid blow-offs
    rsi_short_min: float = 28.0       # avoid capitulations

    # Volume screen (light)
    use_volume: bool = True
    vol_n: int = 50
    vol_z_min: float = -0.1           # allow slightly below-avg too

    # Cooldown between entries
    cooldown_bars: int = 8

    # Optional shorts toggle
    allow_shorts: bool = True

# =================== Helpers ===================

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _wilder_ewm(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0/n, adjust=False).mean()

def _atr_abs(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return _wilder_ewm(tr, n)

def _atr_pct_from_abs(atr_abs: pd.Series, close: pd.Series) -> pd.Series:
    return (atr_abs / close).fillna(0.0)

def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    ph, pl, pc = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - ph
    down_move = pl - low
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low).abs(),
        (high - pc).abs(),
        (low - pc).abs()
    ], axis=1).max(axis=1)

    tr_n = _wilder_ewm(tr, n)
    pos_dm_n = _wilder_ewm(pd.Series(pos_dm, index=df.index), n)
    neg_dm_n = _wilder_ewm(pd.Series(neg_dm, index=df.index), n)

    pos_di = 100.0 * (pos_dm_n / tr_n.replace(0, np.nan))
    neg_di = 100.0 * (neg_dm_n / tr_n.replace(0, np.nan))
    dx = 100.0 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
    return _wilder_ewm(dx.fillna(0.0), n).fillna(0.0)

def _donchian(df: pd.DataFrame, n: int):
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()

def _volume_z(v: pd.Series, n: int) -> pd.Series:
    mu = v.rolling(n).mean()
    sd = v.rolling(n).std(ddof=0)
    return ((v - mu) / sd.replace(0, np.nan)).fillna(0.0)

def _rsi(series: pd.Series, n: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_ewm(gain, n)
    avg_loss = _wilder_ewm(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

# =================== Model Strategy ===================

class StrategyModel:
    """
    Model-style wrapper. Usage:
        sm = StrategyModel()                 # or StrategyModel(custom_params)
        df_with_signals = sm.transform(df)   # adds entry_long/entry_short
    """
    def __init__(self, params: StrategyParams | None = None):
        self.params = params or StrategyParams()

    @property
    def name(self) -> str:
        return "DonchianBreakout+Pullback_v1"

    def get_params(self) -> dict:
        return asdict(self.params)

    # ---------- feature engineering ----------
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()

        # EMAs (5m)
        out["ema_fast"] = _ema(out["close"], p.fast_ema_5m)
        out["ema_slow"] = _ema(out["close"], p.slow_ema_5m)

        # Higher TF alignment via longer spans on the same series
        out["ema_fast_h1"] = _ema(out["close"], p.fast_ema_5m * p.htf1_mult)  # ≈ 1h
        out["ema_slow_h1"] = _ema(out["close"], p.slow_ema_5m * p.htf1_mult)
        out["ema_fast_h4"] = _ema(out["close"], p.fast_ema_5m * p.htf2_mult)  # ≈ 4h
        out["ema_slow_h4"] = _ema(out["close"], p.slow_ema_5m * p.htf2_mult)

        # Trend strength & volatility
        out["adx_5m"] = _adx(out, p.adx_n)
        out["adx_h1_proxy"] = _adx(out, p.adx_n * p.htf1_mult)
        out["atr_abs"] = _atr_abs(out, p.atr_n)
        out["atr_pct"] = _atr_pct_from_abs(out["atr_abs"], out["close"])

        # Donchian
        out["dc_high"], out["dc_low"] = _donchian(out, p.donchian_n)

        # Volume & RSI
        if p.use_volume:
            out["vol_z"] = _volume_z(out["volume"], p.vol_n)
        else:
            out["vol_z"] = 0.0
        out["rsi"] = _rsi(out["close"], p.rsi_n)

        return out

    # ---------- signal logic ----------
    def _compute_signals(self, feats: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = feats.dropna().reset_index(drop=True)

        # Multi-TF trend alignment
        up_5m  = out["ema_fast"] > out["ema_slow"]
        dn_5m  = out["ema_fast"] < out["ema_slow"]
        up_h1  = out["ema_fast_h1"] > out["ema_slow_h1"]
        dn_h1  = out["ema_fast_h1"] < out["ema_slow_h1"]
        up_h4  = out["ema_fast_h4"] > out["ema_slow_h4"]
        dn_h4  = out["ema_fast_h4"] < out["ema_slow_h4"]

        # Strength + regime
        adx_ok = (out["adx_5m"] >= p.adx_min_5m) & (out["adx_h1_proxy"] >= p.adx_min_htf)
        atr_ok = (out["atr_pct"] >= p.min_atr_pct) & (out["atr_pct"] <= p.max_atr_pct)
        vol_ok = (out["vol_z"] >= p.vol_z_min) if p.use_volume else pd.Series(True, index=out.index)

        close = out["close"]
        buf = p.breakout_buffer_bps / 10000.0
        dc_high_prev = out["dc_high"].shift(1)
        dc_low_prev  = out["dc_low"].shift(1)

        # Entry A: Donchian breakout
        brk_up = close > (dc_high_prev * (1.0 + buf))
        brk_dn = close < (dc_low_prev  * (1.0 - buf))
        long_breakout  = up_5m & up_h1 & up_h4 & adx_ok & atr_ok & vol_ok & brk_up
        short_breakout = dn_5m & dn_h1 & dn_h4 & adx_ok & atr_ok & vol_ok & brk_dn

        # Entry B: Pullback -> EMA reclaim
        if p.use_pullback:
            atr_abs  = out["atr_abs"]
            ema_fast = out["ema_fast"]
            prev_close = close.shift(1)

            long_reclaim = (
                up_5m & up_h1 & adx_ok & atr_ok & vol_ok &
                (prev_close < ema_fast.shift(1)) &    # prior below fast EMA
                (close > ema_fast) &                  # reclaim
                ((ema_fast - out["low"]).clip(lower=0.0) <= p.pullback_atr_mult * atr_abs) &
                (out["rsi"] <= p.rsi_long_max)
            )

            short_reclaim = (
                dn_5m & dn_h1 & adx_ok & atr_ok & vol_ok &
                (prev_close > ema_fast.shift(1)) &
                (close < ema_fast) &
                ((out["high"] - ema_fast).clip(lower=0.0) <= p.pullback_atr_mult * atr_abs) &
                (out["rsi"] >= p.rsi_short_min)
            )
        else:
            long_reclaim  = pd.Series(False, index=out.index)
            short_reclaim = pd.Series(False, index=out.index)

        # Combine modes
        long_now  = long_breakout  | long_reclaim
        short_now = short_breakout | (short_reclaim if p.allow_shorts else pd.Series(False, index=out.index))

        # Transitions + cooldown
        long_prev  = long_now.shift(1, fill_value=False)
        short_prev = short_now.shift(1, fill_value=False)
        raw_long   = long_now  & (~long_prev)  & (~short_prev)
        raw_short  = short_now & (~short_prev) & (~long_prev)

        any_sig = (raw_long | raw_short).astype(int)
        recent  = any_sig.shift(1).rolling(p.cooldown_bars).max().fillna(0).astype(bool)

        out["entry_long"]  = raw_long  & (~recent)
        out["entry_short"] = raw_short & (~recent)
        return out

    # ---------- public API ----------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with at least entry_long / entry_short boolean columns."""
        feats = self._add_indicators(df)
        return self._compute_signals(feats)

# =================== Backward compatibility ===================

def add_indicators(df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
    """Deprecated shim for older imports (not used by backtester)."""
    return StrategyModel()._add_indicators(df)

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """What run_backtest.py imports; delegates to StrategyModel."""
    return StrategyModel().transform(df)
