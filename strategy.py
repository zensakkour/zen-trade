import pandas as pd
import numpy as np

# =================== Tunables (balanced defaults) ===================
# 5m base EMAs
FAST_EMA_5M = 50
SLOW_EMA_5M = 200

# Higher TF via longer EMAs on 5m (12*5m≈1h, 48*5m≈4h)
HTF1_MULT = 12
HTF2_MULT = 48

# Trend strength & volatility regime
ADX_N = 14
ADX_MIN_5M = 14.0           # relaxed from 18
ADX_MIN_HTF = 12.0          # relaxed from 18

ATR_N = 14
MIN_ATR_PCT = 0.002         # 0.20% <= ATR% ...
MAX_ATR_PCT = 0.030         # ... <= 3.0%

# Donchian breakout (looser)
DONCHIAN_N = 40             # was 55
BREAKOUT_BUFFER_BPS = 3     # 0.03% buffer (was 0.07%)

# Optional squeeze logic (OFF by default)
USE_SQUEEZE = False
BB_N = 20
BB_SQZ_MAX = 0.008          # 0.8%
SQZ_LOOKBACK = 30
BW_EXP_LOOKBACK = 10

# Volume & momentum
USE_VOLUME_FILTER = True
VOL_N = 50
VOL_Z_MIN = -0.2            # allow slightly-below-average volume too

USE_RSI_GUARD = False
RSI_N = 14
RSI_LONG_MIN, RSI_LONG_MAX = 40.0, 75.0
RSI_SHORT_MIN, RSI_SHORT_MAX = 25.0, 60.0

# Cooldown
COOLDOWN_BARS = 10          # a bit shorter to allow more attempts

# =================== Helpers ===================
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _wilder_ewm(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0/n, adjust=False).mean()

def _atr_percent(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = _wilder_ewm(tr, n)
    return (atr / df["close"]).fillna(0.0)

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
    adx = _wilder_ewm(dx.fillna(0.0), n)
    return adx.fillna(0.0)

def _boll_bandwidth(df: pd.DataFrame, n: int) -> pd.Series:
    mid = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std(ddof=0)
    upper = mid + 2.0 * sd
    lower = mid - 2.0 * sd
    bw = (upper - lower) / mid.replace(0, np.nan)
    return bw.bfill().fillna(0.0)  # avoids deprecated fillna(method=...)

def _donchian(df: pd.DataFrame, n: int):
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()

def _volume_z(df: pd.DataFrame, n: int) -> pd.Series:
    v = df["volume"]
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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

# =================== Main API ===================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # EMAs (5m)
    out["ema_fast"] = _ema(out["close"], FAST_EMA_5M)
    out["ema_slow"] = _ema(out["close"], SLOW_EMA_5M)

    # Higher TF alignment via longer EMAs on 5m
    out["ema_fast_h1"] = _ema(out["close"], FAST_EMA_5M * HTF1_MULT)  # ≈ 1h
    out["ema_slow_h1"] = _ema(out["close"], SLOW_EMA_5M * HTF1_MULT)
    out["ema_fast_h4"] = _ema(out["close"], FAST_EMA_5M * HTF2_MULT)  # ≈ 4h
    out["ema_slow_h4"] = _ema(out["close"], SLOW_EMA_5M * HTF2_MULT)

    # Trend strength & volatility
    out["adx_5m"] = _adx(out, ADX_N)
    out["adx_h1_proxy"] = _adx(out, ADX_N * HTF1_MULT)  # proxy on 5m series
    out["atr_pct"] = _atr_percent(out, ATR_N)

    # Donchian bands
    out["dc_high"], out["dc_low"] = _donchian(out, DONCHIAN_N)

    # Optional squeeze metrics
    if USE_SQUEEZE:
        bw = _boll_bandwidth(out, BB_N)
        out["bb_bw"] = bw
        out["bb_bw_min_recent"] = bw.rolling(SQZ_LOOKBACK).min()
        out["bb_bw_mean_recent"] = bw.rolling(BW_EXP_LOOKBACK).mean()

    # Volume & RSI
    if USE_VOLUME_FILTER:
        out["vol_z"] = _volume_z(out, VOL_N)
    if USE_RSI_GUARD:
        out["rsi"] = _rsi(out["close"], RSI_N)

    return out

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(df).dropna().reset_index(drop=True)

    # Trend alignment (multi-TF)
    up_5m  = out["ema_fast"] > out["ema_slow"]
    dn_5m  = out["ema_fast"] < out["ema_slow"]
    up_h1  = out["ema_fast_h1"] > out["ema_slow_h1"]
    dn_h1  = out["ema_fast_h1"] < out["ema_slow_h1"]
    up_h4  = out["ema_fast_h4"] > out["ema_slow_h4"]
    dn_h4  = out["ema_fast_h4"] < out["ema_slow_h4"]

    # Strength + regime
    adx_ok = (out["adx_5m"] >= ADX_MIN_5M) & (out["adx_h1_proxy"] >= ADX_MIN_HTF)
    atr_ok = (out["atr_pct"] >= MIN_ATR_PCT) & (out["atr_pct"] <= MAX_ATR_PCT)

    # Optional squeeze → expansion
    if USE_SQUEEZE:
        sqz_ok = out["bb_bw_min_recent"] <= BB_SQZ_MAX
        bw_expand = out["bb_bw"] > out["bb_bw_mean_recent"]
        squeeze_filter = sqz_ok & bw_expand
    else:
        squeeze_filter = pd.Series(True, index=out.index)

    # Volume
    if USE_VOLUME_FILTER:
        vol_ok = out["vol_z"] >= VOL_Z_MIN
    else:
        vol_ok = pd.Series(True, index=out.index)

    # RSI guard
    if USE_RSI_GUARD:
        rsi_long_ok  = (out["rsi"] >= RSI_LONG_MIN) & (out["rsi"] <= RSI_LONG_MAX)
        rsi_short_ok = (out["rsi"] >= RSI_SHORT_MIN) & (out["rsi"] <= RSI_SHORT_MAX)
    else:
        rsi_long_ok = rsi_short_ok = pd.Series(True, index=out.index)

    # Donchian breakout with small buffer
    buf = BREAKOUT_BUFFER_BPS / 10000.0
    dc_high_prev = out["dc_high"].shift(1)
    dc_low_prev  = out["dc_low"].shift(1)
    close = out["close"]

    breakout_up = close > (dc_high_prev * (1.0 + buf))
    breakout_dn = close < (dc_low_prev  * (1.0 - buf))

    # Combine now-conditions
    long_now = (
        up_5m & up_h1 & up_h4 &
        adx_ok & atr_ok & squeeze_filter & vol_ok & rsi_long_ok &
        breakout_up
    )
    short_now = (
        dn_5m & dn_h1 & dn_h4 &
        adx_ok & atr_ok & squeeze_filter & vol_ok & rsi_short_ok &
        breakout_dn
    )

    # Transitions only + cooldown
    long_prev  = long_now.shift(1, fill_value=False)
    short_prev = short_now.shift(1, fill_value=False)
    raw_long   = long_now  & (~long_prev)  & (~short_prev)
    raw_short  = short_now & (~short_prev) & (~long_prev)

    any_sig = (raw_long | raw_short).astype(int)
    recent  = any_sig.shift(1).rolling(COOLDOWN_BARS).max().fillna(0).astype(bool)

    out["entry_long"]  = raw_long  & (~recent)
    out["entry_short"] = raw_short & (~recent)
    return out
