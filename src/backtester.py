from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import math

@dataclass
class BacktestParams:
    initial_equity: float = 14000.0
    risk_pct: float = 0.003
    sl_pct: float = 0.006
    tp_pct: float = 0.010
    leverage: int = 5
    margin_buffer: float = 0.90
    taker_fee: float = 0.0005
    slippage_bps: float = 1.0
    entry_on: str = "next_open"

def position_size(equity: float, entry: float, stop: float, risk_pct: float, leverage: int) -> float:
    risk_usd = equity * risk_pct
    stop_dist = abs(entry - stop)
    if stop_dist <= 0 or entry <= 0:
        return 0.0
    qty = (risk_usd * leverage) / stop_dist
    return max(qty, 0.0)

def cap_by_margin(qty: float, equity: float, entry: float, leverage: int, margin_buffer: float) -> float:
    if leverage <= 0:
        return 0.0
    max_notional = equity * leverage * margin_buffer
    max_qty = max_notional / max(1e-12, entry)
    return min(qty, max_qty)

def _conservative_exit_long(high: float, low: float, stop: float, tp: float):
    stop_hit = low <= stop
    tp_hit = high >= tp
    if stop_hit and tp_hit: return True, stop, "SL"
    if stop_hit: return True, stop, "SL"
    if tp_hit: return True, tp, "TP"
    return False, math.nan, ""

def _conservative_exit_short(high: float, low: float, stop: float, tp: float):
    stop_hit = high >= stop
    tp_hit = low <= tp
    if stop_hit and tp_hit: return True, stop, "SL"
    if stop_hit: return True, stop, "SL"
    if tp_hit: return True, tp, "TP"
    return False, math.nan, ""

def run_backtest(df: pd.DataFrame, params: BacktestParams) -> Dict[str, Any]:
    cols_needed = {"ts","open","high","low","close","entry_long","entry_short"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    ts = df["ts"].to_numpy()
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    el = df["entry_long"].to_numpy(dtype=bool)
    es = df["entry_short"].to_numpy(dtype=bool)

    equity = params.initial_equity
    in_pos = False
    side = ""
    qty = 0.0
    entry_price = 0.0
    stop = 0.0
    tp = 0.0

    slippage = params.slippage_bps / 10000.0

    trades: List[Dict[str, Any]] = []
    equity_curve_t: List[pd.Timestamp] = []
    equity_curve_v: List[float] = []

    i = 1
    n = len(df)
    while i < n - 1:
        equity_curve_t.append(pd.Timestamp(ts[i]))
        equity_curve_v.append(equity)

        if not in_pos:
            if el[i]:
                side = "long"
            elif es[i]:
                side = "short"
            else:
                i += 1
                continue

            if params.entry_on == "next_open":
                if i+1 >= n: break
                raw_entry = o[i+1]; i_entry = i+1
            else:
                raw_entry = c[i]; i_entry = i

            if side == "long":
                stop = raw_entry * (1.0 - params.sl_pct)
                tp   = raw_entry * (1.0 + params.tp_pct)
                entry_price = raw_entry * (1.0 + slippage)
            else:
                stop = raw_entry * (1.0 + params.sl_pct)
                tp   = raw_entry * (1.0 - params.tp_pct)
                entry_price = raw_entry * (1.0 - slippage)

            qty_raw = position_size(equity, entry_price, stop, params.risk_pct, params.leverage)
            qty = cap_by_margin(qty_raw, equity, entry_price, params.leverage, params.margin_buffer)

            if qty <= 0.0:
                side = ""; i = i_entry; i += 1; continue

            fee_entry = params.taker_fee * qty * entry_price
            equity -= fee_entry

            in_pos = True
            i = i_entry + 1
            continue

        else:
            exited = False
            j = i
            while j < n:
                if side == "long":
                    exited, exit_price, reason = _conservative_exit_long(h[j], l[j], stop, tp)
                    if exited:
                        exit_price_adj = exit_price * (1.0 - slippage)
                        fee_exit = params.taker_fee * qty * exit_price_adj
                        pnl = qty * (exit_price_adj - entry_price) - fee_exit
                        equity += pnl
                        trades.append({"entry_ts": pd.Timestamp(ts[j]), "exit_ts": pd.Timestamp(ts[j]),
                                       "side": side, "entry_price": entry_price, "exit_price": exit_price_adj,
                                       "qty": qty, "reason": reason, "pnl_usd": pnl})
                        in_pos = False; side = ""; qty = 0.0; entry_price = 0.0; stop = tp = 0.0
                        equity_curve_t.append(pd.Timestamp(ts[j])); equity_curve_v.append(equity)
                        j += 1; i = j; break
                else:
                    exited, exit_price, reason = _conservative_exit_short(h[j], l[j], stop, tp)
                    if exited:
                        exit_price_adj = exit_price * (1.0 + slippage)
                        fee_exit = params.taker_fee * qty * exit_price_adj
                        pnl = qty * (entry_price - exit_price_adj) - fee_exit
                        equity += pnl
                        trades.append({"entry_ts": pd.Timestamp(ts[j]), "exit_ts": pd.Timestamp(ts[j]),
                                       "side": side, "entry_price": entry_price, "exit_price": exit_price_adj,
                                       "qty": qty, "reason": reason, "pnl_usd": pnl})
                        in_pos = False; side = ""; qty = 0.0; entry_price = 0.0; stop = tp = 0.0
                        equity_curve_t.append(pd.Timestamp(ts[j])); equity_curve_v.append(equity)
                        j += 1; i = j; break
                j += 1

            if not exited:
                j = n - 1
                if side == "long":
                    last_px = c[j] * (1.0 - slippage)
                    fee_exit = params.taker_fee * qty * last_px
                    pnl = qty * (last_px - entry_price) - fee_exit
                else:
                    last_px = c[j] * (1.0 + slippage)
                    fee_exit = params.taker_fee * qty * last_px
                    pnl = qty * (entry_price - last_px) - fee_exit
                equity += pnl
                trades.append({"entry_ts": pd.Timestamp(ts[j]), "exit_ts": pd.Timestamp(ts[j]),
                               "side": side, "entry_price": entry_price, "exit_price": last_px,
                               "qty": qty, "reason": "EOD", "pnl_usd": pnl})
                in_pos = False; side = ""; qty = 0.0; entry_price = 0.0; stop = tp = 0.0
                equity_curve_t.append(pd.Timestamp(ts[j])); equity_curve_v.append(equity)
                i = n

    eq = pd.DataFrame({"ts": equity_curve_t, "equity": equity_curve_v}).drop_duplicates(subset=["ts"]).reset_index(drop=True)
    tr = pd.DataFrame(trades)
    metrics = compute_metrics(eq)
    return {"trades": tr, "equity_curve": eq, "metrics": metrics}

def compute_metrics(eq_df: pd.DataFrame) -> dict:
    if len(eq_df) < 2:
        return {"total_return_pct": 0.0, "cagr_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0}

    equity = eq_df["equity"].to_numpy(dtype=float)
    ts = pd.to_datetime(eq_df["ts"])

    total_return = equity[-1] / equity[0] - 1.0
    days = (ts.iloc[-1] - ts.iloc[0]).days / 365.25
    cagr = (equity[-1] / equity[0]) ** (1.0 / max(days, 1e-9)) - 1.0 if days > 0 else total_return

    peak = -np.inf
    dd = []
    for v in equity:
        if v > peak: peak = v
        dd.append((v/peak) - 1.0 if peak > 0 else 0.0)
    max_dd = min(dd) if dd else 0.0

    rets = np.diff(np.log(np.maximum(1e-12, equity)))
    if len(rets) > 1:
        dt = np.diff(ts.values).astype("timedelta64[s]").astype(float)
        median_secs = np.median(dt) if len(dt) else 60.0
        steps_per_year = (365.25*24*3600) / max(median_secs, 1.0)
        mu = np.mean(rets)
        sd = np.std(rets, ddof=1) if np.std(rets) > 0 else 0.0
        sharpe = (mu / sd) * np.sqrt(steps_per_year) if sd > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_return_pct": float(total_return*100.0),
        "cagr_pct": float(cagr*100.0),
        "max_drawdown_pct": float(max_dd*100.0),
        "sharpe": float(sharpe),
        "start_equity": float(equity[0]),
        "end_equity": float(equity[-1]),
    }
