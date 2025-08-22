import os, time
from datetime import datetime, timedelta, timezone
import pandas as pd
import ccxt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _parse_date(d):
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc)
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def backfill_ohlcv(
    exchange_name="binanceusdm", symbol="BTC/USDT", timeframe="5m",
    start="2024-08-22", end="2025-08-22", cache=True, verbose=True
):
    os.makedirs(DATA_DIR, exist_ok=True)

    ex_class = getattr(ccxt, exchange_name)
    ex = ex_class({"enableRateLimit": True})
    if exchange_name == "bybit":
        ex.options["defaultType"] = "swap"
    ex.load_markets()

    sdt = _parse_date(start)
    edt = _parse_date(end)

    sym = symbol.replace("/", "")
    fname = f"{sym}-{timeframe}-{sdt.date()}_{edt.date()}.csv"
    fpath = os.path.join(DATA_DIR, fname)

    if cache and os.path.exists(fpath):
        if verbose: print(f"[cache] Using cached {fpath}")
        df = pd.read_csv(fpath)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    limit = 1000
    since_ms = int(sdt.timestamp()*1000)
    until_ms = int(edt.timestamp()*1000)
    out = []

    while True:
        batch = ex.fetch_ohlcv(sym, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        out += batch
        last_open = batch[-1][0]
        next_ms = last_open + 1
        if next_ms >= until_ms:
            break
        since_ms = next_ms
        time.sleep(0.2)

    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[(df["ts"] >= sdt) & (df["ts"] <= edt)].reset_index(drop=True)

    if cache:
        df.to_csv(fpath, index=False)
        if verbose: print(f"[cache] Saved {fpath} ({len(df)} rows)")
    return df
