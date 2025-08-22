import argparse, os, json
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser as dateparser

from config import DEFAULTS
from data_fetch import backfill_ohlcv
from strategy import generate_signals
from backtester import BacktestParams, run_backtest

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser(description="Run a 1-year backtest on Binance USDT-M perps.")
    ap.add_argument("--symbol", type=str, default=DEFAULTS["symbol"])
    ap.add_argument("--timeframe", type=str, default=DEFAULTS["timeframe"])
    ap.add_argument("--start", type=str, default=DEFAULTS["start"])
    ap.add_argument("--end", type=str, default=DEFAULTS["end"])
    ap.add_argument("--equity", type=float, default=DEFAULTS["initial_equity"])
    ap.add_argument("--risk-pct", type=float, default=DEFAULTS["risk_pct"])
    ap.add_argument("--sl-pct", type=float, default=DEFAULTS["sl_pct"])
    ap.add_argument("--tp-pct", type=float, default=DEFAULTS["tp_pct"])
    ap.add_argument("--leverage", type=int, default=DEFAULTS["leverage"])
    ap.add_argument("--margin-buffer", type=float, default=DEFAULTS["margin_buffer"])
    ap.add_argument("--taker-fee", type=float, default=DEFAULTS["taker_fee"])
    ap.add_argument("--slippage-bps", type=float, default=DEFAULTS["slippage_bps"])
    ap.add_argument("--entry-on", type=str, default=DEFAULTS["entry_on"], choices=["next_open","close"])
    return ap.parse_args()

def main():
    args = parse_args()

    df = backfill_ohlcv(symbol=args.symbol, timeframe=args.timeframe, start=args.start, end=args.end, cache=True)
    if len(df) == 0:
        print("No data. Check symbol/timeframe/date range."); return

    sdf = generate_signals(df)

    params = BacktestParams(
        initial_equity=args.equity, risk_pct=args.risk_pct,
        sl_pct=args.sl_pct, tp_pct=args.tp_pct,
        leverage=args.leverage, margin_buffer=args.margin_buffer,
        taker_fee=args.taker_fee, slippage_bps=args.slippage_bps,
        entry_on=args.entry_on,
    )
    result = run_backtest(sdf, params)

    trades = result["trades"]; eq = result["equity_curve"]; metrics = result["metrics"]

    trades_path = os.path.join(OUT_DIR, "trades.csv")
    eq_path = os.path.join(OUT_DIR, "equity_curve.csv")
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    trades.to_csv(trades_path, index=False)
    eq.to_csv(eq_path, index=False)
    with open(metrics_path, "w") as f: json.dump(metrics, f, indent=2)

    plt.figure()
    plt.plot(eq["ts"], eq["equity"])
    plt.xlabel("Time"); plt.ylabel("Equity (USDT)"); plt.title("Equity Curve")
    fig_path = os.path.join(OUT_DIR, "equity_curve.png")
    plt.tight_layout(); plt.savefig(fig_path, dpi=160); plt.close()

    print(f"Saved trades  -> {trades_path}")
    print(f"Saved equity  -> {eq_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved plot    -> {fig_path}")
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
