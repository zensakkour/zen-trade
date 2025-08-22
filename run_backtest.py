import argparse, os, json
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser as dateparser
from dotenv import load_dotenv

from config import DEFAULTS
from data_fetch import backfill_ohlcv
from strategy import generate_signals
from backtester import BacktestParams, run_backtest

load_dotenv()

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser(description="Run a 1-year backtest on Binance USDT-M perps.")
    ap.add_argument("--exchange", type=str, default=os.getenv("EXCHANGE", DEFAULTS["exchange"]))
    ap.add_argument("--symbol", type=str, default=os.getenv("SYMBOL", DEFAULTS["symbol"]))
    ap.add_argument("--timeframe", type=str, default=os.getenv("TIMEFRAME", DEFAULTS["timeframe"]))
    ap.add_argument("--start", type=str, default=os.getenv("START", DEFAULTS["start"]))
    ap.add_argument("--end", type=str, default=os.getenv("END", DEFAULTS["end"]))
    ap.add_argument("--equity", type=float, default=os.getenv("INITIAL_EQUITY", DEFAULTS["initial_equity"]))
    ap.add_argument("--risk-pct", type=float, default=os.getenv("RISK_PCT", DEFAULTS["risk_pct"]))
    ap.add_argument("--sl-pct", type=float, default=os.getenv("SL_PCT", DEFAULTS["sl_pct"]))
    ap.add_argument("--tp-pct", type=float, default=os.getenv("TP_PCT", DEFAULTS["tp_pct"]))
    ap.add_argument("--leverage", type=int, default=os.getenv("LEVERAGE", DEFAULTS["leverage"]))
    ap.add_argument("--margin-buffer", type=float, default=os.getenv("MARGIN_BUFFER", DEFAULTS["margin_buffer"]))
    ap.add_argument("--taker-fee", type=float, default=os.getenv("TAKER_FEE", DEFAULTS["taker_fee"]))
    ap.add_argument("--slippage-bps", type=float, default=os.getenv("SLIPPAGE_BPS", DEFAULTS["slippage_bps"]))
    ap.add_argument("--entry-on", type=str, default=os.getenv("ENTRY_ON", DEFAULTS["entry_on"]), choices=["next_open","close"])
    return ap.parse_args()

def main():
    args = parse_args()

    df = backfill_ohlcv(
        exchange_name=args.exchange, symbol=args.symbol, timeframe=args.timeframe,
        start=args.start, end=args.end, cache=True
    )
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
