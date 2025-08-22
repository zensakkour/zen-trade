import os, json
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser as dateparser
from dotenv import load_dotenv

from src.config import DEFAULTS
from src.data_fetch import backfill_ohlcv
import importlib
from src.backtester import BacktestParams, run_backtest

load_dotenv()

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    df = backfill_ohlcv(
        exchange_name=os.getenv("EXCHANGE"),
        symbol=os.getenv("SYMBOL"),
        timeframe=os.getenv("TIMEFRAME"),
        start=os.getenv("START"),
        end=os.getenv("END"),
        cache=True
    )
    if len(df) == 0:
        print("No data. Check symbol/timeframe/date range."); return

    strategy_name = os.getenv("STRATEGY")
    strategy_module = importlib.import_module(f"src.strategies.{strategy_name}")
    strategy_class = getattr(strategy_module, f"{strategy_name.replace('_', ' ').title().replace(' ', '')}Strategy")

    strategy_params = {
        "fast": int(os.getenv("SMA_FAST", 20)),
        "slow": int(os.getenv("SMA_SLOW", 60)),
        "slope_window": int(os.getenv("SMA_SLOPE_WINDOW", 5)),
    }
    strategy = strategy_class(strategy_params)
    sdf = strategy.generate_signals(df)

    params = BacktestParams(
        initial_equity=float(os.getenv("INITIAL_EQUITY")),
        risk_pct=float(os.getenv("RISK_PCT")),
        sl_pct=float(os.getenv("SL_PCT")),
        tp_pct=float(os.getenv("TP_PCT")),
        leverage=int(os.getenv("LEVERAGE")),
        margin_buffer=float(os.getenv("MARGIN_BUFFER")),
        taker_fee=float(os.getenv("TAKER_FEE")),
        slippage_bps=float(os.getenv("SLIPPAGE_BPS")),
        entry_on=os.getenv("ENTRY_ON"),
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
